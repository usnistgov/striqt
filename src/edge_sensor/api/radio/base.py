from __future__ import annotations
import functools
from math import ceil
import typing

import labbench as lb
from labbench import paramattr as attr
import msgspec
import numpy as np

from .. import structs, util

if typing.TYPE_CHECKING:
    import iqwaveform
    import pandas as pd
else:
    iqwaveform = util.lazy_import('iqwaveform')
    pd = util.lazy_import('pandas')


class RadioDevice(lb.Device):
    stream = None

    analysis_bandwidth = attr.value.float(
        float('inf'),
        allow_none=False,
        min=1e-6,
        label='Hz',
        help='bandwidth of the digital bandpass filter (or None to bypass)',
    )

    calibration = attr.value.Path(
        None,
        help='path to a calibration file, or None to skip calibration',
    )

    duration = attr.value.float(
        10e-3, min=0, label='s', help='receive waveform capture duration'
    )

    periodic_trigger = attr.value.float(
        None,
        allow_none=True,
        help='if specified, acquisition start times will begin at even multiples of this',
    )

    continuous_trigger = attr.value.float(
        True,
        help='whether to trigger immediately after each call to acquire() when armed',
    )

    lo_offset = attr.value.float(
        0.0,
        label='Hz',
        help='digital frequency shift of the RX center frequency',
    )

    gapless_repeats = attr.value.bool(
        False, help='whether to skip stream disable->renable between identical captures'
    )

    time_sync_every_capture = attr.value.bool(
        False,
        help='whether to synchronize sample timestamps external PPS on each capture',
    )

    _downsample = attr.value.float(1.0, min=0, help='backend_sample_rate/sample_rate')

    # these must be implemented by child classes
    channel = attr.method.int(min=0)
    center_frequency = attr.method.float(
        min=0, label='Hz', help='RF frequency at the center of the RX baseband'
    )
    backend_sample_rate = attr.method.float(
        min=0,
        label='Hz',
        help='sample rate before resampling',
    )
    channel_enabled = attr.method.bool()
    gain = attr.method.float(label='dB', help='SDR hardware gain')
    time_source = attr.method.str(
        only=['host', 'internal', 'external', 'gps'],
        help='time base for sample timestamps',
    )

    transient_holdoff_time = attr.value.float(
        0.0, sets=False, label='s', help='holdoff time before valid data after enable'
    )

    @attr.property.str(sets=False, cache=True, help='unique radio hardware identifier')
    def id(self):
        raise NotImplementedError

    @property
    def base_clock_rate(self):
        return type(self).backend_sample_rate.max

    def open(self):
        self._armed_capture: structs.RadioCapture | None = None

    def setup(self, radio_config: structs.RadioSetup):
        """disarm acquisition and apply the given radio setup"""

        if self.channel() is not None:
            self.channel_enabled(False)

        self.calibration = radio_config.calibration
        self.periodic_trigger = radio_config.periodic_trigger
        self.gapless_repeats = radio_config.gapless_repeats
        self.time_sync_every_capture = radio_config.time_sync_every_capture
        self.time_source(radio_config.time_source)

        if not self.time_sync_every_capture:
            self.sync_time_source()

    def arm(self, capture: structs.RadioCapture):
        """stop the stream, apply a capture configuration, and start it"""

        with lb.stopwatch('arm', logger_level='debug'):
            if self.channel() is not None:
                self.channel_enabled(False)

            if iqwaveform.power_analysis.isroundmod(
                capture.duration * capture.sample_rate, 1
            ):
                self.duration = capture.duration
            else:
                raise ValueError(
                    f'duration {capture.duration} is not an integer multiple of sample period'
                )

            if capture.channel != self.channel():
                self.channel(capture.channel)
            else:
                self.channel_enabled(False)

            if self.gain() != capture.gain:
                self.gain(capture.gain)

            fs_backend, lo_offset, analysis_filter = design_capture_filter(
                self.base_clock_rate, capture
            )

            nfft_out = analysis_filter.get('nfft_out', analysis_filter['nfft'])
            downsample = analysis_filter['nfft'] / nfft_out

            if (
                fs_backend != self.backend_sample_rate()
                or downsample != self._downsample
            ):
                self.channel_enabled(False)
                with attr.hold_attr_notifications(self):
                    self._downsample = 1  # temporarily avoid a potential bounding error
                self.backend_sample_rate(fs_backend)
                self._downsample = downsample

            if capture.sample_rate != self.sample_rate():
                # in this case, it's only a post-processing (GPU resampling) change
                self.sample_rate(capture.sample_rate)

            if (
                self.periodic_trigger is not None
                and capture.duration < self.periodic_trigger
            ):
                self._logger.warning(
                    'periodic trigger duration exceeds capture duration, '
                    'which creates a large buffer of unused samples'
                )

            if lo_offset != self.lo_offset:
                self.lo_offset = lo_offset  # hold update on this one?

            if capture.center_frequency != self.center_frequency():
                self.channel_enabled(False)
                self.center_frequency(capture.center_frequency)

            self.analysis_bandwidth = capture.analysis_bandwidth

            if self.time_sync_every_capture:
                self.channel_enabled(False)
                self.sync_time_source()

            if not self.channel_enabled():
                self._holdover_samples = None

            self._armed_capture = capture
            self._next_time_ns = None

    def read_iq(self, capture) -> tuple['np.ndarray[np.complex64]', float]:
        streamed_count = 0
        awaiting_timestamp = True
        buf_time_ns = self._next_time_ns
        start_ns = self._next_time_ns

        buf_size = get_input_iq_buffer_size(self, capture, include_holdoff=True)
        sample_count = get_input_iq_buffer_size(self, capture, include_holdoff=False)

        samples = np.empty((2 * buf_size,), dtype=np.float32)

        if buf_time_ns is None and self._holdover_samples is not None:
            raise ValueError('holdover samples are missing timestamp')

        if self._holdover_samples is None:
            holdover_count = 0
        else:
            # note: holdover_count.dtype is np.complex64, samples.dtype is np.float32
            holdover_count = self._holdover_samples.size
            samples[: 2 * holdover_count] = self._holdover_samples.view(samples.dtype)

        holdover_size = sample_count - round(
            capture.duration * self.backend_sample_rate()
        )

        # default holdoffs parameters, valid when we already have a clock reading
        stft_pad, _ = _get_stft_padding(self.base_clock_rate, capture)
        holdoff_size = stft_pad

        fs = self.backend_sample_rate()
        chunk_size = sample_count + holdover_count
        timeout_sec = chunk_size / fs + 50e-3
        remaining = sample_count - holdover_count

        while remaining > 0:
            if streamed_count > 0 or self.gapless_repeats:
                on_overflow = 'except'
            else:
                on_overflow = 'ignore'

            request_count = min(chunk_size, remaining)

            if 2*(streamed_count + request_count) > samples.size:
                # this should never happen if samples are tracked and allocated properly
                raise MemoryError(f'about to request {request_count} samples, but buffer has capacity for only {samples.size//2 - streamed_count}')

            # Read the samples from the data buffer
            this_count, ret_time_ns = self._read_stream(
                [samples],
                offset=holdover_count + streamed_count,
                count=request_count,
                timeout_sec=timeout_sec,
                on_overflow=on_overflow,
            )

            if 2*(this_count + streamed_count) > samples.size:
                # this should never happen
                print(f'requested {min(chunk_size, remaining)} samples, but got {remaining}')
                raise MemoryError(f'overfilled receive buffer by {2*(this_count + streamed_count) - samples.size}')

            if buf_time_ns is None:
                # special case for the first read in the stream, since
                # devices may not always return timestamps
                buf_time_ns = ret_time_ns

            if awaiting_timestamp:
                holdoff_size = find_trigger_holdoff(self, buf_time_ns, stft_pad=stft_pad)
                remaining = remaining + holdoff_size

                start_ns = buf_time_ns + round(holdoff_size * 1e9 / fs)
                awaiting_timestamp = False

            remaining = remaining - this_count
            streamed_count += this_count

        sample_offs = holdoff_size - stft_pad
        samples = samples.view('complex64')[sample_offs: sample_offs + sample_count]

        self._holdover_samples = samples[-holdover_size:]
        self._next_time_ns = start_ns + round(1e9 * capture.duration)

        return samples, start_ns

    def acquire(
        self,
        capture: structs.RadioCapture,
        next_capture: typing.Union[structs.RadioCapture, None] = None,
        correction: bool = True,
    ) -> tuple[np.array, 'pd.Timestamp']:
        """arm a capture and enable the channel (if necessary), read the resulting IQ waveform.

        Optionally, calibration corrections can be applied, and the radio can be left ready for the next capture.
        """
        from .. import iq_corrections

        with lb.stopwatch('acquire', logger_level='debug'):
            if capture != self._armed_capture:
                self.arm(capture)

            if self.channel() is None or not self.channel_enabled():
                self.channel_enabled(True)

            iq, time_ns = self.read_iq(capture)
            time_ns = pd.Timestamp(time_ns, unit='ns')

            if next_capture == capture and self.gapless_repeats:
                # the one case where we leave it running
                pass
            else:
                self.channel_enabled(False)

            if next_capture is not None and next_capture != next_capture:
                self.arm(next_capture)

            if correction:
                with lb.stopwatch('resampling', logger_level='debug'):
                    iq = iq_corrections.resampling_correction(iq, capture, self)

            acquired_capture = structs.copy_struct(capture, start_time=time_ns)
            return iq, acquired_capture

    def _read_stream(
        self, buffers, offset, count, timeout_sec, *, on_overflow='except'
    ) -> tuple[int, int]:
        """to be implemented in subclasses"""
        raise NotImplementedError

    def sync_time_source(self):
        raise NotImplementedError

    def get_capture_struct(
        self, cls=structs.RadioCapture
    ) -> structs.RadioCapture | None:
        """generate the currently armed capture configuration for the specified channel"""

        if self.channel() is None:
            return None

        if self.lo_offset == 0:
            lo_shift = 'none'
        elif self.lo_offset < 0:
            lo_shift = 'left'
        elif self.lo_offset > 0:
            lo_shift = 'right'

        return cls(
            # RF and leveling
            center_frequency=self.center_frequency(),
            channel=self.channel(),
            gain=self.gain(),
            # acquisition
            duration=self.duration,
            sample_rate=self.sample_rate(),
            # filtering and resampling
            analysis_bandwidth=self.analysis_bandwidth,
            lo_shift=lo_shift,
        )


def find_trigger_holdoff(radio: RadioDevice, start_time_ns: int, stft_pad: int = 0):
    sample_rate = radio.backend_sample_rate()
    periodic_trigger = radio.periodic_trigger

    if periodic_trigger in (0, None):
        return 0

    periodic_trigger_ns = round(periodic_trigger * 1e9)

    # float rounding errors cause problems here; evaluate based on the 1-ns resolution
    excess_time_ns = start_time_ns % periodic_trigger_ns
    holdoff_ns = (periodic_trigger_ns - excess_time_ns) % periodic_trigger_ns

    holdoff = round(holdoff_ns / 1e9 * sample_rate)

    # wait long enough to meet the fulfill the radio's transient time and any
    # needed stft padding samples
    min_holdoff = round(radio.transient_holdoff_time * sample_rate) + stft_pad
    if holdoff < min_holdoff:
        periodic_trigger_samples = round(periodic_trigger * sample_rate)
        holdoff += (
            ceil(min_holdoff / periodic_trigger_samples) * periodic_trigger_samples
        )

    return holdoff


@functools.lru_cache(30000)
def _design_capture_filter(
    base_clock_rate: float,
    capture: structs.WaveformCapture,
    bw_lo=0.25e6,
    min_oversampling=1.1,
) -> tuple[float, float, dict]:
    """design a filter specified by the capture for a radio with the specified MCR.

    For the return value, see `iqwaveform.fourier.design_cola_resampler`
    """
    if str(capture.lo_shift).lower() == 'none':
        lo_shift = False
    else:
        lo_shift = capture.lo_shift

    if (
        capture.analysis_bandwidth != float('inf')
        and capture.analysis_bandwidth > capture.sample_rate
    ):
        raise ValueError(
            f'analysis bandwidth must be smaller than sample rate in {capture}'
        )

    if capture.host_resample:
        # use GPU DSP to resample from integer divisor of the MCR
        fs_sdr, lo_offset, kws = iqwaveform.fourier.design_cola_resampler(
            fs_base=base_clock_rate,
            fs_target=capture.sample_rate,
            bw=capture.analysis_bandwidth,
            bw_lo=bw_lo,
            shift=lo_shift,
            min_fft_size=4 * 4096 - 1,
            min_oversampling=min_oversampling,
        )

        return fs_sdr, lo_offset, kws

    elif lo_shift:
        raise ValueError('lo_shift requires host_resample=True')
    elif base_clock_rate < capture.sample_rate:
        raise ValueError(
            f'upsampling above {base_clock_rate/1e6:f} MHz requires host_resample=True'
        )
    else:
        # use the SDR firmware to implement the desired sample rate
        return iqwaveform.fourier.design_cola_resampler(
            fs_base=capture.sample_rate,
            fs_target=capture.sample_rate,
            bw=capture.analysis_bandwidth,
            shift=False,
        )


@functools.wraps(_design_capture_filter)
def design_capture_filter(
    base_clock_rate, capture: structs.WaveformCapture, *args, **kws
):
    # cast the struct in case it's a subclass
    fixed_capture = msgspec.convert(
        capture, structs.WaveformCapture, from_attributes=True
    )
    return _design_capture_filter(base_clock_rate, fixed_capture, *args, **kws)


def needs_stft(analysis_filter: dict, capture: structs.RadioCapture):
    is_resample = analysis_filter['nfft'] != analysis_filter['nfft_out']
    return analysis_filter and (
        np.isfinite(capture.analysis_bandwidth)
        or (is_resample and capture.host_resample)
    )


@functools.lru_cache(30000)
def _get_stft_padding(
    base_clock_rate: float, capture: structs.RadioCapture
) -> tuple[int, int]:
    """returns the padding before and after a waveform to achieve an integral number of FFT windows"""
    samples_out = round(capture.duration * capture.sample_rate)

    _, _, analysis_filter = design_capture_filter(base_clock_rate, capture)
    nfft = analysis_filter['nfft']

    min_samples_in = ceil(samples_out * nfft / analysis_filter['nfft_out'])

    # round up to an integral number of FFT windows
    samples_in = ceil(min_samples_in / nfft) * nfft + nfft

    if needs_stft(analysis_filter, capture):
        return nfft // 2, nfft // 2 + (samples_in - min_samples_in)
    else:
        return 0, 0


@functools.lru_cache(30000)
def _get_capture_buffer_sizes_cached(
    base_clock_rate: float,
    periodic_trigger: float | None,
    capture: structs.RadioCapture,
    transient_holdoff: float = 0,
    include_holdoff: bool = False,
):
    if iqwaveform.power_analysis.isroundmod(capture.duration * capture.sample_rate, 1):
        samples_out = round(capture.duration * capture.sample_rate)
    else:
        msg = f'duration must be an integer multiple of the sample period (1/{capture.sample_rate} s)'
        raise ValueError(msg)

    _, _, analysis_filter = design_capture_filter(base_clock_rate, capture)
    if capture.host_resample:
        sample_rate = analysis_filter['fs']
    else:
        sample_rate = capture.sample_rate

    if capture.host_resample and needs_stft(analysis_filter, capture):
        nfft = analysis_filter['nfft']
        pad_before, pad_after = _get_stft_padding(base_clock_rate, capture)
        min_samples_in = ceil(samples_out * nfft / analysis_filter['nfft_out'])
        samples_in = min_samples_in + (pad_before + pad_after)
    else:
        samples_in = round(capture.sample_rate * capture.duration)

    if include_holdoff:
        # accommmodate holdoff samples as needed for the periodic trigger and transient holdoff durations
        samples_in += ceil(
            sample_rate * (transient_holdoff + 2*(periodic_trigger or 0))
        )

    return samples_in


def get_input_iq_buffer_size(
    radio: RadioDevice, capture=None, include_holdoff=False
) -> int:
    if capture is None:
        capture = radio.get_capture_struct()

    return _get_capture_buffer_sizes_cached(
        base_clock_rate=radio.base_clock_rate,
        periodic_trigger=radio.periodic_trigger,
        capture=capture,
        transient_holdoff=radio.transient_holdoff_time,
        include_holdoff=include_holdoff,
    )


def _list_radio_classes(subclass=RadioDevice):
    """returns a list of radio subclasses that have been imported"""

    clsmap = {c.__name__: c for c in subclass.__subclasses__()}

    for subcls in list(clsmap.values()):
        clsmap.update(_list_radio_classes(subcls))

    clsmap = {name: cls for name, cls in clsmap.items() if not name.startswith('_')}

    return clsmap


def find_radio_cls_by_name(
    name: str, parent_cls: type[RadioDevice] = RadioDevice
) -> RadioDevice:
    """returns a list of radio subclasses that have been imported"""

    mapping = _list_radio_classes(parent_cls)

    if name in mapping:
        return mapping[name]
    else:
        raise AttributeError(
            f'invalid driver {repr(name)}. valid names: {tuple(mapping.keys())}'
        )


def is_same_resource(r1: str | dict, r2: str | dict):
    if hasattr(r1, 'items'):
        return set(r1.items()) == set(r2.items())
    else:
        return r1 == r2
