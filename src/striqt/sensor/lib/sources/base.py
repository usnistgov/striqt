from __future__ import annotations
import functools
from math import ceil
import numbers
import typing

import labbench as lb
from labbench import paramattr as attr
import numpy as np

from .. import specs, util
from . import method_attr

from striqt.analysis.lib.util import pinned_array_as_cupy
from striqt.analysis.lib.specs import Analysis
from striqt.analysis.lib import register
from striqt.analysis.lib.dataarrays import IQPair


if typing.TYPE_CHECKING:
    import iqwaveform
    import pandas as pd
    import scipy
else:
    iqwaveform = util.lazy_import('iqwaveform')
    pd = util.lazy_import('pandas')
    scipy = util.lazy_import('scipy')


FILTER_SIZE = 4001
MIN_OARESAMPLE_FFT_SIZE = 4 * 4096 - 1
RESAMPLE_COLA_WINDOW = 'hamming'
FILTER_DOMAIN = 'time'


class _ReceiveBufferCarryover:
    """remember unused samples from the previous IQ capture"""

    samples: 'np.ndarray' | None
    start_time_ns: int | None

    def __init__(self, radio=None):
        self.radio = radio
        if radio is not None:
            attr.observe(radio, self.on_radio_attr_change, type_='set')

        self.clear()

    def apply(self, samples: 'np.ndarray') -> tuple[int | None, int]:
        """carry over samples into `samples` from the previous capture.

        Returns:
            (start_time_ns, number of samples)
        """
        if self.start_time_ns is None and self.samples is not None:
            raise ValueError(
                'carryover time information present, but missing timestamp'
            )

        if self.samples is None:
            carryover_count = 0
        else:
            # note: carryover.samples.dtype is np.complex64, samples.dtype is np.float32
            carryover_count = self.samples.shape[1]
            samples[:, : 2 * carryover_count] = self.samples.view(samples.dtype)

        return self.start_time_ns, carryover_count

    def stash(
        self,
        samples: 'np.ndarray',
        sample_start_ns,
        unused_sample_count: int,
        capture: specs.RadioCapture,
    ):
        """stash data needed to carry over extra samples into the next capture"""
        carryover_count = unused_sample_count
        self.samples = samples[:, -carryover_count:].copy()
        self.start_time_ns = sample_start_ns + round(1e9 * capture.duration)

    def clear(self):
        self.samples = None
        self.start_time_ns = None

    def on_radio_attr_change(self, msg):
        """invalidate on notification of paramattr changes"""

        if self.samples is None:
            return

        # TODO: proper gapless capturing will to manage this case, but
        # there is troubleshooting to do
        # elif msg['name'] == RadioDevice.rx_enabled.name:
        #     return

        if msg['new'] != msg['old']:
            self.clear()

    def __del__(self):
        self.unobserve()

    def unobserve(self):
        if self.radio is None:
            return
        attr.unobserve(self.radio, self.on_radio_attr_change)


def _cast_iq(
    radio: SourceBase, buffer: 'iqwaveform.util.ArrayType'
) -> 'iqwaveform.util.ArrayType':
    """cast the buffer to floating point, if necessary"""
    # array_namespace will categorize cupy pinned memory as numpy
    dtype_in = np.dtype(radio._transport_dtype)

    if radio.array_backend == 'cupy':
        import cupy as xp

        buffer = pinned_array_as_cupy(buffer)
    else:
        import numpy as xp

        buffer = xp.array(buffer)

    print('buffer out: ', buffer[-1, buffer.shape[1]//2:][:10])

    # what follows is some acrobatics to minimize new memory allocation and copy
    if dtype_in.kind == 'i':
        # the same memory buffer, interpreted as int16 without casting
        buffer_int16 = buffer.view('int16')[:, : 2 * buffer.shape[1]]
        buffer_float32 = buffer.view('float32')

        # in-place cast from the int16 samples, filling the extra allocation in self.buffer
        xp.copyto(buffer_float32, buffer_int16, casting='unsafe')

        # re-interpret the interleaved (float32 I, float32 Q) values as a complex value
        buffer_out = buffer_float32.view('complex64')

    else:
        buffer_out = buffer


    

    return buffer_out


class SourceBase(lb.Device):
    _carryover = _ReceiveBufferCarryover()

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
        100e-3, min=0, label='s', help='receive waveform capture duration'
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

    fast_lo = attr.value.bool(
        False,
        help='if False, permit the radio to use slower frequency changes/channel enables to improve LO spurs',
    )

    _downsample = attr.value.float(1.0, min=0, help='backend_sample_rate/sample_rate')

    # these must be implemented by child classes
    channel = method_attr.ChannelMaybeTupleMethod(
        cache=True,
        contained_type=int,
        min=0,
        help='list of RX channel port indexes to acquire',
    )

    gain = method_attr.FloatMaybeTupleMethod(
        label='dB', help='receive gain for each channel in hardware'
    )

    center_frequency = attr.method.float(
        min=0,
        label='Hz',
        help='RF frequency at the center of the RX baseband for all channels',
    )

    backend_sample_rate = attr.method.float(
        min=0,
        label='Hz',
        help='sample rate before resampling',
    )

    rx_enabled = attr.method.bool()

    time_source = attr.method.str(
        only=['host', 'internal', 'external', 'gps'],
        help='time base for sample timestamps',
    )

    clock_source = attr.method.str(
        only=['internal', 'external', 'gps'],
        help='frequency reference source',
    )

    rx_channel_count = attr.value.int(
        1, sets=False, min=1, cache=True, help='number of input ports'
    )

    array_backend = attr.value.str(
        'numpy',
        only=('numpy', 'cupy'),
        help='array module to use, which sets the type of compute device: numpy = cpu, cupy = gpu',
    )

    # constants that can be adjusted by device-specific classes to tune streaming behavior
    _stream_all_rx_channels = attr.value.bool(
        False,
        sets=False,
        help='whether to stream all channels even when only one is selected',
    )

    _transient_holdoff_time = attr.value.float(
        0.0, sets=False, label='s', help='holdoff time before valid data after enable'
    )

    _transport_dtype = attr.value.str(
        'complex64',
        only=('int16', 'float32', 'complex64'),
        sets=False,
        help='buffer and transport dtype',
    )

    _forced_backend_sample_rate = attr.value.float(
        None, help='if specified, only the specified backend sample rate will be used'
    )

    resource: dict = attr.value.dict(
        default={}, help='resource dictionary to specify the device connection'
    )

    @attr.property.str(sets=False, cache=True, help='unique radio hardware identifier')
    def id(self):
        raise NotImplementedError

    @property
    def base_clock_rate(self):
        return type(self).backend_sample_rate.max

    def open(self):
        self._armed_capture: specs.RadioCapture | None = None
        self._carryover = _ReceiveBufferCarryover(self)
        self._buffers = [None, None]
        self._aligner: register.AlignmentCaller | None = None

    def close(self):
        self._carryover = None
        self._buffers = None

    def setup(
        self,
        radio_setup: specs.RadioSetup,
        analysis: Analysis = None,
        **setup_kws: typing.Unpack[specs._RadioSetupKeywords],
    ):
        """disarm acquisition and apply the given radio setup"""

        if radio_setup is None:
            radio_setup = specs.RadioSetup()

        radio_setup = radio_setup.replace(**setup_kws)

        if radio_setup.driver is None:
            driver = self.__class__.__name__
            radio_setup = radio_setup.replace(driver=driver)

        self.calibration = radio_setup.calibration
        self.periodic_trigger = radio_setup.periodic_trigger
        self.gapless_repeats = radio_setup.gapless_repeats
        self.array_backend = radio_setup.array_backend
        self.time_sync_every_capture = radio_setup.time_sync_every_capture
        self.time_source(radio_setup.time_source)
        self.clock_source(radio_setup.clock_source)
        self.fast_lo = radio_setup.fast_lo

        if not self.time_sync_every_capture:
            self.rx_enabled(False)
            self.sync_time_source()

        if radio_setup.alignment_source is None:
            self._aligner = None
        elif analysis is None:
            name = register.get_alignment_measurement_name(radio_setup.alignment_source)
            raise ValueError(
                f'alignment source {name!r} requires an analysis '
                f'specification for {radio_setup.alignment_source!r}'
            )
        else:
            self._aligner = register.get_aligner(
                radio_setup.alignment_source, analysis=analysis
            )

        return radio_setup

    @lb.stopwatch('arm', logger_level='debug')
    def arm(
        self,
        capture: specs.RadioCapture = None,
        **capture_kws: typing.Unpack[specs._RadioCaptureKeywords],
    ) -> specs.RadioCapture:
        """stop the stream, apply a capture configuration, and start it"""

        if capture is None:
            capture = self.get_capture_struct()

        capture = capture.replace(**capture_kws)

        self._forced_backend_sample_rate = capture.backend_sample_rate

        if iqwaveform.power_analysis.isroundmod(
            capture.duration * capture.sample_rate, 1
        ):
            self.duration = capture.duration
        else:
            raise ValueError(
                f'duration {capture.duration} is not an integer multiple of sample period'
            )

        if not self.gapless_repeats:
            self.rx_enabled(False)

        if method_attr._number_if_single(capture.channel) != self.channel():
            # TODO: support the case of multichannel -> single channel elegantly
            self.rx_enabled(False)
            self.channel(capture.channel)

        if method_attr._number_if_single(capture.gain) != self.gain():
            self.rx_enabled(False)
            self.gain(capture.gain)

        resampler_design = design_capture_resampler(self.base_clock_rate, capture)
        fs_backend = resampler_design['fs_sdr']
        lo_offset = resampler_design['lo_offset']

        if (
            lo_offset != self.lo_offset
            or capture.center_frequency != self.center_frequency()
        ):
            self.rx_enabled(False)
            self.center_frequency(capture.center_frequency)
            self.lo_offset = lo_offset

        nfft_out = resampler_design.get('nfft_out', resampler_design['nfft'])
        downsample = resampler_design['nfft'] / nfft_out

        if fs_backend != self.backend_sample_rate() or downsample != self._downsample:
            self.rx_enabled(False)
            with attr.hold_attr_notifications(self):
                self._downsample = 1  # temporarily avoid a potential bounding error
            self.backend_sample_rate(fs_backend)
            self._downsample = downsample

        if capture.sample_rate != self.sample_rate():
            # in this case, it's only a post-processing (GPU resampling) change
            self.rx_enabled(False)
            self.sample_rate(capture.sample_rate)

        if (
            self.periodic_trigger is not None
            and capture.duration < self.periodic_trigger
        ):
            self._logger.warning(
                'periodic trigger duration exceeds capture duration, '
                'which creates a large buffer of unused samples'
            )

        self.analysis_bandwidth = capture.analysis_bandwidth

        self._armed_capture = capture

        return capture

    @lb.stopwatch('read_iq', logger_level='debug')
    def read_iq(
        self,
        capture: specs.RadioCapture,
    ) -> tuple['np.ndarray[np.complex64]', int]:
        # the return buffer
        samples, stream_bufs = self._get_next_buffers(capture)

        # holdoff parameters, valid when we already have a clock reading
        dsp_pad_before, _ = _get_dsp_pad_size(
            self.base_clock_rate, capture, self._aligner
        )

        # carryover from the previous acquisition
        awaiting_timestamp = True
        start_ns, carryover_count = self._carryover.apply(samples)
        buf_time_ns = start_ns

        # the number of holdoff samples from the end of the holdoff period
        # to include with the returned waveform
        included_holdoff = dsp_pad_before

        fs = self.backend_sample_rate()

        # sample counters
        sample_count = get_channel_read_buffer_count(
            self, capture, include_holdoff=False
        )
        received_count = 0
        chunk_count = remaining = sample_count - carryover_count

        if self.time_sync_every_capture:
            self.rx_enabled(False)
            self.sync_time_source()

        if not self.rx_enabled():
            self.rx_enabled(True)

        while remaining > 0:
            if received_count > 0 or self.gapless_repeats:
                on_overflow = 'except'
            else:
                on_overflow = 'ignore'

            request_count = min(chunk_count, remaining)

            if (received_count + request_count) > samples.shape[1]:
                # this could happen if there is a slight mismatch between
                # the requested and realized sample rate
                break

            # Read the samples from the data buffer
            this_count, ret_time_ns = self._read_stream(
                stream_bufs,
                offset=carryover_count + received_count,
                count=request_count,
                timeout_sec=request_count / fs + 10e-3,
                on_overflow=on_overflow,
            )

            if (this_count + received_count) > samples.shape[1]:
                # this should never happen
                raise MemoryError(
                    f'overfilled receive buffer by {(this_count + received_count) - samples.size}'
                )

            if buf_time_ns is None:
                # special case for the first read in the stream, because
                # devices may not always return timestamps
                buf_time_ns = ret_time_ns

            if awaiting_timestamp:
                included_holdoff = find_trigger_holdoff(
                    self, buf_time_ns, dsp_pad_before=dsp_pad_before
                )
                remaining = remaining + included_holdoff - dsp_pad_before

                start_ns = buf_time_ns + round(included_holdoff * 1e9 / fs)
                awaiting_timestamp = False

            remaining = remaining - this_count
            received_count += this_count

        samples = samples.view('complex64')
        sample_offs = included_holdoff - dsp_pad_before
        sample_span = slice(sample_offs, sample_offs + sample_count)

        unused_count = sample_count - round(capture.duration * fs)
        self._carryover.stash(
            samples[:, sample_span],
            start_ns,
            unused_sample_count=unused_count,
            capture=capture,
        )

        # it seems to be important to convert to cupy here in order
        # to get a full view of the underlying pinned memory. cuda
        # memory corruption has been observed when waiting until after
        samples = _cast_iq(self, samples)

        return samples[:, sample_span], start_ns

    def _get_next_buffers(self, capture) -> tuple[np.ndarray, np.ndarray]:
        """swap the buffers, and reallocate if needed"""
        self._buffers = [self._buffers[1], self._buffers[0]]
        self._buffers[0], ret = alloc_empty_iq(self, capture, self._buffers[0])
        return ret

    @lb.stopwatch('acquire', logger_level='debug')
    def acquire(
        self,
        capture: specs.RadioCapture = None,
        next_capture: typing.Union[specs.RadioCapture, None] = None,
        correction: bool = True,
    ) -> tuple[np.array | IQPair, 'pd.Timestamp']:
        """arm a capture and enable the channel (if necessary), read the resulting IQ waveform.

        Optionally, calibration corrections can be applied, and the radio can be left ready for the next capture.
        """
        from .. import iq_corrections

        if capture is None:
            capture = self.get_capture_struct()

        # allocate (and arm the capture if necessary)
        if capture != getattr(self, '_armed_capture', None):
            self.arm(capture)

        # this must be here, _after_ the possible arm call, and before possibly
        # arming the next
        fs = self.backend_sample_rate()

        # the low-level acquisition
        iq, time_ns = self.read_iq(capture)

        if next_capture == capture and self.gapless_repeats:
            # the one case where we leave it running
            pass
        else:
            self.rx_enabled(False)

        if next_capture is not None and capture != next_capture:
            self.arm(next_capture)

        if correction:
            with lb.stopwatch('resample and calibrate', logger_level='debug'):
                iq = iq_corrections.resampling_correction(
                    iq, capture, self, overwrite_x=True
                )

                if self._aligner is None:
                    iq = iq.unsync_span

        acquired_capture = capture.replace(
            start_time=pd.Timestamp(time_ns, unit='ns'),
            backend_sample_rate=fs,
        )

        return iq, acquired_capture

    def _read_stream(
        self, buffers, offset, count, timeout_sec, *, on_overflow='except'
    ) -> tuple[int, int]:
        """to be implemented in subclasses"""
        raise NotImplementedError

    def sync_time_source(self):
        raise NotImplementedError

    def get_capture_struct(
        self, cls=specs.RadioCapture, *, realized: bool = False
    ) -> specs.RadioCapture:
        """generate the currently armed capture configuration for the specified channel.

        If the truth of actual evaluates as False, only the requested value
        of backend_sample_rate is returned in the given radio capture.
        """

        if self.lo_offset == 0:
            lo_shift = 'none'
        elif self.lo_offset < 0:
            lo_shift = 'left'
        elif self.lo_offset > 0:
            lo_shift = 'right'

        if realized:
            backend_sample_rate = self.backend_sample_rate()
        else:
            backend_sample_rate = self._forced_backend_sample_rate

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
            backend_sample_rate=backend_sample_rate,
        )

    def get_array_namespace(self: SourceBase):
        if self.array_backend == 'cupy':
            import cupy

            return cupy
        elif self.array_backend == 'numpy':
            import numpy

            return numpy


def find_trigger_holdoff(
    radio: SourceBase, start_time_ns: int, dsp_pad_before: int = 0
):
    sample_rate = radio.backend_sample_rate()
    min_holdoff = dsp_pad_before

    # transient holdoff if we've rearmed as indicated by the presence of carryover samples
    if radio._carryover.start_time_ns is None:
        min_holdoff = min_holdoff + round(radio._transient_holdoff_time * sample_rate)

    periodic_trigger = radio.periodic_trigger
    if periodic_trigger in (0, None):
        return min_holdoff

    periodic_trigger_ns = round(periodic_trigger * 1e9)

    # float rounding errors cause problems here; evaluate based on the 1-ns resolution
    excess_time_ns = start_time_ns % periodic_trigger_ns
    holdoff_ns = (periodic_trigger_ns - excess_time_ns) % periodic_trigger_ns
    holdoff = round(holdoff_ns / 1e9 * sample_rate)

    if holdoff < min_holdoff:
        periodic_trigger_samples = round(periodic_trigger * sample_rate)
        holdoff += (
            ceil(min_holdoff / periodic_trigger_samples) * periodic_trigger_samples
        )

    return holdoff


@util.lru_cache(30000)
def _design_capture_resampler(
    base_clock_rate: float,
    capture: specs.WaveformCapture,
    bw_lo=0.25e6,
    min_oversampling=1.1,
    window=RESAMPLE_COLA_WINDOW,
    min_fft_size=MIN_OARESAMPLE_FFT_SIZE,
) -> 'iqwaveform.fourier.ResamplerDesign':
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
        # fs_sdr, lo_offset, kws = iqwaveform.fourier.design_cola_resampler(
        design = iqwaveform.fourier.design_cola_resampler(
            fs_base=base_clock_rate,
            fs_target=capture.sample_rate,
            bw=capture.analysis_bandwidth,
            bw_lo=bw_lo,
            shift=lo_shift,
            min_fft_size=min_fft_size,
            min_oversampling=min_oversampling,
            window=window,
            fs_sdr=capture.backend_sample_rate,
        )

        design['window'] = window

        return design

    elif lo_shift:
        raise ValueError('lo_shift requires host_resample=True')
    elif base_clock_rate < capture.sample_rate:
        raise ValueError(
            f'upsampling above {base_clock_rate / 1e6:f} MHz requires host_resample=True'
        )
    else:
        # use the SDR firmware to implement the desired sample rate
        return iqwaveform.fourier.design_cola_resampler(
            fs_base=capture.sample_rate,
            fs_target=capture.sample_rate,
            bw=capture.analysis_bandwidth,
            shift=False,
        )


@functools.wraps(_design_capture_resampler)
def design_capture_resampler(
    base_clock_rate, capture: specs.WaveformCapture, *args, **kws
) -> 'iqwaveform.fourier.ResamplerDesign':
    # cast the struct in case it's a subclass
    fixed_capture = specs.WaveformCapture.fromspec(capture)
    kws.setdefault('window', RESAMPLE_COLA_WINDOW)

    from .. import iq_corrections

    if iq_corrections.USE_OARESAMPLE:
        min_fft_size = MIN_OARESAMPLE_FFT_SIZE
    else:
        # this could probably be set to 1?
        min_fft_size = 256

    return _design_capture_resampler(
        base_clock_rate,
        fixed_capture,
        min_fft_size=min_fft_size,
        *args,
        **kws,
    )


def needs_resample(analysis_filter: dict, capture: specs.RadioCapture) -> bool:
    """determine whether an STFT will be needed to filter or resample"""

    is_resample = analysis_filter['nfft'] != analysis_filter['nfft_out']
    return is_resample and capture.host_resample


def _get_filter_pad(capture: specs.RadioCapture):
    if np.isfinite(capture.analysis_bandwidth):
        return FILTER_SIZE // 2 + 1
    else:
        return 0


@util.lru_cache(30000)
def _get_dsp_pad_size(
    base_clock_rate: float,
    capture: specs.RadioCapture,
    aligner: register.AlignmentCaller | None = None,
) -> tuple[int, int]:
    """returns the padding before and after a waveform to achieve an integral number of FFT windows"""

    from .. import iq_corrections

    min_lag_pad = _get_aligner_pad_size(base_clock_rate, capture, aligner)

    if iq_corrections.USE_OARESAMPLE:
        oa_pad_low, oa_pad_high = _get_oaresample_pad(base_clock_rate, capture)
        return (oa_pad_low, oa_pad_high + min_lag_pad)
    else:
        # this is removed before the FFT, so no need to micromanage its size
        filter_pad = _get_filter_pad(capture)

        # accommodate the large fft by padding to a fast size that includes at least lag_pad
        design = design_capture_resampler(base_clock_rate, capture)
        analysis_size = round(capture.duration * design['fs_sdr'])

        # treat the block size as the minimum number of samples needed for the resampler
        # output to have an integral number of samples
        block_size = design['nfft']
        block_count = analysis_size // block_size
        min_blocks = block_count + iqwaveform.util.ceildiv(min_lag_pad, block_size)

        # since design_capture_resampler gives us a nice fft size
        # for block_size, then if we make sure pad_blocks is also a nice fft size,
        # then the product (pad_blocks * block_size) will also be a product of small
        # primes
        pad_blocks = _get_next_fast_len(min_blocks)
        pad_end = pad_blocks * block_size - analysis_size
        return (filter_pad, pad_end)


def _get_aligner_pad_size(
    base_clock_rate: float,
    capture: specs.RadioCapture,
    aligner: register.AlignmentCaller | None = None,
) -> tuple[int, int]:
    if aligner is None:
        return 0

    max_lag = aligner.max_lag(capture)
    lag_pad = ceil(base_clock_rate * max_lag)

    return lag_pad


def _get_next_fast_len(n):
    try:
        from cupyx import scipy
    except ModuleNotFoundError:
        import scipy

    return scipy.fft.next_fast_len(n)


def _get_oaresample_pad(base_clock_rate: float, capture: specs.RadioCapture):
    resampler_design = design_capture_resampler(base_clock_rate, capture)

    nfft = resampler_design['nfft']
    nfft_out = resampler_design.get('nfft_out', nfft)

    samples_out = round(capture.duration * capture.sample_rate)
    min_samples_in = ceil(samples_out * nfft / resampler_design['nfft_out'])

    # round up to an integral number of FFT windows
    samples_in = ceil(min_samples_in / nfft) * nfft + nfft

    noverlap_out = iqwaveform.fourier._ola_filter_parameters(
        samples_in,
        window=resampler_design['window'],
        nfft_out=nfft_out,
        nfft=nfft,
        extend=True,
    )[1]

    noverlap = ceil(noverlap_out * nfft / nfft_out)

    return (samples_in - min_samples_in) + noverlap + nfft // 2, noverlap


@util.lru_cache(30000)
def _get_input_buffer_count_cached(
    base_clock_rate: float,
    periodic_trigger: float | None,
    capture: specs.RadioCapture,
    transient_holdoff: float = 0,
    include_holdoff: bool = False,
    aligner: register.AlignmentCaller | None = None,
):
    if iqwaveform.power_analysis.isroundmod(capture.duration * capture.sample_rate, 1):
        samples_out = round(capture.duration * capture.sample_rate)
    else:
        msg = f'duration must be an integer multiple of the sample period (1/{capture.sample_rate} s)'
        raise ValueError(msg)

    resampler_design = design_capture_resampler(base_clock_rate, capture)
    if capture.host_resample:
        sample_rate = resampler_design['fs']
    else:
        sample_rate = capture.sample_rate

    pad_size = sum(_get_dsp_pad_size(base_clock_rate, capture, aligner))
    if needs_resample(resampler_design, capture):
        nfft = resampler_design['nfft']
        min_samples_in = ceil(samples_out * nfft / resampler_design['nfft_out'])
        samples_in = min_samples_in + pad_size
    else:
        samples_in = round(capture.sample_rate * capture.duration) + pad_size

    if include_holdoff:
        # pad the buffer for triggering and transient holdoff
        extra_time = transient_holdoff + 2 * (periodic_trigger or 0)
        samples_in += ceil(sample_rate * extra_time)

    return samples_in


def get_channel_resample_buffer_count(radio: SourceBase, capture):
    resampler_design = design_capture_resampler(radio.base_clock_rate, capture)

    if capture.host_resample and needs_resample(resampler_design, capture):
        input_size = get_channel_read_buffer_count(radio, capture, True)
        buf_size = iqwaveform.fourier._istft_buffer_size(
            input_size,
            window=resampler_design['window'],
            nfft_out=resampler_design['nfft_out'],
            nfft=resampler_design['nfft'],
            extend=True,
        )
    else:
        buf_size = round(capture.duration * capture.sample_rate)

    return buf_size


def get_channel_read_buffer_count(
    radio: SourceBase, capture: specs.RadioCapture | None = None, include_holdoff=False
) -> int:
    if capture is None:
        capture = radio.get_capture_struct()

    return _get_input_buffer_count_cached(
        base_clock_rate=radio.base_clock_rate,
        periodic_trigger=radio.periodic_trigger,
        capture=capture,
        transient_holdoff=radio._transient_holdoff_time,
        include_holdoff=include_holdoff,
        aligner=radio._aligner,
    )


@lb.stopwatch('allocate acquisition buffer', logger_level='debug')
def alloc_empty_iq(
    radio: SourceBase,
    capture: specs.RadioCapture,
    prior: typing.Optional[np.ndarray] = None,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """allocate a buffer of IQ return values.

    Returns:
        The buffer and the list of buffer references for streaming.
    """
    count = get_channel_read_buffer_count(radio, capture, include_holdoff=True)

    if radio.array_backend == 'cupy':
        try:
            util.configure_cupy()
            from cupyx import empty_pinned as empty
        except ModuleNotFoundError as ex:
            raise RuntimeError(
                'could not import the configured array backend, "cupy"'
            ) from ex
    else:
        empty = np.empty

    buf_dtype = np.dtype(radio._transport_dtype)

    # fast reinterpretation between dtypes requires the waveform to be in the last axis
    channels = capture.channel
    if isinstance(channels, numbers.Number):
        channels = (channels,)
    else:
        channels = tuple(channels)

    if prior is None or (prior.shape[0] < len(channels) or prior.shape[1] < count):
        all_samples = empty((len(channels), count), dtype=np.complex64)
    else:
        all_samples = prior

    # the subset that we'll return for use by the next acquisition
    samples = all_samples[: len(channels), :count]

    # build the list of channel buffers, including references to the throwaway
    # in case of radio._stream_all_rx_channels
    if radio._stream_all_rx_channels and len(channels) != radio.rx_channel_count:
        if radio._transport_dtype == 'complex64':
            # a throwaway buffer for samples that won't be returned
            extra_count = count
        else:
            extra_count = 2 * count

        extra = np.empty(extra_count, dtype=buf_dtype)
    else:
        extra = None

    buffers = []
    i = 0
    for channel in range(radio.rx_channel_count):
        if channel in channels:
            buffers.append(samples[i].view(buf_dtype))
            i += 1
        elif radio._stream_all_rx_channels:
            buffers.append(extra)

    return all_samples, (samples, buffers)


def _list_radio_classes(subclass=SourceBase):
    """returns a list of radio subclasses that have been imported"""

    clsmap = {c.__name__: c for c in subclass.__subclasses__()}

    for subcls in list(clsmap.values()):
        clsmap.update(_list_radio_classes(subcls))

    clsmap = {name: cls for name, cls in clsmap.items() if not name.startswith('_')}

    return clsmap


def find_radio_cls_helper(
    name: str, parent_cls: type[SourceBase] = SourceBase
) -> SourceBase:
    """returns a list of radio subclasses that have been imported"""

    mapping = _list_radio_classes(parent_cls)

    if name in mapping:
        return mapping[name]
    else:
        raise AttributeError(
            f'invalid driver {repr(name)}. valid names: {tuple(mapping.keys())}'
        )
