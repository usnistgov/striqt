from __future__ import annotations
import contextlib
import functools
from math import ceil
import typing

import labbench as lb
from labbench import paramattr as attr
import msgspec
import numpy as np
from queue import Queue, Empty
import threading

from .. import structs, util

if typing.TYPE_CHECKING:
    import iqwaveform
    import pandas as pd
else:
    iqwaveform = util.lazy_import('iqwaveform')
    pd = util.lazy_import('pandas')


TRANSIENT_HOLDOFF_WINDOWS = 1


def find_trigger_holdoff(start_time, sample_rate, periodic_trigger: float | None):
    if periodic_trigger in (0, None):
        return 0

    periodic_trigger_ns = round(periodic_trigger * 1e9)

    # float rounding errors cause problems here; evaluate based on the 1-ns resolution
    excess_time_ns = start_time % periodic_trigger_ns
    holdoff_ns = (periodic_trigger_ns - excess_time_ns) % periodic_trigger_ns
    holdoff = round(holdoff_ns / 1e9 * sample_rate)

    if holdoff != 0:
        lb.logger.info(f'snapping to periodic trigger with {holdoff}-sample holdoff')

    return holdoff


class ThreadedRXStream:
    def __init__(self, radio: RadioDevice):
        self._thread = None
        self.radio = radio

        self._result = Queue(1)
        self._running = threading.Event()
        self._return_request = threading.Event()
        self._stop_req = threading.Event()
        self._trigger_interrupt_req = threading.Event()
        self.sample_count = None

    def arm(self, capture: structs.RadioCapture):
        with self.pause():
            self.capture = capture
            self.buf_size, _ = get_capture_buffer_sizes(
                self.radio, capture, include_holdoff=True
            )
            self.sample_count, _ = get_capture_buffer_sizes(
                self.radio, capture, include_holdoff=False
            )
            self._thread_exc = None

    def _accumulate_buffer(
        self, buf_time_ns=None, holdover_samples: 'np.ndarray[np.complex64]' = None
    ) -> tuple['np.ndarray[np.complex64]', float]:
        holdoff_size = 0
        streamed_count = 0
        awaiting_timestamp = True
        acq_start_ns = buf_time_ns
        samples = np.empty((2 * self.buf_size,), dtype=np.float32)

        if buf_time_ns is None and holdover_samples is not None:
            raise ValueError(
                'when start time is None, holdover_samples must also be None'
            )

        if holdover_samples is None:
            holdover_count = 0
        else:
            # note: holdover_count.dtype is np.complex64, samples.dtype is np.float32
            holdover_count = holdover_samples.size
            samples[: 2 * holdover_count] = holdover_samples.view(samples.dtype)

        fs = self.radio.backend_sample_rate()
        chunk_size = min(int(200e-3 * fs), self.sample_count + holdover_count)
        timeout_sec = chunk_size / fs + 50e-3
        remaining = self.sample_count - holdover_count

        while remaining > 0:
            if self._stop_req.is_set() or self._trigger_interrupt_req.is_set():
                if buf_time_ns is None:
                    return None, buf_time_ns
                else:
                    return None, buf_time_ns + round(
                        (streamed_count * 1_000_000_000) / fs
                    )

            if streamed_count > 0 or self.radio.gapless_repeats:
                on_overflow = 'except'
            else:
                on_overflow = 'ignore'

            # Read the samples from the data buffer
            this_count, ret_time_ns = self.radio._read_stream(
                [samples],
                offset=holdover_count + streamed_count,
                count=min(chunk_size, remaining),
                timeout_sec=timeout_sec,
                on_overflow=on_overflow,
            )

            if buf_time_ns is None:
                # special case for the first read in the stream, since
                # devices may not always return timestamps
                buf_time_ns = ret_time_ns

            if awaiting_timestamp:
                holdoff_size = find_trigger_holdoff(
                    buf_time_ns, fs, self.radio.periodic_trigger
                )
                remaining = remaining + holdoff_size

                acq_start_ns = buf_time_ns + round((holdoff_size * 1_000_000_000) / fs)
                awaiting_timestamp = False

            remaining = remaining - this_count
            streamed_count += this_count

        samples = samples.view('complex64')[
            holdoff_size : self.sample_count + holdoff_size
        ]
        return samples, acq_start_ns

    def _background_loop(self):
        """acquire samples in a loop"""

        next_time_ns = None

        # to skip transients in the resampler, pass in extra samples from the
        # previous acquisition allow for settling.
        if self.capture.host_resample:
            holdover_size = self.sample_count - round(
                self.capture.duration * self.radio.backend_sample_rate()
            )
        else:
            holdover_size = None

        holdover_samples = None

        try:
            while True:
                try:
                    samples, time_ns = self._accumulate_buffer(
                        next_time_ns, holdover_samples
                    )
                except BaseException:
                    if self._stop_req.is_set():
                        break
                    else:
                        raise

                if holdover_size not in (None, 0):
                    holdover_samples = samples[-holdover_size:]

                if self._stop_req.is_set():
                    break
                elif self._trigger_interrupt_req.is_set():
                    # on trigger, _read_buffer returns the timestamp is at the
                    # end of its (likely incomplete) buffer read
                    # TODO: adjust the holdover buffer appropriately
                    next_time_ns = time_ns
                    self._trigger_interrupt_req.clear()
                    continue

                try:
                    # TODO: revisit this constant
                    self._return_request.wait(timeout=20e-3)
                except TimeoutError:
                    if self.radio.gapless_repeats:
                        raise OverflowError('gapless repeat acquisition failed')
                else:
                    self._result.put((samples, time_ns))
                    # the request flag has been set by get(). copy and allocate
                    # first to ensure proper sequencing with the main thread
                    self._return_request.clear()

                next_time_ns = time_ns + round(1e9 * self.capture.duration)

        except BaseException as ex:
            self._thread_exc = ex
            self._stop_req.set()

    def start(self):
        if self.is_running():
            self.radio._logger.warning(
                'tried to start stream thread, but one is already running'
            )
        self._return_request.clear()
        self._running.clear()
        self._stop_req.clear()
        try:
            self._result.get_nowait()
        except Empty:
            pass
        self._trigger_interrupt_req.clear()
        self._thread_exc = None
        self._thread = threading.Thread(target=self._background_loop)

        self._thread.start()

    @contextlib.contextmanager
    def pause(self):
        restart = self.is_running()
        if restart:
            self.stop()
        yield
        if restart:
            self.start()

    def stop(self):
        self._stop_req.set()
        self._return_request.clear()

        if self._thread is None:
            return
        elif self._thread.is_alive():
            self._thread.join()

        self._thread = None

    def is_running(self):
        thread = self._thread
        if thread is None:
            return False
        running = thread.is_alive() and not self._stop_req.is_set()
        if not running and self._thread_exc is not None:
            raise self._thread_exc
        return running

    def get(self) -> tuple['np.ndarray', float]:
        if not self.is_running():
            raise RuntimeError('stream acquisition is not running')

        self._return_request.set()

        timeout = max(50e-3 + self.sample_count / self.radio.backend_sample_rate(), 1)

        try:
            samples, time_ns = self._result.get(timeout=timeout)
        except Empty:
            exc = TimeoutError('stream thread returned no data')
        else:
            exc = None

        if exc:
            raise exc

        self._return_request.clear()

        if samples is None:
            self.stop()
            if self._thread_exc is not None:
                raise self._thread_exc
            else:
                raise TimeoutError('stream thread returned no data')

        return samples, time_ns

    def trigger(self):
        self._trigger_interrupt_req.set()


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

    @attr.property.str(sets=False, cache=True, help='unique radio hardware identifier')
    def id(self):
        raise NotImplementedError

    @property
    def base_clock_rate(self):
        return type(self).backend_sample_rate.max

    def open(self):
        self.stream = ThreadedRXStream(self)

    def close(self):
        self.stream.stop()

    def acquire(
        self,
        capture: structs.RadioCapture,
        next_capture: typing.Union[structs.RadioCapture, None] = None,
        correction: bool = True,
    ) -> tuple[np.array, 'pd.Timestamp']:
        from .. import iq_corrections

        with lb.stopwatch('acquire', logger_level='debug'):
            if self.channel() is None or not self.channel_enabled():
                self.arm(capture)

            iq, timestamp = self.stream.get()
            timestamp = pd.Timestamp(timestamp, unit='ns')

            if next_capture is None:
                self.channel_enabled(False)
            elif capture != next_capture:
                self.channel_enabled(False)
                self.arm(next_capture)

            if correction:
                with lb.stopwatch('resampling', logger_level='debug'):
                    iq = iq_corrections.resampling_correction(iq, capture, self)

            acquired_capture = structs.copy_struct(capture, start_time=timestamp)
            return iq, acquired_capture

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

        if fs_backend != self.backend_sample_rate() or downsample != self._downsample:
            with attr.hold_attr_notifications(self):
                self._downsample = 1  # temporarily avoid a potential bounding error
            self.backend_sample_rate(fs_backend)
            self._downsample = downsample

        if capture.sample_rate != self.sample_rate():
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
            self.center_frequency(capture.center_frequency)

        self.analysis_bandwidth = capture.analysis_bandwidth

        if self.time_sync_every_capture:
            self.sync_time_source()

        self.stream.arm(capture)
        self.channel_enabled(True)

    # def arm(self, capture: structs.RadioCapture):
    #     """apply a capture configuration"""

    #     if self.channel() is None:
    #         # current channel was unset
    #         self.channel(capture.channel)

    #     if capture == self.get_capture_struct():
    #         return

    #     if iqwaveform.power_analysis.isroundmod(
    #         capture.duration * capture.sample_rate, 1
    #     ):
    #         self.duration = capture.duration
    #     else:
    #         raise ValueError(
    #             f'duration {capture.duration} is not an integer multiple of sample period'
    #         )

    #     if self.gain() != capture.gain:
    #         self.gain(capture.gain)

    #     fs_backend, lo_offset, analysis_filter = design_capture_filter(
    #         self.base_clock_rate, capture
    #     )

    #     nfft_out = analysis_filter.get('nfft_out', analysis_filter['nfft'])

    #     downsample = analysis_filter['nfft'] / nfft_out

    #     if fs_backend != self.backend_sample_rate() or downsample != self._downsample:
    #         with attr.hold_attr_notifications(self):
    #             self._downsample = 1  # temporarily avoid a potential bounding error
    #         self.backend_sample_rate(fs_backend)
    #         self._downsample = downsample

    #     if capture.sample_rate != self.sample_rate():
    #         self.sample_rate(capture.sample_rate)

    #     if lo_offset != self.lo_offset:
    #         self.lo_offset = lo_offset  # hold update on this one?

    #     if capture.center_frequency != self.center_frequency():
    #         self.center_frequency(capture.center_frequency)

    #     self.analysis_bandwidth = capture.analysis_bandwidth

    #     self.stream.arm(capture)

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


@functools.lru_cache(30000)
def _get_capture_buffer_sizes_cached(
    base_clock_rate: float,
    periodic_trigger: float | None,
    capture: structs.RadioCapture,
    include_holdoff: bool = False,
):
    if iqwaveform.power_analysis.isroundmod(capture.duration * capture.sample_rate, 1):
        samples_out = round(capture.duration * capture.sample_rate)
    else:
        msg = f'duration must be an integer multiple of the sample period (1/{capture.sample_rate} s)'
        raise ValueError(msg)

    _, _, analysis_filter = design_capture_filter(base_clock_rate, capture)

    samples_in = ceil(
        samples_out * analysis_filter['nfft'] / analysis_filter['nfft_out']
    )

    if include_holdoff and periodic_trigger is not None:
        # add holdoff samples needed for the periodic trigger
        samples_in += ceil(analysis_filter['fs'] * periodic_trigger)

    if analysis_filter and capture.host_resample:
        samples_in += TRANSIENT_HOLDOFF_WINDOWS * analysis_filter['nfft']
        samples_out = iqwaveform.fourier._istft_buffer_size(
            samples_in,
            window=analysis_filter['window'],
            nfft_out=analysis_filter['nfft_out'],
            nfft=analysis_filter['nfft'],
            extend=True,
        )

    return samples_in, samples_out


def get_capture_buffer_sizes(
    radio: RadioDevice, capture=None, include_holdoff=False
) -> tuple[int, int]:
    if capture is None:
        capture = radio.get_capture_struct()

    return _get_capture_buffer_sizes_cached(
        base_clock_rate=radio.base_clock_rate,
        periodic_trigger=radio.periodic_trigger,
        capture=capture,
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
