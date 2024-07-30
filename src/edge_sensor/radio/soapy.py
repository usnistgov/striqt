import time
from functools import wraps

import cupy as cp
import numpy as np
import numba
import numba.cuda
import labbench as lb
import pandas as pd
import SoapySDR
from iqwaveform import fourier
from iqwaveform.power_analysis import isroundmod
from iqwaveform.util import empty_shared
from labbench import paramattr as attr
from SoapySDR import (
    SOAPY_SDR_CS16,
    SOAPY_SDR_CF32,
    SOAPY_SDR_RX,
    SOAPY_SDR_TX,
    SOAPY_SDR_TIMEOUT,
    errToStr,
    SOAPY_SDR_HAS_TIME,
)
from functools import lru_cache

from .. import structs
from .base import RadioBase

TRANSIENT_HOLDOFF_WINDOWS = 1

channel_kwarg = attr.method_kwarg.int('channel', min=0, help='hardware port number')


def _verify_channel_setting(func: callable) -> callable:
    # TODO: fix typing
    @wraps(func)
    def wrapper(self, *args, **kws):
        if self.channel() is None:
            raise ValueError(f'set {self}.channel first')
        else:
            return func(self, *args, **kws)

    return wrapper


def _verify_channel_for_getter(func: callable) -> callable:
    # TODO: fix typing
    @wraps(func)
    def wrapper(self):
        if self.channel() is None:
            raise ValueError(f'set {self}.channel first')
        else:
            return func(self)

    return wrapper


def _verify_channel_for_setter(func: callable) -> callable:
    # TODO: fix typing
    @wraps(func)
    def wrapper(self, argument):
        if self.channel() is None:
            raise ValueError(f'set {self}.channel first')
        else:
            return func(self, argument)

    return wrapper


def free_cuda_memory():
    cp._default_memory_pool.free_all_blocks()


class SoapyRadioDevice(RadioBase):
    """single-channel sensor waveform acquisition through SoapySDR and pre-processed with iqwaveform"""

    _inbuf = None
    _outbuf = None

    resource = attr.value.dict(
        default={},
        help="SoapySDR resource dictionary to specify the device connection"
    )

    on_overflow = attr.value.str(
        'ignore',
        only=['ignore', 'except', 'log'],
        help='configure behavior on receive buffer overflow',
    )

    calibration_path = attr.value.Path(
        None,
        help='path to a calibration file, or None to skip calibration',
    )

    duration = attr.value.float(
        10e-3, min=0, label='s', help='receive waveform capture duration'
    )

    _downsample = attr.value.float(1.0, min=1, help='backend_sample_rate/sample_rate')

    lo_offset = attr.value.float(
        0.0,
        label='Hz',
        help='digital frequency shift of the RX center frequency',
    )
    analysis_bandwidth = attr.value.float(
        None,
        min=1,
        label='Hz',
        help='bandwidth of the digital bandpass filter (or None to bypass)',
    )

    @attr.method.int(
        min=0,
        allow_none=True,
        cache=True,
        help='RX input port index',
    )
    def channel(self):
        # return none until this is set, then the cached value is returned
        return None

    @channel.setter
    def _(self, channel: int):
        if self.channel() == channel:
            return

        if channel is None and self.rx_stream is not None:
            self.channel_enabled(False)
        else:
            if getattr(self, 'rx_stream', None) is not None:
                self.backend.closeStream(self.rx_stream)
            self.rx_stream = self.backend.setupStream(
                SOAPY_SDR_RX, SOAPY_SDR_CF32, [channel]
            )

    @attr.method.float(
        min=0,
        label='Hz',
        help='direct conversion LO frequency of the RX',
    )
    @_verify_channel_for_getter
    def lo_frequency(self):
        # there is only one RX LO, shared by both channels
        ret = self.backend.getFrequency(SOAPY_SDR_RX, self.channel())
        return ret

    @lo_frequency.setter
    @_verify_channel_for_setter
    def _(self, center_frequency):
        # there is only one RX LO, shared by both channels
        self.backend.setFrequency(SOAPY_SDR_RX, self.channel(), center_frequency)

    center_frequency = lo_frequency.corrected_from_expression(
        lo_frequency + lo_offset,
        help='RF frequency at the center of the RX baseband',
        label='Hz',
    )

    @attr.method.float(
        min=0,
        cache=True,
        label='Hz',
        help='sample rate before resampling',
    )
    @_verify_channel_for_getter
    def backend_sample_rate(self):
        return self.backend.getSampleRate(SOAPY_SDR_RX, self.channel())

    @backend_sample_rate.setter
    @_verify_channel_for_setter
    def _(self, sample_rate):
        self.backend.setSampleRate(SOAPY_SDR_RX, self.channel(), sample_rate)

    sample_rate = backend_sample_rate.corrected_from_expression(
        backend_sample_rate / _downsample,
        label='Hz',
        help='sample rate of acquired waveform',
    )

    @attr.method.float(sets=False, label='Hz')
    def realized_sample_rate(self):
        """the self-reported "actual" sample rate of the radio"""
        return self.backend.getSampleRate(SOAPY_SDR_RX, 0) / self._downsample

    @attr.method.bool(cache=True)
    @_verify_channel_for_getter
    def channel_enabled(self):
        # this is only called at most once, due to cache=True
        raise ValueError('must set channel_enabled once before reading')

    @channel_enabled.setter
    @_verify_channel_for_setter
    def _(self, enable: bool):
        if enable:
            self.backend.activateStream(
                self.rx_stream,
                flags=SOAPY_SDR_HAS_TIME,
                # timeNs=self.backend.getHardwareTime('now'),
            )
        else:
            self.backend.deactivateStream(self.rx_stream)

    @attr.method.float(label='dB', help='SDR hardware gain')
    @_verify_channel_for_getter
    def gain(self):
        return self.backend.getGain(SOAPY_SDR_RX, self.channel())

    @gain.setter
    @_verify_channel_for_setter
    def _(self, gain: float):
        self.backend.setGain(SOAPY_SDR_RX, self.channel(), gain)

    @attr.method.float(label='dB', help='SDR TX hardware gain')
    @channel_kwarg
    def tx_gain(self, gain: float = lb.Undefined, /, *, channel: int = 0):
        if gain is lb.Undefined:
            return self.backend.getGain(SOAPY_SDR_TX, channel)
        else:
            self.backend.setGain(SOAPY_SDR_TX, channel, gain)

    def open(self):
        self._logger.info('connecting')
        self.backend = SoapySDR.Device(self.resource)
        self._logger.info('connected')

        self._reset_stats()

        for channel in 0, 1:
            self.backend.setGainMode(SOAPY_SDR_RX, channel, False)
        self.channel(0)
        self.channel_enabled(False)
        self.backend.setHardwareTime(1, 'now')

    @property
    def _master_clock_rate(self):
        return type(self).backend_sample_rate.max

    def _reset_stats(self):
        self._stream_stats = {'overflow': 0, 'exceptions': 0, 'total': 0}

    @_verify_channel_setting
    def acquire(self, capture, next_capture=None, correction: bool=True) -> tuple[np.array, pd.Timestamp]:
        count, _ = _get_capture_sizes(self._master_clock_rate, capture)

        with lb.stopwatch('acquire', logger_level='debug'):
            self.arm(capture)
            self.channel_enabled(True)
            timestamp = pd.Timestamp('now')        
            self._prepare_buffer(capture)
            iq = self._read_stream(count)
            self.channel_enabled(False)

            if next_capture is not None:
                self.arm(next_capture)

            if correction:
                iq = self.resampling_correction(iq, capture)

            return iq, timestamp

    def arm(self, capture: structs.RadioCapture):
        """apply a capture configuration"""

        if capture == self.get_capture_struct():
            return

        if capture.preselect_if_frequency is not None:
            raise IOError('external frequency conversion is not yet supported')

        if isroundmod(capture.duration * capture.sample_rate, 1):
            self.duration = capture.duration
        else:
            raise ValueError(
                f'duration {capture.duration} is not an integer multiple of sample period'
            )

        if capture.channel != self.channel():
            self.channel(capture.channel)

        if self.gain() != capture.gain:
            self.gain(capture.gain)

        fs_backend, lo_offset, analysis_filter = _design_capture_filter(
            self._master_clock_rate, capture
        )

        fft_size_out = analysis_filter.get('fft_size_out', analysis_filter['fft_size'])
        downsample = analysis_filter['fft_size'] / fft_size_out

        if fs_backend != self.backend_sample_rate() or downsample != self._downsample:
            with attr.hold_attr_notifications(self):
                self._downsample = 1  # temporarily avoid a potential bounding error
            self.backend_sample_rate(fs_backend)
            self._downsample = downsample

        if capture.sample_rate != self.sample_rate():
            self.sample_rate(capture.sample_rate)

        if lo_offset != self.lo_offset:
            self.lo_offset = lo_offset  # hold update on this one?

        if capture.center_frequency != self.center_frequency():
            self.center_frequency(capture.center_frequency)

        self.analysis_bandwidth = capture.analysis_bandwidth

    def resampling_correction(
        self,
        iq: fourier.Array,
        capture: structs.RadioCapture,
        *,
        axis=0,
    ):
        """apply a bandpass filter implemented through STFT overlap-and-add.

        Args:
            iq: the input waveform
            capture: the capture filter specification structure
            axis: the axis of `x` along which to compute the filter
            out: None, 'shared', or an array object to receive the output data

        Returns:
            the filtered IQ capture
        """

        # create a buffer large enough for post-processing seeded with a copy of the IQ
        _, buf_size = _get_capture_sizes(self._master_clock_rate, capture)
        buf = cp.empty(buf_size, dtype='complex64')
        iq = buf[: iq.size] = cp.asarray(iq)

        fs_backend, lo_offset, analysis_filter = _design_capture_filter(
            self._master_clock_rate, capture
        )

        fft_size = analysis_filter['fft_size']

        fft_size_out, noverlap, overlap_scale, _ = fourier._ola_filter_parameters(
            iq.size,
            window=analysis_filter['window'],
            fft_size_out=analysis_filter.get('fft_size_out', fft_size),
            fft_size=fft_size,
            extend=True,
        )

        w = fourier._get_window(
            analysis_filter['window'], fft_size, fftbins=False, xp=cp
        )

        freqs, _, xstft = fourier.stft(
            iq,
            fs=fs_backend,
            window=w,
            nperseg=analysis_filter['fft_size'],
            noverlap=round(analysis_filter['fft_size'] * overlap_scale),
            axis=axis,
            truncate=False,
            out=buf,
        )

        # set the passband roughly equal to the 3 dB bandwidth based on ENBW
        enbw = (
            fs_backend
            / fft_size
            * fourier.equivalent_noise_bandwidth(
                analysis_filter['window'], fft_size, fftbins=False
            )
        )
        passband = analysis_filter['passband']

        if fft_size_out != analysis_filter['fft_size']:
            freqs, xstft = fourier.downsample_stft(
                freqs,
                xstft,
                fft_size_out=fft_size_out,
                passband=passband,
                axis=axis,
                out=buf,
            )
        else:
            fourier.zero_stft_by_freq(
                freqs,
                xstft,
                passband=(passband[0] + enbw, passband[1] - enbw),
                axis=axis,
            )

        iq = fourier.istft(
            xstft,
            iq.shape[axis],
            fft_size=fft_size_out,
            noverlap=noverlap,
            out=buf,
            axis=axis,
        )

        return iq[TRANSIENT_HOLDOFF_WINDOWS * fft_size_out :]

    def get_capture_struct(self) -> structs.RadioCapture:
        """generate the currently armed capture configuration for the specified channel"""
        if self.lo_offset == 0:
            lo_shift = 'none'
        elif self.lo_offset < 0:
            lo_shift = 'left'
        elif self.lo_offset > 0:
            lo_shift = 'right'

        return structs.RadioCapture(
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
            # future: external frequency conversion support
            # if_frequency=None,
            # lo_gain=0,
            # rf_gain=0,
        )

    def close(self):
        try:
            self.channel_enabled(False)
        except ValueError:
            # channel not yet set
            pass

        try:
            self.backend.closeStream(self.rx_stream)
        except ValueError as ex:
            if 'invalid parameter' in str(ex):
                # already closed
                pass
            else:
                raise

    def __del__(self):
        self.close()

    def _check_remaining_samples(self, sr, count: int) -> int:
        """validate the stream response after reading.

        Args:
            sr: the return value from self.backend.readStream
            count: the expected number of samples (1 (I,Q) pair each)

        Returns:
            number of samples left to acquire
        """
        msg = None

        # ensure the proper number of waveform samples was read
        if sr.ret == count:
            self._logger.debug(f'received all {sr.ret} samples')
            return 0
        elif sr.ret > 0:
            self._logger.debug(f'received {sr.ret} samples')
            return count - sr.ret
        elif sr.ret == SoapySDR.SOAPY_SDR_OVERFLOW:
            self._stream_stats['overflow'] += 1
            total_info = (
                f"{self._stream_stats['overflow']}/{self._stream_stats['total']}"
            )
            msg = f'{time.perf_counter()}: overflow (total {total_info}'
            if self.on_overflow == 'except':
                raise OverflowError(msg)
            elif self.on_overflow == 'log':
                self._logger.info(msg)
            return 0
        elif sr.ret < 0:
            self._stream_stats['exceptions'] += 1
            raise IOError(f'Error {sr.ret}: {errToStr(sr.ret)}')
        else:
            raise TypeError(f'did not understand response {sr.ret}')

    def _prepare_buffer(self, capture: structs.RadioCapture):
        samples_in, samples_out = _get_capture_sizes(self._master_clock_rate, capture)

        # total buffer size for 2 values per IQ sample
        size_in =  2 * samples_in

        if self._inbuf is None or self._inbuf.size < size_in:
            self._logger.debug(
                f"allocating input sample buffer ({size_in * 2 /1e6:0.2f} MB)"
            )
            self._inbuf = np.empty((size_in,), dtype=np.float32)
            self._logger.debug('done')

    def _flush_stream(self):
        """attempt to flush the buffer of the receive stream without a slower disable/enable cycle"""
        read_duration = 10e-3
        expected_samples = round(read_duration * self.backend_sample_rate())

        self.on_overflow = 'ignore'
        while True:
            # Read the samples from the data buffer
            sr = self.backend.readStream(
                self.rx_stream,
                [self._inbuf],
                round(read_duration*self.backend_sample_rate()),
                timeoutUs=1,
            )

            if sr.ret > 0 and sr.ret < expected_samples:
                break
            elif sr.ret == SOAPY_SDR_TIMEOUT:
                break
            elif sr.ret < -1:
                raise IOError(f'Error {sr.ret}: {errToStr(sr.ret)}')


    @_verify_channel_setting
    def _read_stream(self, N, raise_on_overflow=False) -> cp.ndarray:
        timeout = max(round(N / self.backend_sample_rate() * 1.5), 50e-3)

        remaining = N

        self.on_overflow = 'ignore'
        while remaining > 0:
            # Read the samples from the data buffer
            sr = self.backend.readStream(
                self.rx_stream,
                [self._inbuf[(N - remaining) * 2 : (N) * 2]],
                remaining,
                timeoutUs=int(timeout * 1e6),
            )

            remaining = self._check_remaining_samples(sr, remaining)
            self.on_overflow = 'except'

        self._stream_stats['total'] += 1

        # # what follows is some acrobatics to minimize new memory allocation and copy
        # buff_int16 = cp.array(self._inbuf, copy=False)[: 2 * N]

        # # 1. the same memory buffer, interpreted as float32 without casting
        # buff_float32 = cp.array(self._inbuf, copy=False)[: 4 * N].view('float32')

        # # 2. in-place casting from the int16 samples, filling in the extra allocation in self.buffer
        # cp.copyto(buff_float32, buff_int16, casting='unsafe')

        # 3. last, re-interpret each interleaved (float32 I, float32 Q) as a complex value
        return self._inbuf.view('complex64')[:N]


@lru_cache(30000)
def _design_capture_filter(
    master_clock_rate: float, capture: structs.RadioCapture
) -> tuple[float, float, dict]:
    if str(capture.lo_shift).lower() == 'none':
        lo_shift = False
    else:
        lo_shift = capture.lo_shift

    # fs_backend, lo_offset, self.analysis_filter
    return fourier.design_cola_resampler(
        fs_base=master_clock_rate,
        fs_target=capture.sample_rate,
        bw=capture.analysis_bandwidth,
        bw_lo=0.75e6,
        shift=lo_shift,
    )


@lru_cache(30000)
def _get_capture_sizes(
    master_clock_rate: float, capture: structs.RadioCapture
) -> tuple[int, int]:
    if isroundmod(capture.duration * capture.sample_rate, 1):
        Nout = round(capture.duration * capture.sample_rate)
    else:
        msg = f'duration must be an integer multiple of the sample period (1/{capture.sample_rate} s)'
        raise ValueError(msg)

    _, _, analysis_filter = _design_capture_filter(master_clock_rate, capture)

    Nin = round(
        np.ceil(Nout * analysis_filter['fft_size'] / analysis_filter['fft_size_out'])
    )

    if analysis_filter:
        Nin += TRANSIENT_HOLDOFF_WINDOWS * analysis_filter['fft_size']
        Nout = fourier._istft_buffer_size(
            Nin,
            window=analysis_filter['window'],
            fft_size_out=analysis_filter['fft_size_out'],
            fft_size=analysis_filter['fft_size'],
            extend=True,
        )

    return Nin, Nout


def empty_capture(radio: SoapyRadioDevice, capture: structs.RadioCapture):
    """evaluate a capture on an empty buffer to warm up a GPU"""

    import cupy as cp

    nin, _ = _get_capture_sizes(radio._master_clock_rate, capture)
    radio._prepare_buffer(capture)
    iq = cp.array(radio._inbuf, copy=False).view('complex64')[:nin]
    ret = radio.resampling_correction(iq, capture)

    return ret
