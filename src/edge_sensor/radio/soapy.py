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
from SoapySDR import SOAPY_SDR_CF32, SOAPY_SDR_RX, SOAPY_SDR_TX, errToStr

from .. import structs
from .base import RadioBase

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


class SoapyRadioDevice(RadioBase):
    """single-channel sensor waveform acquisition through SoapySDR and pre-processed with iqwaveform"""

    TRANSIENT_HOLDOFF_WINDOWS = 2  # one on each side
    _inbuf = None
    _outbuf = None

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
            self.backend.activateStream(self.rx_stream)
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

    @attr.property.float(
        sets=False, cache=True, label='Hz', help='base sample clock rate (MCR)'
    )
    def master_clock_rate(self):
        return self.backend.getMasterClockRate()

    def open(self):
        self._logger.info('connecting')
        self.backend = SoapySDR.Device(dict(driver='SoapyAIRT'))
        self._logger.info('connected')

        self._reset_stats()
        self.analysis_filter = {}

        for channel in 0, 1:
            self.backend.setGainMode(SOAPY_SDR_RX, channel, False)

    @_verify_channel_setting
    def autosample(
        self,
        center_frequency,
        sample_rate,
        analysis_bandwidth,
        lo_shift='none',
    ):
        """automatically configure center frequency and sampling parameters.

        Sampling rates are set to ensure rational resampling relative to the SDR master clock.
        Optionally, a frequency shift with oversampling is implemented to move the LO leakage
        outside of the specified analysis bandwidth.
        """

        self.channel_enabled(False)

        if str(lo_shift).lower() == 'none':
            lo_shift = False

        fs_backend, lo_offset, self.analysis_filter = fourier.design_cola_resampler(
            fs_base=self.master_clock_rate,
            fs_target=sample_rate,
            bw=analysis_bandwidth,
            bw_lo=0.75e6,
            shift=lo_shift,
        )

        fft_size_out = self.analysis_filter.get(
            'fft_size_out', self.analysis_filter['fft_size']
        )

        with lb.paramattr.hold_attr_notifications(self):
            self._downsample = 1  # temporarily avoid a potential bounding error
            self.backend_sample_rate(fs_backend)
            self._downsample = self.analysis_filter['fft_size'] / fft_size_out
            self.lo_offset = lo_offset  # hold update on this one?

        self.center_frequency(center_frequency)
        self.sample_rate(sample_rate)
        self.analysis_bandwidth = analysis_bandwidth

    def _reset_stats(self):
        self._stream_stats = {'overflow': 0, 'exceptions': 0, 'total': 0}

    @_verify_channel_setting
    def acquire(self, calibration_bypass=False) -> tuple[cp.array, pd.Timestamp]:
        if isroundmod(self.duration * self.sample_rate(), 1):
            Nout = round(self.duration * self.sample_rate())
        else:
            msg = f'duration must be an integer multiple of the sample period (1/{self.sample_rate} s)'
            raise ValueError(msg)

        timestamp = pd.Timestamp('now')
        Nin = round(np.ceil(Nout * self._downsample))

        if self.analysis_filter:
            Npad = (
                self.TRANSIENT_HOLDOFF_WINDOWS * self.analysis_filter['fft_size']
            )
            bufsize_in = Nin + 2*Npad            
            bufsize_out = fourier._ola_filter_buffer_size(
                bufsize_in,
                window=self.analysis_filter['window'],
                fft_size_out=self.analysis_filter['fft_size_out'],
                fft_size=self.analysis_filter['fft_size'],
                extend=True
            )
        else:
            bufsize_in = Nin
            bufsize_out = Nout

        self._prepare_buffer(bufsize_in, bufsize_out)
        iq = self._read_stream(Nin)
        if self.calibration_path is not None and not calibration_bypass:
            raise ValueError('calibration not yet supported')
        else:
            pass
        #            iq /= float(np.finfo(np.float32).max)

        if self.analysis_filter:
            # out = cp.array(self.buffer, copy=False).view(iq.dtype)
            iq = fourier.ola_filter(iq, extend=True, out=self._outbuf, **self.analysis_filter)

        trim = Nout - iq.shape[0]
        return iq[-trim // 2 : trim // 2 or None], timestamp

    def arm(self, capture: structs.RadioCapture):
        """apply a capture configuration and enable the channel to receive samples"""

        if capture.preselect_if_frequency is not None:
            raise IOError('external frequency conversion is not yet supported')

        if isroundmod(capture.duration * capture.sample_rate, 1):
            self.duration = capture.duration
        else:
            raise ValueError(
                f'duration {capture.duration} is not an integer multiple of sample period'
            )

        self.channel(capture.channel)
        self.gain(capture.gain)

        self.autosample(
            center_frequency=capture.center_frequency,
            sample_rate=capture.sample_rate,
            analysis_bandwidth=capture.analysis_bandwidth,
            lo_shift=capture.lo_shift,
        )

        self.channel_enabled(True)

    def get_capture_struct(self, duration=None) -> structs.RadioCapture:
        """generate the currently armed capture configuration for the specified channel"""
        if self.lo_offset == 0:
            lo_shift = None
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
            duration=duration,
            sample_rate=self.sample_rate,
            # filtering and resampling
            analysis_bandwidth=self.analysis_bandwidth,
            lo_shift=lo_shift,
            # future: external frequency conversion support
            if_frequency=None,
            lo_gain=0,
            rf_gain=0,
        )

    def close(self):
        try:
            self.channel_enabled(False)
        except ValueError as ex:
            if 'invalid parameter' in str(ex):
                # already closed
                pass
            else:
                raise

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
            return 0
        elif sr.ret > 0:
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

    def _prepare_buffer(self, Nin, Nout):
        if self._inbuf is None or self._inbuf.size < 2 * Nin:
            # create a buffer for received samples that can be shared across CPU<->GPU
            #     ref: https://github.com/cupy/cupy/issues/3452#issuecomment-903273011
            #
            # this is double-sized compared to the usual number of (int16, int16) IQ pairs,
            # because later we want to store upcasted np.float32 without an extra (allocate, copy)
            #     ref: notebooks/profile_cupy.ipynb
            self._inbuf = empty_shared((2 * Nin,), dtype=np.float32, xp=np)

        if self._outbuf is None or self._outbuf.size < Nout:
            self._outbuf = empty_shared((Nout,), dtype=np.complex64, xp=cp)

    @_verify_channel_setting
    def _read_stream(self, N, raise_on_overflow=False) -> cp.ndarray:        
        timeout = max(round(N / self.backend_sample_rate() * 1.5), 50e-3)

        remaining = N

        while remaining > 0:
            # Read the samples from the data buffer
            sr = self.backend.readStream(
                self.rx_stream,
                [self._inbuf[2 * (N - remaining) : 2 * N]],
                remaining,
                timeoutUs=int(timeout * 1e6),
            )

            remaining = self._check_remaining_samples(sr, remaining)

        self._stream_stats['total'] += 1

        # # what follows is some acrobatics to minimize new memory allocation and copy
        # buff_int16 = cp.array(self.buffer, copy=False)[: 2 * N]

        # # 1. the same memory buffer, interpreted as float32 without casting
        # buff_float32 = cp.array(self.buffer, copy=False)[: 4 * N].view('float32')

        # # 2. in-place casting from the int16 samples, filling in the extra allocation in self.buffer
        # cp.copyto(buff_float32, buff_int16, casting='unsafe')

        # # 3. last, re-interpret each interleaved (float32 I, float32 Q) as a complex value
        # buff_complex64 = buff_float32.view('complex64')

        cp_buff = cp.array(self._inbuf, copy=False)[: 2 * N]

        return cp_buff.view('complex64')


def empty_capture(radio: SoapyRadioDevice, capture: structs.RadioCapture):
    """evaluate a capture on an empty buffer to warm up a GPU"""

    import cupy as cp

    fs_backend, lo_offset, analysis_filter = fourier.design_cola_resampler(
        fs_base=type(radio).backend_sample_rate.max,
        fs_target=capture.sample_rate,
        bw=capture.analysis_bandwidth,
        bw_lo=0.75e6,
        shift=capture.lo_shift,
    )    

    Nin = round(capture.duration * fs_backend)
    Nout = round(capture.duration * capture.sample_rate)


    if analysis_filter:
        Npad = (
            radio.TRANSIENT_HOLDOFF_WINDOWS * analysis_filter['fft_size']
        )
        bufsize_in = Nin + 2*Npad            
        bufsize_out = fourier._ola_filter_buffer_size(
            bufsize_in,
            window=analysis_filter['window'],
            fft_size_out=analysis_filter['fft_size_out'],
            fft_size=analysis_filter['fft_size'],
            extend=True
        )
    else:
        bufsize_in = Nin
        bufsize_out = Nout

    radio._prepare_buffer(bufsize_in, bufsize_out)
    iq = cp.array(radio._inbuf, copy=False)[: 2*bufsize_in].view('complex64')
    iq = fourier.ola_filter(iq, extend=True, out=radio._outbuf, **analysis_filter)
    trim = Nout - iq.shape[0]
    return iq[-trim // 2 : trim // 2 or None]