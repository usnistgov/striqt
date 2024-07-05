from __future__ import annotations
from iqwaveform.power_analysis import isroundmod
from iqwaveform import fourier
import time
import cupy as cp
import numpy as np
import numba
import numba.cuda
import SoapySDR
from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_TX, SOAPY_SDR_CS16, errToStr
import labbench as lb
import labbench.paramattr as attr
import pandas as pd

from .base import RadioDevice
from .. import structs


channel_kwarg = attr.method_kwarg.int('channel', min=0, max=1, help='port number')


class AirT7201B(RadioDevice):
    """fast simplified single-channel receive acquisition"""

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
        100e-3, min=0, label='s', help='receive waveform capture duration'
    )

    _downsample = attr.value.float(1.0, min=1, help='backend_sample_rate/sample_rate')

    lo_offset = attr.value.float(
        0.0,
        min=-125e6,
        max=125e6,
        label='Hz',
        help='digital frequency shift of the RX center frequency',
    )
    analysis_bandwidth = attr.value.float(
        None,
        min=1,
        label='Hz',
        help='bandwidth of the digital bandpass filter (or None to bypass)',
    )

    @attr.method.float(
        min=300e6,
        max=6000e6,
        label='Hz',
        help='direct conversion LO frequency of the RX',
    )
    @channel_kwarg
    def lo_frequency(self, center_frequency: float = lb.Undefined):
        # there is only one RX LO, shared by both channels
        if center_frequency is lb.Undefined:
            return self.backend.setFrequency(SOAPY_SDR_RX, 0)
        else:
            self.backend.setFrequency(SOAPY_SDR_RX, 0, center_frequency)

    center_frequency = lo_frequency.corrected_from_expression(
        lo_frequency + lo_offset,
        help='RF frequency at the center of the RX baseband',
        label='Hz',
    )

    @attr.method.float(
        min=3.906250e6,
        max=125e6,
        cache=True,
        label='Hz',
        help='sample rate before resampling',
    )
    @channel_kwarg
    def backend_sample_rate(self, /, *, channel: int = 0):
        return self.backend.getSampleRate(SOAPY_SDR_RX, channel)

    @backend_sample_rate.setter
    def _(self, sample_rate, /, *, channel: int = 0):
        self.backend.setSampleRate(SOAPY_SDR_RX, channel, sample_rate)

    sample_rate = backend_sample_rate.corrected_from_expression(
        backend_sample_rate / _downsample,
        label='Hz',
        help='sample rate of acquired waveform',
    )

    @attr.method.float(sets=False, label='Hz')
    @channel_kwarg
    def realized_sample_rate(self, *, channel=0):
        """the self-reported "actual" sample rate of the radio"""
        return self.backend.getSampleRate(SOAPY_SDR_RX, 0) / self._downsample

    @attr.method.bool(cache=True)
    @channel_kwarg
    def channel_enabled(self, /, *, channel: int = 0):
        # this is only called at most once, due to cache=True
        raise ValueError('must set channel_enabled once before reading')

    @channel_enabled.setter
    def _(self, enable: bool, /, *, channel: int = 0):
        if enable:
            if channel != 0:
                raise ValueError('only channel 0 is supported for now')
            self.backend.activateStream(self.rx_streams[channel])
        else:
            self.backend.deactivateStream(self.rx_streams[channel])

    @attr.method.float(min=-30, max=0, step=0.05, label='dB', help='SDR hardware gain')
    @channel_kwarg
    def gain(self, gain: float = lb.Undefined, /, *, channel: int = 0):
        if gain is lb.Undefined:
            return self.backend.getGain(SOAPY_SDR_RX, channel)
        else:
            self.backend.setGain(SOAPY_SDR_RX, channel, gain)

    @attr.method.float(
        min=-41.95, max=0, step=0.05, label='dB', help='SDR TX hardware gain'
    )
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
        self.buffer = None
        self.reset_counts()
        self.rx_streams = []
        self.analysis_filter = {}

        for channel in 0, 1:
            self.backend.setGainMode(SOAPY_SDR_RX, channel, False)
            self.rx_streams += [
                self.backend.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CS16, [channel])
            ]
            # self.channel_enabled(False, channel=channel)

    def autosample(
        self,
        center_frequency,
        sample_rate,
        analysis_bandwidth,
        lo_shift=False,
        channel=0,
    ):
        """automatically configure center frequency and sampling parameters.

        Sampling rates are set to ensure rational resampling relative to the SDR master clock.
        Optionally, a frequency shift with oversampling is implemented to move the LO leakage
        outside of the specified analysis bandwidth.
        """

        fs_backend, lo_offset, self.analysis_filter = fourier.design_cola_resampler(
            fs_base=self.master_clock_rate,
            fs_target=sample_rate,
            bw=analysis_bandwidth,
            shift=lo_shift,
        )

        fft_size_out = self.analysis_filter.get(
            'fft_size_out', self.analysis_filter['fft_size']
        )

        with lb.paramattr.hold_attr_notifications(self):
            self._downsample = 1  # temporarily avoid a potential bounding error
            self.backend_sample_rate = fs_backend
            self._downsample = self.analysis_filter['fft_size'] / fft_size_out
            self.lo_offset = lo_offset  # hold update on this one?

        self.center_frequency = center_frequency
        self.sample_rate(sample_rate, channel=channel)
        self.analysis_bandwidth = analysis_bandwidth

    def reset_counts(self):
        self.acquisition_counts = {'overflow': 0, 'exceptions': 0, 'total': 0}

    def acquire(
        self, channel=0, calibration_bypass=False
    ) -> tuple[cp.array, pd.Timestamp]:
        if isroundmod(self.duration * self.sample_rate(channel=channel), 1):
            sample_count = round(self.duration * self.sample_rate(channel=channel))
        else:
            msg = f'duration must be an integer multiple of the sample period (1/{self.sample_rate} s)'
            raise ValueError(msg)

        timestamp = pd.Timestamp('now')
        backend_count = round(np.ceil(sample_count * self._downsample))
        iq = self._read_stream(backend_count)

        if self.calibration_path is not None and not calibration_bypass:
            raise ValueError('calibration not yet supported')
        else:
            iq /= float(np.iinfo(np.int16).max)

        if self.analysis_filter:
            # out = cp.array(self.buffer, copy=False).view(iq.dtype)
            return fourier.ola_filter(
                iq, extend=True, **self.analysis_filter
            ), timestamp
        else:
            return iq, timestamp

    def arm(self, capture: structs.RadioCapture, enable=True) -> int:
        """apply a capture configuration and enable the channel to receive samples"""

        if capture.preselect_if_frequency is not None:
            raise IOError('external frequency conversion is not yet supported')

        if isroundmod(capture.duration * capture.sample_rate, 1):
            self.duration = capture.duration
        else:
            raise ValueError(
                f'duration {capture.duration} is not an integer multiple of sample period'
            )

        self.gain = capture.gain

        self.autosample(
            center_frequency=capture.center_frequency,
            sample_rate=capture.sample_rate,
            analysis_bandwidth=capture.analysis_bandwidth,
            lo_shift=capture.lo_shift,
        )

        if enable:
            self.channel_enabled(True, channel=capture.channel)

    def get_current_capture(self, channel=0, duration=None) -> structs.RadioCapture:
        """generate the currently armed capture configuration for the specified channel"""
        if self.lo_offset == 0:
            lo_shift = None
        elif self.lo_offset < 0:
            lo_shift = 'left'
        elif self.lo_offset > 0:
            lo_shift = 'right'

        return structs.RadioCapture(
            # RF and leveling
            center_frequency=self.center_frequency,
            channel=channel,
            gain=self.gain(channel=channel),
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
        pending_ex = None

        for channel in (0, 1):
            try:
                self.channel_enabled(False, channel=channel)
                self.backend.closeStream(self.rx_streams[channel])
            except ValueError as ex:
                if 'invalid parameter' in str(ex):
                    # already closed
                    pass
                else:
                    pending_ex = ex

        if pending_ex is not None:
            raise pending_ex

    def __del__(self):
        self.close()

    def _validate_stream_response(self, sr, count: int) -> None:
        """validate the stream response after reading.

        Args:
            sr: the return value from self.backend.readStream
            count: the expected number of samples (1 (I,Q) pair each)
        """
        msg = None

        # ensure the proper number of waveform samples was read
        if sr.ret == count:
            pass
        elif sr.ret == -4:
            self.acquisition_counts['overflow'] += 1
            total_info = f"{self.acquisition_counts['overflow']}/{self.acquisition_counts['total']}"
            msg = f'{time.perf_counter()}: overflow (total {total_info}'
            if self.on_overflow == 'except':
                raise OverflowError(msg)
            elif self.on_overflow == 'log':
                self._logger.info(msg)
        else:
            self.acquisition_counts['exceptions'] += 1
            raise IOError(f'Error {sr.ret}: {errToStr(sr.ret)}')

    def _read_stream(self, N, raise_on_overflow=False, channel=0) -> cp.ndarray:
        if self.buffer is None or self.buffer.size < 4 * N:
            # create a buffer for received samples that can be shared across CPU<->GPU
            #     ref: https://github.com/cupy/cupy/issues/3452#issuecomment-903273011
            #
            # this is double-sized compared to the usual number of (int16, int16) IQ pairs,
            # because later we want to store upcasted np.float32 without an extra (allocate, copy)
            #     ref: notebooks/profile_cupy.ipynb
            del self.buffer

            self.buffer = numba.cuda.mapped_array(
                (4 * N,),
                dtype=np.int16,
                strides=None,
                order='C',
                stream=0,
                portable=False,
                wc=False,
            )

        timeout = max(round(N / self.backend_sample_rate * 1.5), 50e-3)

        # Read the samples from the data buffer
        sr = self.backend.readStream(
            self.rx_streams[channel],
            [self.buffer[: 2 * N]],
            N,
            timeoutUs=int(timeout * 1e6),
        )

        self._validate_stream_response(sr, N)

        self.acquisition_counts['total'] += 1

        # what follows is some acrobatics to minimize new memory allocation and copy
        buff_int16 = cp.array(self.buffer, copy=False)[: 2 * N]

        # 1. the same memory buffer, interpreted as float32 without casting
        buff_float32 = cp.array(self.buffer, copy=False)[: 4 * N].view('float32')

        # 2. in-place casting from the int16 samples, filling in the extra allocation in self.buffer
        cp.copyto(buff_float32, buff_int16, casting='unsafe')

        # 3. last, re-interpret each interleaved (float32 I, float32 Q) as a complex value
        buff_complex64 = buff_float32.view('complex64')

        return buff_complex64


if __name__ == '__main__':
    airt = AirT7201B(freq=2.44e9, fs=2 * 31.25e6)
    iq = airt.acquire(256 * 1024)
    airt.close()
