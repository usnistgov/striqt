# Import Packages
from __future__ import annotations
import numpy as np
import sys
import SoapySDR
from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CS16, errToStr
import cupy as cp
import time
import numba
import numba.cuda
from iqwaveform.power_analysis import isroundmod
from iqwaveform import fourier
import labbench as lb
import labbench.paramattr as attr
from .base import HardwareSource


channel_kwarg = attr.method_kwarg.int('channel', min=0, max=1, help='port number')


class AirTSource(HardwareSource):
    """fast simplified single-channel receive acquisition"""

    # TODO: sanity-check bounds
    lo_offset = attr.value.float(0., min=-125e6, max=125e6, label='Hz', help='digital frequency shift of the RX center frequency')
    analysis_bandwidth = attr.value.float(None, min=1, label='Hz', help='bandwidth of the digital filter passband (or None to bypass)')

    # TODO: should this be made dependent on backend_sample_rate?
    sample_rate = attr.value.float(15.36e6, min=1, max=125e6, label='Hz', help='downsampled sample rate after processing')

    @attr.method.float(min=300e6, max=6000e6, label='Hz', help='direct conversion LO frequency of the RX')
    def lo_frequency(self, center_frequency: float = lb.Undefined):
        # there is only one RX LO, shared by both channels
        if center_frequency is lb.Undefined:
            return self.backend.setFrequency(SOAPY_SDR_RX, 0)
        else:
            self.backend.setFrequency(SOAPY_SDR_RX, 0, center_frequency)

    center_frequency = lo_frequency.corrected_from_expression(
        lo_frequency + lo_offset,
        help='RF frequency at the center of the RX baseband',
        label='Hz'
    )

    # TODO: check low bound
    @attr.property.float(min=200e3, max=125e6, label='Hz', help='sample rate before resampling')
    def backend_sample_rate(self, sample_rate: float = lb.Undefined):
        # there is only one RX sample clock, shared by both channels?
        return self.backend.getSampleRate(SOAPY_SDR_RX, 0)

    @backend_sample_rate.setter
    def backend_sample_rate(self, sample_rate):
        # there is only one RX sample clock, shared by both channels?
        self.backend.setSampleRate(SOAPY_SDR_RX, 0, sample_rate)

    _downsample = attr.value.float(1., min=1, help='backend_sample_rate/sample_rate')

    sample_rate = backend_sample_rate.corrected_from_expression(
        backend_sample_rate/_downsample,
        label='Hz',
        help='sample rate of acquired waveform'
    )

    @attr.method.bool(gets=False).setter
    @channel_kwarg    
    def channel_enabled(self, enable: bool = lb.Undefined, /, *, channel: int):
        if enable:
            if channel != 0:
                raise ValueError('only channel 0 is supported for now')
            self.backend.activateStream(self.rx_streams[channel])
        else:
            self.backend.deactivateStream(self.rx_streams[channel])

    @attr.method.float(min=-31.5, max=0, step=0.5, label='dB', help='SDR hardware gain')
    @channel_kwarg
    def gain(self, gain: float = lb.Undefined, /, *, channel: int):
        if gain is lb.Undefined:
            return self.backend.getGain(SOAPY_SDR_RX, channel)
        else:
            self.backend.setGain(SOAPY_SDR_RX, channel, gain)

    def open(self):
        self._logger.debug('connecting')
        self.backend = SoapySDR.Device(dict(driver='SoapyAIRT'))
        self._logger.debug('connected')
        self.buffer = None
        self.reset_counts()
        self.rx_streams = []
        self.analysis_filter = {}

        for channel in 0,1:
            self.backend.setGainMode(SOAPY_SDR_RX, channel, False)
            self.rx_streams += [self.backend.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CS16, [channel])]
            # self.channel_enabled(False, channel=channel)

    def autosample(self, center_frequency, sample_rate, analysis_bandwidth, shift=False):
        """automatically configure center frequency and sampling parameters.

        Sampling rates are set to ensure rational resampling relative to the SDR master clock.
        Optionally, a frequency shift with oversampling is implemented to move the LO leakage
        outside of the specified analysis bandwidth.
        """
        fs_backend, lo_offset, self.analysis_filter = fourier.design_cola_resampler(
            fs_base=125e6, fs_target=sample_rate, bw=analysis_bandwidth, shift=shift
        )

        fft_size_out = self.analysis_filter.get('fft_size_out', self.analysis_filter['fft_size'])

        with lb.paramattr.hold_attr_notifications(self):
            self._downsample = 1 # temporarily avoid a potential bounding error
            self.backend_sample_rate = fs_backend
            self._downsample = self.analysis_filter['fft_size']/fft_size_out
            self.lo_offset = lo_offset # hold update on this one?

        self.center_frequency = center_frequency
        self.sample_rate = sample_rate
        self.analysis_bandwidth = analysis_bandwidth

    def reset_counts(self):
        self.acquisition_counts = {'overflow': 0, 'exception': 0, 'total': 0}

    def _read_stream(self, N, full_scale=True, raise_on_overflow=False, channel=0) -> cp.ndarray:
        if self.buffer is None or self.buffer.size < 4 * N:
            # create a buffer for received samples that can be shared across CPU<->GPU
            #     ref: https://github.com/cupy/cupy/issues/3452#issuecomment-903273011
            #
            # this is double sized compared to the usual number of (int16, int16) IQ pairs,
            # because later we want to store upcasted np.float32 without an extra (allocate, copy)
            #     ref: notebooks/profile_cupy.ipynb
            self.buffer = numba.cuda.mapped_array(
                (4 * N,),
                dtype=np.int16,
                strides=None,
                order='C',
                stream=0,
                portable=False,
                wc=False,
            )

        # Read the samples from the data buffer
        sr = self.backend.readStream(
            self.rx_streams[channel],
            [self.buffer[: 2 * N]],
            N,
            timeoutUs=round(self.timeout * 1e6),
        )

        self.acquisition_counts['total'] += 1

        # ensure the proper number of waveform samples was read
        if sr.ret == N:
            pass
        elif sr.ret == -4 and raise_on_overflow:
            raise OverflowError(
                f"buffer overflow on acquisition {self.acquisition_counts['total']}"
            )
        else:
            if sr.ret == -4:
                self.acquisition_counts['overflow'] += 1
                total_info = f"{self.acquisition_counts['overflow']}/{self.acquisition_counts['total']}"
                print(f'{time.perf_counter()}: overflow (total {total_info}')
            else:
                self.acquisition_counts['exceptions'] += 1
                print('Error {}: {}'.format(sr.ret, errToStr(sr.ret)), sys.stderr)
            return None

        # what follows is some acrobatics to minimize new memory allocation and copy
        buff_int16 = cp.array(self.buffer, copy=False)[: 2 * N]

        # 1. the same memory buffer, interpreted as float32 without casting
        buff_float32 = cp.array(self.buffer, copy=False).view('float32')

        # 2. in-place casting from the int16 samples, filling in the extra allocation in self.buffer
        cp.copyto(buff_float32, buff_int16, casting='unsafe')

        # 3. last, re-interpret each interleaved (float32 I, float32 Q) as a complex value
        buff_complex64 = buff_float32.view('complex64')

        return buff_complex64

    def acquire(self, count, calibrate:bool=False):
        if isroundmod(count*self._downsample, 1):
            backend_count = round(count*self._downsample)
        else:
            raise ValueError('duration must be an integer multiple of the sample rate')

        iq = self._read_stream(backend_count, full_scale=(not calibrate))

        if calibrate:
            raise ValueError('calibration not yet supported')
        else:
            iq /= float(np.iinfo(np.int16).max)

        if self.analysis_filter:
            # out = cp.array(self.buffer, copy=False).view(iq.dtype)
            return fourier.ola_filter(iq, extend=True, **self.analysis_filter)
        else:
            return iq


    def __del__(self):
        self.close()

    def close(self):
        try:
            for channel in 0,:
                self.channel_enabled(False, channel=channel)
                self.backend.closeStream(self.rx_streams[channel])

        except ValueError as ex:
            if 'invalid parameter' in str(ex):
                # already closed
                pass
            else:
                raise


if __name__ == '__main__':
    airt = AirTSource(freq=2.44e9, fs=2 * 31.25e6)
    iq = airt.acquire(256 * 1024)
    airt.close()
