import time
from functools import wraps
import typing

import labbench as lb
from labbench import paramattr as attr

from .base import RadioDevice
from .. import structs, iq_corrections
from .util import design_capture_filter, get_capture_buffer_sizes

if typing.TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    import SoapySDR as soapy
    import iqwaveform
else:
    np = lb.util.lazy_import('numpy')
    pd = lb.util.lazy_import('pandas')
    soapy = lb.util.lazy_import('SoapySDR')
    iqwaveform = lb.util.lazy_import('iqwaveform')

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


class SoapyRadioDevice(RadioDevice):
    """single-channel sensor waveform acquisition through SoapySDR and pre-processed with iqwaveform"""

    _inbuf = None
    _outbuf = None

    resource: dict = attr.value.dict(
        default={}, help='SoapySDR resource dictionary to specify the device connection'
    )

    on_overflow = attr.value.str(
        'ignore',
        only=['ignore', 'except', 'log'],
        help='configure behavior on receive buffer overflow',
    )

    _downsample = attr.value.float(1.0, min=1, help='backend_sample_rate/sample_rate')

    lo_offset = attr.value.float(
        0.0,
        label='Hz',
        help='digital frequency shift of the RX center frequency',
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
                soapy.SOAPY_SDR_RX, soapy.SOAPY_SDR_CF32, [channel]
            )

    @attr.method.float(
        min=0,
        label='Hz',
        help='direct conversion LO frequency of the RX',
    )
    @_verify_channel_for_getter
    def lo_frequency(self):
        # there is only one RX LO, shared by both channels
        ret = self.backend.getFrequency(soapy.SOAPY_SDR_RX, self.channel())
        return ret

    @lo_frequency.setter
    @_verify_channel_for_setter
    def _(self, center_frequency):
        # there is only one RX LO, shared by both channels
        self.backend.setFrequency(soapy.SOAPY_SDR_RX, self.channel(), center_frequency)

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
        return self.backend.getSampleRate(soapy.SOAPY_SDR_RX, self.channel())

    @backend_sample_rate.setter
    @_verify_channel_for_setter
    def _(self, sample_rate):
        self.backend.setSampleRate(soapy.SOAPY_SDR_RX, self.channel(), sample_rate)

    sample_rate = backend_sample_rate.corrected_from_expression(
        backend_sample_rate / _downsample,
        label='Hz',
        help='sample rate of acquired waveform',
    )

    @attr.method.float(sets=False, label='Hz')
    def realized_sample_rate(self):
        """the self-reported "actual" sample rate of the radio"""
        return self.backend.getSampleRate(soapy.SOAPY_SDR_RX, 0) / self._downsample

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
                flags=soapy.SOAPY_SDR_HAS_TIME,
                # timeNs=self.backend.getHardwareTime('now'),
            )
        else:
            self.backend.deactivateStream(self.rx_stream)

    @attr.method.float(label='dB', help='SDR hardware gain')
    @_verify_channel_for_getter
    def gain(self):
        return self.backend.getGain(soapy.SOAPY_SDR_RX, self.channel())

    @gain.setter
    @_verify_channel_for_setter
    def _(self, gain: float):
        self.backend.setGain(soapy.SOAPY_SDR_RX, self.channel(), gain)

    @attr.method.float(label='dB', help='SDR TX hardware gain')
    @channel_kwarg
    def tx_gain(self, gain: float = lb.Undefined, /, *, channel: int = 0):
        if gain is lb.Undefined:
            return self.backend.getGain(soapy.SOAPY_SDR_TX, channel)
        else:
            self.backend.setGain(soapy.SOAPY_SDR_TX, channel, gain)

    def open(self):
        self._logger.info('connecting')
        self.backend = soapy.Device(self.resource)
        self._logger.info('connected')

        self._reset_stats()

        for channel in 0, 1:
            self.backend.setGainMode(soapy.SOAPY_SDR_RX, channel, False)
        self.channel(0)
        self.channel_enabled(False)

        # eventually: replace this with GPS time sync
        # self.backend.setHardwareTime(1, 'now')

    @property
    def _master_clock_rate(self):
        return type(self).backend_sample_rate.max

    def _reset_stats(self):
        self._stream_stats = {'overflow': 0, 'exceptions': 0, 'total': 0}

    def setup(self, radio_config: structs.RadioSetup):
        # TODO: the other parameters too
        self.calibration = radio_config.calibration
        self.periodic_trigger = radio_config.periodic_trigger
        if radio_config.preselect_if_frequency is not None:
            raise IOError('external frequency conversion is not yet supported')

    @_verify_channel_setting
    def acquire(
        self,
        capture: structs.RadioCapture,
        next_capture: typing.Union[structs.RadioCapture, None] = None,
        correction: bool = True,
    ) -> tuple[np.array, pd.Timestamp]:
        count, _ = get_capture_buffer_sizes(self, capture)

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
                iq = iq_corrections.resampling_correction(iq, capture, self)

            return iq, timestamp

    def arm(self, capture: structs.RadioCapture):
        """apply a capture configuration"""

        if capture == self.get_capture_struct():
            return

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

        if self.gain() != capture.gain:
            self.gain(capture.gain)

        fs_backend, lo_offset, analysis_filter = design_capture_filter(
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

    @attr.property.str(
        sets=False, cache=True, help='radio hardware UUID or serial number'
    )
    def id(self):
        # this is very radio dependent
        raise NotImplementedError

    def __del__(self):
        self.close()

    def _validate_remaining_samples(self, sr, count: int) -> int:
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
        elif sr.ret == soapy.SOAPY_SDR_OVERFLOW:
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
            raise IOError(f'Error {sr.ret}: {soapy.errToStr(sr.ret)}')
        else:
            raise TypeError(f'did not understand response {sr.ret}')

    def _prepare_buffer(self, capture: structs.RadioCapture):
        samples_in, _ = get_capture_buffer_sizes(self, capture, include_holdoff=True)

        # total buffer size for 2 values per IQ sample
        size_in = 2 * samples_in

        if self._inbuf is None or self._inbuf.size < size_in:
            self._logger.debug(
                f'allocating input sample buffer ({size_in * 2 /1e6:0.2f} MB)'
            )
            self._inbuf = np.empty((size_in,), dtype=np.float32)
            self._logger.debug('done')

    @_verify_channel_setting
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
                round(read_duration * self.backend_sample_rate()),
                timeoutUs=1,
            )

            if sr.ret > 0 and sr.ret < expected_samples:
                break
            elif sr.ret == soapy.SOAPY_SDR_TIMEOUT:
                break
            elif sr.ret < -1:
                raise IOError(f'Error {sr.ret}: {soapy.errToStr(sr.ret)}')

    @_verify_channel_setting
    def _read_stream(self, samples: int) -> np.ndarray:
        timeout = max(round(samples / self.backend_sample_rate() * 1.5), 50e-3)

        timestamp = None
        remaining = samples
        skip = 0

        self.on_overflow = 'ignore'
        while remaining > 0:
            # Read the samples from the data buffer
            rx_result = self.backend.readStream(
                self.rx_stream,
                [self._inbuf[(samples - remaining + skip) * 2 : (samples + skip) * 2]],
                remaining,
                timeoutUs=int(timeout * 1e6),
            )

            if timestamp is None:
                timestamp = rx_result.timeNs / 1e9

                if self.periodic_trigger is not None:
                    excess_time = timestamp % self.periodic_trigger
                    skip = round(
                        self.backend_sample_rate()
                        * (self.periodic_trigger - excess_time)
                    )
                    remaining = remaining + skip

            remaining = self._validate_remaining_samples(rx_result, remaining)
            self.on_overflow = 'except'

        self._stream_stats['total'] += 1

        # # what follows is some acrobatics to minimize new memory allocation and copy
        # buff_int16 = cp.array(self._inbuf, copy=False)[: 2 * N]

        # # 1. the same memory buffer, interpreted as float32 without casting
        # buff_float32 = cp.array(self._inbuf, copy=False)[: 4 * N].view('float32')

        # # 2. in-place casting from the int16 samples, filling in the extra allocation in self.buffer
        # cp.copyto(buff_float32, buff_int16, casting='unsafe')

        # 3. last, re-interpret each interleaved (float32 I, float32 Q) as a complex value
        return self._inbuf.view('complex64')[skip : samples + skip]
