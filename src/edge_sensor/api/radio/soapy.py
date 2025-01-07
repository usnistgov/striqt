from __future__ import annotations

import time
from functools import wraps
import numbers
import typing

import labbench as lb
from labbench import paramattr as attr

from . import base
from .. import structs


if typing.TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    import SoapySDR
    import iqwaveform
else:
    np = lb.util.lazy_import('numpy')
    pd = lb.util.lazy_import('pandas')
    SoapySDR = lb.util.lazy_import('SoapySDR')
    iqwaveform = lb.util.lazy_import('iqwaveform')


channel_kwarg = attr.method_kwarg.int('channel', min=0, help='hardware port number')


def _verify_channel_setting(func: callable) -> callable:
    # TODO: fix typing
    @wraps(func)
    def wrapper(self, *args, **kws):
        if not self.isopen:
            name = getattr(func, '__name__', repr(func))
            raise RuntimeError(f'open radio before calling {name}')
        elif not self._stream_all_rx_channels:
            pass
        elif len(self.channel()) == 0:
            name = getattr(func, '__name__', repr(func))
            raise RuntimeError(f'set {self}.channel before calling {name}')

        return func(self, *args, **kws)

    return wrapper


def _verify_channel_for_getter(func: callable) -> callable:
    # TODO: fix typing
    @wraps(func)
    def wrapper(self):
        if len(self.channel()) == 0:
            name = getattr(func, '__name__', repr(func))
            raise RuntimeError(f'set {self}.channel before calling {name}')
        else:
            return func(self)

    return wrapper


def _verify_channel_for_setter(func: callable) -> callable:
    # TODO: fix typing
    @wraps(func)
    def wrapper(self, argument):
        if len(self.channel()) == 0:
            raise ValueError(f'set {self}.channel first')
        else:
            return func(self, argument)

    return wrapper


def get_stream_result(
    sr: SoapySDR.StreamResult, on_overflow='except'
) -> tuple[int, int]:
    """track the number of samples received and remaining in a read stream.

    Args:
        sr: the return value from self.backend.readStream
        count: the expected number of samples (1 (I,Q) pair each)

    Returns:
        (samples received, start clock timestamp (ns))
    """
    msg = None

    # ensure the proper number of waveform samples was read
    if sr.ret >= 0:
        return sr.ret, sr.timeNs
    elif sr.ret == SoapySDR.SOAPY_SDR_OVERFLOW:
        if on_overflow == 'except':
            msg = f'{time.perf_counter()}: overflow'
            raise OverflowError(msg)
        return 0, sr.timeNs
    else:
        raise IOError(f'{SoapySDR.errToStr(sr.ret)} (error code {sr.ret})')


class SoapyRadioDevice(base.RadioDevice):
    """single-channel sensor waveform acquisition through SoapySDR and pre-processed with iqwaveform"""

    resource: dict = attr.value.dict(
        default={}, help='SoapySDR resource dictionary to specify the device connection'
    )

    on_overflow = attr.value.str(
        'ignore',
        only=['ignore', 'except', 'log'],
        help='configure behavior on receive buffer overflow',
    )

    _rx_enable_delay = attr.value.float(
        None, sets=False, label='s', help='channel activation wait time'
    )

    @attr.method.float(inherit=True)
    @_verify_channel_for_getter
    def backend_sample_rate(self):
        for channel in self.channel():
            return self.backend.getSampleRate(SoapySDR.SOAPY_SDR_RX, channel)

    @backend_sample_rate.setter
    @_verify_channel_for_setter
    def _(self, backend_sample_rate):
        rate = self.base_clock_rate
        if np.isclose(backend_sample_rate, rate):
            # avoid exceptions due to rounding error
            backend_sample_rate = rate

        for channel in self.channel():
            self.backend.setSampleRate(
                SoapySDR.SOAPY_SDR_RX, channel, backend_sample_rate
            )

    def _setup_rx_stream(self, channels):
        with lb.stopwatch(
            f'setup stream for channels {channels}', logger_level='debug'
        ):
            self._rx_stream = self.backend.setupStream(
                SoapySDR.SOAPY_SDR_RX, SoapySDR.SOAPY_SDR_CF32, list(channels)
            )

    @base.ChannelTupleMethod(inherit=True)
    def channel(self):
        # return none until this is set, then the cached value is returned
        return None

    @channel.setter
    def _(self, channels: tuple[int, ...] | None):
        if self._stream_all_rx_channels:
            # in this case, the stream is controlled only on open
            return

        elif getattr(self, '_rx_stream', None) is not None:
            if self.channel() == channels:
                # already set up
                return
            else:
                self.rx_enabled(False)
                self.backend.closeStream(self._rx_stream)

        # if we make it this far, we need to build and enable the RX stream
        self._setup_rx_stream(channels)

    @attr.method.float(
        min=0,
        label='Hz',
        help='direct conversion LO frequency of the RX',
    )
    @_verify_channel_for_getter
    def lo_frequency(self):
        # there is only one RX LO, shared by both channels
        for channel in self.channel():
            ret = self.backend.getFrequency(SoapySDR.SOAPY_SDR_RX, channel)
            return ret

    @lo_frequency.setter
    @_verify_channel_for_setter
    def _(self, center_frequency):
        # there is only one RX LO, shared by both channels
        for channel in self.channel():
            self.backend.setFrequency(SoapySDR.SOAPY_SDR_RX, channel, center_frequency)

    center_frequency = lo_frequency.corrected_from_expression(
        lo_frequency + base.RadioDevice.lo_offset,
        help='RF frequency at the center of the analysis bandwidth',
        label='Hz',
    )

    sample_rate = backend_sample_rate.corrected_from_expression(
        backend_sample_rate / base.RadioDevice._downsample,
        label='Hz',
        help='sample rate of acquired waveform',
    )

    @attr.method.float(sets=False, label='Hz')
    def realized_sample_rate(self):
        """the self-reported "actual" sample rate of the radio"""
        return self.backend.getSampleRate(SoapySDR.SOAPY_SDR_RX, 0) / self._downsample

    @attr.method.str(inherit=True, sets=True, gets=True)
    def time_source(self):
        # there is only one RX LO, shared by both channels
        backend_result = self.backend.getTimeSource()
        if backend_result == 'internal' and self.on_overflow == 'log':
            return 'host'
        else:
            return backend_result.lower()

    @time_source.setter
    def _(self, time_source: str):
        if time_source == 'host':
            self.backend.setTimeSource('internal')
            self.on_overflow = 'log'
            if self.periodic_trigger is not None:
                self._logger.warning(
                    'periodic trigger with host time will suffer from inaccuracy on overflow'
                )
        else:
            self.backend.setTimeSource(time_source)
            self.on_overflow = 'except'

    @attr.method.bool(cache=True, inherit=True)
    def rx_enabled(self):
        # due to cache=True, this is the default before the first set
        return False

    @rx_enabled.setter
    @_verify_channel_for_setter
    def _(self, enable: bool):
        if enable == self.rx_enabled():
            return

        if enable:
            delay = self._rx_enable_delay
            if delay is not None:
                timeNs = self.backend.getHardwareTime('now') + round(delay * 1e9)
                kws = {'flags': SoapySDR.SOAPY_SDR_HAS_TIME, 'timeNs': timeNs}
            else:
                kws = {'flags': SoapySDR.SOAPY_SDR_HAS_TIME}

            self.backend.activateStream(self._rx_stream, **kws)

        elif self._rx_stream is not None:
            self.backend.deactivateStream(self._rx_stream)

    @base.FloatTupleMethod(inherit=True)
    @_verify_channel_for_getter
    def gain(self):
        values = [
            self.backend.getGain(SoapySDR.SOAPY_SDR_RX, c) for c in self.channel()
        ]
        return tuple(values)

    @gain.setter
    @_verify_channel_for_setter
    def _(self, gain: float | tuple[float, ...]):
        channels = self.channel()

        if isinstance(gain, numbers.Number):
            gain = (gain,) * len(channels)
        elif len(gain) == 1:
            gain = gain * len(channels)
        elif len(gain) != len(channels):
            raise ValueError(
                f'number of gain values mismatches the number of RX channels ({len(channels)})'
            )

        for channel, gain in zip(channels, gain):
            self._logger.debug(f'set channel {channel} gain: {gain} dB')
            self.backend.setGain(SoapySDR.SOAPY_SDR_RX, channel, gain)

    @attr.method.float(label='dB', help='SDR TX hardware gain')
    @channel_kwarg
    def tx_gain(self, gain: float = lb.Undefined, /, *, channel: int = 0):
        if gain is lb.Undefined:
            return self.backend.getGain(SoapySDR.SOAPY_SDR_TX, channel)
        else:
            self.backend.setGain(SoapySDR.SOAPY_SDR_TX, channel, gain)

    def open(self):
        class _SoapySDRDevice(SoapySDR.Device):
            def close(self):
                if SoapySDR._SoapySDR is None:
                    return
                else:
                    super().close()

        if self.resource:
            # prevent race conditions in threaded accesses to the Soapy driver
            self.backend = _SoapySDRDevice(self.resource)
        else:
            self.backend = _SoapySDRDevice()

        self._logger.info(f'opened with {self.backend.getDriverKey()}')

        self._post_connect()

        channels = list(range(self.rx_channel_count))
        for channel in channels:
            self.backend.setGainMode(SoapySDR.SOAPY_SDR_RX, channel, False)

        if self._stream_all_rx_channels:
            self._setup_rx_stream(channels)

    def _post_connect(self):
        pass

    @property
    def base_clock_rate(self):
        return type(self).backend_sample_rate.max

    @_verify_channel_setting
    def _read_stream(
        self, buffers, offset, count, timeout_sec, *, on_overflow='except'
    ) -> tuple[int, int]:
        # Read the samples from the data buffer
        rx_result = self.backend.readStream(
            self._rx_stream,
            [buf[offset * 2 :] for buf in buffers],
            count,
            timeoutUs=int((self._rx_enable_delay + timeout_sec) * 1e6),
        )
        return get_stream_result(rx_result, on_overflow=on_overflow)

    def sync_time_source(self):
        if self.time_source() in ('internal', 'host'):
            self._sync_to_os_time_source()
        else:
            self._sync_to_external_time_source()

    def _sync_to_os_time_source(self):
        hardware_time = self.backend.getHardwareTime('now') / 1e9
        if abs(hardware_time - time.time()) >= 0.2:
            self.backend.setHardwareTime(round(time.time() * 1e9), 'now')

    def _sync_to_external_time_source(self):
        # We first wait for a PPS transition to avoid race conditions involving
        # applying the time of the next PPS
        with lb.stopwatch('synchronize sample time to pps'):
            init_pps_time = self.backend.getHardwareTime('pps')
            start_time = time.perf_counter()
            while init_pps_time == self.backend.getHardwareTime('pps'):
                if time.perf_counter() - start_time > 1.5:
                    raise RuntimeError('no pps input detected for external time source')
                else:
                    time.sleep(10e-3)

        # PPS transition occurred, should be safe to snag system time and apply it
        sys_time_now = time.time()
        full_secs = int(sys_time_now)
        frac_secs = sys_time_now - full_secs
        if frac_secs > 0.8:
            # System time is lagging behind the PPS transition
            full_secs += 1
        elif frac_secs > 0.2:
            # System time and PPS are off, warn caller
            self._logger.warning(
                f'system time and PPS out of sync by {frac_secs:0.3f}s, check NTP'
            )
        time_to_set_ns = int((full_secs + 1) * 1e9)
        self.backend.setHardwareTime(time_to_set_ns, 'pps')

    def acquire(
        self,
        capture: structs.RadioCapture,
        next_capture: typing.Union[structs.RadioCapture, None] = None,
        correction: bool = True,
    ) -> tuple[np.array, 'pd.Timestamp']:
        return super().acquire(
            capture=capture, next_capture=next_capture, correction=correction
        )

    def _needs_reenable(self, next_capture: structs.RadioCapture):
        """returns True if the channel needs to be disabled and re-enabled between the specified capture"""

        current = self.get_capture_struct()

        for field in ('center_frequency', 'channel'):
            if getattr(next_capture, field) != getattr(current, field):
                return True

        next_backend_sample_rate = base.design_capture_filter(
            self.base_clock_rate, next_capture
        )[2]['fs']
        if next_backend_sample_rate != self.backend_sample_rate():
            return True

        return False

    def close(self):
        if self.backend is None:
            return

        if (
            SoapySDR._SoapySDR is None
            or SoapySDR._SoapySDR.Device_deactivateStream is None
        ):
            # occurs sometimes when soapy's underlying libraries
            # have been deconstructed too far to proceed
            return

        try:
            self.rx_enabled(False)
        except ValueError:
            # channel not yet set
            pass

        try:
            if getattr(self, '_rx_stream', None) is not None:
                self.backend.closeStream(self._rx_stream)
        except ValueError as ex:
            if 'invalid parameter' in str(ex):
                # already closed
                pass
            else:
                raise

        self.backend.close()
        self._logger.info('closed')

    @attr.property.str(
        sets=False, cache=True, help='radio hardware UUID or serial number'
    )
    def id(self):
        # this is very radio dependent
        raise NotImplementedError

    def __del__(self):
        self.close()
