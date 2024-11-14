import typing

import labbench as lb
from labbench import paramattr as attr

from .base import RadioDevice
from ..util import import_cupy_with_fallback

if typing.TYPE_CHECKING:
    import pandas as pd
    import iqwaveform
else:
    pd = lb.util.lazy_import('pandas')
    iqwaveform = lb.util.lazy_import('iqwaveform')


channel_kwarg = attr.method_kwarg.int('channel', min=0, help='hardware port number')


class NullSource(RadioDevice):
    """emulate a radio with fake data"""

    _inbuf = None

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
        pass

    @attr.method.float(
        min=0,
        label='Hz',
        help='direct conversion LO frequency of the RX',
    )
    def lo_frequency(self):
        # there is only one RX LO, shared by both channels
        return self.backend.setdefault('lo_frequency', 3e9)

    @lo_frequency.setter
    def _(self, value):
        # there is only one RX LO, shared by both channels
        self.backend['lo_frequency'] = value

    center_frequency = lo_frequency.corrected_from_expression(
        lo_frequency + RadioDevice.lo_offset,
        help='RF frequency at the center of the RX baseband',
        label='Hz',
    )

    @attr.method.float(
        min=0,
        cache=True,
        label='Hz',
        help='sample rate before resampling',
    )
    def backend_sample_rate(self):
        return self.backend.setdefault('backend_sample_rate', 25e6)

    @backend_sample_rate.setter
    def _(self, value):
        self.backend['backend_sample_rate'] = value

    sample_rate = backend_sample_rate.corrected_from_expression(
        backend_sample_rate / RadioDevice._downsample,
        label='Hz',
        help='sample rate of acquired waveform',
    )

    @attr.method.float(sets=False, label='Hz')
    def realized_sample_rate(self):
        """the self-reported "actual" sample rate of the radio"""
        return self.sample_rate()

    @attr.method.str(inherit=True, sets=True, gets=True)
    def time_source(self):
        return self.backend.setdefault('time_source', 'internal')

    @time_source.setter
    def _(self, time_source: str):
        self.backend['time_source'] = time_source.lower()

    def sync_time_source(self):
        pass

    @attr.method.bool(sets=True, gets=True)
    def channel_enabled(self):
        return self.backend.get('channel_enabled', False)

    @channel_enabled.setter
    def _(self, enable: bool):
        if enable == self.channel_enabled():
            return
        if enable:
            self.backend['channel_enabled'] = True
            self.stream.arm(self.get_capture_struct())
            self.stream.start()
        else:
            self.stream.stop()
            self.backend['channel_enabled'] = False
            self.reset_sample_counter()

    @attr.method.float(label='dB', help='SDR hardware gain')
    def gain(self):
        return self.backend.setdefault('gain', 0)

    @gain.setter
    def _(self, gain: float):
        self.backend['gain'] = gain

    @attr.method.float(label='dB', help='SDR TX hardware gain')
    @channel_kwarg
    def tx_gain(self, gain: float = lb.Undefined, /, *, channel: int = 0):
        if gain is lb.Undefined:
            return self.backend.setdefault('tx_gain', default=0)
        else:
            self.backend['tx_gain'] = gain

    def open(self):
        self._logger.propagate = False
        self.reset_sample_counter()
        self.backend = {}

    def close(self):
        self.stream.stop()

    @attr.property.str(inherit=True)
    def id(self):
        return 'null'

    @property
    def base_clock_rate(self):
        return 125e6

    def _read_stream(self, buffers, offset, count, timeout_sec=None, *, on_overflow='except') -> tuple[int,int]:
        xp = iqwaveform.util.array_namespace(buffers[0])
        timestamp_ns = (1_000_000_000 * self._sample_count) / float(self.backend_sample_rate())

        for channel, buf in zip([self.channel], buffers):
            values = self.get_waveform(count, self._sample_count, channel=channel, xp=xp)
            buf[2*offset : 2*(offset + count)] = values.view('float32')

        self._sample_count += count

        return count, round(timestamp_ns)

    def get_waveform(self, count, start_index: int, *, channel: int = 0, xp):
        return xp.empty(count, dtype='complex64')
    
    def reset_sample_counter(self):
        self._sample_count = 0


class NullRadio(NullSource):
    # eventually, deprecate this
    pass
