import typing

import labbench as lb
from labbench import paramattr as attr

from .base import RadioDevice
from ..util import import_cupy_with_fallback

if typing.TYPE_CHECKING:
    import numpy as np
    import pandas as pd
else:
    np = lb.util.lazy_import('numpy')
    pd = lb.util.lazy_import('pandas')


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

    time_source = attr.value.str(default='internal', inherit=True)

    @attr.method.bool(cache=True)
    def channel_enabled(self):
        # this is only called at most once, due to cache=True
        raise ValueError('must set channel_enabled once before reading')
    
    def sync_time_source(self):
        pass

    @channel_enabled.setter
    def _(self, enable: bool):
        self.backend['channel_enabled'] = enable

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
        self.backend = {}
        self.channel(0)
        self.channel_enabled(False)

    @attr.property.str(inherit=True)
    def id(self):
        return 'null'

    @property
    def base_clock_rate(self):
        return 125e6

    def _prepare_buffer(self, capture):
        pass

    def _read_stream(self, N):
        xp = import_cupy_with_fallback()
        return xp.empty(N, dtype='complex64'), pd.Timestamp('now')


class NullRadio(NullSource):
    # eventually, deprecate this
    pass
