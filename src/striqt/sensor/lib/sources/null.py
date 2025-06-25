import typing

import labbench as lb
from labbench import paramattr as attr
import time

from . import base, method_attr
from .. import specs
from ..util import lazy_import

if typing.TYPE_CHECKING:
    import pandas as pd
    import numpy as np
    import iqwaveform
else:
    pd = lazy_import('pandas')
    iqwaveform = lazy_import('iqwaveform')
    np = lazy_import('numpy')


channel_kwarg = attr.method_kwarg.int('channel', min=0, help='hardware port number')


class NullSource(base.SourceBase):
    """emulate a radio with fake data"""

    _inbuf = None
    _samples_elapsed = 0
    _transient_holdoff_time: float = attr.value.float(0.0, sets=True, inherit=True)
    _transport_dtype = attr.value.str('complex64', inherit=True)
    rx_channel_count: int = attr.value.int(1, cache=True, help='number of input ports')

    @method_attr.ChannelMaybeTupleMethod(inherit=True)
    def channel(self):
        # return none until this is set, then the cached value is returned
        return (0,)

    @channel.setter
    def _(self, channels: int):
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
        lo_frequency + base.SourceBase.lo_offset,
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
        return self.backend.setdefault('backend_sample_rate', 125e6)

    @backend_sample_rate.setter
    def _(self, value):
        self.backend['backend_sample_rate'] = value

    sample_rate = backend_sample_rate.corrected_from_expression(
        backend_sample_rate / base.SourceBase._downsample,
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

    @attr.method.str(inherit=True, sets=True, gets=True)
    def clock_source(self):
        return self.backend.setdefault('clock_source', 'internal')

    @clock_source.setter
    def _(self, clock_source: str):
        self.backend['clock_source'] = clock_source.lower()

    def sync_time_source(self):
        self._sync_time_ns = time.time_ns()

    @attr.method.bool(sets=True, gets=True)
    def rx_enabled(self):
        return self.backend.get('rx_enabled', False)

    @rx_enabled.setter
    def _(self, enable: bool):
        if enable == self.rx_enabled():
            return
        if enable:
            self.backend['rx_enabled'] = True
        else:
            self.backend['rx_enabled'] = False

    def setup(self, setup: specs.RadioSetup = None, analysis=None, **kws):
        setup = super().setup(setup, analysis, **kws)

        if setup._rx_channel_count is not None:
            self.rx_channel_count = setup._rx_channel_count

        self.reset_sample_counter()

        return setup

    @method_attr.FloatMaybeTupleMethod(inherit=True)
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

    @attr.property.str(inherit=True)
    def id(self):
        return 'null'

    @property
    def base_clock_rate(self):
        return self.resource.get('base_clock_rate', 125e6)

    def set_array_module(self, xp):
        self.xp = xp

    def _read_stream(
        self, buffers, offset, count, timeout_sec=None, *, on_overflow='except'
    ) -> tuple[int, int]:
        fs = float(self.backend_sample_rate())
        sample_period_ns = 1_000_000_000 / fs
        timestamp_ns = self._sync_time_ns + self._samples_elapsed * sample_period_ns

        self._samples_elapsed += count

        return count, round(timestamp_ns)

    def arm(self, capture=None, **kwargs):
        super().arm(capture, **kwargs)
        self.reset_sample_counter()

    def reset_sample_counter(self, value=0):
        self.sync_time_source()
        self._samples_elapsed = value
        self._sample_start_index = value


class NullRadio(NullSource):
    # eventually, deprecate this
    pass
