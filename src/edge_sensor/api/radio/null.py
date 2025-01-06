import typing

import labbench as lb
from labbench import paramattr as attr

from . import base
from ..util import import_cupy_with_fallback, lazy_import

if typing.TYPE_CHECKING:
    import pandas as pd
    import numpy as np
    import iqwaveform
else:
    pd = lazy_import('pandas')
    iqwaveform = lazy_import('iqwaveform')
    np = lazy_import('numpy')


channel_kwarg = attr.method_kwarg.int('channel', min=0, help='hardware port number')


class NullSource(base.RadioDevice):
    """emulate a radio with fake data"""

    _inbuf = None

    _transient_holdoff_time: float = attr.value.float(0.0, sets=True, inherit=True)

    rx_channel_count: int = attr.value.int(2, cache=True, help='number of input ports')

    @base.ChannelListMethod(inherit=True)
    def channels(self):
        # return none until this is set, then the cached value is returned
        return tuple()

    @channels.setter
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
        lo_frequency + base.RadioDevice.lo_offset,
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
        backend_sample_rate / base.RadioDevice._downsample,
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
            self.reset_sample_counter()

    @base.FloatTupleMethod(inherit=True)
    def gains(self):
        return self.backend.setdefault('gain', 0)

    @gains.setter
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

    @attr.property.str(inherit=True)
    def id(self):
        return 'null'

    @property
    def base_clock_rate(self):
        return 125e6

    def set_array_module(self, xp):
        self.xp = xp

    def _read_stream(
        self, buffers, offset, count, timeout_sec=None, *, on_overflow='except'
    ) -> tuple[int, int]:
        capture = self.get_capture_struct()
        _, _, analysis_filter = base.design_capture_filter(
            self.base_clock_rate, capture
        )
        sample_time_offset = -analysis_filter['nfft'] // 2

        timestamp_ns = (
            1_000_000_000 * (self._samples_elapsed - sample_time_offset)
        ) / float(self.backend_sample_rate())

        self._samples_elapsed += count

        return count, round(timestamp_ns)

    def reset_sample_counter(self):
        self._samples_elapsed = 0


class NullRadio(NullSource):
    # eventually, deprecate this
    pass
