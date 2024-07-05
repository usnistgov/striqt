from __future__ import annotations
from channel_analysis.sources import WaveformSource
import labbench as lb
import labbench.paramattr as attr

from .. import host


class HardwareSource(lb.Device, WaveformSource):
    timeout = attr.value.float(5.0, label='s', help='data transport timeout')

    def build_index_variables(self):
        return host.index_variables()

    def build_metadata(self):
        return dict(super().build_metadata(), **host.host_metadata())
