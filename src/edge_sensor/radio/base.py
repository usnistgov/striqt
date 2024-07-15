from __future__ import annotations
import labbench as lb
from .. import host


class RadioBase(lb.Device):
    def build_index_variables(self):
        return host.index_variables()

    def build_metadata(self):
        return dict(super().build_metadata(), **host.host_metadata())