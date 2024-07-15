from __future__ import annotations
import labbench as lb
from .. import host


class RadioDevice(lb.Device):
    def build_index_variables(self):
        return host.index_variables()

    def build_metadata(self):
        return host.host_metadata()
