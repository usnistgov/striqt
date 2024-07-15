from __future__ import annotations
import labbench as lb
from .. import diagnostic_data, structs
from iqwaveform import fourier


class RadioBase(lb.Device):
    _buffer = None

    def build_index_variables(self):
        return diagnostic_data.index_variables()

    def build_metadata(self):
        return dict(
            super().build_metadata(), **diagnostic_data.package_host_resources()
        )

    def _prepare_buffer(self, size):
        raise NotImplementedError
