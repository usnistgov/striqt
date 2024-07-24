from __future__ import annotations
import labbench as lb
from .. import diagnostic_data


class RadioBase(lb.Device):
    _inbuf = None
    _outbuf = None

    def build_index_variables(self):
        return diagnostic_data.index_variables()

    def build_metadata(self):
        return dict(
            super().build_metadata(), **diagnostic_data.package_host_resources()
        )

    def _prepare_buffer(self, input_size, output_size):
        raise NotImplementedError
