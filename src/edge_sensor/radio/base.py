from __future__ import annotations
import labbench as lb
from labbench import paramattr as attr


TRANSIENT_HOLDOFF_WINDOWS = 1


class RadioDevice(lb.Device):
    # _inbuf = None
    # _outbuf = None
    # def build_index_variables(self):
    #     return diagnostic_data.index_variables()

    # def build_metadata(self):
    #     return dict(
    #         super().build_metadata(), **diagnostic_data.package_host_resources()
    #     )

    # def _prepare_buffer(self, input_size, output_size):
    #     raise NotImplementedError

    analysis_bandwidth = attr.value.float(
        None,
        min=1e-6,
        label='Hz',
        help='bandwidth of the digital bandpass filter (or None to bypass)',
    )

    calibration_path = attr.value.Path(
        None,
        help='path to a calibration file, or None to skip calibration',
    )

    duration = attr.value.float(
        10e-3, min=0, label='s', help='receive waveform capture duration'
    )

    @attr.property.str(sets=False, cache=True, help='unique radio hardware identifier')
    def id(self):
        raise NotImplementedError

    @property
    def _master_clock_rate(self):
        return type(self).backend_sample_rate.max