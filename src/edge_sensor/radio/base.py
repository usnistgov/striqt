from __future__ import annotations
import labbench as lb
from labbench import paramattr as attr
from .. import structs

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

    calibration = attr.value.Path(
        None,
        help='path to a calibration file, or None to skip calibration',
    )

    duration = attr.value.float(
        10e-3, min=0, label='s', help='receive waveform capture duration'
    )

    periodic_trigger = attr.value.float(
        None,
        allow_none=True,
        help='if specified, acquisition start times will begin at even multiples of this'
    )

    @attr.property.str(sets=False, cache=True, help='unique radio hardware identifier')
    def id(self):
        raise NotImplementedError

    @property
    def _master_clock_rate(self):
        return type(self).backend_sample_rate.max
    
    def get_capture_struct(self) -> structs.RadioCapture:
        """generate the currently armed capture configuration for the specified channel"""
        if self.lo_offset == 0:
            lo_shift = 'none'
        elif self.lo_offset < 0:
            lo_shift = 'left'
        elif self.lo_offset > 0:
            lo_shift = 'right'

        return structs.RadioCapture(
            # RF and leveling
            center_frequency=self.center_frequency(),
            channel=self.channel(),
            gain=self.gain(),
            # acquisition
            duration=self.duration,
            sample_rate=self.sample_rate(),
            # filtering and resampling
            analysis_bandwidth=self.analysis_bandwidth,
            lo_shift=lo_shift,
            # future: external frequency conversion support
            # if_frequency=None,
            # lo_gain=0,
            # rf_gain=0,
        )
