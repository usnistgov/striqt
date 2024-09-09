from __future__ import annotations
from typing import Literal
from functools import lru_cache

from dataclasses import dataclass
import numpy as np

from xarray_dataclasses import AsDataArray, Coordof, Data, Attr
import iqwaveform

from .. import structs, dataarrays


### Time elapsed dimension and coordinates
TimeElapsedAxis = Literal['time_elapsed']

@dataclass
class TimeElapsedCoords:
    data: Data[TimeElapsedAxis, np.float32]
    standard_name: Attr[str] = 'Time elapsed'
    units: Attr[str] = 's'

    @staticmethod
    @lru_cache
    def factory(
        capture: structs.Capture, *, power_detectors: tuple[str], detector_period: float
    ) -> dict[str, np.ndarray]:
        length = round(capture.duration / detector_period)
        return np.arange(length) * detector_period


### Power detector dimension and coordinates
PowerDetectorAxis = Literal['power_detector']

@dataclass
class PowerDetectorCoords:
    data: Data[PowerDetectorAxis, object]
    standard_name: Attr[str] = 'Power detector'

    @staticmethod
    @lru_cache
    def factory(
        capture: structs.Capture, *, power_detectors: tuple[str], **kws
    ) -> dict[str, list]:
        return np.array(list(power_detectors))


### Dataarray definition
@dataclass
class ChannelPowerTimeSeries(AsDataArray):
    power_time_series: Data[tuple[PowerDetectorAxis, TimeElapsedAxis], np.float32]

    power_detector: Coordof[PowerDetectorCoords]
    time_elapsed: Coordof[TimeElapsedCoords]

    standard_name: Attr[str] = 'Channel power'
    units: Attr[str] = 'dBm'


### iqwaveform implementation
@dataarrays.as_registered_channel_analysis(ChannelPowerTimeSeries)
def channel_power_time_series(
    iq,
    capture: structs.Capture,
    *,
    detector_period: float,
    power_detectors: tuple[str, ...] = ('rms', 'peak'),
):
    kws = {'iq': iq, 'Ts': 1 / capture.sample_rate, 'Tbin': detector_period}

    data = []
    for detector in power_detectors:
        power = iqwaveform.iq_to_bin_power(kind=detector, **kws)
        data.append(iqwaveform.powtodB(power.astype('float32')))

    metadata = {
        'detector_period': detector_period,
        'standard_name': 'Channel power',
        'units': f'dBm/{(capture.analysis_bandwidth or capture.sample_rate)/1e6} MHz',
    }

    return data, metadata
