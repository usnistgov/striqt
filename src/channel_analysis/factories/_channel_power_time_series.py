from __future__ import annotations
from typing import Literal, get_args
from functools import lru_cache

from dataclasses import dataclass
import numpy as np

from xarray_dataclasses import AsDataArray, Coordof, Data, Attr, Name
from iqwaveform import powtodB, iq_to_bin_power

from ..dataarrays import expose_in_yaml, ChannelAnalysisResult, select_parameter_kws
from ..structs import Capture


### iqwaveform wrapper
@expose_in_yaml
def channel_power_time_series(
    iq,
    capture: Capture,
    *,
    detector_period: float,
    power_detectors: tuple[str, ...] = ('rms', 'peak'),
) -> ChannelAnalysisResult:
    params = select_parameter_kws(locals())

    kws = {'iq': iq, 'Ts': 1 / capture.sample_rate, 'Tbin': detector_period}

    data = [
        powtodB(iq_to_bin_power(kind=detector, **kws).astype('float32'))
        for detector in power_detectors
    ]

    metadata = {
        'detector_period': detector_period,
        'standard_name': 'Channel power',
        'units': f'dBm/{(capture.analysis_bandwidth or capture.sample_rate)/1e6} MHz',
    }

    return ChannelAnalysisResult(
        ChannelPowerTimeSeries, data, capture, parameters=params, attrs=metadata
    )


### Time elapsed dimension and coordinates
TimeElapsedAxis = Literal['time_elapsed']


@lru_cache
def time_elapsed_coord_factory(
    capture: Capture, *, power_detectors: tuple[str], detector_period: float
) -> dict[str, np.ndarray]:
    length = round(capture.duration / detector_period)
    return np.arange(length) * detector_period


@dataclass
class TimeElapsedCoords:
    data: Data[TimeElapsedAxis, np.float32]
    name: Name[str] = get_args(TimeElapsedAxis)[0]
    standard_name: Attr[str] = 'Time elapsed'
    units: Attr[str] = 's'

    factory: callable = time_elapsed_coord_factory


### Power detector dimension and coordinates
PowerDetectorAxis = Literal['power_detector']


@lru_cache
def power_detector_coord_factory(
    capture: Capture, *, power_detectors: tuple[str], **kws
) -> dict[str, list]:
    return np.array(list(power_detectors))


@dataclass
class PowerDetectorCoords:
    data: Data[PowerDetectorAxis, object]
    name: Name[str] = get_args(PowerDetectorAxis)[0]
    standard_name: Attr[str] = 'Power detector'

    factory: callable = power_detector_coord_factory


### Dataarray definition
@dataclass
class ChannelPowerTimeSeries(AsDataArray):
    power_time_series: Data[tuple[PowerDetectorAxis, TimeElapsedAxis], np.float32]

    power_detector: Coordof[PowerDetectorCoords]
    time_elapsed: Coordof[TimeElapsedCoords]

    standard_name: Attr[str] = 'Channel power'
    units: Attr[str] = 'dBm'
