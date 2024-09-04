from __future__ import annotations
import typing
from functools import lru_cache

from dataclasses import dataclass
import numpy as np

from xarray_dataclasses import AsDataArray, Coordof, Data, Attr
import iqwaveform

from ..dataarrays import expose_in_yaml, ChannelAnalysisResult, select_parameter_kws
from ..structs import Capture

from ._channel_power_time_series import PowerDetectorAxis, PowerDetectorCoords


### iqwaveform -> ChannelAnalysisResult wrapper
@expose_in_yaml
def cyclic_channel_power(
    iq,
    capture: Capture,
    *,
    cyclic_period: float,
    detector_period: float,
    power_detectors: tuple[str, ...] = ('rms', 'peak'),
    cyclic_statistics: tuple[str, ...] = ('min', 'mean', 'max'),
) -> ChannelAnalysisResult:
    params = select_parameter_kws(locals())

    metadata = {}

    power_detectors = tuple(power_detectors)
    cyclic_statistics = tuple(cyclic_statistics)

    data_dict = iqwaveform.iq_to_cyclic_power(
        iq,
        1 / capture.sample_rate,
        cyclic_period=cyclic_period,
        detector_period=detector_period,
        detectors=power_detectors,
        cycle_stats=cyclic_statistics,
    )

    data_dict = {
        det: {cyc_stat: iqwaveform.powtodB(power) for cyc_stat, power in d.items()}
        for det, d in data_dict.items()
    }

    metadata = {
        'standard_name': 'Channel power',
        # 'units': f'dBm/{(capture.analysis_bandwidth or capture.sample_rate)/1e6} MHz',
        'cyclic_period': cyclic_period,
        'detector_period': detector_period,
    }

    return ChannelAnalysisResult(
        CyclicChannelPower, data_dict, capture, parameters=params, attrs=metadata
    )


### Cyclic statistic axis and coordinate labels
CyclicStatisticAxis = typing.Literal['cyclic_statistic']


@lru_cache
def cyclic_statistic_coord_factory(
    capture: Capture,
    *,
    cyclic_statistics: tuple[str, ...] = ('min', 'mean', 'max'),
    **_,
):
    return list(cyclic_statistics)


@dataclass
class CyclicStatisticCoords:
    data: Data[CyclicStatisticAxis, object]
    standard_name: Attr[str] = 'Cyclic statistic'

    factory: callable = cyclic_statistic_coord_factory


### Cyclic lag axis and coordinate labels
CyclicLagAxis = typing.Literal['cyclic_lag']


@lru_cache
def cyclic_lag_coord_factory(
    capture: Capture, *, cyclic_period: float, detector_period: float, **_
):
    lag_count = int(np.rint(cyclic_period / detector_period))

    return np.arange(lag_count) * detector_period


@dataclass
class CyclicLagCoords:
    data: Data[CyclicLagAxis, np.float32]
    standard_name: Attr[str] = 'Cyclic lag'
    units: Attr[str] = 's'

    factory: callable = cyclic_lag_coord_factory


### DataArray definition
@dataclass
class CyclicChannelPower(AsDataArray):
    power_time_series: Data[
        tuple[PowerDetectorAxis, CyclicStatisticAxis, CyclicLagAxis], np.float32
    ]

    power_detector: Coordof[PowerDetectorCoords]
    cyclic_statistic: Coordof[CyclicStatisticCoords]
    cyclic_lag: Coordof[CyclicLagCoords]

    standard_name: Attr[str] = 'Channel power'
    units: Attr[str] = 'dBm'
