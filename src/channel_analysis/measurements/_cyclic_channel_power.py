from __future__ import annotations
import dataclasses
import functools
import typing

import numpy as np
import iqwaveform
from xarray_dataclasses import AsDataArray, Coordof, Data, Attr

from ._common import as_registered_channel_analysis
from .._api import structs
from ._channel_power_time_series import PowerDetectorAxis, PowerDetectorCoords


### Cyclic statistic axis and coordinate labels
CyclicStatisticAxis = typing.Literal['cyclic_statistic']


@dataclasses.dataclass
class CyclicStatisticCoords:
    data: Data[CyclicStatisticAxis, object]
    standard_name: Attr[str] = 'Cyclic statistic'

    @staticmethod
    @functools.lru_cache
    def factory(
        capture: structs.Capture,
        *,
        cyclic_statistics: tuple[typing.Union[str, float], ...] = (
            'min',
            'mean',
            'max',
        ),
        **_,
    ):
        return list(cyclic_statistics)


### Cyclic lag axis and coordinate labels
CyclicLagAxis = typing.Literal['cyclic_lag']


@dataclasses.dataclass
class CyclicLagCoords:
    data: Data[CyclicLagAxis, np.float32]
    standard_name: Attr[str] = 'Cyclic lag'
    units: Attr[str] = 's'

    @staticmethod
    @functools.lru_cache
    def factory(
        capture: structs.Capture,
        *,
        cyclic_period: float,
        detector_period: float,
        **_,
    ):
        lag_count = int(np.rint(cyclic_period / detector_period))

        return np.arange(lag_count) * detector_period


### Define the data structure of the returned DataArray
@dataclasses.dataclass
class CyclicChannelPower(AsDataArray):
    power_time_series: Data[
        tuple[PowerDetectorAxis, CyclicStatisticAxis, CyclicLagAxis], np.float32
    ]

    power_detector: Coordof[PowerDetectorCoords]
    cyclic_statistic: Coordof[CyclicStatisticCoords]
    cyclic_lag: Coordof[CyclicLagCoords]

    standard_name: Attr[str] = 'Channel power'
    units: Attr[str] = 'dBm'


@as_registered_channel_analysis(CyclicChannelPower)
def cyclic_channel_power(
    iq,
    capture: structs.Capture,
    *,
    cyclic_period: float,
    detector_period: float,
    power_detectors: tuple[str, ...] = ('rms', 'peak'),
    cyclic_statistics: tuple[str, ...] = ('min', 'mean', 'max'),
):
    power_detectors = tuple(power_detectors)
    cyclic_statistics = tuple(cyclic_statistics)

    nested_ret: dict = iqwaveform.iq_to_cyclic_power(
        iq,
        1 / capture.sample_rate,
        cyclic_period=cyclic_period,
        detector_period=detector_period,
        detectors=power_detectors,
        cycle_stats=cyclic_statistics,
    )

    result = {}
    for det_name, cyc_stats in nested_ret.items():
        for cyc_name, power in cyc_stats.items():
            power_dB = iqwaveform.powtodB(power)
            result.setdefault(det_name, {}).setdefault(cyc_name, power_dB)

    return result
