from __future__ import annotations
import dataclasses
import functools
import typing

from xarray_dataclasses import AsDataArray, Coordof, Data, Attr

from ..lib.registry import measurement
from ._channel_power_time_series import PowerDetectorAxis, PowerDetectorCoords

from ..lib import specs, util

if typing.TYPE_CHECKING:
    import iqwaveform
    import numpy as np
else:
    iqwaveform = util.lazy_import('iqwaveform')
    np = util.lazy_import('numpy')


### Cyclic statistic axis and coordinate labels
CyclicStatisticAxis = typing.Literal['cyclic_statistic']


@dataclasses.dataclass
class CyclicStatisticCoords:
    data: Data[CyclicStatisticAxis, object]
    standard_name: Attr[str] = 'Cyclic statistic'

    @staticmethod
    @functools.lru_cache
    def factory(
        capture: specs.Capture,
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
        capture: specs.Capture,
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


@measurement(CyclicChannelPower)
def cyclic_channel_power(
    iq,
    capture: specs.Capture,
    *,
    cyclic_period: float,
    detector_period: float,
    power_detectors: tuple[str, ...] = ('rms', 'peak'),
    cyclic_statistics: tuple[typing.Union[str, float], ...] = ('min', 'mean', 'max'),
):
    xp = iqwaveform.util.array_namespace(iq)

    power_detectors = tuple(power_detectors)
    cyclic_statistics = tuple(cyclic_statistics)

    nested_ret = iqwaveform.iq_to_cyclic_power(
        iq,
        1 / capture.sample_rate,
        cyclic_period=cyclic_period,
        detector_period=detector_period,
        detectors=power_detectors,
        cycle_stats=cyclic_statistics,
        axis=1,
    )

    # pull arrays from the returned nested dict and combine into one ndarray
    x = xp.array([list(d.values()) for d in nested_ret.values()])

    # move the capture axis to the front
    x = xp.moveaxis(x, -2, 0)

    return iqwaveform.powtodB(x).astype('float32')
