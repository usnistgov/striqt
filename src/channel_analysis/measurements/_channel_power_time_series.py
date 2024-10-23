from __future__ import annotations
import dataclasses
import functools
import typing

from xarray_dataclasses import AsDataArray, Coordof, Data, Attr

from ._common import as_registered_channel_analysis

from .._api import structs, util

if typing.TYPE_CHECKING:
    import iqwaveform
    import numpy as np
else:
    iqwaveform = util.lazy_import('iqwaveform')
    np = util.lazy_import('numpy')

### Time elapsed dimension and coordinates
TimeElapsedAxis = typing.Literal['time_elapsed']


@dataclasses.dataclass
class TimeElapsedCoords:
    data: Data[TimeElapsedAxis, np.float32]
    standard_name: Attr[str] = 'Time elapsed'
    units: Attr[str] = 's'

    @staticmethod
    @functools.lru_cache
    def factory(
        capture: structs.Capture, *, detector_period: float, **_
    ) -> dict[str, np.ndarray]:
        import pandas as pd

        length = round(capture.duration / detector_period)
        return pd.RangeIndex(length) * detector_period


### Power detector dimension and coordinates
PowerDetectorAxis = typing.Literal['power_detector']


@dataclasses.dataclass
class PowerDetectorCoords:
    data: Data[PowerDetectorAxis, object]
    standard_name: Attr[str] = 'Power detector'

    @staticmethod
    @functools.lru_cache
    def factory(
        capture: structs.Capture, *, power_detectors: tuple[str], **kws
    ) -> dict[str, list]:
        return np.array(list(power_detectors))


### Dataarray definition
@dataclasses.dataclass
class ChannelPowerTimeSeries(AsDataArray):
    power_time_series: Data[tuple[PowerDetectorAxis, TimeElapsedAxis], np.float32]

    power_detector: Coordof[PowerDetectorCoords]
    time_elapsed: Coordof[TimeElapsedCoords]

    standard_name: Attr[str] = 'Channel power'
    units: Attr[str] = 'dBm'


### iqwaveform implementation
@as_registered_channel_analysis(ChannelPowerTimeSeries)
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
        'units': f'dBm/{(capture.analysis_bandwidth or capture.sample_rate)/1e6} MHz',
    }

    return data, metadata
