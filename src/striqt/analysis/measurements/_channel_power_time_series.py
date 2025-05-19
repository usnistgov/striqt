from __future__ import annotations
import dataclasses
import typing

from xarray_dataclasses import AsDataArray, Coordof, Data, Attr

from ..lib.registry import measurement

from ..lib import specs, util

if typing.TYPE_CHECKING:
    import iqwaveform
    import numpy as np
    import pandas as pd
else:
    iqwaveform = util.lazy_import('iqwaveform')
    np = util.lazy_import('numpy')
    pd = util.lazy_import('pandas')

### Time elapsed dimension and coordinates
TimeElapsedAxis = typing.Literal['time_elapsed']


@dataclasses.dataclass
class TimeElapsedCoords:
    data: Data[TimeElapsedAxis, np.float32]
    standard_name: Attr[str] = 'Time elapsed'
    units: Attr[str] = 's'

    @staticmethod
    @util.lru_cache()
    def factory(capture: specs.Capture, spec: ChannelPowerTimeSeriesSpec):
        length = round(capture.duration / spec.detector_period)
        return pd.RangeIndex(length) * spec.detector_period


### Power detector dimension and coordinates
PowerDetectorAxis = typing.Literal['power_detector']


@dataclasses.dataclass
class PowerDetectorCoords:
    data: Data[PowerDetectorAxis, object]
    standard_name: Attr[str] = 'Power detector'

    @staticmethod
    @util.lru_cache()
    def factory(
        capture: specs.Capture, *, power_detectors: tuple[str], **kws
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


class ChannelPowerTimeSeriesSpec(specs.Analysis, kw_only=True, frozen=True):
    detector_period: float
    power_detectors: tuple[str, ...] = ('rms', 'peak')


@measurement(
    ChannelPowerTimeSeries,
    basis='channel_power',
    spec_type=ChannelPowerTimeSeriesSpec,
)
def channel_power_time_series(
    iq, capture: specs.Capture, **kwargs: typing.Unpack[ChannelPowerTimeSeriesSpec]
):
    spec = ChannelPowerTimeSeriesSpec.fromdict(kwargs)

    results = []
    for d in spec.power_detectors:
        power = iqwaveform.iq_to_bin_power(
            iq, kind=d, Ts=1 / capture.sample_rate, Tbin=spec.detector_period, axis=1
        )
        results.append(power)

    xp = iqwaveform.util.array_namespace(iq)
    results = xp.array(results)
    results = xp.moveaxis(results, -2, 0)
    results = iqwaveform.powtodB(power).astype('float32')

    enbw = capture.analysis_bandwidth or capture.sample_rate
    metadata = {
        'detector_period': spec.detector_period,
        'units': f'dBm/{enbw / 1e6} MHz',
    }

    return results, metadata
