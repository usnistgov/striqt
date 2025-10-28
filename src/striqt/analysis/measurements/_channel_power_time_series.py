from __future__ import annotations
import fractions
import typing

from .shared import registry
from ..lib import dataarrays, register, specs, util

if typing.TYPE_CHECKING:
    import iqwaveform
    import numpy as np
    import pandas as pd
else:
    iqwaveform = util.lazy_import('iqwaveform')
    np = util.lazy_import('numpy')
    pd = util.lazy_import('pandas')


class ChannelPowerTimeSeriesSpec(
    specs.Measurement,
    forbid_unknown_fields=True,
    cache_hash=True,
    kw_only=True,
    frozen=True,
):
    detector_period: fractions.Fraction
    power_detectors: tuple[str, ...] = ('rms', 'peak')


class ChannelPowerTimeSeriesKeywords(specs.AnalysisKeywords):
    # for setting the type hints in **kwargs below
    detector_period: fractions.Fraction
    power_detectors: typing.NotRequired[tuple[str, ...]]


@registry.coordinates(
    dtype='float32', attrs={'standard_name': 'Time elapsed', 'units': 's'}
)
@util.lru_cache()
def time_elapsed(capture: specs.Capture, spec: ChannelPowerTimeSeriesSpec):
    length = round(capture.duration / spec.detector_period)
    return pd.RangeIndex(length) * spec.detector_period


@registry.coordinates(dtype=object, attrs={'standard_name': 'Power detector'})
@util.lru_cache()
def power_detector(
    capture: specs.Capture, spec: ChannelPowerTimeSeriesSpec
) -> 'np.ndarray':
    return np.array(spec.power_detectors)


_channel_power_cache = register.KeywordArgumentCache([dataarrays.CAPTURE_DIM, 'spec'])


@_channel_power_cache.apply
def evaluate_channel_power_time_series(
    iq, capture: specs.Capture, spec: ChannelPowerTimeSeriesSpec
):
    results = []
    for d in spec.power_detectors:
        power = iqwaveform.iq_to_bin_power(
            iq, kind=d, Ts=1 / capture.sample_rate, Tbin=spec.detector_period, axis=1
        )
        results.append(power)

    xp = iqwaveform.util.array_namespace(iq)
    results = xp.array(results)
    results = xp.moveaxis(results, 0, 1)
    results = iqwaveform.powtodB(results).astype('float32')

    return results


@registry.measurement(
    coord_factories=[power_detector, time_elapsed],
    dtype='float32',
    spec_type=ChannelPowerTimeSeriesSpec,
    caches=_channel_power_cache,
    attrs={'standard_name': 'Channel Power', 'units': 'dBm'},
)
def channel_power_time_series(
    iq, capture: specs.Capture, **kwargs: typing.Unpack[ChannelPowerTimeSeriesKeywords]
):
    spec = ChannelPowerTimeSeriesSpec.fromdict(kwargs)

    results = evaluate_channel_power_time_series(iq, capture=capture, spec=spec)

    return results, spec.todict()
