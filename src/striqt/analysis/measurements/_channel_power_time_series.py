from __future__ import annotations as __

import fractions
import typing

from .. import specs

from ..lib import dataarrays, register, util
from .shared import registry, hint_keywords

if typing.TYPE_CHECKING:
    import numpy as np
    import pandas as pd

    import striqt.waveform as iqwaveform
else:
    iqwaveform = util.lazy_import('striqt.waveform')
    np = util.lazy_import('numpy')
    pd = util.lazy_import('pandas')


@registry.coordinates(
    dtype='float32', attrs={'standard_name': 'Time elapsed', 'units': 's'}
)
@util.lru_cache()
def time_elapsed(capture: specs.Capture, spec: specs.ChannelPowerTimeSeries):
    length = round(capture.duration / spec.detector_period)
    return pd.RangeIndex(length) * float(spec.detector_period)


@registry.coordinates(dtype=object, attrs={'standard_name': 'Power detector'})
@util.lru_cache()
def power_detector(
    capture: specs.Capture, spec: specs.ChannelPowerTimeSeries
) -> 'np.ndarray':
    return np.array(spec.power_detectors)


_channel_power_cache = register.KeywordArgumentCache([dataarrays.CAPTURE_DIM, 'spec'])


@_channel_power_cache.apply
def evaluate_channel_power_time_series(
    iq, capture: specs.Capture, spec: specs.ChannelPowerTimeSeries
):
    results = []
    for d in spec.power_detectors:
        power = iqwaveform.iq_to_bin_power(
            iq,
            kind=d,
            Ts=1 / capture.sample_rate,
            Tbin=float(spec.detector_period),
            axis=1,
        )
        results.append(power)

    xp = iqwaveform.util.array_namespace(iq)
    results = xp.array(results)
    results = xp.moveaxis(results, 0, 1)
    results = iqwaveform.powtodB(results).astype('float32')

    return results


@hint_keywords(specs.ChannelPowerTimeSeries)
@registry.measurement(
    coord_factories=[power_detector, time_elapsed],
    dtype='float32',
    spec_type=specs.ChannelPowerTimeSeries,
    caches=_channel_power_cache,
    attrs={'standard_name': 'Channel Power', 'units': 'dBm'},
)
def channel_power_time_series(iq, capture: specs.Capture, **kwargs):
    spec = specs.ChannelPowerTimeSeries.from_dict(kwargs)

    results = evaluate_channel_power_time_series(iq, capture=capture, spec=spec)

    return results, spec.to_dict()
