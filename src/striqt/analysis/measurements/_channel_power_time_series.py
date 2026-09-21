from __future__ import annotations as __

import typing

from .. import specs

from ..lib import dataarrays, register, util
from .shared import registry, hint_keywords

import striqt.waveform as sw

if typing.TYPE_CHECKING:
    import numpy as np
    import pandas as pd
else:
    np = util.lazy_import('numpy')
    pd = util.lazy_import('pandas')


class ChannelPowerBinning(typing.NamedTuple):
    """the detector binning implied by a (capture, channel power spec) combination"""

    bin_size: int
    bin_count: int


def validate_detector_period(
    capture: specs.Capture,
    spec: typing.Union[specs.ChannelPowerTimeSeries, specs.CyclicChannelPower],
) -> int:
    """check that `detector_period` spans a whole number of samples, returning that count"""
    if not sw.isroundmod(float(spec.detector_period), 1 / capture.sample_rate):
        raise ValueError(
            'detector_period must be a counting-number multiple of the sample period'
        )

    return round(float(spec.detector_period) * capture.sample_rate)


def validate_channel_power_time_series(
    capture: specs.Capture, spec: specs.ChannelPowerTimeSeries
) -> ChannelPowerBinning:
    """check that `detector_period` tiles the capture in whole samples.

    `sw.iq_to_bin_power` and `sw.axis_to_blocks` apply these same two rules once IQ is
    in hand; checking them here moves the failure ahead of the acquisition.
    """
    bin_size = validate_detector_period(capture, spec)

    if not sw.isroundmod(capture.duration, float(spec.detector_period)):
        raise ValueError(
            'duration must be a counting-number multiple of detector_period'
        )

    return ChannelPowerBinning(
        bin_size=bin_size,
        bin_count=round(capture.duration / float(spec.detector_period)),
    )


@registry.coordinates(
    dtype='float32', attrs={'standard_name': 'Time elapsed', 'units': 's'}
)
@util.lru_cache()
def time_elapsed(capture: specs.Capture, spec: specs.ChannelPowerTimeSeries):
    binning = validate_channel_power_time_series(capture, spec)
    return pd.RangeIndex(binning.bin_count) * float(spec.detector_period)


@registry.coordinates(dtype=object, attrs={'standard_name': 'Power detector'})
@util.lru_cache()
def power_detector(
    capture: specs.Capture, spec: specs.ChannelPowerTimeSeries
) -> 'np.ndarray':
    return np.array(spec.power_detectors)


_channel_power_cache = register.KwArgCache([dataarrays.CAPTURE_DIM, 'spec'])


@_channel_power_cache.apply
def evaluate_channel_power_time_series(
    iq, capture: specs.Capture, spec: specs.ChannelPowerTimeSeries
):
    results = []
    for d in spec.power_detectors:
        power = sw.iq_to_bin_power(
            iq,
            kind=d,
            Ts=1 / capture.sample_rate,
            Tbin=float(spec.detector_period),
            axis=1,
        )
        results.append(power)

    xp = sw.array_namespace(iq)
    results = xp.array(results)
    results = xp.moveaxis(results, 0, 1)
    results = sw.powtodB(results).astype('float32')

    return results


@hint_keywords(specs.ChannelPowerTimeSeries)
@registry.measurement(
    coord_factories=[power_detector, time_elapsed],
    dtype='float32',
    spec_type=specs.ChannelPowerTimeSeries,
    caches=_channel_power_cache,
    prefer_iq_source='aligned',
    attrs={'standard_name': 'Channel Power', 'units': 'dBm'},
    validate=validate_channel_power_time_series,
)
def channel_power_time_series(iq, capture: specs.Capture, **kwargs):
    """Compute a binned time series of channel power detector measurements.

    Args:
    {args}
    """
    spec = specs.ChannelPowerTimeSeries.from_dict(kwargs)

    results = evaluate_channel_power_time_series(iq, capture=capture, spec=spec)

    return results, spec.to_dict()
