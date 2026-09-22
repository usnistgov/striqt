from __future__ import annotations as __

import typing

from .. import specs

from ..lib import dataarrays, register, util
from . import shared
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


def validated_detector_bin_size(
    capture: specs.Capture,
    spec: typing.Union[specs.ChannelPowerTimeSeries, specs.CyclicChannelPower],
) -> int:
    """check that `detector_period` spans a whole number of samples, returning that count"""
    if not sw.isroundmod(float(spec.detector_period), 1 / capture.sample_rate):
        raise ValueError(
            'detector_period must be a counting-number multiple of the sample period '
            f'(detector_period: {float(spec.detector_period)}, '
            f'sample_rate: {capture.sample_rate})'
        )

    return round(float(spec.detector_period) * capture.sample_rate)


def validated_channel_power_binning(
    capture: specs.Capture, spec: specs.ChannelPowerTimeSeries
) -> ChannelPowerBinning:
    """check that `detector_period` tiles the capture in whole samples, returning the
    derived detector binning.

    `sw.iq_to_bin_power` and `sw.axis_to_blocks` apply these same two rules once IQ is
    in hand; checking them here moves the failure ahead of the acquisition.
    """
    bin_size = validated_detector_bin_size(capture, spec)

    if not sw.isroundmod(capture.duration, float(spec.detector_period)):
        raise ValueError(
            'duration must be a counting-number multiple of detector_period '
            f'(duration: {capture.duration}, '
            f'detector_period: {float(spec.detector_period)})'
        )

    return ChannelPowerBinning(
        bin_size=bin_size,
        bin_count=round(capture.duration / float(spec.detector_period)),
    )


def channel_power_tolerance(
    capture: specs.Capture,
    spec: specs.ChannelPowerTimeSeries,
    *,
    array_backend: sw.typing.ArrayBackend = 'numpy',
    input_error: float = 0.0,
) -> specs.Tolerance:
    """the roundoff budget of `channel_power_time_series` in dB, from the detector
    binning and the relative rms amplitude error `input_error` already in the IQ"""
    binning = validated_channel_power_binning(capture, spec)
    return shared.level_tolerance(
        amplitude_rms=input_error,
        size=binning.bin_count * len(spec.power_detectors),
        power_rtol=sw.bin_power_rtol(np.float32, binning.bin_size),
        log_tol=sw.log_conversion_tol(np.float32, 10, complex_input=True),
        quantization=shared.quantization_dB('float32'),
    )


@registry.coordinates(
    dtype='float32', attrs={'standard_name': 'Time elapsed', 'units': 's'}
)
@specs.helpers.lru_cache_on_converted(specs.Capture)
def time_elapsed(capture: specs.Capture, spec: specs.ChannelPowerTimeSeries):
    binning = validated_channel_power_binning(capture, spec)
    return pd.RangeIndex(binning.bin_count) * float(spec.detector_period)


@registry.coordinates(dtype=object, attrs={'standard_name': 'Power detector'})
@specs.helpers.lru_cache_on_converted(specs.Capture)
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
    validate=validated_channel_power_binning,
    tolerance=channel_power_tolerance,
)
def channel_power_time_series(iq, capture: specs.Capture, **kwargs):
    """Compute a binned time series of channel power detector measurements.

    Args:
    {args}
    """
    spec = specs.ChannelPowerTimeSeries.from_dict(kwargs)

    results = evaluate_channel_power_time_series(iq, capture=capture, spec=spec)

    return results, spec.to_dict()
