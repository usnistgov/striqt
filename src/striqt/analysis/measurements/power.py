from __future__ import annotations as __

import typing

from .. import specs

from ..lib import dataarrays, register, util
from ..lib.util import np, pd
from . import shared
from .shared import registry, hint_keywords

import striqt.waveform as sw


# %% channel_power_time_series
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
    shared.check_statistics('power_detectors', spec.power_detectors)
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
        power_rtol=max(
            sw.power_analysis.bin_power_rtol(np.float32, kind=d)
            for d in spec.power_detectors
        ),
        power_rms=max(
            sw.power_analysis.bin_power_rms(np.float32, binning.bin_size, kind=d)
            for d in spec.power_detectors
        ),
        log_tol=sw.power_analysis.log_conversion_tol(
            np.float32, 10, complex_input=True
        ),
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


# %% channel_power_histogram
@util.lru_cache()
def make_power_bins(power_low, power_high, power_resolution, xp=np):
    """generate the list of power bins"""
    ret = xp.arange(power_low, power_high, power_resolution)
    if power_high - ret[-1] > power_resolution / 2:
        ret = xp.pad(ret, [[0, 1]], mode='constant', constant_values=power_high).copy()

    # catch-all bins for counts outside of the specified range
    bottom_inf = xp.array([float('-inf')])
    top_inf = xp.array([float('inf')])
    ret = xp.concatenate((bottom_inf, ret, top_inf))
    return ret


@util.lru_cache()
def make_power_histogram_bin_edges(power_low, power_high, power_resolution, xp=np):
    """generate the list of power bins"""

    bin_centers = (
        make_power_bins(
            power_low=power_low,
            power_high=power_high,
            power_resolution=power_resolution,
            xp=xp,
        )
        + power_resolution / 2
    )

    top_edge = xp.array([
        float(power_high + power_resolution / 2),
        float(bin_centers[-1]),
    ])
    return xp.concatenate((bin_centers[:-1] - power_resolution, top_edge))


@registry.coordinates(
    dtype='float32', attrs={'standard_name': 'Channel power', 'units': 'dBm'}
)
@specs.helpers.lru_cache_on_converted(specs.Capture)
def channel_power_bin(
    capture: specs.Capture, spec: specs.ChannelPowerHistogram
) -> dict[str, np.ndarray]:
    """returns a dictionary of coordinate values, keyed by axis dimension name"""
    return make_power_bins(spec.power_low, spec.power_high, spec.power_resolution)


@hint_keywords(specs.ChannelPowerHistogram)
@registry.measurement(
    coord_factories=[power_detector, channel_power_bin],
    depends=[channel_power_time_series],
    spec_type=specs.ChannelPowerHistogram,
    dtype='float32',
    prefer_iq_source='aligned',
    attrs={'standard_name': 'Fraction of channel power readings'},
    validate=validated_channel_power_binning,
)
def channel_power_histogram(iq, capture: specs.Capture, **kwargs):
    """evaluate the fraction of channel power readings binned on a uniform grid spacing.

    The outputs correspond to bin centers.

    Args:
    {args}
    """

    spec = specs.ChannelPowerHistogram.from_dict(kwargs)

    xp = sw.array_namespace(iq)

    bin_edges = make_power_histogram_bin_edges(
        power_low=spec.power_low,
        power_high=spec.power_high,
        power_resolution=spec.power_resolution,
        xp=xp,
    )

    pvt_spec = specs.ChannelPowerTimeSeries.from_spec(spec)
    power_dB = evaluate_channel_power_time_series(
        iq,
        capture=capture,
        spec=pvt_spec,
    )

    count_dtype = xp.finfo(iq.dtype).dtype

    data = []
    for i_chan in range(power_dB.shape[0]):
        counts = []
        for i_detector in range(power_dB.shape[1]):
            hist = xp.histogram(power_dB[i_chan, i_detector], bin_edges)[0]
            counts.append(hist)
        counts = xp.asarray(counts, dtype=count_dtype)
        data.append(counts / (xp.sum(counts) / power_dB.shape[1]))

    data = xp.asarray(data, dtype=count_dtype)

    metadata = {'detector_period': spec.detector_period}

    return data, metadata


# %% cyclic_channel_power
@registry.coordinates(dtype=object, attrs={'standard_name': 'Cyclic statistic'})
@specs.helpers.lru_cache_on_converted(specs.Capture)
def cyclic_statistic(capture: specs.Capture, spec: specs.CyclicChannelPower):
    return list(spec.cyclic_statistics)


def validated_cyclic_lag_count(
    capture: specs.Capture, spec: specs.CyclicChannelPower
) -> int:
    """check that the sample, detector and cycle periods nest evenly.

    `sw.iq_to_cyclic_power` applies these same three rules once IQ is in hand; checking
    them here moves the failure ahead of the acquisition.

    Returns:
        the number of detector bins in one cycle, i.e. the length of `cyclic_lag`
    """
    shared.check_statistics('power_detectors', spec.power_detectors)
    shared.check_statistics('cyclic_statistics', spec.cyclic_statistics)
    validated_detector_bin_size(capture, spec)

    detector_period = float(spec.detector_period)

    if not sw.isroundmod(spec.cyclic_period, detector_period):
        raise ValueError(
            'cyclic_period must be a counting-number multiple of detector_period '
            f'(cyclic_period: {spec.cyclic_period}, '
            f'detector_period: {detector_period})'
        )

    if not sw.isroundmod(capture.duration, spec.cyclic_period):
        raise ValueError(
            'duration must be a counting-number multiple of cyclic_period '
            f'(duration: {capture.duration}, cyclic_period: {spec.cyclic_period})'
        )

    return round(spec.cyclic_period / detector_period)


def cyclic_channel_power_tolerance(
    capture: specs.Capture,
    spec: specs.CyclicChannelPower,
    *,
    array_backend: sw.typing.ArrayBackend = 'numpy',
    input_error: float = 0.0,
) -> specs.Tolerance:
    """the roundoff budget of `cyclic_channel_power` in dB, from the detector binning,
    the number of cycles each statistic reduces over, and the relative rms amplitude
    error `input_error` already in the IQ"""
    lag_count = validated_cyclic_lag_count(capture, spec)
    bin_size = validated_detector_bin_size(capture, spec)
    cycle_count = round(capture.duration / spec.cyclic_period)
    power_rtol = max(
        sw.power_analysis.bin_power_rtol(np.float32, kind=d)
        for d in spec.power_detectors
    )
    power_rms = max(
        sw.power_analysis.bin_power_rms(np.float32, bin_size, kind=d)
        for d in spec.power_detectors
    )
    power_rtol += max(
        sw.power_analysis.stat_rtol(np.float32, kind=s) for s in spec.cyclic_statistics
    )
    if any(s in ('mean', 'rms') for s in spec.cyclic_statistics):
        # the cycle axis sits ahead of the lag axis, so the reduction is strided
        power_rtol += sw.arrays.accum_rtol(np.float32, cycle_count)
    return shared.level_tolerance(
        amplitude_rms=input_error,
        size=lag_count * len(spec.power_detectors) * len(spec.cyclic_statistics),
        power_rtol=power_rtol,
        power_rms=power_rms,
        log_tol=sw.power_analysis.log_conversion_tol(
            np.float32, 10, complex_input=True
        ),
        quantization=shared.quantization_dB('float32'),
    )


@registry.coordinates(
    dtype='float32', attrs={'standard_name': 'Cyclic lag', 'units': 's'}
)
@specs.helpers.lru_cache_on_converted(specs.Capture)
def cyclic_lag(capture: specs.Capture, spec: specs.CyclicChannelPower):
    lag_count = validated_cyclic_lag_count(capture, spec)

    return np.arange(lag_count) * float(spec.detector_period)


@hint_keywords(specs.CyclicChannelPower)
@registry.measurement(
    coord_factories=[power_detector, cyclic_statistic, cyclic_lag],
    spec_type=specs.CyclicChannelPower,
    dtype='float32',
    prefer_iq_source='aligned',
    attrs={'standard_name': 'Cyclic channel power', 'units': 'dBm'},
    validate=validated_cyclic_lag_count,
    tolerance=cyclic_channel_power_tolerance,
)
def cyclic_channel_power(iq, capture: specs.Capture, **kwargs):
    """Evaluate cyclic statistics of channel power across the cycles in a capture.

    Each power detector bins the capture on `detector_period`; the binned series is
    folded into cycles of `cyclic_period`, and each cyclic statistic reduces across
    the cycles, following D.G. Kuester et al., "Cyclic Analysis of Power in Radio
    Channels". Per capture, the result has
    dimensions ``(power_detector, cyclic_statistic, cyclic_lag)``. The `cyclic_lag`
    coordinate runs from 0 to ``cyclic_period - detector_period`` in steps of
    `detector_period`, so every trace holds ``cyclic_period / detector_period``
    samples regardless of the capture `duration`.

    Signals whose period divides `cyclic_period` (e.g., 10 ms covers TDD cellular frames,
    5 ms WiMAX frames and the 1 ms CBRS test ``bin 1'' pulse repetition interval) resolve
    at fixed lags with little spread between the 'min' and 'max' statistics, so the
    uplink and downlink levels of a TDD network can be read from disjoint lag
    windows, while occupancy with an incommensurate period smears across all lags.

    Statistics are evaluated in linear power and converted to dBm afterwards. The
    capture `duration` must be a whole number of cycles, `cyclic_period` a whole
    number of detector periods, and `detector_period` a whole number of samples at
    `sample_rate`; write it as a fraction (``1/28000`` s is one OFDM symbol at
    30 kHz subcarrier spacing) to keep that exact.

    Args:
    {args}
    """
    spec = specs.CyclicChannelPower.from_dict(kwargs)

    xp = sw.array_namespace(iq)

    nested_ret = sw.iq_to_cyclic_power(
        iq,
        1 / capture.sample_rate,
        cyclic_period=spec.cyclic_period,
        detector_period=float(spec.detector_period),
        detectors=spec.power_detectors,
        cycle_stats=spec.cyclic_statistics,
        axis=1,
    )

    # pull arrays from the returned nested dict and combine into one ndarray
    x = xp.array([list(d.values()) for d in nested_ret.values()])

    # move the capture axis to the front
    x = xp.moveaxis(x, -2, 0)

    return sw.powtodB(x).astype('float32')
