from __future__ import annotations as __

import typing
from typing import Any

import striqt.waveform as sw

from .. import specs
from ..lib import dataarrays, register, util
from ..lib.util import np
from . import shared
from .shared import hint_keywords, registry

if typing.TYPE_CHECKING:
    from ..lib.typing import Array, Measurement


# %% channel_power_time_series
class ChannelPowerBinning(typing.NamedTuple):
    """the detector binning of one capture."""

    bin_size: int
    """samples per detector bin"""
    bin_count: int
    """detector bins across the capture"""


def validated_detector_bin_size(
    capture: specs.Capture,
    spec: typing.Union[specs.ChannelPowerTimeSeries, specs.CyclicChannelPower],
) -> int:
    """return the number of samples in one `detector_period` at `capture.sample_rate`.

    Raises:
        ValueError: if `detector_period` is not a whole number of sample periods
    """
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
    """return the detector binning of `capture` under `spec`.

    Raises:
        ValueError: if `detector_period` is not a whole number of samples, if
            `duration` is not a whole number of detector periods, or if an entry of
            `power_detectors` is not a supported statistic
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
    """bound the roundoff (in dB) of `channel_power_time_series`.

    Args:
        input_error: relative rms error of the IQ amplitude (linear, unitless)
        array_backend: has no effect on this bound

    Returns:
        the `specs.Tolerance` in dB over the ``(power_detector, time_elapsed)`` output
        elements
    """
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
def time_elapsed(
    capture: specs.Capture, spec: specs.ChannelPowerTimeSeries
) -> np.ndarray:
    binning = validated_channel_power_binning(capture, spec)
    return np.arange(binning.bin_count) * float(spec.detector_period)


@registry.coordinates(dtype=object, attrs={'standard_name': 'Power detector'})
@specs.helpers.lru_cache_on_converted(specs.Capture)
def power_detector(
    capture: specs.Capture, spec: specs.ChannelPowerTimeSeries
) -> np.ndarray:
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
    attrs={'standard_name': 'Channel Power', 'units': 'dBm'},
    validate=validated_channel_power_binning,
    tolerance=channel_power_tolerance,
)
def channel_power_time_series(
    x: Array, capture: specs.Capture, **kwargs: Any
) -> Measurement:
    """evaluate a time series of binned channel power for each power detector.

    The input is the IQ after the `analysis_bandwidth` filter and trigger alignment.
    Each detector reduces the instantaneous power ``|iq|**2`` over consecutive blocks
    of `detector_period`: ``'rms'`` is the mean and ``'peak'`` the maximum. Per
    capture, the output has dimensions ``(power_detector, time_elapsed)`` in dBm as
    float32. `time_elapsed` is in s from the start of the capture in steps of
    `detector_period`. The capture `duration` must be a whole number of detector
    periods. `detector_period` must be a whole number of samples at `sample_rate`.
    The output has no `noise_bandwidth` attr. The noise bandwidth of each capture is
    the `analysis_bandwidth` coordinate, or `sample_rate` where `analysis_bandwidth`
    is infinite.

    Args:
    {args}
    """
    spec = specs.ChannelPowerTimeSeries.from_dict(kwargs)

    results = evaluate_channel_power_time_series(x, capture=capture, spec=spec)

    # no noise_bandwidth attr: the bandwidth is the analysis_bandwidth (or sample_rate)
    # of the capture, which varies between the captures of one sweep, and a variable
    # attr is written once per store
    return results, spec.to_dict()


# %% channel_power_histogram
@util.lru_cache()
def make_power_bins(power_low, power_high, power_resolution, xp=np):
    """return histogram bin centers (in dB) with catch-all bins at each end.

    The finite centers run from `power_low` toward `power_high` in steps of
    `power_resolution`. The last finite center is `power_high` or lies within half a
    step below `power_high`. The returned array is cached and shared between callers
    and must not be modified in place.

    Args:
        power_low: the first finite center, in dB
        power_high: the upper end of the finite centers, in dB
        power_resolution: the step between centers, in dB
        xp: the array namespace of the result

    Returns:
        1-D array of the finite centers bracketed by ``-inf`` and ``inf``
    """
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
    """return the `xp.histogram` bin edges (in dB) that pair with `make_power_bins`.

    The finite edges lie midway between adjacent centers and extend half a step below
    `power_low` and half a step above `power_high`. The first and last edges are
    ``-inf`` and ``inf``. The returned array is cached and shared between callers and
    must not be modified in place.

    Args:
        power_low: as for `make_power_bins`
        power_high: as for `make_power_bins`
        power_resolution: as for `make_power_bins`
        xp: the array namespace of the result

    Returns:
        1-D array one element longer than `make_power_bins`
    """

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
) -> tuple[np.ndarray, dict[str, Any]]:
    """return the `channel_power_bin` coordinate values and attrs.

    Returns:
        the bin centers of `make_power_bins` in dBm, with ``-inf`` and ``inf``
        catch-all bins at each end, and an empty attrs dict
    """
    bins = make_power_bins(spec.power_low, spec.power_high, spec.power_resolution)
    # no noise_bandwidth attr; see channel_power_time_series
    return bins, {}


@hint_keywords(specs.ChannelPowerHistogram)
@registry.measurement(
    coord_factories=[power_detector, channel_power_bin],
    depends=[channel_power_time_series],
    spec_type=specs.ChannelPowerHistogram,
    dtype='float32',
    attrs={'standard_name': 'Fraction of channel power readings'},
    validate=validated_channel_power_binning,
)
def channel_power_histogram(
    x: Array, capture: specs.Capture, **kwargs: Any
) -> Measurement:
    """evaluate the fraction of channel power readings that fall in each power bin.

    The input is the IQ after the `analysis_bandwidth` filter and trigger alignment.
    The readings are those of `channel_power_time_series` with the same detector
    fields. Per capture, the output has dimensions
    ``(power_detector, channel_power_bin)`` as float32 and sums to 1 over the bins
    of each detector. `channel_power_bin` is the bin centers in dBm. The ``-inf`` and
    ``inf`` entries collect the readings more than half a bin below `power_low` or
    above `power_high`. Neither the output nor the bin coordinate has a
    `noise_bandwidth` attr. The noise bandwidth of each capture is the
    `analysis_bandwidth` coordinate, or `sample_rate` where `analysis_bandwidth` is
    infinite.

    Args:
    {args}
    """

    spec = specs.ChannelPowerHistogram.from_dict(kwargs)

    xp = sw.array_namespace(x)

    bin_edges = make_power_histogram_bin_edges(
        power_low=spec.power_low,
        power_high=spec.power_high,
        power_resolution=spec.power_resolution,
        xp=xp,
    )

    pvt_spec = specs.ChannelPowerTimeSeries.from_spec(spec)
    power_dB = evaluate_channel_power_time_series(
        x,
        capture=capture,
        spec=pvt_spec,
    )

    count_dtype = xp.finfo(x.dtype).dtype

    data = []
    for i_chan in range(power_dB.shape[0]):
        counts = []
        for i_detector in range(power_dB.shape[1]):
            hist = xp.histogram(power_dB[i_chan, i_detector], bin_edges)[0]
            counts.append(hist)
        counts = xp.asarray(counts, dtype=count_dtype)
        data.append(counts / (xp.sum(counts) / power_dB.shape[1]))

    data = xp.asarray(data, dtype=count_dtype)

    # no noise_bandwidth attr; see channel_power_time_series
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
    """return the number of detector bins in one `cyclic_period`.

    Returns:
        the length of the `cyclic_lag` coordinate

    Raises:
        ValueError: if `detector_period` is not a whole number of samples, if
            `cyclic_period` is not a whole number of detector periods, if `duration`
            is not a whole number of cycles, or if an entry of `power_detectors` or
            `cyclic_statistics` is not a supported statistic
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
    """bound the roundoff (in dB) of `cyclic_channel_power`.

    Args:
        input_error: relative rms error of the IQ amplitude (linear, unitless)
        array_backend: has no effect on this bound

    Returns:
        the `specs.Tolerance` in dB over the
        ``(power_detector, cyclic_statistic, cyclic_lag)`` output elements
    """
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
        # the cycle axis precedes the lag axis. A strided reduction takes the
        # ordered-sum bound, not the rms bound.
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
def cyclic_lag(capture: specs.Capture, spec: specs.CyclicChannelPower) -> np.ndarray:
    lag_count = validated_cyclic_lag_count(capture, spec)

    return np.arange(lag_count) * float(spec.detector_period)


@hint_keywords(specs.CyclicChannelPower)
@registry.measurement(
    coord_factories=[power_detector, cyclic_statistic, cyclic_lag],
    spec_type=specs.CyclicChannelPower,
    dtype='float32',
    attrs={'standard_name': 'Cyclic channel power', 'units': 'dBm'},
    validate=validated_cyclic_lag_count,
    tolerance=cyclic_channel_power_tolerance,
)
def cyclic_channel_power(
    x: Array, capture: specs.Capture, **kwargs: Any
) -> Measurement:
    """evaluate cyclic statistics of channel power across the cycles in a capture.

    The input is the IQ after the `analysis_bandwidth` filter and trigger alignment.
    Each power detector bins the capture on `detector_period`. The binned series is
    folded into cycles of `cyclic_period`. Each cyclic statistic reduces across the
    cycles (Kuester et al.). Per capture, the output has dimensions
    ``(power_detector, cyclic_statistic, cyclic_lag)`` in dBm as float32. The
    `cyclic_lag` coordinate in s starts at 0 and spans one `cyclic_period` in steps
    of `detector_period`. Those 2 fields alone set the trace length, independent of
    the capture `duration`.

    A signal with a period that divides `cyclic_period` (10 ms covers TDD cellular
    frames, 5 ms WiMAX frames, and the 1 ms CBRS test ``bin 1`` pulse repetition
    interval) resolves at fixed lags with little spread between the ``'min'`` and
    ``'max'`` statistics. The uplink and downlink levels of a TDD network then
    occupy disjoint lag windows. Occupancy with an incommensurate period smears
    across all lags.

    Statistics are evaluated in linear power and converted to dBm afterward. The
    capture `duration` must be a whole number of cycles. `cyclic_period` must be a
    whole number of detector periods, and `detector_period` a whole number of
    samples at `sample_rate`. A `detector_period` written as a fraction
    (``1/28000`` s is one OFDM symbol at 30 kHz subcarrier spacing) stays exact.
    The output has no `noise_bandwidth` attr, as for `channel_power_time_series`.

    References:
        D. G. Kuester et al., "Cyclic Analysis of Power in Radio Channels."

    Args:
    {args}
    """
    spec = specs.CyclicChannelPower.from_dict(kwargs)

    xp = sw.array_namespace(x)

    nested_ret = sw.iq_to_cyclic_power(
        x,
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

    # no noise_bandwidth attr; see channel_power_time_series
    return sw.powtodB(x).astype('float32')
