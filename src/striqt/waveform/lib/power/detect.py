"""Power detectors binned in time along an axis, their cyclic statistics, and the
roundoff models the tests measure them against."""

from __future__ import annotations as __

import typing
from functools import partial
from numbers import Number

from .. import util
from ..arrays import (
    ROUNDOFF_SAFETY,
    accum_rms,
    array_namespace,
    axis_slice,
    axis_to_blocks,
    float_dtype_like,
    isroundmod,
    unit_roundoff,
)
from ..util import np
from .convert import envelope_power_rtol, envtopow

if typing.TYPE_CHECKING:
    from ..typing import _AT, Array, DTypeLike


@util.lru_cache()
def stat_ufunc_from_shorthand(kind: str | float, xp=None, axis=0) -> typing.Callable:
    if xp is None:
        xp = np

    NAMED_UFUNCS = {
        'min': xp.min,
        'max': xp.max,
        'peak': xp.max,
        'mean': xp.mean,
        'rms': xp.mean,
    }

    if hasattr(xp, 'median'):
        NAMED_UFUNCS['median'] = xp.median

    if isinstance(kind, str):
        if kind not in NAMED_UFUNCS:
            valid = NAMED_UFUNCS.keys()
            raise ValueError(f'kind argument must be one of {valid}')
        ufunc = partial(NAMED_UFUNCS[kind], axis=axis)

    elif isinstance(kind, Number):
        if not 0 <= kind <= 1:
            raise ValueError(f'quantile {kind!r} is outside the range [0, 1]')

        # numpy < 2.3 casts a scalar q to a float32 input's dtype
        # (numpy.lib._function_base_impl.quantile), which misplaces the virtual
        # index (n-1)*q by ~n*2**-25 samples in the interpolation. An array q
        # is not cast, but promotes the result, so it is cast back
        q = xp.asarray(kind, dtype='float64')

        def ufunc(x: Array, axis: int = axis) -> Array:
            return xp.quantile(x, q, axis=axis).astype(x.dtype, copy=False)

    elif callable(kind):
        ufunc = partial(kind, axis=axis)

    else:
        raise ValueError(f'invalid statistic ufunc "{kind}"')

    return ufunc


def stat_rtol(dtype: DTypeLike, kind: str | float, n_impl: int = 1) -> float:
    """return the worst-case rtol of the `kind` statistic applied to exact samples.

    `kind` is a shorthand of `stat_ufunc_from_shorthand`. The result is 0 for a
    selection or a mean, whose accumulation roundoff is `bin_power_rms`, and a fixed
    multiple of the unit roundoff of `dtype` for a quantile or the median, which
    interpolate once. `n_impl` is 1 against exact math and 2 against a second
    implementation.

    Raises:
        ValueError: if `kind` is an unknown statistic name
    """
    if kind in ('mean', 'rms', 'min', 'max', 'peak'):
        return 0.0
    elif isinstance(kind, str) and kind != 'median':
        raise ValueError(f'unknown statistic {kind!r}')
    return ROUNDOFF_SAFETY * n_impl * unit_roundoff(dtype)


def bin_power_rtol(
    dtype: DTypeLike, n_impl: int = 1, kind: str | float = 'mean'
) -> float:
    """return the pre-accumulation rtol of the `kind` statistic of ``|x|**2``.

    The bound covers squaring the envelope of complex samples and `stat_rtol` for the
    `kind` shorthand of `stat_ufunc_from_shorthand`; the mean's accumulation roundoff
    is `bin_power_rms`.
    """
    envelope = envelope_power_rtol(dtype, complex_input=True, n_impl=n_impl)
    return envelope + stat_rtol(dtype, kind, n_impl)


def bin_power_rms(
    dtype: DTypeLike, size: int, n_impl: int = 1, kind: str | float = 'mean'
) -> float:
    """return the rms accumulation roundoff of the `kind` statistic of ``|x|**2``.

    The roundoff is relative to the result over `size` samples of a contiguous axis:
    `accum_rms` for the mean (``'rms'`` is the same mean of power), 0 for a
    selection or a quantile.
    """
    if kind in ('mean', 'rms'):
        return accum_rms(dtype, size, n_impl)
    return 0.0


def iq_to_bin_power(
    iq: Array,
    Ts: float,
    Tbin: float,
    randomize: bool = False,
    kind: str = 'mean',
    truncate=False,
    axis=0,
):
    """computes power along the rows of `iq` (time axis) on bins of duration Tbin.

    Args:
        iq: complex-valued input waveform samples
        Ts: sample period of the input waveform
        Tbin: time duration of the bin size
        randomize: if True, randomize the start locations of the bins; otherwise, bins are contiguous
        kind: a named statistic ('max', 'mean', 'median', 'min', 'peak', 'rms'), a quantile, or a callable ufunc
        truncate: if True, truncate the last samples of `iq` to an integer number of bins
    """

    xp = array_namespace(iq)

    if axis < 0:
        axis = iq.ndim + axis

    if truncate or isroundmod(Tbin, Ts):
        N = round(Tbin / Ts)
    else:
        raise ValueError(
            f'bin period ({Tbin} s) must be multiple of waveform sample period ({Ts})'
        )

    # instantaneous power, reshaped into bins
    if randomize:
        if axis != 0:
            raise ValueError('only axis=0 is currently supported when randomize=True')

        size = int(np.floor(iq.shape[0] / N))
        starts = xp.random.randint(0, iq.shape[0] - N, size)
        offsets = xp.arange(N)
        iq_blocks = iq[starts[:, np.newaxis] + offsets[np.newaxis, :]]
    else:
        iq_blocks = axis_to_blocks(iq, N, axis=axis, truncate=truncate)

    detector = stat_ufunc_from_shorthand(kind, xp=xp, axis=axis + 1)
    power_bins = envtopow(iq_blocks)

    return detector(power_bins).astype(float_dtype_like(iq))


def iq_to_cyclic_power(
    x: Array,
    Ts: float,
    detector_period: float,
    cyclic_period: float,
    truncate=False,
    detectors=('rms', 'peak'),
    cycle_stats=('min', 'mean', 'max'),
    axis=0,
) -> dict[str, dict[str, Array]]:
    """return cyclic statistics of the channel power of `x` binned along `axis`.

    Channel power along `axis` is first binned with each power detector on
    `detector_period`, giving a time series of ``K`` detector samples. That series
    is re-indexed as a matrix of cycles and cycle lags: detector sample ``k`` lands
    in cycle ``k // L`` at lag ``(k % L) * detector_period``, where
    ``L = cyclic_period / detector_period`` is the number of detector bins per
    cycle and ``M = K / L`` is the number of cycles in the capture. Each cyclic
    statistic then reduces the cycle axis, so the value at lag index ``m``
    summarizes the ``M`` power samples that share the same time offset from the
    start of a cycle. The result spans one `cyclic_period` at the resolution of
    `detector_period` with ``L`` samples per detector and statistic, independent
    of the capture length.

    Choosing `cyclic_period` as a common multiple of the periods of the expected
    signals (for example 10 ms for the LCM of TDD cellular frames, 5 ms WiMAX
    frames, and the 1 ms pulse repetition interval of the SPN-43 radar) aligns their
    features across cycles. Each then resolves at fixed lags, while occupancy with
    an incommensurate period is spread across all lags. Uplink and downlink levels
    of a TDD network can then be read from disjoint lag windows. Statistics are
    evaluated in linear power units, not dB.

    A mismatch ``dT`` between `cyclic_period` and the true signal period, such as
    a sample clock offset, drifts features by ``M * dT / detector_period`` lags by
    the end of the capture and bleeds power between neighboring lags. Features
    stay within one lag while ``M * dT`` is below `detector_period`, which limits
    the number of cycles ``M``.

    Args:
        x: complex-valued IQ waveform samples
        Ts: sample period of the waveform (in s)
        detector_period: duration of each power detector bin (in s), an integer
            multiple of `Ts`
        cyclic_period: duration of one cycle (in s), an integer multiple of
            `detector_period`
        truncate: True to drop trailing samples that do not complete a detector bin
            or a cycle, and to round `detector_period` to the nearest multiple of `Ts`
        detectors: power detector names accepted by `iq_to_bin_power`
        cycle_stats: statistics accepted by `stat_ufunc_from_shorthand`, evaluated
            across cycles (names such as ``'min'``, ``'mean'``, ``'max'``, or
            quantiles in (0, 1))
        axis: the time axis of `x`

    Returns:
        dict keyed on detector, of dicts keyed on cyclic statistic, of arrays in the
        namespace of `x` with the `axis` dimension replaced by the ``L`` cycle lags,
        in linear power units

    Raises:
        ValueError: if `detectors` is None; if `cyclic_period` is not an integer
            multiple of `detector_period`; or, when `truncate` is False, if
            `detector_period` is not an integer multiple of `Ts` or the capture
            does not contain a whole number of cycles

    References:
        D.G. Kuester et al., "Cyclic Analysis of Power in Radio Channels".
    """

    # apply the detector statistic
    xp = array_namespace(x)

    if axis < 0:
        axis = x.ndim + axis

    # the binned power is computed here from the IQ, so detectors are required
    if detectors is None:
        raise ValueError(
            'supply detectors argument to evaluate binned power from time domain IQ'
        )

    power = {
        d: iq_to_bin_power(x, Ts, detector_period, kind=d, truncate=truncate, axis=axis)
        for d in detectors
    }

    if isroundmod(cyclic_period, detector_period, atol=1e-6):
        cyclic_detector_bins = round(cyclic_period / detector_period)
    else:
        raise ValueError(
            'cyclic period must be positive integer multiple of the detector period'
        )

    power_shape = power[detectors[0]].shape

    if power_shape[axis] % cyclic_detector_bins != 0:
        if truncate:
            N = (power_shape[axis] // cyclic_detector_bins) * cyclic_detector_bins
            power = {d: axis_slice(x, 0, N, axis=axis) for d, x in power.items()}
            power_shape = power[detectors[0]].shape
        else:
            raise ValueError(
                'pass truncate=True to allow truncation to align with cyclic windows'
            )

    shape_by_cycle = (
        power_shape[:axis]
        + (power_shape[axis] // cyclic_detector_bins,)
        + (cyclic_detector_bins,)
        + (x.shape[axis + 1 :] if x.ndim > axis else ())
    )

    power = {d: x.reshape(shape_by_cycle) for d, x in power.items()}

    cycle_stat_ufunc = {
        kind: stat_ufunc_from_shorthand(kind, xp=xp) for kind in cycle_stats
    }

    # apply the cyclic statistic

    ret = {}

    for detector, x in power.items():
        ret[detector] = {}
        for cycle_stat, func in cycle_stat_ufunc.items():
            ret[detector][cycle_stat] = func(x, axis=axis)

    return ret


def sample_ccdf(a: _AT, edges: _AT, density: bool = True) -> _AT:
    """computes the fraction (or total number) of samples in `a` that
    exceed each edge value.

    Args:
        a: the vector of input samples
        edges: sample threshold values at which to characterize the distribution
        density: if True, the sample counts are normalized by `a.size`

    Returns:
        the empirical complementary cumulative distribution
    """

    xp = array_namespace(a)

    # 'left' makes the bin interval open-ended on the left side
    # (the CCDF is "number of samples exceeding interval", and not equal to)
    edge_inds = xp.searchsorted(edges, a, side='left')

    bin_counts = xp.bincount(edge_inds, minlength=edges.shape[0] + 1)
    ccdf = (a.shape[0] - bin_counts.cumsum(0))[:-1]

    if density:
        ccdf = xp.asarray(ccdf, dtype=xp.float64)
        ccdf /= a.shape[0]

    return ccdf
