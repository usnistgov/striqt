"""Transformations and statistical tools for power time series"""

from __future__ import annotations as __

import math
import re
import typing

# import typing
import warnings
from functools import partial
from numbers import Number
from types import ModuleType
from typing import Any, Optional, overload, Sequence
from . import util

from .arrays import (
    ROUNDOFF_SAFETY,
    accum_rtol,
    array_namespace,
    float_dtype_like,
    is_cupy_array,
    isroundmod,
    axis_to_blocks,
    unit_roundoff,
)

if typing.TYPE_CHECKING:
    import numpy as np
    import numexpr as ne
    import pandas as pd
    import xarray as xr

    from .typing import ArrayLike, Array, _AL, _ALN, _AT, Dims, DTypeLike

else:
    pd = util.lazy_import('pandas')
    ne = util.lazy_import('numexpr')
    xr = util.lazy_import('xarray')
    np = util.lazy_import('numpy')

warnings.filterwarnings('ignore', message='.*divide by zero.*')
warnings.filterwarnings('ignore', message='.*invalid value encountered.*')


_DB_UNIT_MAPPING = {'dBm': 'mW', 'dBW': 'W', 'dB': 'unitless'}


def unit_dB_to_linear(s: str):
    for db_unit, lin_unit in _DB_UNIT_MAPPING.items():
        s, _ = re.subn('^' + db_unit, lin_unit, s, count=1)
    return s


def unit_linear_to_dB(s: str):
    for db_unit, lin_unit in _DB_UNIT_MAPPING.items():
        s, _ = re.subn('^' + lin_unit, db_unit, s, count=1)
    return s


def unit_dB_to_wave(s: str):
    for db_unit, lin_unit in _DB_UNIT_MAPPING.items():
        s, _ = re.subn('^' + db_unit, '√' + lin_unit, s, count=1)
    return s


def unit_wave_to_dB(s: str):
    for db_unit, lin_unit in _DB_UNIT_MAPPING.items():
        s, _ = re.subn('^√' + lin_unit, db_unit, s, count=1)
    return s


def unit_wave_to_linear(s: str):
    for db_unit, lin_unit in _DB_UNIT_MAPPING.items():
        s, _ = re.subn('^√' + lin_unit, lin_unit, s, count=1)
    return s


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
        ufunc = partial(xp.quantile, q=kind, axis=axis)

    elif callable(kind):
        ufunc = partial(kind, axis=axis)

    else:
        raise ValueError(f'invalid statistic ufunc "{kind}"')

    return ufunc


# %% roundoff model
#
# Roundoff budgets for the dB conversions, in ulps per library call (1 ulp <= 2u
# relative, u = unit roundoff). Sized to cover CUDA's single-precision bounds (log10f 2,
# powf 8, hypotf 3 ulp) as well as libm (~1 ulp), so the tolerances below hold on both
# backends and take no array namespace. Measured with
# chores/tests/measure_db_accuracy.py: libm log10 1.1-1.9 ulp; CUDA libdevice on a
# Jetson TX2i log10f 2.0, powf 5 (|x| <= 30 dB), complex |z| 1.6 input ulps.
LOG10_ULP = 2
POW_ULP = 8
HYPOT_ULP = 3
DB_PER_NEPER = 10 / math.log(10)


def log_conversion_tol(
    dtype, scale: float, complex_input: bool = False, n_impl: int = 1
) -> dict[str, float]:
    """tolerances for scale*log10(|x|), against exact math (n_impl=1) or a second
    implementation (n_impl=2).

    Roundoff on the input side of the log becomes an absolute dB error, which is what
    bounds the result near 0 dB where ulps of the output are meaningless.
    """
    u = unit_roundoff(dtype)
    rtol = 2 * u * (LOG10_ULP + 0.5)
    input_ulp = (HYPOT_ULP if complex_input else 0) + 0.5
    atol = (scale / math.log(10)) * 2 * u * input_ulp
    return {
        'rtol': ROUNDOFF_SAFETY * n_impl * rtol,
        'atol': ROUNDOFF_SAFETY * n_impl * atol,
    }


def pow_conversion_rtol(dtype, max_abs_dB: float, n_impl: int = 1) -> float:
    """rtol for 10**(x/10) with |x| <= max_abs_dB.

    Rounding x/10 perturbs the exponent, so its effect scales with |x|.
    """
    u = unit_roundoff(dtype)
    rtol = u * (math.log(10) * max_abs_dB / 10 + 2 * POW_ULP)
    return ROUNDOFF_SAFETY * n_impl * rtol


def envelope_power_rtol(dtype, complex_input: bool = False, n_impl: int = 1) -> float:
    """rtol for |x|**2"""
    u = unit_roundoff(dtype)
    rtol = 2 * u * (0.5 + (2 * HYPOT_ULP if complex_input else 0))
    return ROUNDOFF_SAFETY * n_impl * rtol


def linear_stat_tol(
    dtype, max_abs_dB: float, n: int, n_impl: int = 1
) -> dict[str, float]:
    """tolerances in dB for dBlinmean/dBlinsum over n terms with |x| <= max_abs_dB"""
    u = unit_roundoff(dtype)
    rel_linear = pow_conversion_rtol(dtype, max_abs_dB) + ROUNDOFF_SAFETY * n * u
    tol = log_conversion_tol(dtype, 10)
    return {
        'rtol': n_impl * tol['rtol'],
        'atol': n_impl * (tol['atol'] + DB_PER_NEPER * rel_linear),
    }


def roundtrip_power_rtol(dtype, max_abs_dB: float) -> float:
    """rtol for dBtopow(powtodB(x)) with |powtodB(x)| <= max_abs_dB"""
    tol = log_conversion_tol(dtype, 10)
    dB_err = tol['rtol'] * max_abs_dB + tol['atol']
    return math.log(10) / 10 * dB_err + pow_conversion_rtol(dtype, max_abs_dB)


def roundtrip_dB_tol(dtype, max_abs_dB: float) -> dict[str, float]:
    """tolerances for powtodB(dBtopow(x)) with |x| <= max_abs_dB"""
    tol = log_conversion_tol(dtype, 10)
    return {
        'rtol': tol['rtol'],
        'atol': tol['atol'] + DB_PER_NEPER * pow_conversion_rtol(dtype, max_abs_dB),
    }


def level_tolerance_dB(sigma, power: bool = False):
    """express a relative tolerance on an output as the uncertainty of its level in dB.

    `sigma` bounds an amplitude ratio (level 20*log10|x|) unless `power` is True
    (level 10*log10 x, e.g. spectrogram bins).
    """
    return (10 if power else 20) * np.log10(1 + sigma)


def linear_tolerance_dB(rtol):
    """express a relative tolerance on a linear power as a tolerance in dB"""
    return 10 * np.log10(1 + rtol)


def dB_tolerance(rtol: float, atol: float, max_abs_dB: float) -> float:
    """express (rtol, atol) on a dB-valued output as its worst-case tolerance in dB"""
    return atol + rtol * max_abs_dB


def dB_tolerance_below_peak(depth_dBc, err):
    """two-sided dB tolerance on a level `depth_dBc` (>= 0) below the peak of its
    output, given a relative amplitude error `err` at the peak.

    Roundoff bounds an element's amplitude error relative to the output's *peak*, so
    as a share of the element's own amplitude the bound grows with its depth below the
    peak. A relative amplitude error r leaves the power anywhere in
    [(1-r)**2, (1+r)**2] of its exact value, and the low side is what dominates in dB:
    -20*log10(1-r), which diverges as r approaches 1. So an element deeper than
    -20*log10(err) below the peak, where roundoff alone could account for all of its
    amplitude, gets an infinite tolerance and goes unchecked - the floor falls out of
    the bound rather than having to be imposed on top of it. This assumes nothing about
    the error spreading over an FFT's bins, so it suits any dB-valued output.
    """
    r = err * 10 ** (np.asarray(depth_dBc, dtype='float64') / 20)
    with np.errstate(divide='ignore'):
        return -20 * np.log10(np.clip(1 - r, 0, None))


def powtodB(
    x: _ALN,
    *,
    abs: bool = True,
    eps: float = 0,
    overwrite_x: bool = False,
    min_dtype: 'DTypeLike' = 'float32',
) -> _ALN:
    """compute `10*log10(abs(x) + eps)` or `10*log10(x + eps)` with speed optimizations"""

    eps_str = '' if eps == 0 else '+eps'

    values, out, xp = _arraylike_with_buffer(x, overwrite_x, min_dtype=min_dtype)

    if xp is np:
        if abs:
            expr = f'real(10*log10(abs(values){eps_str}))'
        else:
            expr = f'real(10*log10(values{eps_str}))'
        values = ne.evaluate(expr, out=out, casting='unsafe')
    elif _use_cuda_kernels(values):
        from .jit import cuda

        out = _real_buffer(out)
        use_abs = abs or xp.iscomplexobj(values)
        if eps == 0:
            kernel = cuda.powtodB if use_abs else cuda.powtodB_noabs
            kernel(values, out)
        else:
            kernel = cuda.powtodB_eps if use_abs else cuda.powtodB_eps_noabs
            kernel(values, out, eps)
        values = out
    else:
        # torch, dask, ...
        if abs:
            values = xp.abs(values, out=out)
        if eps != 0:
            values += eps
        values = xp.log10(values, out=out)
        values *= 10

    return _repackage_arraylike(values, x, unit_transform=unit_linear_to_dB)


def dBtopow(
    x: _ALN, *, overwrite_x: bool = False, min_dtype: 'DTypeLike' = 'float32'
) -> _ALN:
    """compute `10**(x/10)` with speed optimizations"""

    values, out, xp = _arraylike_with_buffer(x, overwrite_x, min_dtype=min_dtype)

    if xp is np:
        expr = '10**(values/10)'
        values = ne.evaluate(expr, out=out, casting='unsafe')
    elif _use_cuda_kernels(values):
        from .jit import cuda

        out = _real_buffer(out)
        cuda.dBtopow(values, out)
        values = out
    else:
        # torch, dask, ...
        values = xp.divide(
            values,
            10,
            out=out,
        )
        values = xp.power(10, values, out=out)

    return _repackage_arraylike(values, x, unit_transform=unit_dB_to_linear)


def envtopow(
    x: _ALN, *, overwrite_x: bool = False, min_dtype: 'DTypeLike' = 'float32'
) -> _ALN:
    """Computes abs(x)**2 with speed optimizations"""

    values, out, xp = _arraylike_with_buffer(x, overwrite_x, min_dtype=min_dtype)

    if xp is np:
        # numpy, pandas
        expr = 'real(abs(values)**2)'
        values = ne.evaluate(expr, out=out, casting='unsafe')

        if xp.iscomplexobj(values):
            values = values.real  # pyright: ignore
    elif _use_cuda_kernels(values):
        from .jit import cuda

        out = _real_buffer(out)
        cuda.envtopow(values, out)
        values = out
    else:
        # torch, dask, ...
        values = xp.abs(values, out=out)
        values *= values

    return _repackage_arraylike(values, x, unit_transform=unit_wave_to_linear)


def envtodB(
    x: _ALN,
    *,
    abs: bool = True,
    eps: float = 0,
    overwrite_x: bool = False,
    min_dtype: 'DTypeLike' = 'float32',
) -> _ALN:
    """compute `20*log10(abs(x) + eps)` or `20*log10(x + eps)` with speed optimizations"""

    eps_str = '' if eps == 0 else '+eps'

    values, out, xp = _arraylike_with_buffer(
        x, overwrite_x=overwrite_x, min_dtype=min_dtype
    )

    if xp is np:
        if abs:
            expr = f'real(20*log10(abs(values){eps_str}))'
        else:
            expr = f'real(20*log10(values{eps_str}))'
        values = ne.evaluate(expr, out=out, casting='unsafe')
    elif _use_cuda_kernels(values):
        from .jit import cuda

        out = _real_buffer(out)
        use_abs = abs or xp.iscomplexobj(values)
        if eps == 0:
            kernel = cuda.envtodB if use_abs else cuda.envtodB_noabs
            kernel(values, out)
        else:
            kernel = cuda.envtodB_eps if use_abs else cuda.envtodB_eps_noabs
            kernel(values, out, eps)
        values = out
    else:
        # torch, dask, ...
        if abs:
            values = xp.abs(values, out=out)
        if eps != 0:
            values += eps
        values = xp.log10(values, out=out)
        values *= 20

    return _repackage_arraylike(values, x, unit_transform=unit_wave_to_dB)


@overload
def dBlinmean(
    x_dB: 'xr.Dataset', axis: 'Dims|None' = None, overwrite_x=...
) -> 'xr.Dataset': ...


@overload
def dBlinmean(
    x_dB: 'xr.DataArray', axis: 'Dims|None' = None, overwrite_x=..., min_dtype=...
) -> 'xr.DataArray': ...


@overload
def dBlinmean(
    x_dB: 'np.ndarray',
    axis: 'int|Sequence[int]|None' = None,
    overwrite_x=...,
    min_dtype=...,
) -> 'np.ndarray': ...


@overload
def dBlinmean(
    x_dB: 'pd.Series',
    axis: 'int|Sequence[int]|None' = None,
    overwrite_x=...,
    min_dtype=...,
) -> 'pd.Series': ...


@overload
def dBlinmean(
    x_dB: 'pd.DataFrame',
    axis: 'int|Sequence[int]|None' = None,
    overwrite_x=...,
    min_dtype=...,
) -> 'pd.DataFrame': ...


def dBlinmean(
    x_dB: _AL,
    axis: 'Dims|int|Sequence[int]|None' = None,
    overwrite_x: bool = False,
    min_dtype: 'DTypeLike' = 'float32',
) -> _AL:
    """evaluate the mean in linear power space given power in dB.

    This is equivalent to:
        powtodB(dBtopow(x).mean(axis))

    Returns:
        array-like object with same shape as x_dB, reduced by the
        dimension at the specified axes
    """

    x = dBtopow(x_dB, overwrite_x=overwrite_x, min_dtype=min_dtype)
    linmean = x.mean(axis)  # type: ignore
    return powtodB(linmean, overwrite_x=True, min_dtype=min_dtype)  # pyright: ignore


@overload
def dBlinsum(
    x_dB: 'xr.Dataset', axis: 'Dims|None' = None, overwrite_x=...
) -> 'xr.Dataset': ...


@overload
def dBlinsum(
    x_dB: 'xr.DataArray', axis: 'Dims|None' = None, overwrite_x=...
) -> 'xr.DataArray': ...


@overload
def dBlinsum(
    x_dB: 'np.ndarray',
    axis: 'int|Sequence[int]|None' = None,
    overwrite_x=...,
    min_dtype=...,
) -> 'np.ndarray': ...


@overload
def dBlinsum(
    x_dB: 'pd.Series',
    axis: 'int|Sequence[int]|None' = None,
    overwrite_x=...,
    min_dtype=...,
) -> 'pd.Series': ...


@overload
def dBlinsum(
    x_dB: 'pd.DataFrame',
    axis: 'int|Sequence[int]|None' = None,
    overwrite_x=...,
    min_dtype=...,
) -> 'pd.DataFrame': ...


def dBlinsum(
    x_dB: _AL, axis=None, overwrite_x=False, min_dtype: 'DTypeLike' = 'float32'
) -> _AL:
    """evaluate the sum in linear power space given power in dB.

    This is equivalent to:
        powtodB(dBtopow(x).sum(axis))

    Returns:
        array-like object with same shape as x_dB, reduced by the
        dimension at the specified axes
    """

    x_lin = dBtopow(x_dB, overwrite_x=overwrite_x, min_dtype=min_dtype)
    x_sum = x_lin.sum(axis)  # type: ignore
    return powtodB(x_sum, overwrite_x=True, min_dtype=min_dtype)  # type: ignore


def bin_power_rtol(dtype, size: int, n_impl: int = 1) -> float:
    """rtol for a statistic of |x|**2 over `size` samples against exact arithmetic"""
    envelope = envelope_power_rtol(dtype, complex_input=True, n_impl=n_impl)
    return envelope + accum_rtol(dtype, size, n_impl)


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
    """Evaluate cyclic statistics of binned channel power.

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
    frames and the 1 ms pulse repetition interval of the SPN-43 radar) aligns their
    features across cycles, so that each resolves at fixed lags while occupancy
    with an incommensurate period is spread across all lags. Uplink and downlink
    levels of a TDD network can then be read from disjoint lag windows. Statistics
    are evaluated in linear power units, so convert to dB afterwards.

    A mismatch ``dT`` between `cyclic_period` and the true signal period, such as
    a sample clock offset, drifts features by ``M * dT / detector_period`` lags by
    the end of the capture and bleeds power between neighbouring lags. Keep
    ``M * dT`` below `detector_period` by limiting the number of cycles ``M``.

    Reference: D.G. Kuester et al., "Cyclic Analysis of Power in Radio Channels".

    Args:
        x: complex-valued input waveform samples
        Ts: sample period of the waveform
        detector_period: duration of each power detector bin
        cyclic_period: duration of one cycle, an integer multiple of `detector_period`
        truncate: if True, drop trailing detector bins that do not complete a cycle
        detectors: power detector names accepted by `iq_to_bin_power`
        cycle_stats: statistics accepted by `stat_ufunc_from_shorthand`, evaluated
            across cycles (names such as 'min', 'mean', 'max', or quantiles in
            (0, 1))
        axis: the time axis of `x`

    Raises:
        ValueError: if `detector_period` is not an integer multiple of `Ts`,
            `cyclic_period` is not an integer multiple of `detector_period`, or
            the capture does not hold a whole number of cycles and `truncate` is
            False

    Returns:
        dict keyed on detector, of dicts keyed on cyclic statistic, of arrays whose
        `axis` dimension has been replaced by the ``L`` cycle lags
    """

    # apply the detector statistic
    xp = array_namespace(x)

    # compute the binned power ourselves
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

    if power_shape[1] % cyclic_detector_bins != 0:
        if truncate:
            N = (power_shape[1] // cyclic_detector_bins) * cyclic_detector_bins
            power = {d: x[:N] for d, x in power.items()}
        else:
            raise ValueError(
                'pass truncate=True to allow truncation to align with cyclic windows'
            )

    if axis < 0:
        axis = x.ndim + axis

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


# %% module-local helper functions
def _infer_contained_array(x: Any) -> Array:
    if hasattr(type(x), 'values'):
        # first, guess at xarray/pandas types before expensive imports
        if hasattr(type(x), 'data') and isinstance(x, (xr.DataArray, xr.Dataset)):
            return x.data
        elif isinstance(x, (pd.DataFrame, pd.Series)):
            return x.values
        else:
            raise TypeError('unable to associate an array type with input')


def _arraylike_with_buffer(
    x: ArrayLike | Number, overwrite_x: bool = False, min_dtype: 'DTypeLike' = 'float32'
) -> 'tuple[Array, Array, ModuleType]':
    """interpret the array-like input and output buffer arguments.

    Args:
        x: the input array-like or dataframe-like object
        out: the output buffer, or True to use the extracted array, or False force None
    Returns:
        Array objects pointing to the underlying array-type objects,
        and the module to work with them
    """
    # infer the array object and namespace
    if min_dtype is None:
        raise TypeError('must pass a dtype as min_dtype')
    if np.dtype(min_dtype) == np.dtype('float16'):
        raise TypeError('min_dtype must be at least float32 or larger')

    type_ = type(x)
    if hasattr(type_, '__array_function__') or hasattr(type_, '__array_namespace__'):
        values: Array = x
        xp = array_namespace(values)
        if xp.ndim(values) == 0:
            overwrite_x = False
    elif isinstance(x, (int, float)):
        values: Array = np.array(x)
        xp = np
    elif hasattr(type_, 'values'):
        values = _infer_contained_array(x)
        xp = array_namespace(values)
    else:
        raise TypeError(f'unable to associate an array with type {type_!r} with input')
    values = typing.cast('Array', values)

    # do we need to upcast?
    dtype = values.dtype
    if np.dtype(min_dtype) <= dtype:
        promote_dtype = None
    else:
        promote_dtype = np.dtype(min_dtype)

    if promote_dtype is not None:
        # cupy.fuse evaluates in the input dtype and casts only when assigning
        # into `out`, so the input has to be widened before the kernel runs
        values = values.astype(promote_dtype)

    if xp.__name__.startswith('dask'):
        return values, None, xp
    elif promote_dtype is not None or overwrite_x:
        return values, values, xp
    elif xp is np or _use_cuda_kernels(values):
        # numexpr promotes to float64 if out=None, and the cupy fused kernels
        # assign into `out` in place. the results are real even for complex input
        out = xp.empty_like(values, dtype=float_dtype_like(values))
        return values, out, xp
    else:
        return values, None, xp


def _real_buffer(out: Array) -> Array:
    """`out`, or its real part if it is complex.

    `_arraylike_with_buffer` hands back a complex buffer only when overwriting
    complex input in place. The fused kernels compute real values and cannot
    assign into it, so they write the real part, which is also what the numpy
    path returns.
    """
    if out.dtype.kind == 'c':
        return out.real
    return out


def _use_cuda_kernels(values: Array) -> bool:
    """whether to evaluate on `values` with the fused kernels in `.jit.cuda`.

    The kernels assign through `out[:]`, which 0-d arrays do not support, so
    scalars take the generic array-API path.
    """
    return is_cupy_array(values) and values.ndim > 0


def _repackage_arraylike(
    values: Array,
    obj: _ALN,
    *,
    unit_transform: Optional[typing.Callable] = None,
) -> _ALN:
    """package `values` into a data type matching `obj`"""

    # accessing each of these forces imports of each module.
    # work through progressively more expensive imports
    if isinstance(obj, Number):
        return values.item()
    elif not hasattr(type(obj), 'values'):
        return typing.cast('_ALN', values)
    elif isinstance(obj, pd.Series):
        return pd.Series(values, index=obj.index)  # type: ignore
    elif isinstance(obj, pd.DataFrame):
        return pd.DataFrame(values, index=obj.index, columns=obj.columns)  # type: ignore
    elif isinstance(obj, xr.DataArray):
        ret = obj.copy(deep=False, data=values)
        units = ret.attrs.get('units', None)
        if units is not None and unit_transform is not None:
            ret.attrs['units'] = unit_transform(units)
        return ret
    else:
        raise TypeError(f'unrecognized input type {type(obj)}')


# %%
