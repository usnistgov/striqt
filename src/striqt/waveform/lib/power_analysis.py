"""Transformations and statistical tools for power time series"""

from __future__ import annotations as __

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
    array_namespace,
    float_dtype_like,
    is_cupy_array,
    isroundmod,
    axis_to_blocks,
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
def stat_ufunc_from_shorthand(kind, xp=None, axis=0) -> typing.Callable:
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


def powtodB(x: _ALN, *, abs: bool = True, eps: float = 0, overwrite_x: bool = False, min_dtype: 'DTypeLike|None'='float32') -> _ALN:
    """compute `10*log10(abs(x) + eps)` or `10*log10(x + eps)` with speed optimizations"""

    eps_str = '' if eps == 0 else '+eps'

    values, out, xp = _arraylike_with_buffer(x, overwrite_x, min_dtype=min_dtype)

    if xp is np:
        if abs:
            expr = f'real(10*log10(abs(values){eps_str}))'
        else:
            expr = f'real(10*log10(values+eps){eps_str})'
        values = ne.evaluate(expr, out=out, casting='unsafe')
    elif is_cupy_array(xp):
        from .jit import cuda

        if eps == 0:
            if abs:
                values = cuda.powtodB(x, out)
            else:
                values = cuda.powtodB_noabs(x, out)
        else:
            if abs:
                values = cuda.powtodB_eps(x, out, eps)
            else:
                values = cuda.powtodB_eps_noabs(x, out, eps)
    else:
        # torch, dask, ...
        if abs:
            values = xp.abs(values, out=out)
        if eps != 0:
            values += eps
        values = xp.log10(values, out=out)
        values *= 10

    return _repackage_arraylike(values, x, unit_transform=unit_linear_to_dB)


def dBtopow(x: _ALN, *, overwrite_x: bool = False, min_dtype: 'DTypeLike|None'='float32') -> _ALN:
    """compute `10**(x/10)` with speed optimizations"""

    values, out, xp = _arraylike_with_buffer(x, overwrite_x, min_dtype=min_dtype)

    if xp is np:
        expr = '10**(values/10.)'
        values = ne.evaluate(expr, out=out, casting='unsafe')
    elif is_cupy_array(xp):
        from .jit import cuda

        values = cuda.dBtopow(x, out)
    else:
        # torch, dask, ...
        values = xp.divide(values, 10, out=out)
        values = xp.power(10, values, out=out)

    return _repackage_arraylike(values, x, unit_transform=unit_dB_to_linear)


def envtopow(x: _ALN, *, overwrite_x: bool = False, min_dtype: 'DTypeLike|None'='float32') -> _ALN:
    """Computes abs(x)**2 with speed optimizations"""

    values, out, xp = _arraylike_with_buffer(x, overwrite_x, min_dtype=min_dtype)

    if xp is np:
        # numpy, pandas
        expr = 'real(abs(values)**2)'
        values = ne.evaluate(expr, out=out, casting='unsafe')

        if xp.iscomplexobj(values):
            values = values.real  # pyright: ignore
    elif is_cupy_array(xp):
        from .jit import cuda

        values = cuda.envtopow(x, out)
    else:
        # torch, dask, ...
        values = xp.abs(x, out=out)
        values *= values

    return _repackage_arraylike(values, x, unit_transform=unit_wave_to_linear)


def envtodB(x: _ALN, *, abs: bool = True, eps: float = 0, overwrite_x: bool = False, min_dtype: 'DTypeLike|None'='float32') -> _ALN:
    """compute `20*log10(abs(x) + eps)` or `20*log10(x + eps)` with speed optimizations"""

    eps_str = '' if eps == 0 else '+eps'

    values, out, xp = _arraylike_with_buffer(x, overwrite_x=overwrite_x, min_dtype=min_dtype)

    if xp is np:
        if abs:
            expr = f'real(20*log10(abs(values){eps_str}))'
        else:
            expr = f'real(20*log10(values+eps){eps_str})'
        values = ne.evaluate(expr, out=out, casting='unsafe')
    elif is_cupy_array(xp):
        from .jit import cuda

        if eps == 0:
            if abs:
                values = cuda.envtodB(x, out)
            else:
                values = cuda.envtodB_noabs(x, out)
        else:
            if abs:
                values = cuda.envtodB_eps(x, out, eps)
            else:
                values = cuda.envtodB_eps_noabs(x, out, eps)
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
    x_dB: 'np.ndarray', axis: 'int|Sequence[int]|None' = None, overwrite_x=..., min_dtype=...
) -> 'np.ndarray': ...


@overload
def dBlinmean(
    x_dB: 'pd.Series', axis: 'int|Sequence[int]|None' = None, overwrite_x=..., min_dtype=...
) -> 'pd.Series': ...


@overload
def dBlinmean(
    x_dB: 'pd.DataFrame', axis: 'int|Sequence[int]|None' = None, overwrite_x=..., min_dtype=...
) -> 'pd.DataFrame': ...


def dBlinmean(
    x_dB: _AL, axis: 'Dims|int|Sequence[int]|None' = None, overwrite_x: bool = False, min_dtype: 'DTypeLike|None'='float32'
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
    return powtodB(linmean, overwrite_x=True)  # pyright: ignore


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
    x_dB: 'np.ndarray', axis: 'int|Sequence[int]|None' = None, overwrite_x=..., min_dtype=...
) -> 'np.ndarray': ...


@overload
def dBlinsum(
    x_dB: 'pd.Series', axis: 'int|Sequence[int]|None' = None, overwrite_x=..., min_dtype=...
) -> 'pd.Series': ...


@overload
def dBlinsum(
    x_dB: 'pd.DataFrame', axis: 'int|Sequence[int]|None' = None, overwrite_x=..., min_dtype=...
) -> 'pd.DataFrame': ...


def dBlinsum(x_dB: _AL, axis=None, overwrite_x=False, min_dtype: 'DTypeLike|None'='float32') -> _AL:
    """evaluate the sum in linear power space given power in dB.

    This is equivalent to:
        powtodB(dBtopow(x).sum(axis))

    Returns:
        array-like object with same shape as x_dB, reduced by the
        dimension at the specified axes
    """

    x_lin = dBtopow(x_dB, overwrite_x=overwrite_x, min_dtype=min_dtype)
    return powtodB(x_lin.sum(axis), overwrite_x=True)  # type: ignore


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
    """computes a time series of periodic frame power statistics.

    The time axis on the cyclic time lag [0, cyclic_period) is binned with step size
    `detector_period`, for a total of `cyclic_period/detector_period` samples.

    RMS and peak power detector data are returned. For each type of detector, a time
    series is returned for (min, mean, max) statistics, which are computed across the
    number of frames (`cyclic_period/Ts`).

    Args:
        iq: complex-valued input waveform samples
        Ts: sample period of the iq waveform
        detector_period: sampling period within the frame
        cyclic_period: the cyclic period to analyze

    Raises:
        ValueError: if detector_period%Ts != 0 or cyclic_period%detector_period != 0

    Returns:
        dict keyed on detector type, with values (dict of np.arrays keyed on cyclic statistic)
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
    x: ArrayLike | Number, overwrite_x: bool = False, min_dtype: 'DTypeLike|None' = None
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
    if hasattr(type(x), '__array_function__'):
        values: Array = x
        xp = array_namespace(values)
    elif isinstance(x, (int, float)):
        values: Array = np.array(x)
        xp = np
    elif hasattr(type(x), 'values'):
        values = _infer_contained_array(x)
        xp = array_namespace(values)
    else:
        raise TypeError('unable to associate an array type with input')
    values = typing.cast('Array', values)

    # do we need to upcast?
    dtype = values.dtype
    if min_dtype is None or np.dtype(min_dtype) <= dtype:
        promote_dtype = None
    else:
        promote_dtype = np.dtype(min_dtype)

    if overwrite_x:
        if xp.__name__.startswith('dask'):
            out = None
            cast_dtype = promote_dtype
        elif promote_dtype:
            out = xp.empty(values.shape, dtype=promote_dtype)
            cast_dtype = None
        else:
            out = values
            cast_dtype = None
    else:
        out = None
        cast_dtype = promote_dtype

    if cast_dtype is not None:
        return values.astype(cast_dtype), out, xp
    else:
        return values, out, xp


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
