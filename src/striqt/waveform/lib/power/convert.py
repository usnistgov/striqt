"""Elementwise conversions between linear and dB scales for array-like inputs
(numbers, arrays, and pandas or xarray objects wrapping them), evaluated through
numexpr on numpy and fused kernels on cupy, with the roundoff models that bound
their results on both backends."""

from __future__ import annotations as __

import math
import re
import typing
import warnings
from numbers import Number
from types import ModuleType
from typing import Any, Literal, Optional, Sequence
from .. import util
from ..util import np, pd, xr

from ..arrays import (
    ROUNDOFF_SAFETY,
    array_namespace,
    float_dtype_like,
    is_cupy_array,
    tiny_like,
    unit_roundoff,
)

if typing.TYPE_CHECKING:
    import numexpr as ne

    from typing import Callable, TypeVar, Union

    from ..typing import ArrayLike, Array, _ALN, Dims, DTypeLike

    _ALD = TypeVar('_ALD', bound=Union[ArrayLike, xr.Dataset])

else:
    ne = util.lazy_import('numexpr')

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


# %% roundoff model
#
# Roundoff budgets for the dB conversions, in ulps per library call (1 ulp <= 2u
# relative, u = unit roundoff). Sized to cover CUDA's single-precision bounds (logf 1,
# powf 8, hypotf 3 ulp) as well as libm (~1 ulp), so the tolerances below hold on both
# backends and take no array namespace. The dB conversions evaluate
# `scale_ln*log(x)` rather than `scale*log10(x)`: on the Jetson TX2i (glibc 2.27) that
# is 0.55-0.65x the numexpr time in float32 and 0.82x in float64, neutral on the GPU,
# and more accurate in float64 (chores/tests/profile_log_scaling.py, 2026-10-19).
# Measured with chores/tests/measure_db_accuracy.py and that script, in spacing ulps
# (up to twice the `2u*k` factor): libm log 1.8 (float32, Apple and glibc) and 1.0
# (float64, glibc); CUDA libdevice on the TX2i logf 2.0, log 1.0, powf 5
# (|x| <= 30 dB), complex |z| 1.6 input ulps.
HYPOT_ULP = 3
LOG_ULP = 1
# DB_PER_NEPER rounded to the computation dtype; float32 carries 0.41 ulp
SCALE_CONST_ULP = 0.5
# 10/ln(10) correctly rounded; the computed `10 / math.log(10)` is 0.87 ulp off
DB_PER_NEPER = 4.342944819032518


def log_conversion_tol(
    dtype: DTypeLike, scale: float, complex_input: bool = False, n_impl: int = 1
) -> dict[str, float]:
    """return the ``rtol`` and ``atol`` (in dB) on ``scale*log10(|x|)`` as a dict.

    Both are fixed multiples of the unit roundoff of `dtype`, and ``atol`` also
    scales with `scale`. `n_impl` is 1 against exact math and 2 against a second
    implementation; `complex_input` adds the roundoff of ``|x|``. Roundoff on the
    input side of the log becomes an absolute dB error, which is what bounds the
    result near 0 dB where ulps of the output are meaningless.
    """
    u = unit_roundoff(dtype)
    # the log call, the rounded scale constant, and the product rounding
    rtol = 2 * u * (LOG_ULP + SCALE_CONST_ULP + 0.5)
    input_ulp = (HYPOT_ULP if complex_input else 0) + 0.5
    atol = (DB_PER_NEPER * scale / 10) * 2 * u * input_ulp
    return {
        'rtol': ROUNDOFF_SAFETY * n_impl * rtol,
        'atol': ROUNDOFF_SAFETY * n_impl * atol,
    }


def pow_conversion_rtol(dtype: DTypeLike, max_abs_dB: float, n_impl: int = 1) -> float:
    """return the rtol on ``10**(x/10)`` for ``|x| <= max_abs_dB``.

    Rounding the exponent perturbs the result in proportion to its magnitude, so the
    bound grows linearly with `max_abs_dB` from a floor of a fixed number of unit
    roundoffs of `dtype`. `n_impl` is 1 against exact math and 2 against a second
    implementation.
    """
    pow_ulp = 8
    u = unit_roundoff(dtype)
    rtol = u * (math.log(10) * max_abs_dB / 10 + 2 * pow_ulp)
    return ROUNDOFF_SAFETY * n_impl * rtol


def envelope_power_rtol(
    dtype: DTypeLike, complex_input: bool = False, n_impl: int = 1
) -> float:
    """return the rtol on ``|x|**2``.

    The bound is a fixed multiple of the unit roundoff of `dtype`, larger when
    `complex_input` adds the roundoff of ``|x|``. `n_impl` is 1 against exact math
    and 2 against a second implementation.
    """
    u = unit_roundoff(dtype)
    rtol = 2 * u * (0.5 + (2 * HYPOT_ULP if complex_input else 0))
    return ROUNDOFF_SAFETY * n_impl * rtol


def linear_stat_tol(
    dtype: DTypeLike, max_abs_dB: float, n: int, n_impl: int = 1
) -> dict[str, float]:
    """return the ``rtol`` and ``atol`` in dB on `dBlinmean`/`dBlinsum` over `n` terms.

    `max_abs_dB` bounds ``|x|`` of the dB-valued input. ``rtol`` is that of the final
    dB conversion; ``atol`` grows with `max_abs_dB` and linearly with `n` for the
    accumulation of the linear terms. `n_impl` is 1 against exact math and 2 against
    a second implementation.
    """
    u = unit_roundoff(dtype)
    rel_linear = pow_conversion_rtol(dtype, max_abs_dB) + ROUNDOFF_SAFETY * n * u
    tol = log_conversion_tol(dtype, 10)
    return {
        'rtol': n_impl * tol['rtol'],
        'atol': n_impl * (tol['atol'] + DB_PER_NEPER * rel_linear),
    }


def roundtrip_power_rtol(dtype: DTypeLike, max_abs_dB: float) -> float:
    """return the rtol on ``dBtopow(powtodB(x))`` for ``|powtodB(x)| <= max_abs_dB``"""
    tol = log_conversion_tol(dtype, 10)
    dB_err = tol['rtol'] * max_abs_dB + tol['atol']
    return math.log(10) / 10 * dB_err + pow_conversion_rtol(dtype, max_abs_dB)


def roundtrip_dB_tol(dtype: DTypeLike, max_abs_dB: float) -> dict[str, float]:
    """return the ``rtol`` and ``atol`` in dB on ``powtodB(dBtopow(x))``.

    `max_abs_dB` bounds ``|x|``, and ``atol`` grows with it.
    """
    tol = log_conversion_tol(dtype, 10)
    return {
        'rtol': tol['rtol'],
        'atol': tol['atol'] + DB_PER_NEPER * pow_conversion_rtol(dtype, max_abs_dB),
    }


def level_tolerance_dB(sigma: float, power: bool = False) -> float:
    """convert a relative tolerance `sigma` on an output to a level uncertainty in dB.

    `sigma` bounds an amplitude ratio (20 dB per decade) unless `power` is True for a
    power ratio (10 dB per decade, e.g. spectrogram bins). The result is
    non-negative for non-negative `sigma`.
    """
    return (10 if power else 20) * np.log10(1 + sigma)


def linear_tolerance_dB(rtol: float) -> float:
    """convert a relative tolerance `rtol` on a linear power to a tolerance in dB"""
    return 10 * np.log10(1 + rtol)


def dB_tolerance(rtol: float, atol: float, max_abs_dB: float) -> float:
    """return the worst-case tolerance in dB of ``(rtol, atol)`` on a dB-valued output.

    `max_abs_dB` bounds the magnitude of the output, and the result grows with it.
    """
    return atol + rtol * max_abs_dB


@typing.overload
def off_peak_dB_tolerance(depth_dBc: float, err: float) -> float: ...
@typing.overload
def off_peak_dB_tolerance(depth_dBc: Array, err: float) -> Array: ...
def off_peak_dB_tolerance(depth_dBc: float | Array, err: float) -> float | Array:
    """bound the dB error of an element `depth_dBc` (>= 0) below the peak of its output.

    The tolerance is two-sided, given a relative amplitude error `err` at the peak.
    The peak is the matched-filter bin of the matched input, so the same bound serves
    a tone's FFT bin and an impulse's detector bin. The result is non-negative and
    grows with `depth_dBc`: a float for a scalar `depth_dBc` and a float64 numpy
    array otherwise.

    Roundoff bounds an element's amplitude error relative to the output's *peak*, so
    as a share of the element's own amplitude the bound grows with its depth below the
    peak. A relative amplitude error ``r`` leaves the power anywhere between
    ``(1-r)**2`` and ``(1+r)**2`` of its exact value, and the low side is what
    dominates in dB, diverging as ``r`` approaches 1. So an element deep enough
    below the peak that roundoff alone could account for all of its amplitude gets
    an infinite tolerance and goes unchecked - the floor falls out of the bound
    rather than having to be imposed on top of it. This assumes nothing about the
    error spreading over an FFT's bins, so it suits any dB-valued output.
    """
    r = err * 10 ** (np.asarray(depth_dBc, dtype='float64') / 20)
    with np.errstate(divide='ignore'):
        return -20 * np.log10(np.clip(1 - r, 0, None))


def _cuda_log_kernel(abs: bool, bounded: bool) -> Callable[..., Array]:
    """select the fused kernel in `cuda_jit` for one dB conversion."""
    from . import cuda_jit

    if bounded:
        return cuda_jit.lin_to_dB_eps if abs else cuda_jit.lin_to_dB_eps_noabs
    return cuda_jit.lin_to_dB if abs else cuda_jit.lin_to_dB_noabs


def _lin_to_dB(
    x: _ALN,
    *,
    scale: Literal[10, 20],
    abs: bool,
    bounded: bool,
    overwrite_x: bool,
    min_dtype: DTypeLike,
) -> _ALN:
    """compute ``scale*log10(abs(x) + eps)`` or ``scale*log10(x + eps)``.

    The evaluation is ``scale_ln*log(...)`` with ``scale_ln = scale/ln(10)`` rounded
    to the real computation dtype. `eps` is 0 unless `bounded`, and then the smallest
    normal value of the computation dtype for `scale` 10 and its square root for
    `scale` 20, as a scalar of that dtype. With `overwrite_x`, the result is written
    into `x` unless `min_dtype` widens it first; a complex `x` overwritten in place
    comes back as the real view of its buffer (`_real_buffer`) on every backend.
    """

    values, out, xp = _arraylike_with_buffer(x, overwrite_x, min_dtype=min_dtype)

    # numpy scalars of the computation dtype keep a float32 evaluation in float32 on
    # every backend, where a Python float would promote it to double
    real_dtype = np.dtype(float_dtype_like(values))
    scale_ln = real_dtype.type(DB_PER_NEPER * (scale // 10))
    if bounded:
        eps = tiny_like(values)
        if scale == 20:
            eps = np.sqrt(eps)
    else:
        eps = None

    if xp is np:
        # the expression is one of six constant strings so the numexpr compile cache
        # hits; `values`, `scale_ln` and `eps` are looked up from this frame by name.
        # numexpr's `real` and `abs` of a complex operand both yield complex128, so
        # complex input needs `real(...)` and is always converted by magnitude; on a
        # real operand `real` would upcast through complex128, so it is omitted there
        eps_str = '' if eps is None else '+eps'
        if np.iscomplexobj(values):
            expr = f'real(scale_ln*log(abs(values){eps_str}))'
        elif abs:
            expr = f'scale_ln*log(abs(values){eps_str})'
        else:
            expr = f'scale_ln*log(values{eps_str})'
        out = _real_buffer(out)
        values = ne.evaluate(expr, out=out, casting='unsafe')
    elif _use_cuda_kernels(values):
        out = _real_buffer(out)
        use_abs = abs or xp.iscomplexobj(values)
        kernel = _cuda_log_kernel(use_abs, eps is not None)
        if eps is None:
            kernel(values, out, scale_ln)
        else:
            kernel(values, out, scale_ln, eps)
        values = out
    else:
        # torch, dask, ...
        if abs:
            values = xp.abs(values, out=out)
        if eps is not None:
            values += eps
        values = xp.log(values, out=out)
        values *= scale_ln

    return _repackage_arraylike(
        values,
        x,
        unit_transform=unit_linear_to_dB if scale == 10 else unit_wave_to_dB,
    )


def powtodB(
    x: _ALN,
    *,
    abs: bool = True,
    bounded: bool = False,
    overwrite_x: bool = False,
    min_dtype: 'DTypeLike' = 'float32',
) -> _ALN:
    """compute ``10*log10(abs(x))``, or ``10*log10(x)`` when `abs` is False.

    The computation is elementwise through numexpr on numpy and fused kernels on
    cupy, in a floating-point dtype no narrower than `min_dtype`.

    Args:
        x: values in linear power units, as a number, array, or pandas/xarray object
        abs: True to convert ``abs(x)``; a complex `x` is converted by magnitude in
            either case
        bounded: True to floor the result at ``10*log10(tiny)``, where ``tiny`` is the
            smallest normal value of the computation dtype (about -379 dB in float32
            and -3077 dB in float64); zeros then map to the floor instead of ``-inf``
        overwrite_x: True to write the result into `x` when its dtype is at least
            `min_dtype`; a complex `x` then returns the real view of its buffer
        min_dtype: the narrowest floating-point dtype of the computation; a narrower
            `x` is widened to it

    Returns:
        the dB values, real-valued, in the container type of `x` (a Python number for
        a number, otherwise the array, `pandas` or `xarray` type of `x`, with an
        xarray ``units`` attribute converted from linear power to dB)
    """
    return _lin_to_dB(
        x,
        scale=10,
        abs=abs,
        bounded=bounded,
        overwrite_x=overwrite_x,
        min_dtype=min_dtype,
    )


def dBtopow(
    x: _ALN, *, overwrite_x: bool = False, min_dtype: 'DTypeLike' = 'float32'
) -> _ALN:
    """compute ``10**(x/10)``.

    The computation is elementwise through numexpr on numpy and a fused kernel on
    cupy, in a floating-point dtype no narrower than `min_dtype`.

    Args:
        x: values in dB, as a number, array, or pandas/xarray object
        overwrite_x: True to write the result into `x` when its dtype is at least
            `min_dtype`
        min_dtype: the narrowest floating-point dtype of the computation; a narrower
            `x` is widened to it

    Returns:
        the values in linear power units, in the container type of `x` (a Python
        number for a number, otherwise the array, `pandas` or `xarray` type of `x`,
        with an xarray ``units`` attribute converted from dB to linear power)
    """

    values, out, xp = _arraylike_with_buffer(x, overwrite_x, min_dtype=min_dtype)

    if xp is np:
        expr = '10**(values/10)'
        values = ne.evaluate(expr, out=out, casting='unsafe')
    elif _use_cuda_kernels(values):
        from . import cuda_jit

        out = _real_buffer(out)
        cuda_jit.dBtopow(values, out)
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
    """compute ``abs(x)**2``.

    The computation is elementwise through numexpr on numpy and a fused kernel on
    cupy, in a floating-point dtype no narrower than `min_dtype`.

    Args:
        x: values in linear amplitude (envelope) units, real or complex, as a
            number, array, or pandas/xarray object
        overwrite_x: True to write the result into `x` when its dtype is at least
            `min_dtype`; a complex `x` then returns the real view of its buffer
        min_dtype: the narrowest floating-point dtype of the computation; a narrower
            `x` is widened to it

    Returns:
        the power values, real-valued, in the container type of `x` (a Python number
        for a number, otherwise the array, `pandas` or `xarray` type of `x`, with an
        xarray ``units`` attribute converted from amplitude to linear power)
    """

    values, out, xp = _arraylike_with_buffer(x, overwrite_x, min_dtype=min_dtype)

    if xp is np:
        # numpy, pandas
        expr = 'real(abs(values)**2)'
        values = ne.evaluate(expr, out=out, casting='unsafe')

        if xp.iscomplexobj(values):
            values = values.real
    elif _use_cuda_kernels(values):
        from . import cuda_jit

        out = _real_buffer(out)
        cuda_jit.envtopow(values, out)
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
    bounded: bool = False,
    overwrite_x: bool = False,
    min_dtype: 'DTypeLike' = 'float32',
) -> _ALN:
    """compute ``20*log10(abs(x))``, or ``20*log10(x)`` when `abs` is False.

    The computation is elementwise through numexpr on numpy and fused kernels on
    cupy, in a floating-point dtype no narrower than `min_dtype`.

    Args:
        x: values in linear amplitude (envelope) units, as a number, array, or
            pandas/xarray object
        abs: True to convert ``abs(x)``; a complex `x` is converted by magnitude in
            either case
        bounded: True to floor the result at the same dB value as `powtodB` with
            `bounded`, ``10*log10(tiny)`` for the smallest normal value ``tiny`` of the
            computation dtype; zeros then map to the floor instead of ``-inf``
        overwrite_x: True to write the result into `x` when its dtype is at least
            `min_dtype`; a complex `x` then returns the real view of its buffer
        min_dtype: the narrowest floating-point dtype of the computation; a narrower
            `x` is widened to it

    Returns:
        the dB values, real-valued, in the container type of `x` (a Python number for
        a number, otherwise the array, `pandas` or `xarray` type of `x`, with an
        xarray ``units`` attribute converted from amplitude to dB)
    """
    return _lin_to_dB(
        x,
        scale=20,
        abs=abs,
        bounded=bounded,
        overwrite_x=overwrite_x,
        min_dtype=min_dtype,
    )


def dBlinmean(
    x_dB: _ALD,
    axis: 'Dims|int|Sequence[int]|None' = None,
    overwrite_x: bool = False,
    min_dtype: 'DTypeLike' = 'float32',
) -> _ALD:
    """evaluate the mean in linear power space given power in dB.

    This is equivalent to:
        powtodB(dBtopow(x).mean(axis))

    Returns:
        array-like object with same shape as x_dB, reduced by the
        dimension at the specified axes
    """

    x = dBtopow(x_dB, overwrite_x=overwrite_x, min_dtype=min_dtype)
    linmean = x.mean(axis)  # type: ignore
    return powtodB(linmean, overwrite_x=True, min_dtype=min_dtype)


def dBlinsum(
    x_dB: _ALD,
    axis: 'Dims|int|Sequence[int]|None' = None,
    overwrite_x: bool = False,
    min_dtype: 'DTypeLike' = 'float32',
) -> _ALD:
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
    """resolve an array-like input to its array, an output buffer and its namespace.

    Args:
        x: a number, an array, or a pandas/xarray object wrapping one
        overwrite_x: True to hand back the array of `x` as the output buffer;
            ignored for a 0-d array
        min_dtype: the narrowest floating-point dtype of the computation; an array
            narrower than it is widened, and the widened copy becomes the buffer

    Returns:
        ``(values, out, xp)``: the array to compute on, with dtype at least
        `min_dtype`; the buffer to evaluate into, which is `values` itself when
        overwriting or widened, a new real-valued array shaped like `values` on the
        numexpr and fused-kernel paths, and None on dask or any other namespace;
        and the array namespace of `values`

    Raises:
        TypeError: if `min_dtype` is None or narrower than float32, or if no array
            can be associated with the type of `x`
    """
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
    """return `out`, or the real view of `out` if it is complex.

    `_arraylike_with_buffer` hands back a complex buffer only when overwriting
    complex input in place. The conversions produce real values, so both the numexpr
    and the fused-kernel paths write into this view and return it.
    """
    if out.dtype.kind == 'c':
        return out.real
    return out


def _use_cuda_kernels(values: Array) -> bool:
    """report whether `values` is evaluated with the fused kernels in `cuda_jit`.

    The kernels assign through ``out[:]``, which 0-d arrays do not support, so cupy
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
