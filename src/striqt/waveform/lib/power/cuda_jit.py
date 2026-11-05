"""cupy.fuse kernels for the dB and power conversions, reached through `get_kernel`.

Each kernel assigns through ``out[:]`` and requires ``out.ndim > 0``.
"""

from __future__ import annotations as __

import typing

import cupy  # ty: ignore[unresolved-import]
import numpy as np

if typing.TYPE_CHECKING:
    from typing import Callable, Literal


@cupy.fuse()
def _envtopow(x: cupy.ndarray, out: cupy.ndarray) -> cupy.ndarray:
    out[:] = cupy.abs(x) ** 2
    return out


@cupy.fuse()
def _lin_to_dB(
    x: cupy.ndarray, out: cupy.ndarray, scale_ln: np.floating
) -> cupy.ndarray:
    out[:] = scale_ln * cupy.log(cupy.abs(x))
    return out


@cupy.fuse()
def _lin_to_dB_noabs(
    x: cupy.ndarray, out: cupy.ndarray, scale_ln: np.floating
) -> cupy.ndarray:
    out[:] = scale_ln * cupy.log(x)
    return out


@cupy.fuse()
def _lin_to_dB_eps(
    x: cupy.ndarray, out: cupy.ndarray, scale_ln: np.floating, eps: np.floating
) -> cupy.ndarray:
    out[:] = scale_ln * cupy.log(cupy.abs(x) + eps)
    return out


@cupy.fuse()
def _lin_to_dB_eps_noabs(
    x: cupy.ndarray, out: cupy.ndarray, scale_ln: np.floating, eps: np.floating
) -> cupy.ndarray:
    out[:] = scale_ln * cupy.log(x + eps)
    return out


@cupy.fuse()
def _dBtopow(x: cupy.ndarray, out: cupy.ndarray) -> cupy.ndarray:
    out[:] = cupy.power(10.0, x / 10)
    return out


_BIN_POWER_KERNELS: dict[tuple[str, bool, str], cupy.ReductionKernel] = {}


def bin_power_kernel(
    stat: Literal['mean', 'max'], complex_input: bool, out_dtype: np.dtype
) -> cupy.ReductionKernel:
    """return the reduction kernel of the `stat` of ``|x|**2`` over one axis.

    The kernel squares each sample and reduces in one pass, with no buffer of the
    size of the input. The accumulation runs in `out_dtype`.

    Args:
        stat: ``'mean'`` or ``'max'``
        complex_input: True for complex samples (``re**2 + im**2``); False for real
            samples (``x**2``)
        out_dtype: the floating-point dtype of the result and the accumulation

    Returns:
        a kernel called as ``kernel(x, axis=...)`` that returns the statistic of each
        row along ``axis``
    """
    dtype = np.dtype(out_dtype).name
    key = (stat, complex_input, dtype)
    kernel = _BIN_POWER_KERNELS.get(key)
    if kernel is not None:
        return kernel

    power = 'x.real() * x.real() + x.imag() * x.imag()' if complex_input else 'x * x'
    if stat == 'mean':
        # _in_ind.size() / _out_ind.size() is the length of the reduced axis
        reduce_expr, post = 'a + b', 'y = a / (_in_ind.size() / _out_ind.size())'
    elif stat == 'max':
        # 0 is the identity because every mapped value is a power
        reduce_expr, post = 'max(a, b)', 'y = a'
    else:
        raise ValueError(f"stat must be 'mean' or 'max', not {stat!r}")

    kind = 'c' if complex_input else 'r'
    kernel = cupy.ReductionKernel(
        'T x',
        f'{dtype} y',
        power,
        reduce_expr,
        post,
        '0',
        f'striqt_bin_power_{stat}_{kind}_{dtype}',
    )
    _BIN_POWER_KERNELS[key] = kernel
    return kernel


def get_kernel(
    conversion: Literal['lin_to_dB', 'envtopow', 'dBtopow'],
    *,
    abs: bool = True,
    bounded: bool = False,
) -> Callable[..., cupy.ndarray]:
    """return the fused kernel for `conversion`.

    Args:
        conversion: the conversion evaluated by the kernel
        abs: True selects the ``lin_to_dB`` variant that converts ``abs(x)``; ignored
            for the other conversions
        bounded: True selects the ``lin_to_dB`` variant with a trailing ``eps``
            argument; ignored for the other conversions

    Returns:
        a kernel called as ``(x, out, scale_ln)`` or ``(x, out, scale_ln, eps)`` for
        ``lin_to_dB``, and as ``(x, out)`` for ``envtopow`` and ``dBtopow``
    """
    if conversion == 'envtopow':
        return _envtopow
    if conversion == 'dBtopow':
        return _dBtopow
    if bounded:
        return _lin_to_dB_eps if abs else _lin_to_dB_eps_noabs
    return _lin_to_dB if abs else _lin_to_dB_noabs
