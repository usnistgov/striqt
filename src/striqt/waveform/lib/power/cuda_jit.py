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
