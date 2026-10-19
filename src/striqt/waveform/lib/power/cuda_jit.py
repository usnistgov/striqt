"""cupy.fuse kernels for the dB and power conversions.

Each kernel assigns through ``out[:]`` and requires ``out.ndim > 0``.
"""

from __future__ import annotations as __

import cupy  # ty: ignore[unresolved-import]
import numpy as np


@cupy.fuse()
def envtopow(x: cupy.ndarray, out: cupy.ndarray) -> cupy.ndarray:
    out[:] = cupy.abs(x) ** 2
    return out


@cupy.fuse()
def lin_to_dB(
    x: cupy.ndarray, out: cupy.ndarray, scale_ln: np.floating
) -> cupy.ndarray:
    out[:] = scale_ln * cupy.log(cupy.abs(x))
    return out


@cupy.fuse()
def lin_to_dB_noabs(
    x: cupy.ndarray, out: cupy.ndarray, scale_ln: np.floating
) -> cupy.ndarray:
    out[:] = scale_ln * cupy.log(x)
    return out


@cupy.fuse()
def lin_to_dB_eps(
    x: cupy.ndarray, out: cupy.ndarray, scale_ln: np.floating, eps: np.floating
) -> cupy.ndarray:
    out[:] = scale_ln * cupy.log(cupy.abs(x) + eps)
    return out


@cupy.fuse()
def lin_to_dB_eps_noabs(
    x: cupy.ndarray, out: cupy.ndarray, scale_ln: np.floating, eps: np.floating
) -> cupy.ndarray:
    out[:] = scale_ln * cupy.log(x + eps)
    return out


@cupy.fuse()
def dBtopow(x: cupy.ndarray, out: cupy.ndarray) -> cupy.ndarray:
    out[:] = cupy.power(10.0, x / 10)
    return out
