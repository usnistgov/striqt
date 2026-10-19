"""Cyclic-prefix correlation at index sets.

`correlate_cyclic_shift` dispatches to the numba kernels in the sibling modules
`cp_cpu_jit` (numpy) and `cp_cuda_jit` (cupy).
"""

from __future__ import annotations as __

import typing

from ..arrays import ROUNDOFF_SAFETY, array_namespace, unit_roundoff
from ..util import array_api_compat

if typing.TYPE_CHECKING:
    from ..typing import Array, DTypeLike


def correlate_cyclic_shift(
    inds: Array, x: Array, nfft: int, norm: bool = True, out: Array | None = None
) -> Array:
    """correlate `x` against its `nfft`-sample cyclic shift at the index sets in `inds`.

    Args:
        inds: integer sample indices into `x`, with the cyclic-prefix sample on the
            last axis; the last-axis size ``ncp`` is the number of index pairs summed
            per output lag, and the leading axes are flattened before use
        x: complex waveform, 1-D
        nfft: shift in samples between a sample and its cyclic copy
        norm: True normalizes each lag by the geometric mean of the powers in the
            two windows; False divides by the number of index pairs
        out: buffer of length ``nfft + ncp`` and the dtype of `x`

    Returns:
        correlation at each lag, length ``nfft + ncp``, in the dtype and array
        namespace of `x`; `out` when given
    """
    xp = array_namespace(x)
    assert xp is array_namespace(inds)

    # the number of waveform samples per cyclic prefix
    ncp = inds.shape[-1]
    flat_inds = inds.flatten()

    if out is None:
        out = xp.empty(nfft + ncp, dtype=x.dtype)

    if array_api_compat.is_numpy_array(x):
        from .cp_cpu_jit import _corr_at_indices as func
    else:
        from .cp_cuda_jit import _corr_at_indices

        tpb = 32
        bpg = max((x.size + (tpb - 1)) // tpb, 1)

        func = _corr_at_indices[bpg, tpb]

    func(flat_inds, x, int(nfft), int(ncp), bool(norm), out)

    return out


def corr_atol(
    dtype: DTypeLike, n_inds: int, norm: bool, scale: float = 1.0, n_impl: int = 1
) -> float:
    """bound the absolute roundoff of `correlate_cyclic_shift` against exact arithmetic.

    The bound grows linearly with `n_inds` in unit roundoffs of `dtype`. Each
    product ``a*conj(b)`` and the power terms are rounded at the input precision
    before the complex128 accumulation, and the result is rounded once more on
    output. With ``norm=True`` the Cauchy-Schwarz bound ``sum|a||b| <= sqrt(Pa*Pb)``
    makes the error relative to a unit-scale output, and `scale` is ignored. With
    ``norm=False`` the error is relative to the largest product magnitude, which the
    caller passes as ``scale = max|x|**2``.

    Args:
        dtype: the input dtype (real or complex)
        n_inds: the number of index pairs summed at each lag
        norm: the `correlate_cyclic_shift` normalization flag
        scale: ``max|x|**2`` when `norm` is False
        n_impl: 1 against exact arithmetic, 2 against a second implementation
    """
    u = unit_roundoff(dtype)
    scale = 1.0 if norm else scale
    return ROUNDOFF_SAFETY * n_impl * (n_inds + 3) * u * scale
