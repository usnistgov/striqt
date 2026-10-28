"""Cyclic-prefix correlation at index sets, evaluated by numba kernels (CPU for numpy
arrays, CUDA for cupy arrays)."""

from __future__ import annotations as __

import typing

from ..arrays import ROUNDOFF_SAFETY, array_namespace, unit_roundoff
from ..util import array_api_compat

if typing.TYPE_CHECKING:
    from ..typing import Array, DTypeLike


def correlate_cyclic_shift(
    inds: Array, x: Array, nfft: int, norm: bool = True, out: Array | None = None
) -> Array:
    """correlate `x` with a copy of `x` delayed `nfft` samples at the index sets `inds`.

    At lag ``j``, the result sums ``x[i + j] * conj(x[i + j + nfft])`` over every
    index ``i`` in `inds`. Index pairs that run past the end of `x` contribute zero.

    Args:
        inds: integer sample indices into `x` of cyclic-prefix samples, in the array
            namespace of `x`, with the sample within each prefix on the last axis of
            size ``ncp``
        x: complex waveform, 1-D
        nfft: the delay (in samples) from a sample to the cyclic copy of that sample
        norm: True divides each lag by the geometric mean of the summed powers at
            the two index sets; False divides by the number of indices
        out: buffer of length ``nfft + ncp`` in the dtype of `x`

    Returns:
        the correlation at each of the ``nfft + ncp`` lags, in the dtype and array
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

    The bound grows linearly with `n_inds` in unit roundoffs of `dtype`. With
    ``norm=True`` the output is unit-scale and `scale` is ignored. With ``norm=False``
    the bound is relative to the largest product magnitude `scale`.

    Args:
        dtype: the input dtype (real or complex)
        n_inds: the number of index pairs summed at each lag
        norm: the `correlate_cyclic_shift` normalization flag
        scale: ``max|x|**2`` when `norm` is False
        n_impl: 1 against exact arithmetic, 2 against a second implementation
    """
    # each product a*conj(b) and each power term is rounded once at the input
    # precision before the complex128 accumulation, and the output is rounded once
    # more; with norm=True the Cauchy-Schwarz bound sum|a||b| <= sqrt(Pa*Pb) keeps
    # the output at unit scale
    u = unit_roundoff(dtype)
    scale = 1.0 if norm else scale
    return ROUNDOFF_SAFETY * n_impl * (n_inds + 3) * u * scale
