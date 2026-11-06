"""Cyclic-prefix correlation of a waveform against a delayed copy of itself at
selected index sets, in the array namespace of the waveform."""

from __future__ import annotations as __

import typing

from ..arrays import ROUNDOFF_SAFETY, array_namespace, unit_roundoff

if typing.TYPE_CHECKING:
    from ..typing import Array, DTypeLike


def correlate_cyclic_shift(
    inds: Array, x: Array, nfft: int, norm: bool = True, out: Array | None = None
) -> Array:
    """correlate `x` with a copy of `x` delayed `nfft` samples at the index sets `inds`.

    The value at lag ``j`` is the sum of ``x[i + j] * conj(x[i + j + nfft])`` over
    every index ``i`` in `inds`. Index pairs that run past the end of `x` contribute
    zero to the sum and to the powers of the normalization. The sum is accumulated in
    complex128 and rounded once into the dtype of `x`. With ``norm=True``, a lag at
    which every pair runs past the end of `x` is ``nan``.

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

    nfft = int(nfft)
    ncp = inds.shape[-1]
    n_lags = nfft + ncp
    rows = inds.reshape(-1, ncp)

    # a row of consecutive indices is gathered once from its first index and summed
    # over the run length by a sliding window. The gathered array is
    # (rows, run + lags - 1) rather than (rows * ncp, lags)
    if bool(xp.all(rows == rows[:, :1] + xp.arange(ncp))):
        starts, run = rows[:, 0], ncp
    else:
        starts, run = inds.reshape(-1), 1

    lo = int(starts.min())
    span = int(starts.max()) - lo + run + n_lags - 1
    seg = x[lo : min(lo + span + nfft, x.shape[0])].astype(xp.complex128)
    n_valid = max(seg.shape[0] - nfft, 0)
    a = seg[:n_valid]
    b = seg[nfft : nfft + n_valid]

    # the sliding sum over the run requires run - 1 lags past the last output lag
    start_gather = (starts - lo)[:, None] + xp.arange(run + n_lags - 1)[None, :]
    run_gather = xp.arange(run)[:, None] + xp.arange(n_lags)[None, :]

    def sum_over_pairs(values: Array) -> Array:
        padded = xp.zeros(span, dtype=values.dtype)
        padded[:n_valid] = values
        by_lag = padded[start_gather].sum(axis=0)
        if run == 1:
            return by_lag
        return by_lag[run_gather].sum(axis=0)

    corr = sum_over_pairs(a * xp.conj(b))

    if norm:
        power_a = sum_over_pairs(xp.abs(a) ** 2)
        power_b = sum_over_pairs(xp.abs(b) ** 2)
        corr /= xp.sqrt(power_a * power_b)
    else:
        corr /= inds.size

    if out is None:
        out = xp.empty(n_lags, dtype=x.dtype)
    out[:] = corr
    return out


def corr_atol(
    dtype: DTypeLike, n_inds: int, norm: bool, scale: float = 1.0, n_impl: int = 1
) -> float:
    """bound the absolute roundoff of `correlate_cyclic_shift` against exact arithmetic.

    The bound grows linearly with `n_inds` in unit roundoffs of `dtype`, scaled by
    `scale` when `norm` is False.

    Args:
        dtype: the input dtype (real or complex)
        n_inds: the number of index pairs summed at each lag
        norm: the `norm` passed to `correlate_cyclic_shift`
        scale: ``max|x|**2`` when `norm` is False; ignored when `norm` is True
        n_impl: 1 against exact arithmetic, 2 against a second implementation

    Returns:
        the absolute tolerance, in the units of the correlation
    """
    # the bound allows one rounding at the input precision for each product a*conj(b)
    # and each power term, and one for the output. The complex128 accumulation in
    # `correlate_cyclic_shift` rounds by less than that. With norm=True, |output| <= 1
    # by Cauchy-Schwarz: sum|a||b| <= sqrt(Pa*Pb)
    u = unit_roundoff(dtype)
    scale = 1.0 if norm else scale
    return ROUNDOFF_SAFETY * n_impl * (n_inds + 3) * u * scale
