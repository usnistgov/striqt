import numba as nb
import math
import numpy as np


@nb.njit(parallel=True, cache=True)
def _corr_at_indices(
    inds: np.ndarray, x: np.ndarray, nfft: int, ncp: int, norm: bool, out: np.ndarray
):
    # j: autocorrelation sequence (output) index
    for j in nb.prange(nfft + ncp):
        accum_corr = nb.complex128(0 + 0j)
        accum_power_a = nb.float64(0.0)
        accum_power_b = nb.float64(0.0)

        # i: the sample index of each waveform sample to compare against its shift
        for i in nb.prange(inds.shape[0]):
            ix = inds[i] + j
            ix_next = ix + nfft

            if ix_next < x.shape[0]:
                a = x[ix]
                b = x[ix_next]
            else:
                a = 0
                b = 0

            bconj = b.conjugate()
            accum_corr += a * bconj
            if norm:
                accum_power_a += (a * a.conjugate()).real
                accum_power_b += (b * bconj).real

        if norm:
            # normalize by the standard deviation under the assumption
            # that the voltage has a mean of zero
            accum_corr /= math.sqrt(accum_power_a * accum_power_b)
        else:
            # power normalization: scale by number of indices
            accum_corr /= inds.shape[0]

        out[j] = accum_corr
