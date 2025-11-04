import numba as nb
import numba.cuda
import cupy
import math


@numba.cuda.jit(cache=True)
def _corr_at_indices(inds, x, nfft: int, ncp: int, norm: bool, out):
    # iterate on parallel across the points in the output correlation
    j = nb.cuda.grid(1)

    if j < nfft + ncp:
        # from here on, the numba code is identical to _corr_at_indices_cpu
        accum_corr = nb.complex128(0 + 0j)
        accum_power_a = nb.float64(0.0)
        accum_power_b = nb.float64(0.0)

        # i: the sample index of each waveform sample to compare against its cyclic shift
        for i in range(inds.shape[0]):
            ix = inds[i] + j
            ix_next = ix + nfft

            if ix_next >= x.shape[0]:
                break

            a = x[ix]
            b = x[ix_next]

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


@cupy.fuse()
def envtopow(x, out):
    out[:] = cupy.abs(x) ** 2
    return out


@cupy.fuse()
def envtodB(x, out):
    out[:] = 20 * cupy.log10(cupy.abs(x))
    return out


@cupy.fuse()
def envtodB_noabs(x, out):
    out[:] = 20 * cupy.log10(x)
    return out


@cupy.fuse()
def envtodB_eps(x, out, eps):
    out[:] = 20 * cupy.log10(cupy.abs(x) + eps)
    return out


@cupy.fuse()
def envtodB_eps_noabs(x, out, eps):
    out[:] = 20 * cupy.log10(x + eps)
    return out


@cupy.fuse()
def powtodB(x, out):
    out[:] = 10 * cupy.log10(cupy.abs(x))
    return out


@cupy.fuse()
def powtodB_eps_noabs(x, out, eps=0):
    out[:] = 10 * cupy.log10(x + eps)
    return out


@cupy.fuse()
def powtodB_eps(x, out, eps=0):
    out[:] = 10 * cupy.log10(cupy.abs(x) + eps)
    return out


@cupy.fuse()
def powtodB_noabs(x, out):
    out[:] = 10 * cupy.log10(x)
    return out


@cupy.fuse()
def dBtopow(x, out):
    out[:] = 10 ** (x / 10)
    return out
