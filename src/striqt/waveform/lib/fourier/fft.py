"""FFT evaluation on numpy and cupy, the exact `fftfreq` grid with its band selectors,
and the roundoff model that budgets FFT error.

The cupy path calls into cuFFT through `cupy.fft._fft`, which honors the `out=` buffer,
in chunks of `MAX_CUPY_FFT_SAMPLES` when that is set.
"""

from __future__ import annotations as __

import decimal
import typing
from os import cpu_count

import array_api_compat

from .. import util
from ..arrays import (
    array_namespace,
    axis_slice,
    axis_to_blocks,
    cp,
    grouped_views_along_axis,
    is_cupy_array,
    isroundmod,
    unit_roundoff,
)
from ..util import np, scipy

if typing.TYPE_CHECKING:
    from types import ModuleType

    from ..typing import _AT, Array, ArrayBackend, DTypeLike


CPU_COUNT = cpu_count() or 1

# this tunes a tradeoff between CPU and memory consumption
# as of cupy 12
MAX_CUPY_FFT_SAMPLES = None


# %% roundoff model
#
# The FFT roundoff models below are rms models with fitted constants (see the functions
# for their provenance); FFT_ROUNDOFF_SAFETY is the margin the tolerances built on them
# carry, and independent backends' errors add in quadrature (measured 0.83-1.0).
FFT_ROUNDOFF_SAFETY = 3


def fft_roundoff_rms(
    dtype: DTypeLike,
    nffts: typing.Iterable[int],
    n_elementwise: int = 0,
    *,
    array_backend: ArrayBackend = 'numpy',
) -> float:
    """return the expected rms roundoff of one backend, relative to the output rms.

    Args:
        dtype: the working dtype (a complex dtype selects its real component)
        nffts: the size of each FFT pass in the computation
        n_elementwise: the number of elementwise roundings in the computation
        array_backend: the backend whose FFT constant applies, named as in a source
            spec so that a budget can be evaluated where that backend is not installed

    Returns:
        a dimensionless fraction of the output rms, proportional to the unit
        roundoff of `dtype`, that grows with each FFT size in `nffts` and with
        `n_elementwise`, and is larger for cupy than for numpy
    """
    # Per FFT pass the rms error relative to the output rms is c*eps*sqrt(log2 N), eps
    # the unit roundoff (Gentleman & Sande 1966; FFTW accuracy notes), and each
    # elementwise rounding adds (eps/sqrt(3))**2 of error variance. c was fitted with
    # chores/tests/measure_fft_accuracy.py per backend: pocketfft gives 0.55-0.65 (1.2
    # for sizes with a prime factor >= 128); cuFFT on a Jetson TX2i reaches 1.9 at
    # N=512 and 2.1 for Bluestein sizes.
    c = {'numpy': 1.2, 'cupy': 2.2}[array_backend]
    eps = unit_roundoff(dtype)
    var = n_elementwise * (eps / np.sqrt(3)) ** 2
    var += sum((c * eps * np.sqrt(np.log2(n))) ** 2 for n in nffts)
    return float(np.sqrt(var))


def fft_tolerance_rms(
    dtype: DTypeLike,
    nffts: typing.Iterable[int],
    n_elementwise: int = 0,
    *,
    array_backend: ArrayBackend = 'numpy',
) -> float:
    """return the rms tolerance on one backend's error against an exact reference.

    The tolerance is relative to the output rms and exceeds `fft_roundoff_rms`,
    which documents the arguments, by the module's safety margin.
    """
    return FFT_ROUNDOFF_SAFETY * fft_roundoff_rms(
        dtype, nffts, n_elementwise, array_backend=array_backend
    )


def peak_factor(size: int) -> float:
    """bound the max/rms ratio of `size` complex gaussian errors with 2x tail margin"""
    return float(2 * np.sqrt(np.log(size)))


def on_peak_roundoff(dtype: DTypeLike) -> float:
    """bound the structured roundoff in the matched-filter bin of the matched input.

    The bound is relative to that peak's amplitude and is a fixed multiple of the
    unit roundoff of `dtype`. This is where roundoff concentrates: the bin of a tone
    in the frequency domain (an FFT), or the sample of an impulse in the time domain
    (a resampler or detector).
    """
    # an FFT of a tone measured up to 6.7 units of roundoff of the tone amplitude at
    # the tone (cuFFT, N=512) and 3.9 in a far bin (cuFFT, N=1024), against ~2 for
    # pocketfft
    on_peak_roundings = 8
    return FFT_ROUNDOFF_SAFETY * on_peak_roundings * unit_roundoff(dtype)


def rms_tolerance_dBc(sigma: float) -> float:
    """convert a relative rms amplitude tolerance `sigma` to error power in dBc.

    `sigma` is relative to the output rms, so the result is the error power relative
    to the signal: negative for `sigma` below 1, and unbounded below as `sigma`
    approaches 0.
    """
    return float(20 * np.log10(sigma))


def off_peak_floor_dBc(
    sigma: float, nfft: int, size: int | None = None, dtype: DTypeLike = 'complex64'
) -> float:
    """express the roundoff floor in off-peak bins of a matched-filter output, in dBc.

    The matched input (a bin-centered tone) occupies one bin while roundoff of
    relative rms `sigma` spreads evenly over all `nfft` bins, so the floor below the
    peak deepens with `nfft`. With `size`, the result is the peak tolerance over that
    many off-peak bins, which is the larger of the white-noise tail and the
    structured `on_peak_roundoff` of `dtype`.
    """
    if size is None:
        return rms_tolerance_dBc(sigma / np.sqrt(nfft))
    white = peak_factor(size) * sigma / np.sqrt(nfft)
    return rms_tolerance_dBc(max(white, on_peak_roundoff(dtype)))


# %% FFTs
def set_max_cupy_fft_chunk(count: int | None):
    global MAX_CUPY_FFT_SAMPLES
    MAX_CUPY_FFT_SAMPLES = count


def get_max_cupy_fft_chunk():
    return MAX_CUPY_FFT_SAMPLES


def fft(
    x,
    n=None,
    axis=-1,
    out=None,
    overwrite_x=False,
    plan=None,
    workers: int | None = None,
):
    if is_cupy_array(x):
        assert cp is not None, ImportError('cupy is not installed')

        return _cupy_fftn_helper(
            x,
            n=n,
            axis=axis,
            direction=cp.cuda.cufft.CUFFT_FORWARD,
            out=out,
            overwrite_x=overwrite_x,
            plan=plan,
        )

    else:
        if workers is None:
            workers = CPU_COUNT // 2
        return scipy.fft.fft(
            x, n=n, axis=axis, workers=workers, overwrite_x=overwrite_x, plan=plan
        )


def ifft(
    x,
    n=None,
    axis=-1,
    out=None,
    overwrite_x=False,
    plan=None,
    workers=None,
):
    if is_cupy_array(x):
        assert cp is not None, ImportError('cupy is not installed')

        return _cupy_fftn_helper(
            x,
            n=n,
            axis=axis,
            direction=cp.cuda.cufft.CUFFT_INVERSE,
            out=out,
            overwrite_x=overwrite_x,
            plan=plan,
        )
    else:
        if workers is None:
            workers = CPU_COUNT // 2
        return scipy.fft.ifft(
            x, n=n, axis=axis, workers=workers, overwrite_x=overwrite_x, plan=plan
        )


def _cupy_fftn_helper(
    x,
    n,
    axis,
    direction,
    out=None,
    overwrite_x=False,
    plan=None,
):
    assert cp is not None, ImportError('cupy is not installed')
    from cupy.fft import _fft  # type: ignore

    if n is not None:
        raise Exception('implement n here')

    kws = dict(overwrite_x=overwrite_x, plan=plan, order='C')
    args = (None,), (axis,), None, direction

    # TODO: see about upstream question on this
    if out is None:
        return _fft._fftn(x, out=out, *args, **kws)
    else:
        out = out.reshape(x.shape)

    if MAX_CUPY_FFT_SAMPLES is None:
        return _fft._fftn(x, out=out, *args, **kws)

    x_views = grouped_views_along_axis(x, MAX_CUPY_FFT_SAMPLES, axis=axis)
    out_views = grouped_views_along_axis(out, MAX_CUPY_FFT_SAMPLES, axis=axis)

    for x_view, out_view in zip(x_views, out_views):
        out_view[:] = _fft._fftn(x_view, out=out_view, *args, **kws)

    return out


@util.lru_cache()
def fftfreq(
    nfft: int,
    fs: float,
    dtype='float64',
    as_index: bool = True,
    xp: ModuleType = np,
) -> Array:
    """compute fftfreq for a specified sample rate.

    This supports varying levels of accuracy:
    - with as_index=True, the result is accurate to full supported
      floating point precision using fixed-point evaluation
    - with as_index=False, the result is at least as accurate as
      `scipy.fft.fftfreq`, which includes rounding error relative
      to floating point precision
    """

    if not as_index:
        dtype = np.dtype(dtype)
        fnyq = fs / 2
        if nfft % 2 == 0:
            return xp.linspace(-fnyq, fnyq - 2 * fnyq / nfft, nfft, dtype=dtype)
        else:
            return xp.linspace(
                -fnyq + fnyq / nfft, fnyq - fnyq / nfft, nfft, dtype=dtype
            )

    if not array_api_compat.is_numpy_namespace(xp):
        return xp.asarray(fftfreq(nfft, fs, dtype, as_index=as_index))

    # high resolution rational representation of frequency resolution
    fres = decimal.Decimal(fs) / nfft
    span = range(-nfft // 2, -nfft // 2 + nfft)
    if nfft % 2 == 0:
        values = [fres * n for n in span]
    else:
        values = [fres * (n + 1) for n in span]
    return xp.array(values, dtype=dtype)


def time_fftshift(x, scale: Array | float | None = None, overwrite_x=False, axis=0):
    if scale is None:
        if overwrite_x:
            # short path: no scale and write directly to x
            xview = axis_slice(x, start=1, step=2, axis=axis)
            xview *= -1
            return x
        else:
            scale = 1

    xp = array_namespace(x)

    if overwrite_x:
        out = x
    else:
        out = xp.empty_like(x)

    if np.ndim(scale) > 1:
        raise ValueError('scale must be 1-D or scalar')

    xview = axis_to_blocks(x, 2, axis=axis)
    outview = axis_to_blocks(out, 2, axis=axis)
    scale = _broadcast_onto(xp.atleast_1d(scale), outview, axis=max(axis - 1, 0))
    shift = _broadcast_onto(xp.array([1, -1]), outview, axis=axis + 1)
    xp.multiply(xview, scale * shift, out=outview)
    return out


time_ifftshift = time_fftshift


# %% Frequency grid selection
@util.lru_cache()
def slice_freqs(
    nfft: int, fs: float, bandwidth: float, *, offset: float = 0.0
) -> slice:
    """return the slice of `bandwidth` about `offset` on an `nfft`-bin frequency axis.

    The axis is the ascending `fftfreq` grid at sample rate `fs`; `fs`, `bandwidth`
    and `offset` are in Hz. Validators call this for its exceptions.

    Raises:
        ValueError: if `bandwidth` is negative, if `offset` is off the ``fs/nfft``
            grid, or if the band extends outside ``±fs/2``
    """
    if bandwidth < 0:
        raise ValueError('invalid negative bandwidth')

    ic = nfft * (0.5 + offset / fs)
    if isroundmod(ic, 1):
        ic = round(ic)
    else:
        raise ValueError('offset center frequency is not a multiple of fs/nfft')

    count = round(nfft * (bandwidth / fs))

    start = ic - count // 2
    stop = start + count
    if start < 0:
        raise ValueError(f'offset - bandwidth/2 < fs/2')
    if stop > nfft:
        raise ValueError(f'offset + bandwidth/2 > fs/2')

    return slice(start, stop)


def truncate_freqs(
    x: _AT,
    nfft: int,
    fs: float,
    bandwidth: float,
    *,
    offset: float = 0.0,
    axis: int = 0,
) -> _AT:
    """trim an array outside of the specified bandwidth on a frequency axis"""

    s = slice_freqs(nfft, fs, bandwidth, offset=offset)
    return typing.cast('_AT', axis_slice(x, s.start, s.stop, axis=axis))


def null_lo(
    x: Array,
    nfft: int,
    fs: float,
    bandwidth: float,
    *,
    offset: float = 0,
    axis: int = 0,
):
    """sets samples within the specified bandwidth on a frequency axis to nan in-place"""
    # to make the top bound inclusive
    nfft = x.shape[axis]
    s = slice_freqs(nfft, fs, bandwidth, offset=offset)
    view = axis_slice(x, s.start, s.stop, axis=axis)
    view[:] = float('nan')


@util.lru_cache()
def _freq_band_edges(n, d, cutoff_low, cutoff_hi, *, xp=None):
    if xp is None:
        xp = np
    freqs = fftfreq(n, d, xp=xp)

    if cutoff_low is None:
        ilo = None
    else:
        ilo = xp.where(freqs >= cutoff_low)[0][0]

    if cutoff_hi is None:
        ihi = None
    elif cutoff_hi >= freqs[-1]:
        ihi = freqs.size
    else:
        ihi = int(xp.searchsorted(freqs, cutoff_hi))

    return ilo, ihi


# %% Helpers
def _broadcast_onto(a: _AT, other: _AT, *, axis: int) -> _AT:
    """reshape a 1-D array to support broadcasting onto a specified axis of `other`"""

    if a.ndim != 1:
        raise ValueError('input array a must be 1-D')

    xp = array_namespace(a)

    slices = [xp.newaxis] * int(other.ndim)
    slices[axis] = slice(None, None)
    return typing.cast('_AT', a[tuple(slices)])
