from __future__ import annotations
import functools
import typing

from os import cpu_count
from math import ceil

from . import power_analysis

from .util import (
    array_namespace,
    axis_index,
    axis_slice,
    Domain,
    dtype_change_float,
    find_float_inds,
    grouped_views_along_axis,
    get_input_domain,
    lazy_import,
    lru_cache,
    pad_along_axis,
    sliding_window_view,
    to_blocks,
    isroundmod,
    is_cupy_array
)

from .windows import register_extra_windows

if typing.TYPE_CHECKING:
    import numpy as np
    import scipy
    from scipy import signal
    from ._typing import ArrayType

else:
    np = lazy_import('numpy')
    scipy = lazy_import('scipy')
    signal = lazy_import('signal', 'scipy')

CPU_COUNT = cpu_count() or 1
OLA_MAX_FFT_SIZE = 128 * 1024
INF = float('inf')

# this tunes a tradeoff between CPU and memory consumption
# as of cupy 12
MAX_CUPY_FFT_SAMPLES = None


# required divisors
_COLA_WINDOW_SIZE_DIVISOR = {
    None: 1,
    'rect': 1,
    'hamming': 2,
    'blackman': 3,
    'blackmanharris': 5,
}


def set_max_cupy_fft_chunk(count: int | None):
    global MAX_CUPY_FFT_SAMPLES
    MAX_CUPY_FFT_SAMPLES = count


def get_max_cupy_fft_chunk():
    return MAX_CUPY_FFT_SAMPLES


def _get_window_uncached(
    name_or_tuple,
    nwindow: int,
    nzero: int = 0,
    *,
    fftshift: bool = False,
    center_zeros=False,
    fftbins=True,
    norm=True,
    dtype='float32',
    xp=None,
):
    """build an window function with optional zero-padding or parameter finding.

    Arguments:

    See also:
        `scipy.signal.get_window`
    """

    register_extra_windows()

    if xp is not None:
        w = _get_window_uncached(
            name_or_tuple,
            nwindow,
            nzero=nzero,
            fftbins=fftbins,
            norm=norm,
            fftshift=fftshift,
            dtype=dtype,
        )

        if hasattr(xp, 'asarray'):
            w = xp.asarray(w)
        else:
            w = xp.array(w)

        return w

    if isinstance(name_or_tuple, tuple):
        # maybe evaluate the window argument needed to realize the specified ENBW
        window_name, *suffix = name_or_tuple[0].rsplit('_by_enbw', 1)

        if len(suffix) > 0:
            enbw = name_or_tuple[1]
            param = find_window_param_from_enbw(window_name, enbw, nfft=nwindow)
            name_or_tuple = (window_name, param)

    ws = signal.windows.get_window(name_or_tuple, nwindow, fftbins=fftbins)

    ntotal = nwindow + nzero

    if nzero == 0:
        w = ws
    elif center_zeros:
        w = np.empty(ntotal, dtype=ws.dtype)
        w[nzero // 2 : nzero // 2 + nwindow] = ws
        w[: nzero // 2] = 0
        w[nzero // 2 + nwindow :] = 0
    else:
        w = np.empty(ntotal, dtype=ws.dtype)
        w[:nwindow] = ws
        w[nwindow:] = 0

    if norm:
        # scale the time-averaged power to 1
        w /= np.sqrt(np.mean(np.abs(w) ** 2))

    if fftshift:
        delay = scipy.ndimage.fourier_shift(np.ones_like(w), ntotal // 2)

        if ntotal % 2 == 0:
            # really just [1, -1, 1, -1, 1, ...]
            delay = delay.real

        w = delay * w

    if dtype is not None:
        dtype_out = dtype_change_float(w.dtype, dtype)
        w = w.astype(dtype_out)

    return w


get_window = functools.wraps(_get_window_uncached)(
    lru_cache(1024)(_get_window_uncached)
)


def _truncated_buffer(x: ArrayType, shape, dtype=None):
    if dtype is not None:
        x = x.view(dtype)
    out_size = np.prod(shape)
    assert x.size >= out_size
    return x.flatten()[:out_size].reshape(shape)


def _cupy_fftn_helper(
    x,
    axis,
    direction,
    out=None,
    overwrite_x=False,
    plan=None,
):
    import cupy as cp  # pyright: ignore[reportMissingImports]

    kws = dict(overwrite_x=overwrite_x, plan=plan, order='C')
    args = (None,), (axis,), None, direction

    # TODO: see about upstream question on this
    if out is None:
        args = (None,), (axis,), None, direction
        return cp.fft._fft._fftn(x, out=out, *args, **kws)
    else:
        out = out.reshape(x.shape)

    if MAX_CUPY_FFT_SAMPLES is None:
        return cp.fft._fft._fftn(x, out=out, *args, **kws)

    x_views = grouped_views_along_axis(x, MAX_CUPY_FFT_SAMPLES, axis=axis)
    out_views = grouped_views_along_axis(out, MAX_CUPY_FFT_SAMPLES, axis=axis)

    for x_view, out_view in zip(x_views, out_views):
        out_view[:] = cp.fft._fft._fftn(x_view, out=out_view, *args, **kws)

    return out


def fft(x, axis=-1, out=None, overwrite_x=False, plan=None, workers: int | None = None):
    if is_cupy_array(x):
        import cupy as cp  # pyright: ignore[reportMissingImports]

        return _cupy_fftn_helper(
            x,
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
            x, axis=axis, workers=workers, overwrite_x=overwrite_x, plan=plan
        )


def ifft(
    x,
    axis=-1,
    out=None,
    overwrite_x=False,
    plan=None,
    workers=None,
):
    if is_cupy_array(x):
        import cupy as cp  # pyright: ignore[reportMissingImports]

        return _cupy_fftn_helper(
            x,
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
            x, axis=axis, workers=workers, overwrite_x=overwrite_x, plan=plan
        )


def fftfreq(n, d, *, xp=None, dtype='float64') -> ArrayType:
    """A replacement for `scipy.fft.fftfreq` that mitigates
    some rounding errors underlying `np.fft.fftfreq`.

    Further, no `fftshift` is needed for complex-valued data;
    the return result is monotonic beginning in the negative
    frequency half-space.

    Args:
        n: fft size
        d: sample spacing (inverse of sample rate)
        xp: the array module to use, dictating the return type

    Returns:
        an array of type `xp.ndarray`
    """
    if xp is None:
        xp = np

    dtype = np.dtype(dtype)
    fnyq = 1 / (2 * dtype.type(d))
    if n % 2 == 0:
        return xp.linspace(-fnyq, fnyq - 2 * fnyq / n, n, dtype=dtype)
    else:
        return xp.linspace(-fnyq + fnyq / n, fnyq - fnyq / n, n, dtype=dtype)


def _enbw_uncached(
    window: str | tuple[str, float], N, fftbins=True, cached=True, xp=None
):
    """return the equivalent noise bandwidth (ENBW) of a window, in bins"""
    if cached:
        w = get_window(window, N, fftbins=fftbins, xp=xp)
    else:
        w = _get_window_uncached(window, N, fftbins=fftbins, xp=xp or None)
    return len(w) * xp.sum(w**2) / xp.sum(w) ** 2


# allow access to the uncached version for find_window_param_from_enbw
equivalent_noise_bandwidth = functools.wraps(_enbw_uncached)(
    functools.lru_cache()(_enbw_uncached)
)


@lru_cache()
def find_window_param_from_enbw(
    window_name: str, enbw: float, *, nfft: int = 4096, atol=1e-6, xp=None
) -> float:
    """find the parameter that satistifes the specified equivalent-noise bandwidth (ENBW)
    for a given single-parameter window.

    The estimate is performed for the given `nfft`; when it is
    larger, the result will tend to converge toward a central value.

    Typically, where enbw is at least 1.1:
        * when `window_name == 'dpss'`: the result will be slightly smaller than `(enbw)**2`
        * when `window_name == 'kaiser'`: the result will be slightly smaller than `pi * (enbw)**2`

    Arguments:
        window_name: one of 'kaiser', 'dpss', or 'chebwin'
        enbw: the desired equivalent noise bandwidth (in FFT bins)
        nfft: the window size used to estimate ENBW
        atol: the absolute error tolerance in the estimated NW

    Returns:
        result parameter suited for `get_window((window_name, result), ...)`
    """
    from scipy.optimize import bisect

    if enbw < 1 + 1 / nfft:
        raise ValueError('enbw must be greater than 1')

    def err(x):
        return _enbw_uncached((window_name, x), nfft, cached=False, xp=xp or np) - enbw

    if window_name == 'kaiser':
        a = np.pi * 1e-2
        b = min(enbw**2, nfft // 2 - 1) * np.pi
    elif window_name == 'dpss':
        a = 1e-2
        b = min(enbw**2, nfft // 2 - 1)
    elif window_name == 'chebwin':
        a = 45
        b = 1000
    else:
        raise ValueError('window_name must be one of ("kaiser", "dpss", "chebwin")')

    result = bisect(err, a, b, xtol=atol)
    assert isinstance(result, float)

    return result


def broadcast_onto(a: ArrayType, other: ArrayType, *, axis: int) -> ArrayType:
    """reshape a 1-D array to support broadcasting onto a specified axis of `other`"""

    if a.ndim != 1:
        raise ValueError('input array a must be 1-D')

    xp = array_namespace(a)

    slices = [xp.newaxis] * other.ndim
    slices[axis] = slice(None, None)
    return a[tuple(slices)]


@lru_cache(16)
def _get_stft_axes(
    fs: float, nfft: int, time_size: int, overlap_frac: float = 0, *, xp=None
) -> tuple[ArrayType, ArrayType]:
    """returns stft (freqs, times) array tuple appropriate to the array module xp"""
    if xp is None:
        xp = np

    freqs = fftfreq(nfft, 1 / fs, xp=xp)
    times = xp.arange(time_size) * ((1 - overlap_frac) * nfft / fs)

    return freqs, times


@lru_cache()
def _prime_fft_sizes(min=2, max=OLA_MAX_FFT_SIZE):
    s = np.arange(3, max, 2)

    for m in range(3, int(np.sqrt(max) + 1), 2):
        if s[(m - 3) // 2]:
            s[(m * m - 3) // 2 :: m] = 0

    return s[(s > min)]


class ResamplerDesign(typing.TypedDict):
    fs_sdr: float
    lo_offset: float
    window: str | tuple[str, float]
    nfft: int
    nfft_out: int
    frequency_shift: ShiftType
    passband: tuple[float | None, float | None]
    fs: float


ShiftType = (
    typing.Literal['left']
    | typing.Literal['right']
    | typing.Literal['none']
    | typing.Literal[False]
)


@lru_cache()
def design_cola_resampler(
    fs_base: float,
    fs_target: float,
    bw: float = INF,
    bw_lo: float = 0,
    min_oversampling: float = 1.1,
    min_fft_size=2 * 4096 - 1,
    shift: ShiftType = False,
    avoid_primes=True,
    window=None,
    fs_sdr: typing.Optional[float] = None,
) -> ResamplerDesign:
    """designs sampling and RF center frequency parameters that shift LO leakage outside of the specified bandwidth.

    The result includes the integer-divided SDR sample rate to request from the SDR, the LO frequency offset,
    and the keyword arguments needed to realize resampling with `ola_filter`.

    Args:
        fs_base: the base clock rate (sometimes known as master clock rate, MCR) of the receiver
        fs_target: the desired sample rate after resampling
        bw: the analysis bandwidth to protect from LO leakage
        bw_lo: the spectral leakage/phase noise bandwidth of the LO
        shift: the direction to shift the LO
        avoid_primes: whether to avoid large prime numbered FFTs for performance reasons
        fs_sdr: force the given sample rate (in Hz), or None to select automatically

    Returns:
        (SDR sample rate, RF LO frequency offset in Hz, ola_filter_kws)
    """

    if bw == INF and shift:
        raise ValueError(
            'frequency shifting may only be applied when an analysis bandwidth is specified'
        )

    if shift:
        fs_sdr_min = fs_target + min_oversampling * bw / 2 + bw_lo / 2
    else:
        fs_sdr_min = fs_target

    if fs_sdr is not None:
        pass
    elif fs_base <= fs_target:
        fs_sdr = fs_base
    elif shift and fs_sdr_min > fs_base:
        msg = f"""LO frequency shift with the requested parameters
        requires running the radio at a minimum {fs_sdr_min / 1e6:0.2f} MS/s,
        but its maximum rate is {fs_base / 1e6:0.2f} MS/s"""

        raise ValueError(msg)
    else:
        # the source permits arbitrary sample rates
        decimation = int(fs_base / fs_sdr_min)
        fs_sdr = fs_base / decimation

    if bw != INF and bw > fs_base:
        raise ValueError(
            'passband bandwidth exceeds Nyquist bandwidth at maximum sample rate'
        )

    resample_ratio = fs_sdr / fs_target

    # the following returns the modulos closest to either 0 or 1, accommodating downward rounding errors (e.g., 0.999)
    trial_noverlap = resample_ratio * np.arange(1, OLA_MAX_FFT_SIZE + 1)
    check_mods = isroundmod(trial_noverlap, 1) & (
        trial_noverlap > min_fft_size * resample_ratio
    )

    # all valid noverlap size candidates
    valid_noverlap_out = 1 + np.where(check_mods)[0]
    if avoid_primes:
        reject = _prime_fft_sizes(100)
        valid_noverlap_out = np.setdiff1d(valid_noverlap_out, reject, True)
    if len(valid_noverlap_out) == 0:
        raise ValueError('no rational FFT sizes satisfied design constraints')

    nfft_out = valid_noverlap_out[0]
    nfft_in = round(resample_ratio * nfft_out)

    divisor = _COLA_WINDOW_SIZE_DIVISOR[window]
    if nfft_out % divisor > 0 or nfft_in % divisor > 0:
        nfft_out *= divisor
        nfft_in *= divisor

    # the following LO shift arguments assume that a hamming COLA window is used
    if shift == 'left':
        sign = -1
    elif shift == 'right':
        sign = +1
    elif shift in ('none', False, None):
        sign = 0
    else:
        raise ValueError(f'shift argument must be "left" or "right", not {repr(shift)}')

    if sign != 0 and bw == INF:
        raise ValueError('a passband bandwidth must be set to design a LO shift')

    if bw == INF:
        lo_offset = 0
        passband = (None, None)
    else:
        lo_offset = sign * (
            bw / 2 + bw_lo / 2
        )  # fs_sdr / nfft_in * (nfft_in - nfft_out)
        passband = (lo_offset - bw / 2, lo_offset + bw / 2)

    return ResamplerDesign(
        fs_sdr=fs_sdr,
        lo_offset=lo_offset,
        window=window or 'hamming',
        nfft=int(nfft_in),
        nfft_out=int(nfft_out),
        frequency_shift=shift,
        passband=passband,
        fs=fs_sdr,
    )


def design_fir_resampler(
    fs_base: float,
    fs_target: float,
    bw: float = INF,
    bw_lo: float = 0,
    min_oversampling: float = 1.04,
) -> tuple[float, dict]:
    """designs sampling and RF center frequency parameters that shift LO leakage outside of the specified bandwidth.

    The result includes the integer-divided SDR sample rate to request from the SDR, the LO frequency offset,
    and the keyword arguments needed to realize resampling with `ola_filter`.

    Args:
        fs_base: the base clock rate (sometimes known as master clock rate, MCR) of the receiver
        fs_target: the desired sample rate after resampling
        bw: the analysis bandwidth to protect from LO leakage
        bw_lo: the spectral leakage/phase noise bandwidth of the LO
        shift: the direction to shift the LO
        avoid_primes: whether to avoid large prime numbered FFTs for performance reasons

    Returns:
        (SDR sample rate, upfirdn keywords)
    """

    design = design_cola_resampler(
        fs_base,
        fs_target,
        bw=bw,
        bw_lo=bw_lo,
        min_oversampling=min_oversampling,
        min_fft_size=1,
        avoid_primes=False,
    )

    fir_params = {
        'up': design['nfft_out'],
        'down': design['nfft'],
    }

    return design['fs'], fir_params


def _stack_stft_windows(
    x: ArrayType,
    window: ArrayType,
    nperseg: int,
    noverlap: int,
    norm=None,
    axis=0,
    out=None,
) -> ArrayType:
    """add overlapping windows at appropriate offset _to_overlapping_windows, returning a waveform.

    Compared to the underlying stft implementations in scipy and cupyx.scipy, this has been simplified
    to a reduced set of parameters for speed.

    Args:
        x: the 1-D waveform (or N-D tensor of waveforms)
        axis: the waveform axis; stft will be evaluated across all other axes
    """

    xp = array_namespace(x)

    hop_size = nperseg - noverlap

    strided = sliding_window_view(x, nperseg, axis=axis)
    xstacked = axis_slice(strided, start=0, step=hop_size, axis=axis)

    if norm is None:
        scale = xp.abs(window[::hop_size]).sum()
    elif norm == 'power':
        scale = 1
    else:
        raise ValueError(
            f"invalid normalization argument '{norm}' (should be 'cola' or 'psd')"
        )

    w = broadcast_onto(window / scale, xstacked, axis=axis + 1)
    return xp.multiply(xstacked, w.astype(xstacked.dtype), out=out)


def _unstack_stft_windows(
    y: ArrayType, noverlap: int, nperseg: int, axis=0, out=None, extra=0
) -> ArrayType:
    """reconstruct the time-domain waveform from its STFT representation.

    Compared to the underlying istft implementations in scipy and cupyx.scipy, this has been simplified
    for speed at the expense of memory consumption.

    Args:
        y: the stft output, containing at least 2 dimensions
        noverlap: the overlap size that was used to generate the STFT (see scipy.signal.stft)
        axis: the axis of the first dimension of the STFT (the second is at axis+1)
        out: if specified, the output array that will receive the result. it must have at least the same allocated size as y
        extra: total number of extra samples to include at the edges
    """

    xp = array_namespace(y)

    nfft = nperseg
    hop_size = nperseg - noverlap

    waveform_size = y.shape[axis] * y.shape[axis + 1] * hop_size // nfft + noverlap
    target_shape = y.shape[:axis] + (waveform_size,) + y.shape[axis + 2 :]

    if out is None:
        xr = xp.empty(target_shape, dtype=y.dtype)
    else:
        xr = _truncated_buffer(out, target_shape)

    xr_slice = axis_slice(
        xr,
        start=0,
        stop=noverlap,
        axis=axis,
    )
    xp.copyto(xr_slice, 0)

    xr_slice = axis_slice(
        xr,
        start=-noverlap,
        stop=None,
        axis=axis,
    )
    xp.copyto(xr_slice, 0)

    # for speed, sum up in groups of non-overlapping windows
    for offs in range(nfft // hop_size):
        yslice = axis_slice(y, start=offs, step=nfft // hop_size, axis=axis)
        yshape = yslice.shape

        yslice = yslice.reshape(
            yshape[:axis] + (yshape[axis] * yshape[axis + 1],) + yshape[axis + 2 :]
        )
        xr_slice = axis_slice(
            xr,
            start=offs * hop_size,
            stop=offs * hop_size + yslice.shape[axis],
            axis=axis,
        )

        if offs == 0:
            xp.copyto(xr_slice, yslice[: xr_slice.size])
        else:
            xr_slice += yslice[: xr_slice.size]

    return xr  # axis_slice(xr, start=noverlap-extra//2, stop=(-noverlap+extra//2) or None, axis=axis)


@lru_cache()
def _ola_filter_parameters(
    array_size: int, *, window, nfft_out: int, nfft: int, extend: bool
) -> tuple:
    if nfft_out is None:
        nfft_out = nfft

    try:
        divisor = _COLA_WINDOW_SIZE_DIVISOR[window]
    except KeyError:
        raise TypeError(
            'ola_filter argument "window" must be one of ("hamming", "blackman", or "blackmanharris")'
        )

    if nfft_out % divisor != 0:
        raise ValueError(
            f'{window!r} window COLA requires output nfft_out % {divisor} == 0'
        )

    if window is None or window == 'rect':
        overlap_scale = 1
    if window == 'hamming':
        overlap_scale = 1 / 2
    elif window == 'blackman':
        overlap_scale = 2 / 3
    elif window == 'blackmanharris':
        overlap_scale = 4 / 5
    else:
        raise ValueError('unexpected matching error')

    noverlap = round(nfft_out * overlap_scale)

    if array_size % noverlap != 0:
        if extend:
            pad_out = array_size % noverlap
        else:
            raise ValueError(
                f'x.size ({array_size}) is not an integer multiple of noverlap ({noverlap})'
            )
    else:
        pad_out = 0

    return nfft_out, noverlap, overlap_scale, pad_out


def _istft_buffer_size(
    array_size: int, *, window, nfft_out: int, nfft: int, extend: bool
):
    nfft_out, _, overlap_scale, pad_out = _ola_filter_parameters(**locals())
    nfft_max = max(nfft_out, nfft)
    fft_count = 2 + ((array_size + pad_out) / nfft_max) / overlap_scale
    size = ceil(fft_count * nfft_max)
    return size


def zero_stft_by_freq(
    freqs: ArrayType, xstft: ArrayType, *, passband: tuple[float, float], axis=0
) -> ArrayType:
    """apply a bandpass filter in the STFT domain by zeroing frequency indices"""
    xp = array_namespace(xstft)

    freq_step = float(freqs[1] - freqs[0])
    fs = xstft.shape[axis] * freq_step
    ilo, ihi = _freq_band_edges(freqs.size, fs, *passband, xp=xp)

    xp.copyto(axis_slice(xstft, 0, ilo, axis=axis + 1), 0)
    xp.copyto(axis_slice(xstft, ihi, None, axis=axis + 1), 0)
    return xstft


@lru_cache()
def design_fir_lpf(
    bandwidth,
    sample_rate,
    *,
    numtaps=4001,
    transition_bandwidth=250e3,
    dtype='float32',
    xp=None,
):
    if xp is None:
        xp = np
    edges = [
        0,
        bandwidth / 2 - transition_bandwidth / 2,
        bandwidth / 2 + transition_bandwidth / 2,
        sample_rate / 2,
    ]
    bands = list(zip(edges[:-1], edges[1:]))
    desired = [1, 1, 1, 0, 0, 0]

    b = signal.firls(numtaps, bands=bands, desired=desired, fs=sample_rate)

    return xp.asarray(b.astype(dtype))


@lru_cache()
def _fir_lowpass_fft(
    size: int,
    sample_rate: float,
    *,
    cutoff: float,
    transition: float,
    window='hamming',
    xp=None,
    dtype='complex64',
):
    """returns the complex frequency response of an FIR filter suited for filtering in the frequency domain

    Arguments:
        size: window size
        sample_rate: sample rate (in Hz)
        cutoff: filter cutoff (in Hz)
        transition: bandwidth of the transition (in Hz)

    Returns:
        a frequency-domain window
    """

    if xp is None:
        xp = np

    if cutoff == float('inf'):
        h = np.ones(size, dtype=dtype)
    else:
        freqs = [
            0,
            # cutoff - transition / 2,
            cutoff,
            cutoff + transition,
            sample_rate / 2,
        ]
        h = signal.firwin2(
            size, freqs, [1.0, 1, 0.0, 0.0], window=window, fs=sample_rate
        )

    taps = xp.array(h).astype(dtype)
    w = get_window('rect', size, xp=xp, dtype=dtype, fftshift=True)
    H = xp.fft.fft(taps * w)
    return H * w


def stft_fir_lowpass(
    xstft: ArrayType,
    *,
    sample_rate: float,
    bandwidth: float,
    transition_bandwidth: float,
    axis=0,
    out=None,
):
    xp = array_namespace(xstft)

    H = _fir_lowpass_fft(
        xstft.shape[axis + 1],
        sample_rate=sample_rate,
        cutoff=bandwidth / 2,
        transition=transition_bandwidth,
        dtype=xstft.dtype,
        window='rect',
        xp=xp,
    )

    H = broadcast_onto(H, xstft, axis=axis + 1)

    return xp.multiply(xstft, H, out=out)


@lru_cache(100)
def _find_downsample_copy_range(
    nfft_in: int, nfft_out: int, edge_in_start: int | None, edge_in_end: int | None
):
    if edge_in_start is None:
        edge_in_start = 0
    if edge_in_end is None:
        edge_in_end = nfft_in
    passband_size = edge_in_end - edge_in_start
    passband_center = (edge_in_end + edge_in_start) // 2

    # passband_center_error = (passband_end - passband_start) % 2

    # copy input indexes, taken from the passband
    max_copy_size = min(passband_size, nfft_out)
    copy_in_start = max(passband_center - max_copy_size // 2, 0)
    copy_in_end = min(passband_center - max_copy_size // 2 + max_copy_size, nfft_in)
    copy_size = copy_in_end - copy_in_start

    assert copy_size <= nfft_out, (copy_size, nfft_out)
    assert copy_size >= 0, copy_size
    assert copy_in_end - copy_in_start == copy_size

    # copy output indexes
    output_zeros_size = max(nfft_out - copy_size, 0)
    copy_out_start = output_zeros_size // 2
    copy_out_end = copy_out_start + copy_size

    assert copy_out_end - copy_out_start == copy_size
    assert copy_out_start >= 0
    assert copy_out_end <= nfft_out

    return (copy_out_start, copy_out_end), (copy_in_start, copy_in_end), passband_center


@lru_cache(16)
def _find_downsampled_freqs(nfft_out, freq_step, xp=None):
    return fftfreq(nfft_out, 1.0 / (freq_step * nfft_out), xp=xp or None)


def _same_base_memory(a: ArrayType, b: ArrayType) -> bool:
    if b is None:
        return False
    elif a is b:
        return True
    elif getattr(b, 'base', None) is a:
        return True
    else:
        return False


def downsample_stft(
    freqs: ArrayType,
    y: ArrayType,
    nfft_out: int,
    *,
    passband: tuple[float | None, float | None] = (None, None),
    axis=0,
    out=None,
) -> tuple[ArrayType, ArrayType]:
    """downsample and filter an STFT representation of a filter in the frequency domain.

    * This is rational downsampling by a factor of `nout/xstft.shape[axis+1]`,
      shifted if necessary to center the passband.
    * One approach to selecting `nfft_out` for this purpose is the use
      of `design_ola_filter`.

    Arguments:
        freqs: the list of FFT bin center frequencies
        y: the stft
        nfft_out: the number of points in the output fft

    Returns:
        A tuple containing the new `freqs` range and trimmed `xstft`
    """
    xp = array_namespace(y)
    ax = axis + 1

    shape_out = list(y.shape)
    shape_out[ax] = nfft_out

    # passband indexes in the input
    freq_step = float(freqs[1] - freqs[0])
    fs = y.shape[ax] * freq_step
    passband_start, passband_end = _freq_band_edges(y.shape[ax], 1 / fs, *passband)
    bounds_out, bounds_in, _ = _find_downsample_copy_range(
        y.shape[ax], nfft_out, passband_start, passband_end
    )
    freqs_out = _find_downsampled_freqs(nfft_out, freq_step, xp=xp)

    if tuple(bounds_out) == (0, shape_out[ax]) and _same_base_memory(y, out):
        # fast path: a view if both no zeroing is needed and the
        # output buffer shares underlying y
        return freqs_out, axis_slice(y, *bounds_in, axis=ax)

    if out is None:
        xout = xp.empty(shape_out, dtype=y.dtype)
    else:
        xout = _truncated_buffer(out, shape_out[ax], y.dtype)

    # copy first before zeroing, in case of input-output buffer reuse
    xp.copyto(
        axis_slice(xout, *bounds_out, axis=ax),  #
        axis_slice(y, *bounds_in, axis=ax),
    )

    xp.copyto(axis_slice(xout, 0, bounds_out[0], axis=ax), 0)
    xp.copyto(axis_slice(xout, bounds_out[1], None, axis=ax), 0)

    return freqs_out, xout


def stft(
    x: ArrayType,
    *,
    fs: float,
    window: ArrayType | str | tuple[str, float],
    nperseg: int = 256,
    noverlap: int = 0,
    nzero: int = 0,
    axis: int = 0,
    truncate: bool = True,
    norm: str | None = None,
    overwrite_x=False,
    return_axis_arrays=True,
    out=None,
) -> tuple[ArrayType, ArrayType, ArrayType]:
    """Implements a stripped-down subset of scipy.fft.stft in order to avoid
    some overhead that comes with its generality and allow use of the generic
    python array-api for interchangable numpy/cupy support.

    For additional information, see help for scipy.fft.

    Args:
        x: input array

        fs: sampling rate

        window: a window array, or a name or (name, parameter) pair as in `scipy.signal.get_window`

        nperseg: the size of the FFT (= segment size used if overlapping)

        noverlap: if nonzero, compute windowed FFTs that overlap by this many bins (only 0 and nperseg//2 supported)

        axis: the axis on which to compute the STFT

        truncate: whether to allow truncation of `x` to enforce full fft block sizes

    Raises:
        NotImplementedError: if axis != 0

        ValueError: if truncate == False and x.shape[axis] % nperseg != 0

    Returns:
        stft (see scipy.fft.stft)

    """

    xp = array_namespace(x)

    # # For reference: this is probably the same
    # freqs, times, X = signal.spectral._spectral_helper(
    #     x,
    #     x,
    #     fs,
    #     window,
    #     nperseg,
    #     noverlap,
    #     nperseg,
    #     scaling="spectrum",
    #     axis=axis,
    #     mode="stft",
    #     padded=True,
    # )

    nfft = nperseg

    if norm not in ('power', None):
        raise TypeError('norm must be "power" or None')

    if window is None:
        window = 'rect'

    if isinstance(window, str) or (
        isinstance(window, tuple) and isinstance(window[0], str)
    ):
        should_norm = norm == 'power'
        w = get_window(
            window,
            nfft - nzero,
            nzero=nzero,
            xp=xp,
            dtype=x.dtype,
            norm=should_norm,
            fftshift=True,
        )
    else:
        w = window * get_window(
            'rect', nfft - nzero, nzero=nzero, xp=xp, dtype=x.dtype, fftshift=True
        )

    if noverlap == 0:
        # special case for speed
        xstack = to_blocks(x, nfft, axis=axis, truncate=truncate)
        wstack = broadcast_onto(w / nfft, xstack, axis=axis + 1)

        if out is None and overwrite_x:
            out = xstack

        xstack = xp.multiply(
            xstack,
            wstack.astype(xstack.dtype),
            out=xstack if overwrite_x else out,
        )

    else:
        xstack = _stack_stft_windows(
            x,
            window=w / nfft,
            nperseg=nperseg,
            noverlap=noverlap,
            axis=axis,
            norm=norm,
            out=out,
        )
    assert xstack.dtype == x.dtype
    del x

    # no fftshift needed since it was baked into the window
    y = fft(xstack, axis=axis + 1, overwrite_x=True, out=xstack)

    if not return_axis_arrays:
        return y

    freqs, times = _get_stft_axes(
        fs,
        nfft=nfft,
        time_size=y.shape[axis],
        overlap_frac=noverlap / nfft,
        xp=np,
    )

    return freqs, times, y


def istft(
    y: ArrayType,
    size=None,
    *,
    nfft: int,
    noverlap: int,
    out=None,
    overwrite_x=False,
    axis=0,
) -> ArrayType:
    """reconstruct and return a waveform given its STFT and associated parameters"""

    xp = array_namespace(y)

    # give the stacked NFFT-sized time domain vectors in axis + 1
    xstack = ifft(
        y,
        axis=axis + 1,
        overwrite_x=overwrite_x,
        out=y if overwrite_x else None,
    )

    # correct the fft shift in the time domain, since the
    # multiply operation can be applied in-place
    w = get_window('rect', nfft, xp=xp, dtype=y.dtype, fftshift=True)
    wstack = broadcast_onto(w, xstack, axis=axis + 1)
    xstack = xp.multiply(
        xstack,
        wstack,
        out=xstack,
        dtype=xstack.dtype,
    )
    assert xstack.dtype == y.dtype

    x = _unstack_stft_windows(
        xstack, noverlap=noverlap, nperseg=nfft, axis=axis, out=out
    )
    assert x.dtype == y.dtype

    if size is not None:
        trim = x.shape[axis] - size
        if trim > 0:
            x = axis_slice(x, start=trim // 2, stop=-trim // 2, axis=axis)

    return x


def ola_filter(
    x: ArrayType,
    *,
    fs: float,
    nfft: int,
    window: str | tuple = 'hamming',
    passband: tuple[float, float],
    nfft_out: int | None = None,
    frequency_shift=False,
    axis=0,
    extend=False,
    out=None,
    overwrite_x=False,
):
    """apply a bandpass filter implemented through STFT overlap-and-add.

    Args:
        x: the input waveform
        fs: the sample rate of the input waveform, in Hz
        noverlap: the size of overlap between adjacent FFT windows, in samples
        window: the type of COLA window to apply, 'hamming', 'blackman', or 'blackmanharris'
        passband: a tuple of low-pass cutoff and high-pass cutoff frequency (or None to skip either)
        nfft_out: implement downsampling by adjusting the size of overlap between adjacent FFT windows
        frequency_shift: the direction to shift the downsampled frequencies ('left' or 'right', or False to center)
        axis: the axis of `x` along which to compute the filter
        extend: if True, allow use of zero-padded samples at the edges to accommodate a non-integer number of overlapping windows in x
        out: None, 'shared', or an array object to receive the output data

    Returns:
        an Array of the same shape as X
    """

    nfft_out, noverlap, overlap_scale, _ = _ola_filter_parameters(
        x.size,
        window=window,
        nfft_out=nfft_out or nfft,
        nfft=nfft,
        extend=extend,
    )

    enbw = equivalent_noise_bandwidth(window, nfft_out, fftbins=False)

    freqs, _, y = stft(
        x,
        fs=fs,
        window=window,
        nperseg=nfft,
        noverlap=round(nfft * overlap_scale),
        axis=axis,
        truncate=False,
        overwrite_x=overwrite_x,
    )

    zero_stft_by_freq(
        freqs, y, passband=(passband[0] + enbw, passband[1] - enbw), axis=axis
    )

    if nfft_out != nfft or frequency_shift:
        freqs, y = downsample_stft(
            freqs,
            y,
            nfft_out=nfft_out or nfft,
            passband=passband,
            axis=axis,
            out=y,
        )

    return istft(
        y,
        round(x.shape[axis] * nfft_out / nfft),
        nfft=nfft_out or nfft,
        noverlap=noverlap,
        overwrite_x=True,
        axis=axis,
    )


@lru_cache()
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
        ihi = xp.where(freqs <= cutoff_hi)[0][-1]

    return ilo, ihi


def spectrogram(
    x: ArrayType,
    *,
    fs: float,
    window: ArrayType | str | tuple[str, float],
    nperseg: int = 256,
    noverlap: int = 0,
    nzero: int = 0,
    axis: int = 0,
    truncate: bool = True,
    return_axis_arrays: bool = True,
):
    """evaluate the power spectrogram of x with the given arguments.

    The output is scaled such that the noise bandwidth is equal to the
    frequency resolution.
    """
    kws = dict(locals())

    ret = stft(norm='power', **kws)
    if return_axis_arrays:
        freqs, times, X = ret
        return freqs, times, power_analysis.envtopow(X)
    else:
        return power_analysis.envtopow(ret)


def power_spectral_density(
    x: ArrayType,
    *,
    fs: float,
    bandwidth=INF,
    window,
    resolution: float,
    fractional_overlap=0,
    fractional_window: float = 1,
    statistics: list[float],
    truncate=True,
    dB=True,
    axis=0,
) -> ArrayType:
    if isroundmod(fs, resolution):
        nfft = round(fs / resolution)
        noverlap = round(fractional_overlap * nfft)
    else:
        # need sample_rate_Hz/resolution to give us a counting number
        raise ValueError('sample_rate_Hz/resolution must be a counting number')

    if isroundmod((1 - fractional_window) * nfft, 1):
        nzero = round((1 - fractional_window) * nfft)
    else:
        raise ValueError(
            '(1-fractional_window) * (sample_rate/frequency_resolution) must be a counting number'
        )

    xp = array_namespace(x)
    domain = get_input_domain()

    if domain == Domain.TIME:
        freqs, _, X = spectrogram(
            x,
            window=window,
            fs=fs,
            nperseg=nfft,
            nzero=nzero,
            noverlap=noverlap,
            axis=axis,
        )
    elif domain == Domain.FREQUENCY:
        X = x
        freqs, _ = _get_stft_axes(
            fs=fs,
            nfft=nfft,
            time_size=X.shape[axis],
            overlap_frac=noverlap / nfft,
            xp=np,
        )
    else:
        raise ValueError('unsupported persistence spectrum domain "{domain}')

    if truncate:
        if bandwidth == INF:
            bw_args = (None, None)
        else:
            bw_args = (-bandwidth / 2, +bandwidth / 2)
        ilo, ihi = _freq_band_edges(freqs.size, 1.0 / fs, *bw_args)
        X = axis_slice(X, ilo, ihi, axis=axis + 1)

    if domain == Domain.TIME:
        if dB:
            spg = power_analysis.powtodB(X, eps=1e-25, out=X).real
        else:
            spg = X.astype('float32')
    elif domain == Domain.FREQUENCY:
        if dB:
            # here X is complex-valued; use the first-half of its buffer
            spg = power_analysis.envtodB(X, eps=1e-25, out=X).real
        else:
            spg = power_analysis.envtopow(X, out=X.real)
    else:
        raise ValueError(f'unhandled dB and domain: {dB}, {domain}')

    isquantile = find_float_inds(tuple(statistics))

    shape = list(spg.shape)
    shape[axis] = len(statistics)
    out = xp.empty(tuple(shape), dtype='float32')

    quantiles = list(np.asarray(statistics)[isquantile].astype('float32'))

    out_quantiles = axis_index(out, isquantile, axis=axis).swapaxes(0, 1)
    out_quantiles[:] = xp.quantile(spg, xp.array(quantiles), axis=axis)

    for i, isquantile in enumerate(isquantile):
        if not isquantile:
            ufunc = power_analysis.stat_ufunc_from_shorthand(statistics[i], xp=xp)
            axis_index(out, i, axis=axis)[...] = ufunc(spg, axis=axis)

    return out


def channelize_power(
    iq: ArrayType,
    Ts: float,
    fft_size_per_channel: int,
    *,
    analysis_bins_per_channel: int,
    window: ArrayType,
    fft_overlap_per_channel=0,
    channel_count: int = 1,
    axis=0,
):
    """Channelizes the input waveform and returns a time series of power in each channel.

    The calculation is performed by transformation into the frequency domain. Power is
    summed across the bins in the analysis bandwidth, ignoring those in bins outside
    of the analysis bandwidth.

    The total analysis bandwidth (i.e., covering all channels) is equal to
    `(analysis_bins_per_channel/fft_size_per_channel)/Ts`,
    occupying the center of the total sampling bandwidth. The bandwidth in each power bin is equal to
    `(analysis_bins_per_channel/fft_size_per_channel)/Ts/channel_count`.

    The time spacing of the power samples is equal to `Ts * fft_size_per_channel * channel_count`
    if `fft_overlap_per_channel` is 0, otherwise, `Ts * fft_size_per_channel * channel_count / 2`.

    Args:
        iq: an input waveform or set of input waveforms, with time along axis 0

        Ts: the sampling period (1/sampling_rate)

        fft_size_per_channel: the size of the fft to use in each channel; total fft size is (channel_count * fft_size_per_channel)

        channel_count: the number of channels to analyze

        fft_overlap_per_channel: equal to 0 to disable overlapping windows, or to disable overlap, or fft_size_per_channel // 2)

        analysis_bins_per_channel: the number of bins to keep in each channel

        window: typing.Callable window function to use in the analysis

        axis: the axis along which to perform the FFT (for now, require axis=0)

    Raises:
        NotImplementedError: if axis != 0

        NotImplementedError: if fft_overlap_per_channel is not one of (0, fft_size_per_channel//2)

        ValueError: if analysis_bins_per_channel > fft_size_per_channel

        ValueError: if channel_count * (fft_size_per_channel - analysis_bins_per_channel) is not even
    """
    if axis != 0:
        raise NotImplementedError('sorry, only axis=0 implemented for now')

    if analysis_bins_per_channel > fft_size_per_channel:
        raise ValueError('the number of analysis bins cannot be greater than FFT size')

    freqs, times, X = stft(
        iq,
        fs=1.0 / Ts,
        window=window,
        nperseg=fft_size_per_channel * channel_count,
        noverlap=fft_overlap_per_channel * channel_count,
        norm='power',
        axis=axis,
    )

    # extract only the bins inside the analysis bandwidth
    skip_bins = channel_count * (fft_size_per_channel - analysis_bins_per_channel)
    if skip_bins % 2 == 1:
        raise ValueError('must pass an even number of bins to skip')
    X = X[:, skip_bins // 2 : -skip_bins // 2]
    freqs = freqs[skip_bins // 2 : -skip_bins // 2]

    if channel_count == 1:
        channel_power = power_analysis.envtopow(X).sum(axis=axis + 1)

        return times, channel_power

    else:
        freqs = to_blocks(freqs, analysis_bins_per_channel)
        X = to_blocks(X, analysis_bins_per_channel, axis=axis + 1)

        channel_power = power_analysis.envtopow(X).sum(axis=axis + 2)

        return freqs[0], times, channel_power


def upfirdn(h, x, up=1, down=1, axis=-1, mode='constant', cval=0, overwrite_x=False):
    kws = dict(locals())
    del kws['overwrite_x']

    xp = array_namespace(x)

    if is_cupy_array(x):
        from . import cuda

        del kws['x'], kws['axis']
        if overwrite_x and 2 * up > down > up:
            # skip new allocation if possible and not "too" wasteful
            kws['out'] = x

        func = lambda array: cuda.upfirdn(x=array, **kws)  # noqa: E731
        y = xp.apply_along_axis(func, axis, x)
    else:
        y = signal.upfirdn(**kws)

    return y


def oaconvolve(x1, x2, mode='full', axes=-1):
    """convolve x1 and x2, and return the result.

    See also:
        `scipy.signal.oaconvolve`
    """
    if is_cupy_array(x1):
        from cupyx.scipy.signal import oaconvolve as func  # pyright: ignore[reportMissingImports]
    else:
        from scipy.signal import oaconvolve as func

    return func(x1, x2, mode=mode, axes=axes)


def time_fftshift(x, scale: ArrayType | float | None = None, overwrite_x=False, axis=0):
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

    xview = to_blocks(x, 2, axis=axis)
    outview = to_blocks(out, 2, axis=axis)
    scale = broadcast_onto(xp.atleast_1d(scale), outview, axis=max(axis - 1, 0))
    shift = broadcast_onto(xp.array([1, -1]), outview, axis=axis + 1)
    xp.multiply(xview, scale * shift, out=outview)
    return out


time_ifftshift = time_fftshift


def resample(
    x,
    num,
    axis: int = 0,
    window: str | tuple[str, float] | None = None,
    domain: typing.Literal['time'] | typing.Literal['frequency'] = 'time',
    overwrite_x=False,
    scale: ArrayType | float = 1,
    shift: float = 0,
):
    """limited reimplementation of scipy.signal.resample optimized for reduced memory.

    No new buffers are allocated when downsampling if `overwrite_x` is `False`.

    The window argument is not supported.
    """
    if domain not in ('time', 'freq'):
        raise ValueError(
            "Acceptable domain flags are 'time' or 'freq', not domain={}".format(domain)
        )

    if x.shape[axis] == num:
        return x

    xp = array_namespace(x)

    x = xp.asarray(x)
    nfft_in = x.shape[axis]
    nfft_out = num
    newshape = list(x.shape)
    newshape[axis] = nfft_out

    if nfft_in % 2 != 0:
        raise ValueError('x.shape[axis] must be even')

    if window is not None:
        raise ValueError('window argument is not supported')

    if shift == 0:
        # no frequency shift
        edge_low = edge_high = None
    elif nfft_out > nfft_in:
        raise ValueError('shift is only supported when downsampling')
    else:
        edge_low = nfft_in // 2 - nfft_out // 2 + shift
        edge_high = edge_low + nfft_out

        if edge_low < 0:
            raise ValueError('shift is too small')
        if edge_high > nfft_in:
            raise ValueError('shift is too large')

    resample_scale = float(nfft_out) / float(nfft_in) * scale

    if domain == 'time':
        # apply fftshift in the time domain, where we can avoid a copy.
        # the fftshift is needed to enable clean slice-driven downsampling
        x = time_fftshift(x, resample_scale, overwrite_x=overwrite_x, axis=axis)
        y = fft(x, axis=axis, overwrite_x=True, out=x)
    else:  # domain == 'freq'
        if overwrite_x:
            out = x
        else:
            out = None
        y = xp.multiply(x, resample_scale, out=out)

    del x

    if nfft_out < nfft_in:
        # downsample by trimming frequency
        bounds = _find_downsample_copy_range(nfft_in, nfft_out, edge_low, edge_high)[1]
        y = axis_slice(y, *bounds, axis=axis)

    elif nfft_out > nfft_in:
        # upsample by zero-padding frequency.
        # this requires a copy, since not all platforms support the
        # fft `out` argument is non-standard
        pad_left = (nfft_out - nfft_in) // 2
        pad_right = pad_left + (nfft_out - nfft_in) % 2
        y = pad_along_axis(y, [[pad_left, pad_right]], axis=axis)

    # Inverse transform
    xout = ifft(y, axis=axis, overwrite_x=True, out=y)

    return time_ifftshift(xout, overwrite_x=True, axis=axis)


def oaresample(
    x: ArrayType,
    up,
    down,
    fs,
    # analysis_filter: dict,
    *,
    window: str | tuple[str, float] = 'hamming',
    overwrite_x: bool = False,
    axis: int = 1,
    frequency_shift: float = 0,
    filter_bandwidth: float | None = None,
    transition_bandwidth: float = 250e3,
    scale: ArrayType | float = 1.0,
):
    """apply resampler implemented through STFT overlap-and-add.

    Args:
        iq: the input waveform, as a pinned array
        capture: the capture filter specification structure
        radio: the radio instance that performed the capture
        force_calibration: if specified, this calibration dataset is used rather than loading from file
        axis: the axis of `x` along which to compute the filter

    Returns:
        the filtered IQ capture
    """

    nfft = down
    nfft_out = up
    size_in = x.size

    nfft_out, noverlap, overlap_scale, _ = _ola_filter_parameters(
        x.size,
        window=window,
        nfft_out=nfft_out,
        nfft=nfft,
        extend=True,
    )

    if frequency_shift == 0:
        # no frequency shift
        edge_low = edge_high = None
    elif down < up:
        raise ValueError('frequency_shift is only supported when downsampling')
    elif isroundmod(frequency_shift, fs / nfft):
        shift = round(frequency_shift / (fs / nfft))
        edge_low = nfft // 2 - nfft_out // 2 + shift
        edge_high = edge_low + nfft_out

        if edge_low < 0:
            raise ValueError('frequency_shift is too small')
        if edge_high > nfft:
            raise ValueError('frequency_shift is too large')
    else:
        raise ValueError('frequency_shift must be a multiple of fs/up')

    y = stft(
        x,
        fs=fs,
        window=window,
        nperseg=nfft,
        noverlap=round(nfft * overlap_scale),
        axis=axis,
        truncate=False,
        overwrite_x=overwrite_x,
        return_axis_arrays=False,
    )

    if nfft_out < nfft:
        # downsample
        bounds = _find_downsample_copy_range(nfft, nfft_out, edge_low, edge_high)[1]
        y = axis_slice(y, *bounds, axis=axis + 1)

    elif nfft_out > nfft:
        # upsample
        pad_left = (nfft_out - nfft) // 2
        pad_right = pad_left + (nfft_out - nfft) % 2

        y = pad_along_axis(y, [[pad_left, pad_right]], axis=axis + 1)

    del x

    if filter_bandwidth is not None and np.isfinite(filter_bandwidth):
        y = stft_fir_lowpass(
            y,
            sample_rate=fs * up / down,
            bandwidth=filter_bandwidth,
            transition_bandwidth=transition_bandwidth,
            axis=axis,
            out=y,
        )

    # reconstruct into a resampled waveform
    x = istft(y, nfft=nfft_out, noverlap=noverlap, axis=axis, overwrite_x=True)

    x *= x.size / size_in * scale

    return x
