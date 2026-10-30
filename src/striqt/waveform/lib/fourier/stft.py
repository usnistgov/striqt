"""The short-time Fourier transform, the inverse transform, and edits of an STFT in
the frequency domain (band zeroing and rational downsampling).

`stft(x, axis=a)` stacks windows on axis `a` and frequency bins, in ascending `fftfreq`
order, on `a + 1`. An `out=` buffer is viewed when contiguous and copied otherwise.
"""

from __future__ import annotations as __

import typing

from .. import util
from ..arrays import array_namespace, axis_slice, axis_to_blocks, sliding_window_view
from ..util import np
from .fft import _broadcast_onto, _freq_band_edges, fft, fftfreq, ifft
from .windows import get_window

if typing.TYPE_CHECKING:
    from ..typing import _AT, Array, XpType


def _truncated_buffer(x: Array, shape, dtype=None):
    if dtype is not None:
        x = x.view(dtype)
    out_size = np.prod(shape)
    assert x.size >= out_size
    return x.ravel()[:out_size].reshape(shape)


def _same_base_memory(a: Array, b: Array) -> bool:
    if b is None:
        return False
    return bool(a is b or getattr(b, 'base', None) is a)


# %% STFT
@util.lru_cache(16)
def _get_stft_axes(
    fs: float, nfft: int, time_size: int, overlap_frac: float = 0, *, xp=None
) -> tuple[Array, Array]:
    """returns stft (freqs, times) array tuple appropriate to the array module xp"""
    if xp is None:
        xp = np

    freqs = fftfreq(nfft, fs, xp=xp)
    times = xp.arange(time_size) * ((1 - overlap_frac) * nfft / fs)

    return freqs, times


def stft(
    x: _AT,
    *,
    fs: float,
    window: Array | str | tuple[str, float],
    nperseg: int = 256,
    noverlap: int = 0,
    nzero: int = 0,
    axis: int = 0,
    truncate: bool = True,
    norm: str | None = None,
    overwrite_x=False,
    return_axis_arrays=True,
    out=None,
) -> tuple[_AT, _AT, _AT]:
    """return the short-time Fourier transform of `x` along `axis`.

    The transform follows `scipy.signal.stft` with a reduced parameter set.

    Args:
        x: the waveform, with time on `axis`
        fs: the sample rate (in S/s) of `x`, which scales the returned axes
        window: a window array of ``nperseg - nzero`` samples, or a name or
            ``(name, parameter)`` pair for `get_window`; None selects ``'rect'``
        nperseg: the segment length and FFT size (in samples)
        noverlap: the overlap (in samples) of consecutive segments
        nzero: the number of trailing samples of each segment that are zeroed; the
            window spans the first ``nperseg - nzero``
        axis: the time axis of `x`
        truncate: True drops trailing samples of `x` that do not fill a segment;
            False raises `ValueError` for a partial segment
        norm: None, or ``'power'`` to scale the window to unit time-averaged power
        overwrite_x: True writes the result into the buffer of `x` when `noverlap` is
            0 and `out` is None; ignored otherwise
        return_axis_arrays: True returns the frequency and time axes with the
            transform; False returns the transform alone
        out: a buffer with the shape and dtype of the transform

    Returns:
        ``(freqs, times, y)``: `freqs` is the ascending bin center frequencies (in
        Hz), `times` the segment start times (in s), and `y` the transform with
        segments on `axis` and `nperseg` bins on ``axis + 1``, in the array
        namespace of `x`; `y` alone when `return_axis_arrays` is False

    Raises:
        TypeError: if `norm` is neither None nor ``'power'``
        ValueError: if `truncate` is False and ``x.shape[axis]`` is not a whole
            number of segments at `nperseg` and `noverlap`
    """

    xp = array_namespace(x)

    nfft = nperseg
    dtype = typing.cast(str, x.dtype)

    if norm is not None and norm != 'power':
        raise TypeError('norm must be "power" or None')

    if window is None:
        window = 'rect'

    if isinstance(window, (str, tuple)):
        should_norm = norm == 'power'
        w = get_window(
            window,
            nfft - nzero,
            nzero=nzero,
            xp=xp,
            dtype=dtype,
            norm=should_norm,
            fftshift=True,
        )
    else:
        w = window * get_window(
            'rect', nfft - nzero, nzero=nzero, xp=xp, dtype=dtype, fftshift=True
        )

    if noverlap == 0:
        # special case for speed
        xstack = axis_to_blocks(x, nfft, axis=axis, truncate=truncate)
        wstack = _broadcast_onto(w / nfft, xstack, axis=axis + 1)

        if out is None and overwrite_x:
            out = xstack

        xstack = xp.multiply(
            xstack,
            wstack.astype(xstack.dtype),
            out=xstack if overwrite_x else out,
        )

    else:
        hop = nperseg - noverlap
        if not truncate and (x.shape[axis] - nperseg) % hop != 0:
            raise ValueError(
                f'axis {axis} size {x.shape[axis]} is not a whole number of segments '
                f'with nperseg={nperseg} and noverlap={noverlap}'
            )
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
        xp=xp,
    )

    return (freqs, times, y)  # type: ignore


def istft(
    y: Array,
    size=None,
    *,
    nfft: int,
    noverlap: int,
    out=None,
    overwrite_x=False,
    axis=0,
    fftshift: bool = True,
) -> Array:
    """reconstruct the waveform from the STFT `y` by inverse FFT and overlap-add.

    `y` has windows on `axis` and `nfft` frequency bins on ``axis + 1``. The windows
    are overlap-added at a hop of ``nfft - noverlap`` samples.

    Args:
        y: the STFT, with at least 2 dimensions
        size: the number of output samples; a longer reconstruction is trimmed about
            the center, and None keeps every sample
        nfft: the FFT size of the STFT, which is the window length in samples
        noverlap: the overlap (in samples) of consecutive windows in the STFT
        out: a buffer of at least the untrimmed output size in the dtype of `y`; a
            non-contiguous buffer is copied, and the result is then not a view of
            `out`
        overwrite_x: if True, `y` is overwritten by the inverse FFT of each window
        axis: the window axis of `y`
        fftshift: True for bins in the ascending-frequency order that `stft` returns,
            which multiplies each inverse transform by the half-band shift of the
            analysis window; False for bins in `fftfreq` order, which skips that
            multiply

    Returns:
        the waveform on `axis`, in the dtype and array namespace of `y`, of
        ``y.shape[axis] * (nfft - noverlap) + noverlap`` samples before trimming to
        `size`
    """

    xp = array_namespace(y)
    dtype = typing.cast(str, y.dtype)

    # give the stacked NFFT-sized time domain vectors in axis + 1
    xstack = ifft(
        y,
        axis=axis + 1,
        overwrite_x=overwrite_x,
        out=y if overwrite_x else None,
    )

    if fftshift:
        # correct the fft shift in the time domain, since the
        # multiply operation can be applied in-place
        w = get_window('rect', nfft, xp=xp, dtype=dtype, fftshift=True)
        wstack = _broadcast_onto(w, xstack, axis=axis + 1)
        xstack = xp.multiply(
            xstack,
            wstack,
            out=xstack,
            dtype=xstack.dtype,
        )
        assert xstack.dtype == dtype

    x = _unstack_stft_windows(
        xstack, noverlap=noverlap, nperseg=nfft, axis=axis, out=out
    )
    assert x.dtype == dtype

    if size is not None:
        trim = x.shape[axis] - size
        if trim > 0:
            x = axis_slice(x, start=trim // 2, stop=-trim // 2, axis=axis)

    return x


def _stack_stft_windows(
    x: Array,
    window: Array,
    nperseg: int,
    noverlap: int,
    norm: typing.Literal['power'] | None = None,
    axis: int = 0,
    out=None,
) -> Array:
    """stack the windowed segments of `x` on `axis`, with the samples of each on ``axis + 1``.

    Args:
        x: the waveform, with time on `axis`
        window: the window of `nperseg` samples applied to each segment
        nperseg: the segment length (in samples)
        noverlap: the overlap (in samples) of consecutive segments
        norm: None divides the window by the sum of its magnitude at the hop
            stride; ``'power'`` applies the window unscaled
        axis: the time axis of `x`
        out: a buffer with the shape and dtype of the result
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

    w = _broadcast_onto(window / scale, xstacked, axis=axis + 1)
    if out is None:
        # a fresh C-contiguous buffer: numpy and cupy would otherwise lay the result
        # out like the strided view of `x`, and a Fortran-ordered `x` then leaves the
        # FFT axis strided for every later stage
        out = xp.empty(xstacked.shape, dtype=xstacked.dtype)
    return xp.multiply(xstacked, w.astype(xstacked.dtype), out=out)


def _unstack_stft_windows(
    y: Array, noverlap: int, nperseg: int, axis: int = 0, out: Array | None = None
) -> Array:
    """overlap-add the time-domain windows stacked on `axis` of `y` into one waveform.

    Args:
        y: time-domain windows on `axis`, each of `nperseg` samples on ``axis + 1``
        noverlap: the overlap (in samples) of consecutive windows
        nperseg: the window length (in samples)
        axis: the window axis of `y`
        out: a buffer of at least the output size in the dtype of `y`; a
            non-contiguous buffer is copied, and the result is then not written into
            `out`

    Returns:
        the waveform on `axis`, of ``y.shape[axis] * y.shape[axis + 1] * hop // nperseg
        + noverlap`` samples with ``hop = nperseg - noverlap``
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

    # sum in groups of non-overlapping windows. The strided group of windows is
    # added into the output span viewed as (windows, nfft): splitting the contiguous
    # time axis is a view, where merging the strided window axis would copy
    group_count = nfft // hop_size
    for offs in range(group_count):
        windows = axis_slice(y, start=offs, step=group_count, axis=axis)
        start = offs * hop_size
        span = axis_slice(
            xr, start=start, stop=start + windows.shape[axis] * nfft, axis=axis
        )
        span = span.reshape(windows.shape)

        if offs == 0:
            xp.copyto(span, windows)
        else:
            span += windows

    return xr


# %% Frequency-domain edits
def zero_stft_by_freq(
    freqs: Array, xstft: Array, *, passband: tuple[float, float], axis=0
) -> Array:
    """apply a bandpass filter in the STFT domain by zeroing frequency indices"""
    xp = array_namespace(xstft)

    freq_step = float(freqs[1] - freqs[0])
    fs = xstft.shape[axis + 1] * freq_step
    ilo, ihi = _freq_band_edges(freqs.size, fs, *passband)

    xp.copyto(axis_slice(xstft, 0, ilo, axis=axis + 1), 0)
    xp.copyto(axis_slice(xstft, ihi, None, axis=axis + 1), 0)
    return xstft


def downsample_stft(
    freqs: Array,
    y: Array,
    nfft_out: int,
    *,
    passband: tuple[float | None, float | None] = (None, None),
    axis=0,
    out=None,
) -> tuple[Array, Array]:
    """downsample an STFT in the frequency domain by trimming to `nfft_out` bins.

    The downsampling ratio is ``nfft_out / y.shape[axis + 1]``. The kept bins are
    centered on the passband, with the passband center at the output DC bin.

    Args:
        freqs: the ascending bin center frequencies (in Hz) of `y`
        y: the STFT, with segments on `axis` and bins on ``axis + 1``
        nfft_out: the number of bins kept
        passband: the band edges (in Hz) of the bins that are kept nonzero; None at
            either edge extends the band to that end of `freqs`
        axis: the segment axis of `y`
        out: a buffer of at least the output size in the dtype of `y`

    Returns:
        ``(freqs_out, y_out)``: the `nfft_out` bin center frequencies at the bin
        spacing of `freqs`, and the STFT with `nfft_out` bins on ``axis + 1`` and
        zeros outside the passband; `y_out` is a view of `y` when no bin is zeroed
        and `out` shares memory with `y`
    """
    xp = array_namespace(y)
    ax = axis + 1

    shape_out = list(y.shape)
    shape_out[ax] = nfft_out

    # passband indexes in the input
    freq_step = float(freqs[1] - freqs[0])
    fs = y.shape[ax] * freq_step
    passband_start, passband_end = _freq_band_edges(y.shape[ax], fs, *passband)
    bounds_out, bounds_in, _ = _find_downsample_copy_range(
        y.shape[ax], nfft_out, passband_start, passband_end
    )
    freqs_out = _find_downsampled_freqs(nfft_out, freq_step, xp=xp)

    if tuple(bounds_out) == (0, shape_out[ax]) and _same_base_memory(y, out):
        # fast path: a view when no bins are zeroed and `out` shares memory with `y`
        return freqs_out, axis_slice(y, *bounds_in, axis=ax)

    if out is None:
        xout = xp.empty(shape_out, dtype=y.dtype)
    else:
        xout = _truncated_buffer(out, shape_out, y.dtype)

    # copy first before zeroing, in case of input-output buffer reuse
    xp.copyto(
        axis_slice(xout, *bounds_out, axis=ax),
        axis_slice(y, *bounds_in, axis=ax),
    )

    xp.copyto(axis_slice(xout, 0, bounds_out[0], axis=ax), 0)
    xp.copyto(axis_slice(xout, bounds_out[1], None, axis=ax), 0)

    return freqs_out, xout


@util.lru_cache(100)
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

    # copy output indexes, anchored so that the passband center lands on the
    # output DC bin. Splitting the zeroed bins evenly would put it one bin low when
    # their count is odd, which inverts alternate frames at 50% overlap
    copy_out_start = nfft_out // 2 - (passband_center - copy_in_start)
    copy_out_end = copy_out_start + copy_size

    assert copy_out_end - copy_out_start == copy_size
    assert copy_out_start >= 0
    assert copy_out_end <= nfft_out

    return (copy_out_start, copy_out_end), (copy_in_start, copy_in_end), passband_center


@util.lru_cache(16)
def _find_downsampled_freqs(
    nfft_out: int, freq_step: float, xp: XpType = None
) -> Array:
    return fftfreq(nfft_out, freq_step * nfft_out, xp=xp or np)
