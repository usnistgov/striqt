"""FIR low-pass designs and the STFT overlap-add filters that apply a design.

The overlap of each window is set by one constant-overlap-add (COLA) divisor table.
"""

from __future__ import annotations as __

import typing

from .. import util
from ..arrays import (
    array_namespace,
    axis_slice,
    convert_np_to_xp,
    is_cupy_array,
    pad_along_axis,
)
from ..util import np
from .fft import _broadcast_onto
from .stft import downsample_stft, istft, stft, zero_stft_by_freq
from .windows import equivalent_noise_bandwidth, get_window

if typing.TYPE_CHECKING:
    from ..typing import _AT, Array, DTypeLike, WindowType, XpType


_COLA_WINDOW_SIZE_DIVISOR = {
    None: 1,
    'rect': 1,
    'hann': 2,
    'blackman': 3,
    'blackmanharris': 5,
}


@util.lru_cache()
def design_oafilter(
    array_size: int, *, window, nfft_out: int, nfft: int, extend: bool
) -> tuple[int, int, float, int]:
    """return the block sizes, overlap, and zero pad of an STFT overlap-add filter.

    Args:
        array_size: the number of input samples along the filter axis
        window: the COLA window, ``'hann'``, ``'blackman'``, ``'blackmanharris'``,
            ``'rect'``, or None for ``'rect'``
        nfft_out: the output block size (in samples), an even multiple of the COLA
            divisor of `window`; None selects `nfft`
        nfft: the input block size (in samples), an even multiple of the COLA
            divisor of `window`
        extend: True pads a partial trailing hop with zeros; False rejects it

    Returns:
        ``(nfft_out, noverlap, overlap_scale, pad)``: the output block size, the
        overlap (in samples) of consecutive output blocks, the overlap as a
        fraction of the block, and the number of zeros that bring `array_size` to a
        whole number of input hops (0 without `extend`)

    Raises:
        TypeError: if `window` is not a key of the COLA divisor table
        ValueError: if `nfft` or `nfft_out` is odd or not a multiple of the COLA
            divisor, or if `array_size` is not a whole number of input hops without
            `extend`
    """
    if nfft_out is None:
        nfft_out = nfft

    try:
        divisor = _COLA_WINDOW_SIZE_DIVISOR[window]
    except KeyError:
        raise TypeError(
            f'window must be one of {tuple(_COLA_WINDOW_SIZE_DIVISOR)}, not {window!r}'
        )

    for name, size in (('nfft', nfft), ('nfft_out', nfft_out)):
        if size % divisor != 0:
            raise ValueError(
                f'{window!r} window COLA requires {name} % {divisor} == 0, not {size}'
            )
        if size % 2 != 0:
            # the half-band shift of `stft` and `istft` is self-inverse only on an
            # even block
            raise ValueError(f'{name} must be even, not {size}')

    if window is None or window == 'rect':
        overlap_scale = 1
    elif window == 'hann':
        overlap_scale = 1 / 2
    elif window == 'blackman':
        overlap_scale = 2 / 3
    elif window == 'blackmanharris':
        overlap_scale = 4 / 5
    else:
        raise ValueError(
            f'window must be one of {tuple(_COLA_WINDOW_SIZE_DIVISOR)}, not {window!r}'
        )

    noverlap = round(nfft_out * overlap_scale)
    hop = nfft - round(nfft * overlap_scale)

    pad = (-array_size) % hop
    if pad and not extend:
        raise ValueError(
            f'x.shape[axis] ({array_size}) is not an integer multiple of the hop ({hop})'
        )

    return nfft_out, noverlap, overlap_scale, pad


@convert_np_to_xp
@util.lru_cache()
@util.persistent_cache()
def design_fir_lpf(
    bw: float,
    fs: float,
    *,
    numtaps: int = 4001,
    transition_bw: float = 250e3,
    dtype: DTypeLike = 'float32',
    xp: XpType = None,
) -> Array:
    """return the taps of a least-squares linear-phase FIR low-pass of width `bw`.

    The gain falls linearly from 1 at ``bw/2 - transition_bw/2`` to 0 at
    ``bw/2 + transition_bw/2``, so the gain at ``bw/2`` is 1/2. The transition band
    must lie strictly inside ``(0, fs/2)``.

    Args:
        bw: the passband width (in Hz), centered on 0 Hz
        fs: the sample rate (in S/s)
        numtaps: the number of taps, odd for a type I design
        transition_bw: the width (in Hz) of the transition band
        dtype: the dtype of the taps
        xp: the array namespace of the result; None selects numpy

    Returns:
        the `numtaps` taps as a 1-D array of `dtype` in `xp`

    Raises:
        ValueError: if the transition band reaches 0 Hz or ``fs/2``
    """
    from scipy import signal

    lower = bw / 2 - transition_bw / 2
    upper = bw / 2 + transition_bw / 2
    if lower <= 0:
        raise ValueError(
            f'the {transition_bw} Hz transition band leaves no passband below it for '
            f'bw={bw} Hz'
        )
    if upper >= fs / 2:
        raise ValueError(
            f'the {transition_bw} Hz transition band reaches fs/2 for bw={bw} Hz at '
            f'fs={fs} S/s'
        )

    edges = [0, lower, upper, fs / 2]
    bands = list(zip(edges[:-1], edges[1:]))
    desired = [1, 1, 1, 0, 0, 0]

    b = signal.firls(numtaps, bands=bands, desired=desired, fs=fs)

    return b.astype(dtype)


@util.lru_cache()
def _fir_lowpass_fft(
    size: int,
    sample_rate: float,
    *,
    cutoff: float,
    transition: float,
    window: WindowType = 'hamming',
    xp: XpType = None,
    dtype: DTypeLike = 'complex64',
) -> Array:
    """return the zero-phase response of an FIR low-pass on the `size` bins of an STFT.

    The taps are a `scipy.signal.firwin2` design. The result is the real amplitude
    response of that linear-phase design on the ascending-frequency bin grid of
    `stft`. Multiplying an STFT by the response filters each segment with no delay.
    The transition band is centered on `cutoff`, as in `design_fir_lpf`. The gain at
    `cutoff` is 1/2.

    Args:
        size: the number of taps and of bins
        sample_rate: the sample rate (in S/s) of the filtered waveform
        cutoff: the -6 dB edge (in Hz); at or above Nyquist, the result is all ones
        transition: the width (in Hz) over which the gain falls linearly from 1 to
            0, from ``cutoff - transition/2`` to ``cutoff + transition/2``; a
            transition band that runs past Nyquist ends at Nyquist
        window: the `scipy.signal.firwin2` window of the design
        xp: the array namespace of the result; None selects numpy
        dtype: the dtype of the result

    Returns:
        a 1-D array of `size` samples

    Raises:
        ValueError: if ``cutoff - transition/2`` is at or below 0 Hz
    """

    if xp is None:
        xp = np

    nyquist = sample_rate / 2
    if cutoff >= nyquist:
        return xp.ones(size, dtype=dtype)

    from scipy import signal

    lower = cutoff - transition / 2
    upper = cutoff + transition / 2
    if lower <= 0:
        raise ValueError(
            f'the {transition} Hz transition band leaves no passband below it for '
            f'cutoff={cutoff} Hz'
        )

    # firwin2 requires a grid that rises from 0 to nyquist, so a transition band that
    # runs past nyquist is cut short there
    if upper < nyquist:
        freqs = [0, lower, upper, nyquist]
        gains = [1.0, 1.0, 0.0, 0.0]
    else:
        freqs = [0, lower, nyquist]
        gains = [1.0, 1.0, 0.0]
    h = signal.firwin2(size, freqs, gains, window=window, fs=sample_rate)

    # the shift baked into the window rotates the spectrum by size // 2 bins, which
    # is fftshift order only for even sizes
    w = get_window('rect', size, dtype='complex128', fftshift=True)
    omega = 2 * np.pi * np.roll(np.fft.fftfreq(size), -(size // 2))
    # the design is symmetric about (size - 1) / 2, a half sample for even sizes, so
    # its delay is removed on the shifted grid rather than by a time-domain rotation
    H = np.fft.fft(h * w) * np.exp(1j * omega * (size - 1) / 2)

    return xp.asarray(H.real, dtype=dtype)


def _stft_fir_lowpass(
    xstft: _AT,
    *,
    sample_rate: float,
    bandwidth: float,
    transition_bandwidth: float,
    axis: int = 0,
    out=None,
) -> _AT:
    xp = array_namespace(xstft)

    H = _fir_lowpass_fft(
        xstft.shape[axis + 1],
        sample_rate=sample_rate,
        cutoff=bandwidth / 2,
        transition=transition_bandwidth,
        dtype=typing.cast(str, xstft.dtype),
        window='rect',
        xp=xp,
    )

    H = _broadcast_onto(H, xstft, axis=axis + 1)

    return xp.multiply(xstft, H, out=out)


def oafilter(
    x: Array,
    *,
    fs: float,
    nfft: int,
    window: str | tuple = 'hann',
    passband: tuple[float, float],
    nfft_out: int | None = None,
    frequency_shift=False,
    axis=0,
    extend=False,
    overwrite_x=False,
):
    """apply a bandpass filter to `x` through STFT overlap-add.

    The bins outside `passband` are zeroed in each STFT block, with the band narrowed
    at each edge by the equivalent noise bandwidth of `window`.

    Args:
        x: the waveform, with time on `axis`
        fs: the sample rate (in S/s) of `x`
        nfft: the STFT block size (in samples), an even multiple of the COLA divisor
            of `window`
        window: the COLA window, ``'hann'``, ``'blackman'``, or ``'blackmanharris'``
        passband: the ``(low, high)`` band edges (in Hz) kept nonzero
        nfft_out: the output block size (in samples), an even multiple of the COLA
            divisor; a value other than `nfft` resamples by ``nfft_out / nfft`` with
            the passband centered on the output DC bin; None keeps `nfft`
        frequency_shift: True centers the passband on the output DC bin when
            `nfft_out` equals `nfft`
        axis: the time axis of `x`
        extend: True pads `x` with zeros along `axis` to a whole number of STFT
            hops and trims the output to ``round(x.shape[axis] * nfft_out / nfft)``
            samples; False raises `ValueError` for a partial trailing hop
        overwrite_x: True permits the STFT to consume the buffer of `x`

    Returns:
        the filtered waveform on `axis`, at ``fs * nfft_out / nfft``, in the array
        namespace of `x`

    Raises:
        TypeError: if `window` is not a key of the COLA divisor table
        ValueError: if `nfft` or `nfft_out` is odd or not a multiple of the COLA
            divisor, or if ``x.shape[axis]`` is not a whole number of STFT hops
            without `extend`
    """

    size_in = x.shape[axis]
    nfft_out, noverlap, overlap_scale, pad = design_oafilter(
        size_in,
        window=window,
        nfft_out=nfft_out or nfft,
        nfft=nfft,
        extend=extend,
    )

    if pad:
        x = pad_along_axis(x, [[0, pad]], axis=axis)
        # the padded array is a fresh allocation
        overwrite_x = True

    # the ENBW is in bins of fs / nfft_out, and the passband edges are in Hz
    enbw_bins = equivalent_noise_bandwidth(window, nfft_out, fftbins=False)
    guard = float(enbw_bins) * fs / nfft_out

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
        freqs, y, passband=(passband[0] + guard, passband[1] - guard), axis=axis
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

    x = istft(
        y,
        round((size_in + pad) * nfft_out / nfft),
        nfft=nfft_out or nfft,
        noverlap=noverlap,
        overwrite_x=True,
        axis=axis,
    )

    if pad:
        x = axis_slice(x, 0, round(size_in * nfft_out / nfft), axis=axis)

    if nfft_out != nfft:
        x *= nfft_out / nfft

    return x


def oaconvolve(x1, x2, mode='full', axes=-1):
    """convolve `x1` and `x2` by overlap-add, as `scipy.signal.oaconvolve`.

    The result is in the array namespace of `x1`.
    """
    if is_cupy_array(x1):
        from cupyx.scipy.signal import oaconvolve as func  # type: ignore
    else:
        from scipy.signal import oaconvolve as func

    return func(x1, x2, mode=mode, axes=axes)
