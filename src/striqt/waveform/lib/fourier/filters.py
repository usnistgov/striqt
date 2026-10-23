"""FIR low-pass designs and the STFT overlap-add filters that apply a design.

Window overlaps come from one constant-overlap-add (COLA) divisor table, which the
resampler design shares.
"""

from __future__ import annotations as __

import typing

from .. import util
from ..arrays import array_namespace, convert_np_to_xp, is_cupy_array
from ..util import np
from .fft import _broadcast_onto
from .stft import downsample_stft, istft, stft, zero_stft_by_freq
from .windows import equivalent_noise_bandwidth, get_window

if typing.TYPE_CHECKING:
    from ..typing import _AT, Array, DTypeLike, WindowType, XpType


# required divisors
_COLA_WINDOW_SIZE_DIVISOR = {
    None: 1,
    'rect': 1,
    'hamming': 2,
    'blackman': 3,
    'blackmanharris': 5,
}


@util.lru_cache()
def design_oafilter(
    array_size: int, *, window, nfft_out: int, nfft: int, extend: bool
) -> tuple:
    if nfft_out is None:
        nfft_out = nfft

    try:
        divisor = _COLA_WINDOW_SIZE_DIVISOR[window]
    except KeyError:
        raise TypeError(
            f'window must be one of {tuple(_COLA_WINDOW_SIZE_DIVISOR)}, not {window!r}'
        )

    if nfft_out % divisor != 0:
        raise ValueError(
            f'{window!r} window COLA requires output nfft_out % {divisor} == 0'
        )

    if window is None or window == 'rect':
        overlap_scale = 1
    elif window == 'hamming':
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


@convert_np_to_xp
@util.lru_cache()
@util.persistent_cache()
def design_fir_lpf(
    bw, fs, *, numtaps=4001, transition_bw=250e3, dtype='float32', xp=None
):
    from scipy import signal

    edges = [
        0,
        bw / 2 - transition_bw / 2,
        bw / 2 + transition_bw / 2,
        fs / 2,
    ]
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

    The taps are a `scipy.signal.firwin2` design with unit gain below `cutoff` (in Hz)
    falling linearly to zero over `transition` (in Hz). A transition band that would
    run past Nyquist ends at Nyquist. The result is the real amplitude response of
    that linear-phase design on the bin grid of `stft` (ascending frequency).
    Multiplying an STFT by the response filters each segment with no delay. A
    `cutoff` at or above Nyquist gives all ones.

    Returns:
        a 1-D array of `size` samples with dtype `dtype` in `xp` (numpy when None)
    """

    if xp is None:
        xp = np

    nyquist = sample_rate / 2
    if cutoff >= nyquist:
        return xp.ones(size, dtype=dtype)

    from scipy import signal

    # firwin2 requires a nondecreasing grid that ends at nyquist, so a transition
    # band that runs past it is cut short there
    if cutoff + transition < nyquist:
        freqs = [0, cutoff, cutoff + transition, nyquist]
        gains = [1.0, 1.0, 0.0, 0.0]
    else:
        freqs = [0, cutoff, nyquist]
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
    window: str | tuple = 'hamming',
    passband: tuple[float, float],
    nfft_out: int | None = None,
    frequency_shift=False,
    axis=0,
    extend=False,
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

    nfft_out, noverlap, overlap_scale, _ = design_oafilter(
        x.size,
        window=window,
        nfft_out=nfft_out or nfft,
        nfft=nfft,
        extend=extend,
    )

    # the ENBW is in bins of fs / nfft_out; the passband edges are in Hz
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
        round(x.shape[axis] * nfft_out / nfft),
        nfft=nfft_out or nfft,
        noverlap=noverlap,
        overwrite_x=True,
        axis=axis,
    )

    if nfft_out != nfft:
        x *= nfft_out / nfft

    return x


def oaconvolve(x1, x2, mode='full', axes=-1):
    """convolve x1 and x2, and return the result.

    See also:
        `scipy.signal.oaconvolve`
    """
    if is_cupy_array(x1):
        from cupyx.scipy.signal import oaconvolve as func  # type: ignore
    else:
        from scipy.signal import oaconvolve as func

    return func(x1, x2, mode=mode, axes=axes)
