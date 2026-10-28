"""Rational resampling by FFT and by STFT overlap-add from one COLA design.

The FFT sizes of a `ResamplerDesign` are in the exact ratio of the input to the
output sample rate.
"""

from __future__ import annotations as __

import math
import typing
from math import inf, isfinite

from .. import util
from ..arrays import array_namespace, axis_slice, isroundmod, pad_along_axis
from ..util import np
from .fft import fft, ifft, time_fftshift, time_ifftshift
from .filters import _COLA_WINDOW_SIZE_DIVISOR, _stft_fir_lowpass, design_oafilter
from .stft import _find_downsample_copy_range, istft, stft

if typing.TYPE_CHECKING:
    from ..typing import _AT, Array, ShiftType


OLA_MAX_FFT_SIZE = 128 * 1024


@util.lru_cache()
def _prime_fft_sizes(min=2, max=OLA_MAX_FFT_SIZE):
    s = np.arange(3, max, 2)

    for m in range(3, int(np.sqrt(max) + 1), 2):
        if s[(m - 3) // 2]:
            s[(m * m - 3) // 2 :: m] = 0

    return s[(s > min)]


class ResamplerDesign(typing.TypedDict):
    """the radio and COLA resampler settings that realize a target sample rate.

    ``fs_sdr`` and ``fs`` are both the radio sample rate (in S/s) at the resampler
    input. ``nfft`` and ``nfft_out`` are the input and output FFT sizes, in the exact
    ratio of the radio rate to the target rate and both divisible by the COLA
    divisor of ``window``. ``lo_offset`` is the offset (in Hz) by which the radio is
    tuned below the center frequency: a whole number of ``fs_sdr / nfft`` bins, or 0
    without an LO shift. ``frequency_shift`` is the requested shift direction.
    ``passband`` is the pair of protected band edges (in Hz) relative to the tuned
    LO, or ``(None, None)`` for an unbounded bandwidth.
    """

    fs_sdr: float
    lo_offset: float
    window: str | tuple[str, float]
    nfft: int
    nfft_out: int
    frequency_shift: 'ShiftType'
    passband: tuple[float | None, float | None]
    fs: float


@util.lru_cache()
def design_cola_resampler(
    fs_base: float,
    fs_target: float,
    bw: float = inf,
    bw_lo: float = 0,
    min_oversampling: float = 1.1,
    min_fft_size=2 * 4096 - 1,
    shift: 'ShiftType' = False,
    avoid_primes=True,
    window=None,
    fs_sdr: typing.Optional[float] = None,
) -> ResamplerDesign:
    """design the radio sample rate, LO offset, and COLA FFT sizes of a resampler.

    The design resamples from the radio rate ``fs_sdr`` (an integer division of
    `fs_base`, unless given) to `fs_target` with FFT sizes in the ratio
    ``nfft / nfft_out == fs_sdr / fs_target`` exactly, both divisible by the COLA
    divisor of `window`. With `shift`, the radio is tuned ``lo_offset`` below the
    center frequency, and ``fs_sdr`` is the largest integer division of `fs_base`
    at or above ``fs_target + min_oversampling * bw + bw_lo``. The magnitude of
    ``lo_offset`` is ``bw / 2 + bw_lo / 2`` rounded up to a whole number of
    ``fs_sdr / nfft`` bins, up to one bin more than that sum.

    Args:
        fs_base: the master clock rate (in S/s) that ``fs_sdr`` divides
        fs_target: the sample rate (in S/s) after resampling
        bw: the bandwidth (in Hz) to protect from LO leakage; ``inf`` disables the
            LO shift and leaves the passband unbounded
        bw_lo: the bandwidth (in Hz) of the LO leakage and phase noise kept out of
            the protected band
        min_oversampling: the factor applied to `bw` when sizing ``fs_sdr`` for an
            LO shift
        min_fft_size: the smallest ``nfft_out`` candidate considered
        shift: ``'left'`` or ``'right'`` places the protected band below or above
            the tuned LO; ``'none'``, None, or False leaves the LO at the center
            frequency
        avoid_primes: True rejects ``nfft_out`` candidates that are large primes
        window: a key of the COLA window divisor table; None selects ``'hamming'``
        fs_sdr: the radio sample rate (in S/s) to use instead of selecting one

    Returns:
        the `ResamplerDesign` that resamples from the selected radio rate to
        `fs_target`

    Raises:
        ValueError: if `shift` is set while `bw` is infinite, `shift` is not a
            recognized direction, `window` has no COLA divisor, `bw` exceeds
            `fs_base`, the LO shift needs a radio rate above `fs_base`, no FFT size
            up to `OLA_MAX_FFT_SIZE` realizes the rate ratio, or the LO shift does
            not fit the Nyquist band of the radio at ``fs_sdr``
    """

    if bw == inf and shift:
        raise ValueError(
            'frequency shifting may only be applied when an analysis bandwidth is specified'
        )

    if window not in _COLA_WINDOW_SIZE_DIVISOR:
        raise ValueError(
            f'window must be one of {tuple(_COLA_WINDOW_SIZE_DIVISOR)}, not {window!r}'
        )

    if shift:
        # the output band is fs_target wide and centered on lo_offset
        fs_sdr_min = fs_target + min_oversampling * bw + bw_lo
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

    if bw != inf and bw > fs_base:
        raise ValueError(
            'passband bandwidth exceeds Nyquist bandwidth at maximum sample rate'
        )

    resample_ratio = fs_sdr / fs_target

    # isroundmod admits ratios that round to a whole number, which covers downward
    # rounding errors (e.g., 0.999)
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

    # the LO shift arguments below assume a hamming COLA window
    if shift == 'left':
        sign = -1
    elif shift == 'right':
        sign = +1
    elif shift in ('none', False, None):
        sign = 0
    else:
        raise ValueError(f'shift argument must be "left" or "right", not {repr(shift)}')

    if sign != 0 and bw == inf:
        raise ValueError('a passband bandwidth must be set to design a LO shift')

    if bw == inf:
        lo_offset = 0.0
        passband = (None, None)
    else:
        # records handed to `resample` are padded to a multiple of nfft_in, so an
        # offset on the fs_sdr / nfft_in grid is a whole number of their bins too
        bin_size = fs_sdr / nfft_in
        lo_offset = sign * math.ceil((bw / 2 + bw_lo / 2) / bin_size) * bin_size
        passband = (lo_offset - bw / 2, lo_offset + bw / 2)

    if sign != 0 and abs(lo_offset) + fs_target / 2 > fs_sdr / 2:
        raise ValueError(
            f'LO frequency shift of {lo_offset / 1e6:0.3f} MHz does not fit the '
            f'{fs_target / 1e6:0.2f} MS/s output band in the radio Nyquist band at '
            f'{fs_sdr / 1e6:0.2f} MS/s'
        )

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
    bw: float = inf,
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


def resample_edges(
    nfft_in: int, nfft_out: int, shift: int = 0
) -> tuple[int, int] | None:
    """return the input bin range ``[edge_low, edge_high)`` that `resample` keeps.

    The range indexes the `nfft_in`-bin spectrum in ascending frequency order. The
    range is `nfft_out` bins wide and centered `shift` bins above the input center.
    The result is None for a `shift` of 0.

    Raises:
        ValueError: if `nfft_in` is odd, if `shift` is nonzero while upsampling
            (``nfft_out > nfft_in``), or if the shifted band runs outside the input
            spectrum
    """
    if nfft_in % 2 != 0:
        raise ValueError(f'the input length must be even, not {nfft_in}')

    if shift == 0:
        return None
    if nfft_out > nfft_in:
        raise ValueError('shift is only supported when downsampling')

    edge_low = nfft_in // 2 - nfft_out // 2 + shift
    edge_high = edge_low + nfft_out

    if edge_low < 0:
        raise ValueError('shift is too small')
    if edge_high > nfft_in:
        raise ValueError('shift is too large')

    return edge_low, edge_high


def resample(
    x: _AT,
    num: int,
    axis: int = 0,
    window: None = None,
    domain: typing.Literal['time', 'frequency'] = 'time',
    overwrite_x: bool = False,
    scale: Array | float = 1,
    shift: int = 0,
) -> _AT:
    """resample `x` to `num` samples along `axis` by FFT, as `scipy.signal.resample`.

    Args:
        x: the waveform, or with ``domain='frequency'`` the spectrum of the waveform
            in ascending frequency order
        num: the number of output samples along `axis`
        axis: the axis to resample
        window: must be None
        domain: ``'time'`` takes `x` as a waveform; ``'frequency'`` takes `x` as a
            spectrum
        overwrite_x: True evaluates in the buffer of `x`; when downsampling, no new
            array is then allocated
        scale: a gain applied to the output
        shift: the offset (in input bins) of the kept band above the input center
            when downsampling, as in `resample_edges`

    Returns:
        the resampled waveform with `num` samples on `axis`, in the dtype and array
        namespace of `x`; `x` itself when ``x.shape[axis] == num``

    Raises:
        ValueError: if `domain` is not ``'time'`` or ``'frequency'``, if `window` is
            not None, or if `resample_edges` rejects `shift`
    """
    if domain not in ('time', 'freq', 'frequency'):
        raise ValueError(
            f"Acceptable domain flags are 'time' or 'frequency', not domain={domain!r}"
        )

    if x.shape[axis] == num:
        return x

    xp = array_namespace(x)

    x = xp.asarray(x)
    nfft_in = x.shape[axis]
    nfft_out = num

    if window is not None:
        raise ValueError('window argument is not supported')

    edges = resample_edges(nfft_in, nfft_out, shift)
    if edges is None:
        edge_low = edge_high = None
    else:
        edge_low, edge_high = edges

    resample_scale = float(nfft_out) / float(nfft_in) * scale

    if domain == 'time':
        # fftshift in the time domain avoids a copy and makes downsampling a slice
        x = time_fftshift(x, resample_scale, overwrite_x=overwrite_x, axis=axis)
        y = fft(x, axis=axis, overwrite_x=True, out=x)
    else:
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
        # upsample by zero-padding frequency, which allocates a copy
        pad_left = (nfft_out - nfft_in) // 2
        pad_right = pad_left + (nfft_out - nfft_in) % 2
        y = pad_along_axis(y, [[pad_left, pad_right]], axis=axis)

    # Inverse transform
    xout = ifft(y, axis=axis, overwrite_x=True, out=y)

    return time_ifftshift(xout, overwrite_x=True, axis=axis)


def oaresample(
    x: Array,
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
    scale: Array | float = 1.0,
):
    """apply resampler implemented through STFT overlap-and-add.

    Returns:
        the filtered IQ capture
    """

    nfft = down
    nfft_out = up
    size_in = x.size

    nfft_out, noverlap, overlap_scale, _ = design_oafilter(
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
        raise ValueError('frequency_shift must be a multiple of fs/down')

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

    if filter_bandwidth is not None and isfinite(filter_bandwidth):
        y = _stft_fir_lowpass(
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
