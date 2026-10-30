"""Rational resampling by FFT and by STFT overlap-add from one COLA design.

The FFT sizes of a `ResamplerDesign` are in the exact ratio of the input to the
output sample rate.
"""

from __future__ import annotations as __

import math
import typing
from fractions import Fraction
from math import inf, isfinite

from .. import util
from ..arrays import array_namespace, axis_slice, isroundmod, pad_along_axis
from ..util import np
from .fft import _broadcast_onto, fft, ifft, time_fftshift, time_ifftshift
from .filters import _COLA_WINDOW_SIZE_DIVISOR, _fir_lowpass_fft, design_oafilter
from .stft import _find_downsample_copy_range, istft, stft

if typing.TYPE_CHECKING:
    from ..typing import _AT, Array, ShiftType


OLA_MAX_FFT_SIZE = 128 * 1024


def _next_smooth_size(n: int) -> int:
    """return the smallest 2·3·5·7-smooth integer at or above `n`."""
    # scipy.fft.next_fast_len admits a factor of 11; cuFFT is tuned for sizes of the
    # form 2^a 3^b 5^c 7^d
    size = max(n, 1)
    while True:
        remainder = size
        for prime in (2, 3, 5, 7):
            while remainder % prime == 0:
                remainder //= prime
        if remainder == 1:
            return size
        size += 1


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
    frequency_shift: ShiftType
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
    shift: ShiftType = False,
    window=None,
    fs_sdr: typing.Optional[float] = None,
) -> ResamplerDesign:
    """design the radio sample rate, LO offset, and COLA FFT sizes of a resampler.

    The design resamples from the radio rate ``fs_sdr`` (an integer division of
    `fs_base`, unless given) to `fs_target` with FFT sizes in the ratio
    ``nfft / nfft_out == fs_sdr / fs_target`` exactly. The sizes are ``k * p`` and
    ``k * q`` for the reduced ratio ``p / q`` of ``fs_sdr`` to `fs_target`. The
    multiplier ``k`` is the smallest even multiple of the COLA divisor of `window`
    that is 2·3·5·7-smooth and puts ``nfft_out`` above `min_fft_size`. With
    `shift`, the radio is tuned ``lo_offset`` below the center frequency, and
    ``fs_sdr`` is the largest integer division of `fs_base` at or above
    ``fs_target + min_oversampling * bw + bw_lo``. The magnitude of
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
        min_fft_size: the size that ``nfft_out`` must exceed
        shift: ``'left'`` or ``'right'`` places the protected band below or above
            the tuned LO; ``'none'``, None, or False leaves the LO at the center
            frequency
        window: a key of the COLA window divisor table; None selects ``'hann'``
        fs_sdr: the radio sample rate (in S/s) to use instead of selecting one

    Returns:
        the `ResamplerDesign` that resamples from the selected radio rate to
        `fs_target`

    Raises:
        ValueError: if `shift` is set while `bw` is infinite, `shift` is not a
            recognized direction, `window` has no COLA divisor, `bw` exceeds
            `fs_base`, the LO shift needs a radio rate above `fs_base`, no
            fraction with denominator up to `OLA_MAX_FFT_SIZE` realizes the rate
            ratio, or the LO shift does not fit the Nyquist band of the radio at
            ``fs_sdr``
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

    # isroundmod admits a float ratio that rounds to the fraction, which covers
    # downward rounding errors (e.g., 0.999)
    ratio = Fraction(resample_ratio).limit_denominator(OLA_MAX_FFT_SIZE)
    p, q = ratio.numerator, ratio.denominator
    if not isroundmod(resample_ratio * q, 1):
        raise ValueError('no rational FFT sizes satisfied design constraints')

    # the half-band shift that stft and istft apply is self-inverse only on an even
    # block, so both sizes are even multiples of the COLA divisor. p and q are
    # coprime, so the divisor divides k * p and k * q only when it divides k
    divisor = math.lcm(2, _COLA_WINDOW_SIZE_DIVISOR[window])
    k = divisor * _next_smooth_size(min_fft_size // (q * divisor) + 1)
    nfft_in = k * p
    nfft_out = k * q

    if shift == 'left':
        sign = -1
    elif shift == 'right':
        sign = +1
    elif shift in ('none', False, None):
        sign = 0
    else:
        raise ValueError(f'shift argument must be "left" or "right", not {shift!r}')

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
        window=window or 'hann',
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
    """design the radio sample rate and the FIR rate-change factors of a resampler.

    The factors are the FFT sizes of `design_cola_resampler` with ``min_fft_size``
    of 1 and no LO shift.

    Args:
        fs_base: the master clock rate (in S/s) that the radio rate divides
        fs_target: the sample rate (in S/s) after resampling
        bw: the bandwidth (in Hz) to protect from LO leakage; ``inf`` leaves the
            passband unbounded
        bw_lo: the bandwidth (in Hz) of the LO leakage and phase noise kept out of
            the protected band
        min_oversampling: the factor applied to `bw` when sizing the radio rate

    Returns:
        the radio sample rate (in S/s), and the ``{'up': nfft_out, 'down': nfft}``
        keywords of ``scipy.signal.upfirdn``

    Raises:
        ValueError: if `bw` exceeds `fs_base`, or no fraction with denominator up
            to `OLA_MAX_FFT_SIZE` realizes the rate ratio
    """

    design = design_cola_resampler(
        fs_base,
        fs_target,
        bw=bw,
        bw_lo=bw_lo,
        min_oversampling=min_oversampling,
        min_fft_size=1,
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
    window: str | tuple[str, float] = 'hann',
    overwrite_x: bool = False,
    axis: int = 1,
    frequency_shift: float = 0,
    filter_bandwidth: float | None = None,
    transition_bandwidth: float = 250e3,
    scale: Array | float = 1.0,
):
    """resample `x` by ``up / down`` through STFT overlap-add.

    Each STFT block of `down` samples is cropped or zero-padded to `up` bins, scaled
    by the low-pass response and `scale`, and overlap-added at the output rate. The
    response has the band edges of `design_fir_lpf`: the gain is 1/2 at
    ``filter_bandwidth / 2`` and the transition is centered there.

    Args:
        x: the waveform, with time on `axis`; its length must be a whole number of
            STFT hops, ``down + k * (down - noverlap)``
        up: the output block size (in samples), an even multiple of the COLA divisor
            of `window`
        down: the input block size (in samples), an even multiple of the COLA
            divisor of `window`
        fs: the sample rate (in S/s) of `x`
        window: the COLA window of the STFT, a key of the divisor table
        overwrite_x: True permits the STFT to consume the buffer of `x`
        axis: the time axis of `x`
        frequency_shift: the frequency (in Hz) moved to the output DC, a multiple of
            ``fs / down``; 0 keeps the center frequency
        filter_bandwidth: the passband width (in Hz) of the low-pass; None or
            ``inf`` applies no low-pass
        transition_bandwidth: the width (in Hz) of the low-pass transition band
        scale: a voltage gain, a scalar or one value per entry of the axis before
            `axis`

    Returns:
        the waveform at ``fs * up / down`` on `axis`, of ``x.shape[axis] * up //
        down`` samples, in the dtype and array namespace of `x`

    Raises:
        ValueError: if `frequency_shift` is nonzero while upsampling, is not a
            multiple of ``fs / down``, or moves the output band past an edge of the
            input band; if `up` or `down` is odd or not a multiple of the COLA
            divisor; or if `window` has no COLA divisor
    """

    nfft = down
    nfft_out = up

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

    xp = array_namespace(x)
    y = typing.cast(
        'Array',
        stft(
            x,
            fs=fs,
            window=window,
            nperseg=nfft,
            noverlap=round(nfft * overlap_scale),
            axis=axis,
            truncate=False,
            overwrite_x=overwrite_x,
            return_axis_arrays=False,
        ),
    )
    del x

    if nfft_out <= nfft:
        bounds_out, bounds_in, _ = _find_downsample_copy_range(
            nfft, nfft_out, edge_low, edge_high
        )
    else:
        pad_left = (nfft_out - nfft) // 2
        bounds_in = (0, nfft)
        bounds_out = (pad_left, pad_left + nfft)

    # the low-pass, the COLA rate scale, and the caller's scale are one multiply
    # applied while the bins are copied
    if filter_bandwidth is not None and isfinite(filter_bandwidth):
        gain = _fir_lowpass_fft(
            nfft_out,
            sample_rate=fs * up / down,
            cutoff=filter_bandwidth / 2,
            transition=transition_bandwidth,
            dtype=typing.cast(str, y.dtype),
            window='rect',
            xp=xp,
        )
    else:
        gain = xp.ones(nfft_out, dtype=y.dtype)
    gain = _broadcast_onto(gain * (nfft_out / nfft), y, axis=axis + 1)
    if np.ndim(scale) == 1:
        # one gain per entry of the axis that is not resampled, as in `resample`
        gain = gain * _broadcast_onto(xp.asarray(scale), y, axis=max(axis - 1, 0))
    else:
        gain = gain * scale

    y = _copy_bins_to_fftfreq_order(
        y, nfft_out, bounds_in, bounds_out, gain=gain.astype(y.dtype), axis=axis
    )

    return istft(
        y, nfft=nfft_out, noverlap=noverlap, axis=axis, overwrite_x=True, fftshift=False
    )


def _copy_bins_to_fftfreq_order(
    y: Array,
    nfft_out: int,
    bounds_in: tuple[int, int],
    bounds_out: tuple[int, int],
    *,
    gain: Array,
    axis: int,
) -> Array:
    """return the `bounds_in` bins of each STFT segment of `y` as `nfft_out` bins in
    `fftfreq` order.

    `y` has segments on `axis` and bins in ascending frequency order on ``axis + 1``.
    The copied bins occupy the ascending-order range `bounds_out` of the output, and
    the other output bins are zero. The output is a new contiguous array of `nfft_out`
    bins on ``axis + 1`` and the dtype of `y`.

    Args:
        y: the STFT
        nfft_out: the even number of output bins
        bounds_in: the ``(start, stop)`` ascending-order bins of `y` that are copied
        bounds_out: the ``(start, stop)`` ascending-order output bins that receive
            the copy, of the same length as `bounds_in`
        gain: the multiplier of the copied bins, in ascending order on ``axis + 1``
            and broadcastable onto the output
        axis: the segment axis of `y`
    """
    xp = array_namespace(y)
    shape = list(y.shape)
    shape[axis + 1] = nfft_out
    out = xp.empty(shape, dtype=y.dtype)

    half = nfft_out // 2
    bin_axis = axis + 1

    def fftfreq_order_chunks(start: int, stop: int):
        # split an ascending-order range at the DC bin, since each half maps onto one
        # contiguous range of fftfreq-order bins
        for a, b in ((start, min(stop, half)), (max(start, half), stop)):
            if b > a:
                yield a, b, (a + half) % nfft_out

    for a, b, dest in (
        *fftfreq_order_chunks(0, bounds_out[0]),
        *fftfreq_order_chunks(bounds_out[1], nfft_out),
    ):
        xp.copyto(axis_slice(out, dest, dest + b - a, axis=bin_axis), 0)

    offset = bounds_in[0] - bounds_out[0]
    for a, b, dest in fftfreq_order_chunks(*bounds_out):
        xp.multiply(
            axis_slice(y, a + offset, b + offset, axis=bin_axis),
            axis_slice(gain, a, b, axis=bin_axis),
            out=axis_slice(out, dest, dest + b - a, axis=bin_axis),
        )

    return out
