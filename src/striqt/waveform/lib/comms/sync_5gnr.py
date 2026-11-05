"""5G NR synchronization against the synchronization signal block (SSB).

The PSS and SSS sequences follow 3GPP TS 38.211 §7.4.2, and the SSB symbol positions
follow the cell search cases of TS 38.213 §4.1.
"""

from __future__ import annotations as __

import dataclasses
import math
import typing
from fractions import Fraction
from math import ceil, isclose

import array_api_compat

from .. import arrays, fourier, power, util
from ..arrays import array_namespace, is_cupy_array, isroundmod, pad_along_axis
from ..typing import CellSSBIndexes
from ..util import np
from .phy_3gpp import get_3gpp_phy, slot_period
from .sequences import m_sequence

if typing.TYPE_CHECKING:
    from types import ModuleType

    from ..typing import Array, WindowSpecType


def _min_diff(x: typing.Sequence[int]) -> int | None:
    """return the minimum difference between neighbors in x"""

    if len(x) <= 1:
        return None

    return min(b - a for a, b in zip(x, x[1:]))


@dataclasses.dataclass
class SyncParams:
    min_cp_size: int
    cp_offsets: list[int]
    frame_size: int
    slot_count: int
    corr_size: int
    frames_per_sync: int
    duration: float
    symbol_indexes: list[int]
    short_symbol_size: int
    lag_count: int
    subcarrier_spacing: float
    sample_rate: float = 2 * 7.68e6
    discovery_periodicity: float = 20e-3
    shared_spectrum: bool = False
    max_lag_symbols: int | None = None


def _pss_m_sequence(N_id2: int) -> list[int]:
    """return the bipolar PSS m-sequence for `N_id2`.

    Args:
        N_id2: the sector part of the cell ID, 0 to 2

    Returns:
        127 values in {-1, 1}, one per active subcarrier in frequency-domain order

    References:
        3GPP TS 38.211 §7.4.2.2
    """
    x = m_sequence([0, 1, 1, 0, 1, 1, 1], (3, 7), 127)

    m = [(n + 43 * N_id2) % 127 for n in range(127)]

    pss = [(1 - 2 * x[_m]) for _m in m]

    return pss


def _sss_m_sequence(N_id: int) -> list[int]:
    """return the bipolar SSS m-sequence for the cell ID `N_id`.

    The cell ID is ``N_id = 3*N_id1 + N_id2``, where ``N_id1`` is the cell
    identity group and ``N_id2`` is the sector ID of `_pss_m_sequence`.

    Args:
        N_id: the cell ID, 0 to 1007

    Returns:
        127 values in {-1, 1}, one per active subcarrier in frequency-domain order

    References:
        3GPP TS 38.211 §7.4.2.3
    """
    x_0 = m_sequence([1, 0, 0, 0, 0, 0, 0], (3, 7), 127)
    x_1 = m_sequence([1, 0, 0, 0, 0, 0, 0], (6, 7), 127)

    N_id_1 = N_id // 3
    N_id_2 = N_id % 3

    m_0 = 15 * (N_id_1 // 112) + 5 * N_id_2
    m_1 = N_id_1 % 112

    sss = [
        (1 - 2 * x_0[(n + m_0) % 127]) * (1 - 2 * x_1[(n + m_1) % 127])
        for n in range(127)
    ]

    return sss


def _generate_5g_nr_sync_sequence(
    seq_func: typing.Callable[[int], list[int]],
    max_id: int,
    sample_rate: float,
    subcarrier_spacing: float,
    center_frequency: float | None = None,
    *,
    xp=None,
    dtype='complex64',
):
    """return time-domain sync sequences for every id from an m-sequence generator.

    Each sequence is normalized to unit mean power over the 127 active subcarriers
    of the sequence and placed on the `sample_rate` FFT grid, centered at
    `center_frequency`.

    Args:
        seq_func: maps a sequence id to the 127 bipolar frequency-domain values of
            that sequence
        max_id: sequences are built for ids 0 through `max_id`
        sample_rate: in S/s; a multiple of `subcarrier_spacing` and at least
            ``127 * subcarrier_spacing``
        subcarrier_spacing: in Hz; a multiple of 15e3
        center_frequency: offset (in Hz) of the sequence center from baseband; a
            whole multiple of `subcarrier_spacing`, or ``None`` or 0 for centered
        xp: array namespace of the result; ``None`` selects numpy
        dtype: complex dtype of the result

    Returns:
        array of shape ``(max_id + 1, nfft + cp)``, where ``nfft`` is
        ``sample_rate / subcarrier_spacing`` and ``cp`` is the shortest 5G NR
        cyclic prefix at `sample_rate`; the leading ``cp`` samples are zero

    Raises:
        ValueError: `subcarrier_spacing` is not a multiple of 15 kHz
        ValueError: `sample_rate` is below ``127 * subcarrier_spacing`` or not a
            multiple of `subcarrier_spacing`
        ValueError: `center_frequency` is not a whole multiple of
            `subcarrier_spacing`
        ValueError: the `center_frequency` shift places part of the sequence
            outside the Nyquist band
    """
    if xp is None:
        xp = np

    # number of occupied subcarriers in the PSS
    SC_COUNT = 127

    if not isroundmod(subcarrier_spacing, 15e3):
        raise ValueError('subcarrier_spacing must be a multiple of 15000')

    min_sample_rate = SC_COUNT * subcarrier_spacing
    if sample_rate < min_sample_rate:
        raise ValueError(f'sample_rate must be at least {min_sample_rate} S/s')

    if isroundmod(sample_rate, subcarrier_spacing):
        size_out = round(sample_rate / subcarrier_spacing)
    else:
        raise ValueError('sample_rate must be a multiple of subcarrier spacing')

    if center_frequency is None or center_frequency == 0:
        frequency_offset = 0
    elif isroundmod(center_frequency, subcarrier_spacing):
        # pad_lo and pad_hi below enforce the frequency bounds
        frequency_offset = round(center_frequency / subcarrier_spacing)
    else:
        raise ValueError(
            'center_frequency must be a whole multiple of subcarrier_spacing'
        )

    if size_out == SC_COUNT and frequency_offset == 0:
        pad_lo = 0
        pad_hi = 0
    else:
        pad_lo = size_out // 2 - 120 + 56 + frequency_offset
        pad_hi = size_out - SC_COUNT - pad_lo

    if pad_lo < 0 or pad_hi < 0:
        raise ValueError(
            'center_frequency shift pushes M-sequence outside of Nyquist sample rate'
        )

    norm = xp.sqrt(xp.float32(SC_COUNT))
    m_seqs = xp.array([seq_func(i) for i in range(max_id + 1)], dtype=dtype)
    norm *= xp.sqrt(xp.mean(xp.abs(m_seqs) ** 2))

    seq_freq = pad_along_axis(m_seqs / norm, [(pad_lo, pad_hi)], axis=1)

    seq_freq = xp.fft.fftshift(seq_freq, axes=1)
    x = fourier.fft.ifft(seq_freq, axis=1, out=seq_freq)

    # prepend the shortest cyclic prefix
    phy = get_3gpp_phy(
        1,
        subcarrier_spacing=subcarrier_spacing,
        sample_rate=sample_rate,
        generation='5G',
        xp=xp,
    )
    cp_size = min(phy.cp_sizes)
    return xp.concatenate([xp.zeros_like(x[:, -cp_size:]), x], axis=1)


@util.lru_cache()
def pss_5g_nr(
    sample_rate: float,
    subcarrier_spacing: float,
    center_frequency=0,
    *,
    xp=None,
    dtype='complex64',
):
    """return the time-domain PSS templates at `sample_rate` for ``N_id2`` 0 through 2.

    Each template has unit mean power over its 127 active subcarriers. The result is
    cached and must not be modified in place.

    Args:
        sample_rate: in S/s; a multiple of `subcarrier_spacing` and at least
            ``127 * subcarrier_spacing``
        subcarrier_spacing: in Hz; a multiple of 15e3
        center_frequency: offset (in Hz) of the PSS center from baseband; a whole
            multiple of `subcarrier_spacing`, or 0 for centered
        xp: array namespace of the result; None selects numpy
        dtype: complex dtype of the result

    Returns:
        array of shape ``(3, nfft + cp)``, where ``nfft`` is
        ``sample_rate / subcarrier_spacing`` and ``cp`` is the shortest 5G NR cyclic
        prefix at `sample_rate`; the leading ``cp`` samples are zero

    Raises:
        ValueError: if `subcarrier_spacing`, `sample_rate`, or `center_frequency` is
            outside the constraints above
    """

    xp = xp or np

    return _generate_5g_nr_sync_sequence(
        seq_func=_pss_m_sequence,
        max_id=2,
        sample_rate=sample_rate,
        subcarrier_spacing=subcarrier_spacing,
        center_frequency=center_frequency,
        xp=xp,
        dtype=dtype,
    )


@util.lru_cache()
def sss_5g_nr(
    sample_rate: float,
    subcarrier_spacing: float,
    center_frequency: float | None = None,
    *,
    xp=None,
    dtype='complex64',
):
    """return the time-domain SSS templates at `sample_rate` for cell IDs 0 through 1007.

    Each template has unit mean power over its 127 active subcarriers. The result is
    cached and must not be modified in place.

    Args:
        sample_rate: in S/s; a multiple of `subcarrier_spacing` and at least
            ``127 * subcarrier_spacing``
        subcarrier_spacing: in Hz; a multiple of 15e3
        center_frequency: offset (in Hz) of the SSS center from baseband; a whole
            multiple of `subcarrier_spacing`, or None or 0 for centered
        xp: array namespace of the result; None selects numpy
        dtype: complex dtype of the result

    Returns:
        array of shape ``(1008, nfft + cp)``, where ``nfft`` is
        ``sample_rate / subcarrier_spacing`` and ``cp`` is the shortest 5G NR cyclic
        prefix at `sample_rate`; the leading ``cp`` samples are zero

    Raises:
        ValueError: if `subcarrier_spacing`, `sample_rate`, or `center_frequency` is
            outside the constraints above
    """

    return _generate_5g_nr_sync_sequence(
        seq_func=_sss_m_sequence,
        max_id=1007,
        sample_rate=sample_rate,
        subcarrier_spacing=subcarrier_spacing,
        center_frequency=center_frequency,
        xp=xp or np,
        dtype=dtype,
    )


def index_pss_symbols(
    subcarrier_spacing: float,
    shared_spectrum: bool = False,
    symbol_indexes: CellSSBIndexes = 'auto',
    center_frequency: tuple[float, ...] | float | None = None,
) -> tuple[int, ...]:
    """return the symbol indexes of the PSS within a 3GPP TS 38.213 §4.1 SSB burst.

    A tuple or list in `symbol_indexes` is returned unchanged. ``'auto'`` selects
    the cell search case from `subcarrier_spacing`: 15 kHz is case A, 30 kHz is
    case C and requires `shared_spectrum`, 120 kHz is case D, 240 kHz is case E,
    480 kHz is case F, and 960 kHz is case G. The strings ``'a'``, ``'b'``, and
    ``'c'`` select that case directly. `shared_spectrum` is accepted for cases A
    and C only. For cases A, B, and C, `center_frequency` at or below 3 GHz (1.88 GHz
    for case C) selects the reduced candidate count of the standard.

    Args:
        subcarrier_spacing: in Hz; one of the SSB subcarrier spacings of the standard
        shared_spectrum: True selects the shared-spectrum candidate count of cases
            A and C
        symbol_indexes: ``'auto'``, a case letter, or an explicit sequence of
            symbol indexes
        center_frequency: in Hz; the maximum is used when given as a tuple, and
            ``None`` selects the full candidate count

    Returns:
        tuple of symbol indexes relative to the start of the half-frame, in
        increasing order

    Raises:
        TypeError: `symbol_indexes` is not a str, tuple, or list
        ValueError: 30 kHz spacing with ``'auto'`` and `shared_spectrum` False,
            a spacing without a standardized case, an unknown case string, or
            `shared_spectrum` with a case other than A or C

    References:
        3GPP TS 38.213 §4.1
    """
    if isinstance(symbol_indexes, str):
        symbol_indexes = symbol_indexes.lower()  # type: ignore
    elif isinstance(symbol_indexes, (tuple, list)):
        return symbol_indexes
    else:
        raise TypeError('symbol_indexes must be a str, tuple or list')

    if isinstance(center_frequency, tuple):
        center_frequency = max(center_frequency)

    if symbol_indexes == 'auto':
        if isclose(subcarrier_spacing, 15e3):
            case = 'a'
        elif isclose(subcarrier_spacing, 30e3):
            if shared_spectrum:
                case = 'c'
            else:
                raise ValueError('choose case "b" or "c" for 30 kHz subcarrier spacing')
        elif isclose(subcarrier_spacing, 120e3):
            case = 'd'
        elif isclose(subcarrier_spacing, 240e3):
            case = 'e'
        elif isclose(subcarrier_spacing, 480e3):
            case = 'f'
        elif isclose(subcarrier_spacing, 960e3):
            case = 'g'
        else:
            scs_khz = round(subcarrier_spacing / 1e3)
            raise ValueError(
                f'standard cell search parameters do not exist for SCS {scs_khz} kHz'
            )
    elif symbol_indexes in ('a', 'b', 'c'):
        case = symbol_indexes
    else:
        raise ValueError('symbol_indexes is an invalid str')

    if shared_spectrum and case not in ('a', 'c'):
        raise ValueError(f'shared_spectrum unsupported by cell search case {case}')

    # The standardized cell search cases in 3GPP TS 38.213 §4.1
    if case == 'a':
        offsets = [2, 8]
        mult = 14
        if shared_spectrum:
            nrange = range(5)
        else:
            # the upper 2 candidates are unused below 3 GHz (1.88 GHz in unpaired
            # operation)
            if center_frequency is None or center_frequency > 3e9:
                nrange = range(4)
            else:
                nrange = range(2)
    elif case == 'b':
        offsets = [4, 8, 16, 20]
        mult = 28
        if center_frequency is None or center_frequency > 3e9:
            nrange = range(2)
        else:
            nrange = range(1)
    elif case == 'c':
        offsets = [2, 8]
        mult = 14
        if shared_spectrum:
            nrange = range(10)
        else:
            if center_frequency is None or center_frequency > 1.88e9:
                nrange = range(4)
            else:
                # TS 38.213 §4.1 also assigns paired-spectrum operation between 1.88
                # and 3 GHz to this branch, which the center frequency alone cannot
                # identify
                nrange = range(2)
    # cases D through G are FR2
    elif case == 'd':
        offsets = [4, 8, 16, 20]
        mult = 28
        # TS 38.213 §4.1 skips n = 4, 9, 14 so that L_max = 64
        nrange = [n for n in range(19) if n % 5 != 4]
    elif case == 'e':
        offsets = [8, 12, 16, 20, 32, 36, 40, 44]
        mult = 56
        # TS 38.213 §4.1 skips n = 4 so that L_max = 64
        nrange = [n for n in range(9) if n != 4]
    elif case == 'f' or case == 'g':
        offsets = [2, 9]
        mult = 14
        nrange = range(32)
    else:
        raise TypeError

    inds = []
    for n in nrange:
        for offset in offsets:
            inds.append(offset + mult * n)

    return tuple(inds)


@util.lru_cache()
def pss_params(
    *,
    sample_rate: float = 2 * 7.68e6,
    subcarrier_spacing: float,
    discovery_periodicity: float = 20e-3,
    shared_spectrum: bool = False,
    max_lag_symbols: int | None = None,
    symbol_indexes: CellSSBIndexes = 'auto',
    center_frequency: tuple[float, ...] | float | None = None,
) -> SyncParams:
    if not isroundmod(subcarrier_spacing, 15e3):
        raise ValueError('subcarrier_spacing must be multiple of 15000')

    if isroundmod(sample_rate, 128 * subcarrier_spacing):
        frame_size = round(10e-3 * sample_rate)
    else:
        raise ValueError(
            f'sample_rate must be a multiple of {128 * subcarrier_spacing}'
        )

    symbol_indexes = index_pss_symbols(
        subcarrier_spacing, shared_spectrum, symbol_indexes, center_frequency
    )

    if max_lag_symbols is None:
        ssb_spacing = _min_diff(sorted(symbol_indexes))

        if ssb_spacing is None or ssb_spacing == 4:
            # at least 4 symbols per PSS or SSS
            max_lag_symbols = 4
        elif ssb_spacing >= 5:
            # interference observed in 6th symbol (Case C)
            max_lag_symbols = 5
        else:
            raise ValueError(
                f'symbol_indexes {tuple(symbol_indexes)} are spaced only '
                f'{ssb_spacing} symbols apart, closer than the 4-symbol synchronization '
                'block; pass max_lag_symbols explicitly to search fewer lag symbols'
            )

    slot_count = ceil((symbol_indexes[-1] + max_lag_symbols + 1) / 14)
    slot_duration = slot_period(subcarrier_spacing)
    duration = slot_count * slot_duration
    corr_size = round(duration * sample_rate)
    short_symbol_size = round(slot_duration * sample_rate) // 14

    if isroundmod(discovery_periodicity, 10e-3):
        frames_per_sync = round(discovery_periodicity / 10e-3)
    else:
        raise ValueError('discovery_periodicity must be a multiple of 10e-3')

    phy = get_3gpp_phy(
        1,
        subcarrier_spacing=subcarrier_spacing,
        sample_rate=sample_rate,
        generation='5G',
    )
    min_cp_size = min(phy.cp_sizes)
    cp_offsets = list(np.cumsum([n - min_cp_size for n in phy.cp_sizes]))
    lag_count = short_symbol_size * max_lag_symbols

    return SyncParams(
        min_cp_size=min_cp_size,
        cp_offsets=cp_offsets,
        frame_size=frame_size,
        slot_count=slot_count,
        corr_size=corr_size,
        frames_per_sync=frames_per_sync,
        symbol_indexes=list(symbol_indexes),
        duration=duration,
        short_symbol_size=short_symbol_size,
        lag_count=lag_count,
        sample_rate=sample_rate,
        subcarrier_spacing=subcarrier_spacing,
        discovery_periodicity=discovery_periodicity,
        shared_spectrum=shared_spectrum,
        max_lag_symbols=max_lag_symbols,
    )


@util.lru_cache()
def sss_params(
    *,
    sample_rate: float = 2 * 7.68e6,
    subcarrier_spacing: float,
    discovery_periodicity: float = 20e-3,
    shared_spectrum: bool = False,
    max_lag_symbols: int | None = 2,
    symbol_indexes: CellSSBIndexes = 'auto',
    center_frequency: tuple[float, ...] | float | None = None,
) -> SyncParams:
    # Match PSS except that the symbol indexes are incremented by 2

    if symbol_indexes == 'auto':
        template = pss_params(
            sample_rate=sample_rate,
            subcarrier_spacing=subcarrier_spacing,
            discovery_periodicity=discovery_periodicity,
            shared_spectrum=shared_spectrum,
            max_lag_symbols=max_lag_symbols,
            center_frequency=center_frequency,
        )
        symbol_indexes = tuple(i + 2 for i in template.symbol_indexes)

    return pss_params(
        sample_rate=sample_rate,
        subcarrier_spacing=subcarrier_spacing,
        discovery_periodicity=discovery_periodicity,
        shared_spectrum=shared_spectrum,
        max_lag_symbols=max_lag_symbols,
        symbol_indexes=symbol_indexes,
        center_frequency=center_frequency,
    )


def sync_frame_count(sample_count: int, params: SyncParams) -> int:
    """count the 10 ms frames in a synchronization block of `sample_count` samples.

    Raises:
        ValueError: if `sample_count` is not a whole, nonzero number of frames of
            ``params.frame_size`` samples; if the correlation span ``params.corr_size``
            exceeds a frame; or if trimming the excess cyclic prefix of each slot from
            that span leaves fewer than ``params.lag_count`` correlation samples
    """
    frame_count, remainder = divmod(sample_count, params.frame_size)
    if remainder != 0 or frame_count == 0:
        raise ValueError(
            f'the synchronization block ({sample_count} samples) must hold a whole '
            f'number of 10 ms frames ({params.frame_size} samples each at '
            f'{params.sample_rate!r} S/s)'
        )

    span = (
        f'the {params.slot_count}-slot correlation span ({params.corr_size} samples) '
        f'for symbol_indexes {tuple(params.symbol_indexes)} with '
        f'max_lag_symbols={params.max_lag_symbols}'
    )
    if params.corr_size > params.frame_size:
        raise ValueError(
            f'{span} does not fit in a 10 ms frame ({params.frame_size} samples)'
        )

    # the correlator splits the span into slots and trims the excess cyclic prefix of
    # each slot before the lag search
    per_slot = params.corr_size // params.slot_count - max(params.cp_offsets)
    available = params.slot_count * per_slot
    if available < params.lag_count:
        raise ValueError(
            f'{span} leaves {available} correlation samples, fewer than the '
            f'{params.lag_count} lags searched'
        )

    return frame_count


def get_5g_ssb_spans(
    iq: Array,
    *,
    params: SyncParams,
    fs_in: float,
    frequency_offset: float = 0,
    delay: float = 0,
    max_block_count: int | None = None,
    margin: int | None = None,
) -> Array:
    """return the correlated span of each synchronization block of `iq` at baseband.

    A synchronization block begins every ``params.frames_per_sync`` frames of
    ``params.frame_size`` samples at ``params.sample_rate``, and a span is the first
    ``params.corr_size`` samples of a block, the samples that `correlate_sync_spans`
    reads. Each span is cut from `iq` with `margin` input samples at each end, shifted
    and resampled by FFT in one batch over the ports and blocks, and trimmed to the
    span. The margin absorbs the edge ringing of the FFT resampler. Where the record
    ends inside a margin, the missing samples are zero. The frequency shift is a whole
    number of bins of ``fs_in / (span + 2 * margin)``.

    Args:
        iq: the waveform of shape ``(port, sample)`` at `fs_in`
        params: the synchronization parameters from `pss_params` or `sss_params`,
            with the output sample rate in ``sample_rate``
        fs_in: the sample rate (in S/s) of `iq`
        frequency_offset: the offset (in Hz) of the SSB center from the center of `iq`
        delay: the time (in s) into `iq` at which the first block begins
        max_block_count: the number of blocks kept from `delay`; None keeps every
            block that begins in `iq`
        margin: the input samples kept at each end of a span before the trim; None
            selects one slot, rounded up to a whole number of output samples

    Returns:
        the spans of shape ``(port, sync block, corr_size)`` in the dtype and array
        namespace of `iq`

    Raises:
        ValueError: if a frame or a span is not a whole number of samples at `fs_in`,
            or if `iq` after `delay` is not a whole, nonzero number of frames
    """
    xp = array_namespace(iq)
    fs_out = params.sample_rate
    ratio = Fraction(fs_in / fs_out).limit_denominator(10**6)
    p, q = ratio.numerator, ratio.denominator

    frame_in = Fraction(params.frame_size) * ratio
    corr_in = Fraction(params.corr_size) * ratio
    if frame_in.denominator != 1 or corr_in.denominator != 1:
        raise ValueError(
            f'a frame of {params.frame_size} samples at {fs_out!r} S/s is not a whole '
            f'number of samples at {fs_in!r} S/s'
        )
    frame_in, corr_in = int(frame_in), int(corr_in)
    block_in = frame_in * params.frames_per_sync

    # the margin is an even input count and a whole number of output samples: the
    # trim lands on output samples and `resample` takes an even span
    step = math.lcm(2, p)
    if margin is None:
        margin = round(slot_period(params.subcarrier_spacing) * fs_in)
    margin_in = -(-margin // step) * step
    margin_out = margin_in * q // p
    span_in = corr_in + 2 * margin_in
    span_out = span_in * q // p

    offs = round(delay * fs_in)
    frames, remainder = divmod(iq.shape[-1] - offs, frame_in)
    if remainder != 0 or frames <= 0:
        raise ValueError(
            f'the record ({iq.shape[-1]} samples less a delay of {offs}) must hold a '
            f'whole number of 10 ms frames ({frame_in} samples each at {fs_in!r} S/s)'
        )
    block_count = -(-frames // params.frames_per_sync)
    if max_block_count is not None:
        block_count = min(block_count, max_block_count)

    spans = xp.zeros((iq.shape[0], block_count, span_in), dtype=iq.dtype)
    for block in range(block_count):
        lo = offs + block * block_in - margin_in
        hi = lo + span_in
        src_lo, src_hi = max(lo, 0), min(hi, iq.shape[-1])
        dst_lo = src_lo - lo
        spans[:, block, dst_lo : dst_lo + src_hi - src_lo] = iq[:, src_lo:src_hi]

    shift = round(span_in * frequency_offset / fs_in)
    y = fourier.resamplers.resample(
        spans, num=span_out, axis=2, overwrite_x=True, shift=shift
    )
    return y[..., margin_out : margin_out + params.corr_size]


def _correlation_fft_size(size: int, xp: ModuleType) -> int:
    """return the FFT length at or above `size` for the correlator on `xp`."""
    if array_api_compat.is_cupy_namespace(xp):
        # cuFFT runs a power of 2 faster than the nearest 2·3·5·7-smooth size
        # (32768 against 31360 on a Tegra X2); pocketfft runs the smooth size faster
        return 1 << (size - 1).bit_length()
    return int(util.scipy.fft.next_fast_len(size, real=False))


class _TemplateSpectra:
    """a memo of template spectra keyed on the identity of the template array"""

    MAX_BYTES = 16 * 2**20
    MAX_ENTRIES = 4

    def __init__(self) -> None:
        self._entries: dict[tuple, tuple[Array, Array]] = {}

    def __call__(self, sync_seq: Array, nfft: int) -> Array:
        xp = array_namespace(sync_seq)
        key = (id(sync_seq), sync_seq.shape, nfft, xp.__name__)
        hit = self._entries.get(key)
        if hit is not None and hit[0] is sync_seq:
            return hit[1]

        buf = xp.zeros((sync_seq.shape[0], nfft), dtype=sync_seq.dtype)
        buf[:, : sync_seq.shape[1]] = sync_seq
        spectra = fourier.fft.fft(buf, axis=-1, out=buf, overwrite_x=True)

        if spectra.nbytes <= self.MAX_BYTES:
            while len(self._entries) >= self.MAX_ENTRIES:
                self._entries.pop(next(iter(self._entries)))
            # the array reference keeps the id valid for the life of the entry
            self._entries[key] = (sync_seq, spectra)
        return spectra

    def cache_clear(self) -> None:
        self._entries.clear()


_sync_template_spectra = _TemplateSpectra()
util._caches[_sync_template_spectra] = _sync_template_spectra


def _lag_window_pieces(
    params: SyncParams, head: int, offs: int
) -> list[list[tuple[slice, slice]]]:
    """return the slice pairs that copy each lag window out of the inverse FFT.

    The correlation of one span is the full linear convolution rolled by ``-offs`` and
    cut to ``params.corr_size``, with the excess cyclic-prefix samples of each slot
    removed. A lag window starts at ``symbol * params.short_symbol_size`` in that
    trimmed sequence and runs for ``params.lag_count`` lags. A window splits into a
    new piece at each slot boundary and at the roll boundary within the window.

    Args:
        params: the synchronization parameters from `pss_params` or `sss_params`
        head: the number of lags read from the convolution starting at index `offs`;
            the remaining ``params.corr_size - head`` lags are read starting at index 0
        offs: the index in the convolution of the first lag

    Returns:
        one list per entry of ``params.symbol_indexes``, each a list of ``(dst, src)``
        slice pairs: `dst` indexes the lag axis of the window and `src` the last
        axis of the inverse FFT

    Raises:
        ValueError: if ``params.corr_size`` is not a multiple of ``params.slot_count``,
            or if a lag window ends past the trimmed sequence
    """
    per_slot, rem = divmod(params.corr_size, params.slot_count)
    if rem != 0:
        raise ValueError(
            f'corr_size {params.corr_size} is not a multiple of the slot_count '
            f'{params.slot_count}'
        )
    excess = int(params.cp_offsets[params.symbol_indexes[0] % 14])
    trimmed = per_slot - excess
    span = params.slot_count * trimmed

    pieces = []
    for sym in params.symbol_indexes:
        t0 = sym * params.short_symbol_size
        t1 = t0 + params.lag_count
        if t1 > span:
            raise ValueError(
                f'the lag window of symbol {sym} ends at {t1}, past the {span} '
                'correlation samples that remain after the excess CP is removed'
            )
        window = []
        t = t0
        while t < t1:
            k, i = divmod(t, trimmed)
            n = min(t1, (k + 1) * trimmed) - t
            r = k * per_slot + excess + i
            if r < head:
                n = min(n, head - r)
                src0 = offs + r
            else:
                src0 = r - head
            window.append((slice(t - t0, t - t0 + n), slice(src0, src0 + n)))
            t += n
        pieces.append(window)
    return pieces


def correlate_sync_spans(
    spans: Array,
    sync_seq: Array,
    *,
    params: SyncParams,
    max_chunk_bytes: int = 64 * 2**20,
) -> Array:
    """correlate the span of each synchronization block against synchronization templates.

    The correlation is a linear convolution by FFT. The spectrum of the spans is
    computed once and multiplied by the spectrum of every template, in chunks of cell
    IDs of at most `max_chunk_bytes`. The template spectra are memoized on the
    identity of `sync_seq` in a cache that `clear_caches` empties. `pss_5g_nr` and
    `sss_5g_nr` return one array object per argument set, so the memo applies to the
    templates from either.

    Args:
        spans: the waveform of shape ``(port, sync block, corr_size)`` at
            ``params.sample_rate``, as `get_5g_ssb_spans` returns
        sync_seq: the templates of shape ``(cell id, sample)`` from `pss_5g_nr` or
            `sss_5g_nr` at the same sample rate
        params: the synchronization parameters from `pss_params` or `sss_params`
        max_chunk_bytes: the size of the product buffer that bounds the cell IDs
            correlated at once

    Returns:
        complex64 correlation of shape ``(port, cell id, sync block, beam, lag)``
        with ``params.lag_count`` lags; the last ``params.min_cp_size`` lags are
        zero

    Raises:
        ValueError: if the last axis of `spans` is not ``params.corr_size``, if
            the symbols in ``params.symbol_indexes`` do not share one excess
            cyclic-prefix length, if ``params.corr_size`` is not a multiple of
            ``params.slot_count``, or if a lag window ends past the correlation
            samples that remain after the excess cyclic prefix is removed
    """
    xp = array_namespace(spans)

    ports, sync_count, corr_size = spans.shape
    if corr_size != params.corr_size:
        raise ValueError(
            f'spans hold {corr_size} samples, not the corr_size {params.corr_size} '
            'of params'
        )
    excess_cp = [params.cp_offsets[i % 14] for i in params.symbol_indexes]
    if len(set(excess_cp)) != 1:
        raise ValueError('expect all 5G sync symbols to have the same excess CP')

    cell_count, template_size = sync_seq.shape
    full_size = corr_size + template_size - 1
    nfft = _correlation_fft_size(full_size, xp)

    X = xp.zeros((ports, sync_count, nfft), dtype=spans.dtype)
    X[..., :corr_size] = spans
    X = fourier.fft.fft(X, axis=-1, out=X, overwrite_x=True)
    H = _sync_template_spectra(sync_seq, nfft)

    # the single offset is specific to the 5G NR prefix layout and does not cover
    # 4G LTE
    offs = (
        round(params.sample_rate / params.subcarrier_spacing) + 2 * params.min_cp_size
    )
    # the lag window is the full convolution rolled by -offs and cut to corr_size.
    # The roll wraps the tail of the window onto the first lags of the convolution
    head = min(corr_size, full_size - offs)

    pieces = _lag_window_pieces(params, head, offs)

    # Slices rather than a sliding window view over the lags: that view has one window
    # per sample, above 2**31 elements for 2 ports and 5 blocks at 15.36 MS/s, and
    # cupy 12 returns wrong values for the second port from a fancy index on it
    result = xp.empty(
        (ports, cell_count, sync_count, len(params.symbol_indexes), params.lag_count),
        dtype='complex64',
    )
    chunk = max(1, max_chunk_bytes // (ports * sync_count * nfft * X.dtype.itemsize))
    chunk = min(chunk, cell_count)
    prod = xp.empty((ports, chunk, sync_count, nfft), dtype=X.dtype)
    for start in range(0, cell_count, chunk):
        stop = min(start + chunk, cell_count)
        shape = (ports, stop - start, sync_count, nfft)
        # the FFT requires a C-contiguous buffer, which prod[:, :n] is not for ports > 1
        buf = prod.reshape(-1)[: math.prod(shape)].reshape(shape)
        xp.multiply(X[:, xp.newaxis], H[xp.newaxis, start:stop, xp.newaxis, :], out=buf)
        y = fourier.fft.ifft(buf, axis=-1, out=buf, overwrite_x=True)
        for j, window in enumerate(pieces):
            for dst, src in window:
                result[:, start:stop, :, j, dst] = y[..., src]

    if params.min_cp_size:
        result[..., -params.min_cp_size :] = 0
    return result


def correlate_sync_sequence(
    ssb_iq: Array, sync_seq: Array, *, params: SyncParams, cell_id_split: int | None = 1
) -> Array:
    """correlate the IQ of a synchronization block against synchronization templates.

    The first ``params.corr_size`` samples of every ``params.frames_per_sync`` frames
    are correlated by `correlate_sync_spans`.

    Args:
        ssb_iq: the SSB-band waveform of shape ``(port, sample)`` at
            ``params.sample_rate``, spanning a whole number of 10 ms frames
        sync_seq: the templates of shape ``(cell id, sample)`` from `pss_5g_nr` or
            `sss_5g_nr` at the same sample rate
        params: the synchronization parameters from `pss_params` or `sss_params`
        cell_id_split: ignored

    Returns:
        complex64 correlation of shape ``(port, cell id, sync block, beam, lag)``
        with ``params.lag_count`` lags; the last ``params.min_cp_size`` lags are
        zero

    Raises:
        ValueError: if `sync_frame_count` rejects ``ssb_iq.shape[-1]``, or if the
            symbols in ``params.symbol_indexes`` do not share one excess
            cyclic-prefix length
    """
    sync_frame_count(ssb_iq.shape[-1], params)

    blocks = ssb_iq.reshape((ssb_iq.shape[0], -1, params.frame_size))
    spans = blocks[:, :: params.frames_per_sync, : params.corr_size]
    return correlate_sync_spans(spans, sync_seq, params=params)


COARSE_LAG_DIM = -2
FINE_LAG_DIM = -1


def weighted_ssb_detect(
    rssb: Array,
    params: SyncParams,
    *,
    max_beams: int | None = None,
    per_port: bool = True,
    window: WindowSpecType = 'triang',
    window_fill: float | Fraction = 1,
) -> Array:
    """apply weighted reductions to the SSB correlator output.

    Returns:
        Correlator output power with dims (..., symbol lag index, sample lag index)
    """

    PORT_DIM = -6
    NID2_DIM = -5
    SYNC_DIM = -4
    BEAM_DIM = -3

    xp = arrays.array_namespace(rssb)

    if rssb.ndim == 4:
        # assume a missing dimension is the port index
        rssb = rssb[np.newaxis, ...]
    elif rssb.ndim < 5:
        raise TypeError('input array must have 5 dimensions')
    symbol_count = round(params.lag_count / params.short_symbol_size)
    rssb = rssb.reshape(rssb.shape[:-1] + (symbol_count, -1))

    if max_beams is None or max_beams >= rssb.shape[BEAM_DIM]:
        pass
    else:
        rssb = arrays.axis_slice(rssb, 0, max_beams, axis=BEAM_DIM)

    r = power.envtopow(rssb)

    if is_cupy_array(rssb):
        from cupyx.scipy import ndimage  # type: ignore
    else:
        from scipy import ndimage

    if not per_port:
        r = r.mean(axis=PORT_DIM, keepdims=True)

    # .min on SYNC_DIM would reduce thermal noise but requires close sample clock
    # alignment with the base station
    r = r.mean(axis=(NID2_DIM, BEAM_DIM), keepdims=True)
    r = xp.median(r, axis=SYNC_DIM, keepdims=True)
    rmed = xp.median(r, axis=(FINE_LAG_DIM), keepdims=True)

    # only peaks with at least 3 dB prominence count, which limits spectral bleeding
    # from individual strong sources
    rpeak = r.copy()
    limits = r.max((FINE_LAG_DIM, COARSE_LAG_DIM), keepdims=True)
    threshold = xp.clip(2 * rmed, a_min=limits / 2, a_max=None)
    rpeak[xp.where(rpeak < threshold)] = 0

    # evaluate the sub-symbol IQ offset
    nfine = rpeak.shape[FINE_LAG_DIM]
    # an odd window has a single peak sample, so neighboring lags never tie;
    # even zero padding keeps that peak at the correlate1d origin
    nfill = max(1, 2 * ((round(window_fill * nfine) - 1) // 2) + 1)
    nzero = nfine - nfill
    nzero -= nzero % 2
    w = fourier.windows.get_window(
        window,
        nwindow=nfill,
        nzero=nzero,
        norm=False,
        center_zeros=True,
        fftbins=False,
        xp=xp,
    )

    return ndimage.correlate1d(rpeak, w, mode='wrap', axis=-1).squeeze((
        NID2_DIM,
        SYNC_DIM,
        BEAM_DIM,
    ))


def choose_ssb_offset(
    rssb: Array,
    params: SyncParams,
    *,
    per_port: bool = True,
    max_beams: int | None = None,
    window: WindowSpecType = 'triang',
    window_fill: float | Fraction = 1,
) -> Array:
    """return the SSB start offset (in samples) from the correlator output `rssb`.

    The correlation power is averaged over the cell id, sync block, and beam axes
    and weighted across lag by `window`. The offset is the start of the symbol lag
    with the largest weighted peak plus the fine lag with the largest mean across
    symbol lags.

    Args:
        rssb: the output of `correlate_sync_sequence`
        params: the synchronization parameters that produced `rssb`
        per_port: True returns one offset per port; False averages the ports first
        max_beams: the number of leading beams averaged; None averages all
        window: the window applied across lag in the weighted average
        window_fill: the fraction of one symbol period, 0 to 1, spanned by the
            nonzero part of `window`

    Returns:
        integer offsets (in samples) with shape ``(port,)``, of length 1 when
        `per_port` is False
    """

    offsets = weighted_ssb_detect(**locals())
    nfine = offsets.shape[FINE_LAG_DIM]

    weighted_fine = offsets.mean(axis=COARSE_LAG_DIM)
    ifine = weighted_fine.argmax(-1)

    # evaluate the symbol index offset: the symbol with the largest peak, or 0
    isymbol = nfine * offsets.max(FINE_LAG_DIM).argmax(-1)

    return ifine + isymbol
