"""5G NR synchronization against the synchronization signal block (SSB).

The PSS and SSS sequences follow 3GPP TS 38.211 §7.4.2, and the SSB symbol positions
follow the cell search cases of TS 38.213 §4.1.
"""

from __future__ import annotations as __

import dataclasses
import typing
from fractions import Fraction
from math import ceil, isclose

from .. import arrays, fourier, power, util
from ..arrays import array_namespace, is_cupy_array, isroundmod, pad_along_axis
from ..typing import CellSSBIndexes
from ..util import np
from .phy_3gpp import get_3gpp_phy, slot_period
from .sequences import m_sequence

if typing.TYPE_CHECKING:
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


def get_5g_ssb_iq(
    iq: Array,
    *,
    discovery_periodicity: float,
    fs_out: float,
    fs_in: float,
    subcarrier_spacing: float,
    frequency_offset: float = 0,
    delay: float = 0,
    max_block_count: int | None = None,
    oaresample: bool = False,
) -> Array:
    """return the SSB band of `iq`, recentered to baseband and resampled to `fs_out`.

    Args:
        iq: waveform of shape ``(port, sample)`` at `fs_in`
        discovery_periodicity: the SSB burst period (in s), which sizes the kept
            span with `max_block_count`
        fs_out: the output sample rate (in S/s)
        fs_in: the sample rate (in S/s) of `iq`
        subcarrier_spacing: the SSB subcarrier spacing (in Hz)
        frequency_offset: the offset (in Hz) of the SSB center from the center of
            `iq`
        delay: the time (in s) into `iq` at which the kept span begins
        max_block_count: the number of SSB periods kept from `delay`; None keeps
            the rest of `iq`
        oaresample: True resamples by overlap-add with a Blackman window; False
            resamples by a single FFT

    Returns:
        the baseband SSB waveform of shape ``(port, sample)`` at `fs_out`, in the
        dtype and array namespace of `iq`
    """

    xp = arrays.array_namespace(iq)

    offs = round(delay * fs_in)

    if oaresample:
        down = round(fs_in / subcarrier_spacing / 8)
        up = round(down * (fs_out / fs_in))

        if up % 3 > 0:
            # ensure compatibility with the blackman window overlap of 2/3
            down = down * 3
            up = up * 3

        if max_block_count is not None:
            size_in = round(max_block_count * discovery_periodicity * fs_in)
            iq = iq[..., offs : offs + size_in]
        else:
            size_in = iq.shape[-1]

        size_out = round(up / down * size_in)

        out = xp.empty((iq.shape[0], size_out), dtype=iq.dtype)

        for i in range(out.shape[0]):
            out[i] = fourier.resamplers.oaresample(
                iq[i],
                fs=fs_in,
                up=up,
                down=down,
                axis=0,
                window='blackman',
                frequency_shift=frequency_offset,
            )

    else:
        if max_block_count is not None:
            size_in = round(max_block_count * discovery_periodicity * fs_in)
            iq = iq[..., offs : offs + size_in]
        else:
            size_in = iq.shape[-1]

        size_out = round(size_in * fs_out / fs_in)
        out = xp.empty((iq.shape[0], size_out), dtype=iq.dtype)
        shift = round(iq.shape[1] * frequency_offset / fs_in)

        for i in range(out.shape[0]):
            out[i] = fourier.resamplers.resample(
                iq[i], num=size_out, axis=0, overwrite_x=False, shift=shift
            )

    return out


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


def correlate_sync_sequence(
    ssb_iq: Array, sync_seq: Array, *, params: SyncParams, cell_id_split: int | None = 1
) -> Array:
    """correlate the IQ of a synchronization block against synchronization templates.

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
    xp = array_namespace(ssb_iq)

    sync_frame_count(ssb_iq.shape[-1], params)

    slot_count = params.slot_count
    corr_size = params.corr_size
    frames_per_sync = params.frames_per_sync

    # set up broadcasting on dimensions:
    # (port index, cell Nid, sync block index, IQ sample index)
    iq_bcast = ssb_iq.reshape((ssb_iq.shape[0], -1, params.frame_size))
    iq_bcast = iq_bcast[:, xp.newaxis, ::frames_per_sync, :corr_size]

    template_bcast = sync_seq[xp.newaxis, :, xp.newaxis, :]
    # pad_size = template_bcast.shape[-1]
    # template_bcast  = iqwaveform.util.pad_along_axis(template_bcast, [[0,pad_size]], axis=3)

    # the single offset is specific to the 5G NR prefix layout and does not cover
    # 4G LTE
    offs = (
        round(params.sample_rate / params.subcarrier_spacing) + 2 * params.min_cp_size
    )

    R_shape = list(max(a, b) for a, b in zip(iq_bcast.shape, template_bcast.shape))
    R_shape[-1] = iq_bcast.shape[-1] + template_bcast.shape[-1] - 1
    R = xp.empty(tuple(R_shape), dtype='complex64')

    for cell_id in range(template_bcast.shape[1]):
        R[:, cell_id] = fourier.filters.oaconvolve(
            iq_bcast[:, 0], template_bcast[:, cell_id], axes=2, mode='full'
        )
    R = xp.roll(R, -offs, axis=-1)[..., :corr_size]

    # add slot index dimension: -> (port index, cell Nid, sync block index, slot index, IQ sample index)
    excess_cp = [params.cp_offsets[i % 14] for i in params.symbol_indexes]
    if len(set(excess_cp)) != 1:
        raise ValueError('expect all 5G sync symbols to have the same excess CP')

    Rperslot = R.reshape(R.shape[:-1] + (slot_count, -1))[..., excess_cp[0] :]
    R = Rperslot.reshape(R.shape[:-1] + (-1,))

    # final dims (port index, cell Nid, sync block index, beam index, IQ sample index)
    start_inds = xp.array(params.symbol_indexes) * params.short_symbol_size

    Rwins = arrays.sliding_window_view(R, params.lag_count, axis=-1)
    result = Rwins[..., start_inds, :]
    if params.min_cp_size:
        result[..., -params.min_cp_size :] = 0
    return result


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
