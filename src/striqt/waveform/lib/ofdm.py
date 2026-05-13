from __future__ import annotations as __

import dataclasses
from fractions import Fraction
from math import ceil, isclose
from numbers import Number
import typing

from . import fourier

from . import arrays, power_analysis, util
from .typing import CellSSBIndexes
from .arrays import array_namespace, is_cupy_array, isroundmod, pad_along_axis

if typing.TYPE_CHECKING:
    import array_api_compat
    import numpy as np

    from .typing import Array, WindowSpecType
else:
    np = util.lazy_import('numpy')
    array_api_compat = util.lazy_import('array_api_compat')


def _min_diff(x: typing.Sequence[int]) -> int | None:
    """return the minimum difference between neighbors in x"""

    if len(x) <= 1:
        return None

    return min(b - a for a, b in zip(x, x[1:]))


def _isclosetoint(v, atol=1e-6):
    xp = array_namespace(v)
    return xp.isclose(v % 1, (0, 1), atol=atol).any()


def correlate_along_axis(a, b, axis=0):
    """cross-correlate `a` and `b` along the specified axis.
    this implementation is optimized for small sequences to replace for
    loop across scipy.signal.correlate.
    """
    xp = array_namespace(a)
    if axis == 0:
        # xp.vdot conjugates b for us
        return xp.array([xp.vdot(a[:, i], b[:, i]) for i in range(a.shape[1])])
    else:
        return xp.array([xp.vdot(a[i], b[i]) for i in range(a.shape[0])])


def indexsum2d(ix, iy):
    """take 2 1-D arrays of shape (M,) and (N,) and return a
    2-D array of shape (M,N) with elements (m,n) equal to ix[m,:] + iy[:,n]
    """
    return ix[:, np.newaxis] + iy[np.newaxis, :]


def call_by_block(func, x, size, *args, **kws):
    """repeatedly call `func` on the 1d array `x`, with arguments and keyword arguments args, and kws,
    and concatenate the result
    """
    xp = array_namespace(x)

    out_chunks = []
    input_chunks = xp.split(x, xp.mgrid[: x.size : size][1:])

    if len(input_chunks[-1]) != len(input_chunks[0]):
        input_chunks = input_chunks[:-1]
    for i, chunk in enumerate(input_chunks):
        out_chunks.append(func(chunk, *args, **kws))

    return xp.concatenate(out_chunks)


def subsample_shift(x, shift):
    """FFT-based subsample shift in x"""
    xp = array_namespace(x)

    N = len(x)

    f = xp.fft.fftshift(xp.arange(x.size))
    z = xp.exp((-2j * np.pi * shift / N) * f)
    return xp.fft.ifft(xp.fft.fft(x) * z)


def to_blocks(y, size, truncate=False):
    size = int(size)
    if not truncate and y.shape[-1] % size != 0:
        raise ValueError(
            'last axis size {} is not integer multiple of block size {}'.format(
                y.shape[-1], size
            )
        )

    new_size = size * (y.shape[-1] // size)
    new_shape = y.shape[:-1] + (y.shape[-1] // size, size)

    return y[..., :new_size].reshape(new_shape)


def _index_or_all(inds: tuple[int, ...] | typing.Literal['all'], name, size, xp=None):
    if xp is None:
        xp = np

    if isinstance(inds, (tuple, list)):
        x = xp.array(inds)
    elif isinstance(inds, str) and inds == 'all':
        if size is None:
            raise ValueError('must set max to allow "all" value')
        x = xp.arange(size)
    else:
        raise ValueError(f'{name} argument must be an array of indices or "all"')

    if x.ndim not in (0, 1):
        raise ValueError(f'{name} argument must be a sequence of indices')

    if x.max() > size:
        raise ValueError(f'{name} value {inds} exceeds the maximum {size}')
    if (-x).max() > size:
        raise ValueError(f'{name} value {inds} is below the minimum {-size}')

    return x


def corr_at_indices(inds, x, nfft, norm=True, out=None):
    xp = array_namespace(x)
    assert xp is array_namespace(inds)

    # the number of waveform samples per cyclic prefix
    ncp = inds.shape[-1]
    flat_inds = inds.flatten()

    if out is None:
        out = xp.empty(nfft + ncp, dtype=x.dtype)

    if array_api_compat.is_numpy_array(x):
        from .jit.cpu import _corr_at_indices as func
    else:
        from .jit.cuda import _corr_at_indices

        tpb = 32
        bpg = max((x.size + (tpb - 1)) // tpb, 1)

        func = _corr_at_indices[bpg, tpb]  # pyright: ignore

    func(flat_inds, x, int(nfft), int(ncp), bool(norm), out)

    return out


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
    """compute the M-sequence used as the 5G-NR primary synchronization sequence.

    These express frequency-domain values of the active subcarriers, spaced at the
    subcarrier spacing.

    Args:
        N_id2: one of (0,1,2), expressing the sector portion of the cell ID
    """
    x = [0, 1, 1, 0, 1, 1, 1]

    for i in range(7, 127):
        x.append((x[i - 3] + x[i - 7]) % 2)

    m = [(n + 43 * N_id2) % 127 for n in range(127)]

    pss = [(1 - 2 * x[_m]) for _m in m]

    return pss


def _sss_m_sequence(N_id: int) -> list[int]:
    """compute the M-sequence used as the 5G-NR secondary synchronization sequence.

    These express frequency-domain values of the active subcarriers at the
    channel's subcarrier spacing.

    The cell ID is $N_\text{id} = 3 N_{id}^1 + N_{id}^2$, where $N_{id}^1$
    is the gnode-b ID and the $N_{id}^2$ is the sector ID as in _pss_m_sequence.

    Args:
        N_id: the cell ID in range(1008)

    Returns:
        a list of integers in the set {-1, 1} with length 127
    """

    x_0 = [1, 0, 0, 0, 0, 0, 0]
    x_1 = [1, 0, 0, 0, 0, 0, 0]

    N_id_1 = N_id // 3
    N_id_2 = N_id % 3

    for i in range(7, 127):
        x_0.append((x_0[i - 3] + x_0[i - 7]) % 2)
        x_1.append((x_1[i - 6] + x_1[i - 7]) % 2)

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
    """returns the time-domain PSS sequence.

    The shortest cyclic prefix is prepended.
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
        # check frequency bounds later via pad_*
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
    x = fourier.ifft(seq_freq, axis=1, out=seq_freq)

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
    """compute the PSS correlation sequences at the given sample rate for each N_id2.

    The sequence can be convolved with an IQ waveform of the same sample rate
    along the last axis to compute a synchronization correlation sequence. The
    result would be normalized to the IQ input power.

    Args:
        sample_rate: the desired output sample rate (in S/s), a multiple of subcarrier_spacing and at least (127*subcarrier_spacing)
        subcarrier_spacing: the subcarrier spacing (in Hz), a multiple of 15e3

    Returns:
        xp.ndarray with dimensions (N_id2 index, PSS sample index)
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
    """compute the PSS correlation sequences at the given sample rate for each N_id2.

    The sequence can be convolved with an IQ waveform of the same sample rate
    along the last axis to compute a synchronization correlation sequence. The
    result would be normalized to the IQ input power.

    Args:
        sample_rate: the desired output sample rate (in S/s), a multiple of subcarrier_spacing and at least (127*subcarrier_spacing)
        subcarrier_spacing: the subcarrier spacing (in Hz), a multiple of 15e3

    Returns:
        xp.ndarray with dimensions (cell ID index, sync sample index)
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


def _index_pss_symbols(
    subcarrier_spacing: float,
    shared_spectrum: bool = False,
    symbol_indexes: CellSSBIndexes = 'auto',
    center_frequency: float | None = None,
) -> tuple[int, ...]:
    """returns indexes of PSS symbols relative to frame start.

    Optionally, symbol_indexes may be determined by specifying one of the
    standardized cell search cases in 3GPP TS 138 213: Section 4.1. If
    symbol_indexes is 'auto', then a case
    """
    if isinstance(symbol_indexes, str):
        symbol_indexes = symbol_indexes.lower()  # type: ignore
    elif isinstance(symbol_indexes, (tuple, list)):
        return symbol_indexes
    else:
        raise TypeError('symbol_index has valid type')

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

    # The standardized cell search cases in 3GPP TS 138 213: Section 4.1
    if case == 'a':
        offsets = [2, 8]
        mult = 14
        if shared_spectrum:
            nrange = range(5)
        else:
            # for center frequencies < 3 GHz (or 1.88 GHz in unpaired operation)
            # the upper 2 can be ignored
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
                # NOTE: In theory, TDD between 1.88 - 3 GHz should also go here
                nrange = range(2)
    # Below here FR-2
    elif case == 'd':
        offsets = [4, 8, 16, 20]
        mult = 28
        nrange = range(19)
    elif case == 'e':
        offsets = [8, 12, 16, 20, 32, 36, 40, 44]
        mult = 56
        nrange = range(9)
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
    center_frequency: float | None = None,
) -> SyncParams:
    if not isroundmod(subcarrier_spacing, 15e3):
        raise ValueError('subcarrier_spacing must be multiple of 15000')

    if isroundmod(sample_rate, 128 * subcarrier_spacing):
        frame_size = round(10e-3 * sample_rate)
    else:
        raise ValueError(
            f'sample_rate must be a multiple of {128 * subcarrier_spacing}'
        )

    symbol_indexes = _index_pss_symbols(
        subcarrier_spacing, shared_spectrum, symbol_indexes, center_frequency
    )

    if max_lag_symbols is None:
        # 4 === minimum possible separation between any 2 PSS or SSS symbols
        max_lag_symbols = _min_diff(sorted(symbol_indexes)) or 4

    slot_count = ceil((symbol_indexes[-1] + max_lag_symbols + 1) / 14)
    slot_duration = 10e-3 / (10 * subcarrier_spacing / 15e3)
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
    center_frequency: float | None = None,
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
    """return a sync block waveform, which returns IQ that is recentered
    at baseband frequency spec.frequency_offset and downsampled to spec.sample_rate."""

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
            out[i] = fourier.oaresample(
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
            out[i] = fourier.resample(
                iq[i], num=size_out, axis=0, overwrite_x=False, shift=shift
            )

    return out


def correlate_sync_sequence(
    ssb_iq: Array, sync_seq: Array, *, params: SyncParams, cell_id_split: int | None = 1
) -> Array:
    """correlate the IQ of a synchronization block against a synchronization sequence.

    Arguments:
        ssb_iq: The synchronization block IQ waveform (e.g., from `get_5g_ssb_iq`)
        sync_seq: The reference sequence (e.g. from `striqt.waveform.ofdm.pss_5g_nr` or `striqt.waveform.ofdm.sss_5g_nr`)
        params: The cell synchronization parameters (e.g. from `striqt.waveform.ofdm.pss_params` or `striqt.waveform.ofdm.sss_params`)
    """
    xp = array_namespace(ssb_iq)

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

    # TODO: this would need to support multiple different prefixes for 4G LTE
    offs = (
        round(params.sample_rate / params.subcarrier_spacing) + 2 * params.min_cp_size
    )

    R_shape = list(max(a, b) for a, b in zip(iq_bcast.shape, template_bcast.shape))
    R_shape[-1] = iq_bcast.shape[-1] + template_bcast.shape[-1] - 1
    R = xp.empty(tuple(R_shape), dtype='complex64')

    for cell_id in range(template_bcast.shape[1]):
        R[:, cell_id] = fourier.oaconvolve(
            iq_bcast[:, 0], template_bcast[:, cell_id], axes=2, mode='full'
        )
    R = xp.roll(R, -offs, axis=-1)[..., :corr_size]
    R = R[..., :corr_size]

    # add slot index dimension: -> (port index, cell Nid, sync block index, slot index, IQ sample index)
    excess_cp = [params.cp_offsets[i % 14] for i in params.symbol_indexes]
    if len(set(excess_cp)) != 1:
        raise ValueError('expect all 5G sync symbols to have the same excess CP')

    Rperslot = R.reshape(R.shape[:-1] + (slot_count, -1))[..., excess_cp[0] :]
    R = Rperslot.reshape(R.shape[:-1] + (-1,))

    # take the desired symbol indexes from a sliding window views on the correlation sequence
    # final dims (port index, cell Nid, sync block index, beam index, IQ sample index)
    start_inds = xp.array(params.symbol_indexes) * params.short_symbol_size

    Rwins = arrays.sliding_window_view(R, params.lag_count, axis=-1)
    result = Rwins[..., start_inds, :]
    if params.min_cp_size:
        result[..., -params.min_cp_size :] = 0
    return result


def choose_ssb_offset(
    rssb: Array,
    params: SyncParams,
    *,
    per_port: bool = True,
    window: WindowSpecType = 'triang',
    window_fill: float | Fraction = 1,
) -> Array:
    """Given correlator output, apply averaging reductions to {NID2, SSB, beam} indexes
    and a weighting function to the lag.

    The approach is to use a weighted average of nearby peaks across lag
    in order to reduce mis-alignment errors in measurements of aggregate interference.

    The underlying heuristic is a triangular weighting function to include energy
    within +/- 1/4 symbol of each peak. Outside of this range, spectrogram errors
    due to "ISI" begin to increase quickly.

    Args:
        rssb: output from
        window: the window function to use for weighting
        per_port: whether to return a weighted correlation separately per port
        window_fill: the fraction of the window to apply

    Returns:
        an array with dimensions (port index, symbol lag, iq lag)
    """
    # input dims: (port index, cell Nid, sync block index, beam index, IQ sample index)
    # transform to these dimensions
    PORT_DIM = -6
    NID2_DIM = -5
    SYNC_DIM = -4
    BEAM_DIM = -3
    COARSE_LAG_DIM = -2
    FINE_LAG_DIM = -1

    xp = arrays.array_namespace(rssb)

    if rssb.ndim == 4:
        # assume a missing dimension is the port index
        rssb = rssb[np.newaxis, ...]
    elif rssb.ndim != 5:
        raise TypeError('input array must have 5 dimensions')
    symbol_count = round(params.lag_count / params.short_symbol_size)
    rssb = rssb.reshape(rssb.shape[:-1] + (symbol_count, -1))

    r = power_analysis.envtopow(rssb)

    if is_cupy_array(rssb):
        from cupyx.scipy import ndimage  # type: ignore
    else:
        from scipy import ndimage

    if not per_port:
        r = r.mean(axis=PORT_DIM, keepdims=True)

    r = r.mean(axis=(NID2_DIM, SYNC_DIM, BEAM_DIM))
    rmed = np.median(r, axis=(FINE_LAG_DIM), keepdims=True)
    assert r.ndim == 3

    # to avoid spectral bleeding from individual strong sources,
    # consider only obvious peaks with at least 3 dB prominence
    rpeak = r.copy()
    threshold = np.clip(2 * rmed, min=r.max() / 2)
    rpeak[np.where(rpeak < threshold)] = 0

    # evaluate the sub-symbol IQ offset
    nfill = round(window_fill * rpeak.shape[FINE_LAG_DIM])
    nfine = rpeak.shape[FINE_LAG_DIM]
    w = fourier.get_window(
        window,
        nwindow=nfill,
        nzero=nfine - nfill,
        norm=False,
        center_zeros=True,
        fftbins=False,
        xp=xp,
    )
    offsets = ndimage.correlate1d(rpeak, w, mode='wrap', axis=-1)
    weighted_fine = offsets.mean(axis=COARSE_LAG_DIM)
    ifine = weighted_fine.argmax(-1)

    # evaluate the symbol index offset
    isymbol = nfine * offsets.max(FINE_LAG_DIM).argmax(-1)  # the peakiest symbol or 0

    return ifine + isymbol


class PhyOFDM:
    def __init__(
        self,
        *,
        channel_bandwidth: float,
        sample_rate: float,
        nfft: float,
        cp_sizes: Array,
        frame_duration: float | None = None,
        contiguous_size: int | None = None,
    ):
        xp = array_namespace(cp_sizes)

        self.channel_bandwidth = channel_bandwidth
        self.sample_rate = sample_rate

        self.nfft: float = nfft
        self.frame_duration = frame_duration

        self.subcarrier_spacing: float = self.sample_rate / nfft
        if frame_duration is None:
            self.frame_size = None
        else:
            self.frame_size = round(sample_rate * frame_duration)

        self.cp_sizes = cp_sizes

        if contiguous_size is not None:
            self.contiguous_size = contiguous_size
        else:
            # if not specified, assume no padding is needed to complete a contiguos block of symbols
            self.contiguous_size = int(np.sum(cp_sizes) + len(cp_sizes) * nfft)

        # build a (start_idx, size) pair for each CP
        pair_sizes = xp.concatenate((xp.array((0,)), self.cp_sizes + self.nfft))
        self.cp_start_idx = (pair_sizes.cumsum()).astype(int)[:-1]
        start_and_size = zip(self.cp_start_idx, self.cp_sizes)

        idx_range = xp.arange(self.contiguous_size).astype(int)

        # indices in the contiguous range that are CP
        self.cp_idx = xp.concatenate([
            idx_range[start : start + size] for start, size in start_and_size
        ])

        # indices in the contiguous range that are not CP
        self.symbol_idx = np.setdiff1d(idx_range, self.cp_idx)

    def index_cyclic_prefix(self) -> Array:
        raise NotImplementedError


@util.lru_cache(4)
def get_3gpp_phy(
    channel_bandwidth: float,
    subcarrier_spacing=15e3,
    generation: typing.Literal['4G', '5G'] = '4G',
    sample_rate=None,
    xp=None,
) -> Phy3GPP:
    return Phy3GPP(**locals())


@util.lru_cache(4)
def _get_3gpp_index_cyclic_prefix(
    phy: Phy3GPP,
    *,
    frames: tuple[int, ...] = (0,),
    symbols: tuple[int, ...] | typing.Literal['all'] = 'all',
    slots: tuple[int, ...] | typing.Literal['all'] = 'all',
):
    """build an indexing tensor for performing cyclic prefix correlation across various axes"""
    xp = array_namespace(phy.cp_sizes)

    frames = xp.array(frames)
    frame_size = round(phy.sample_rate * 10e-3)

    slot_inds = _index_or_all(
        slots,
        '"slots" argument',
        size=phy.SCS_TO_SLOTS_PER_FRAME[phy.subcarrier_spacing],
        xp=xp,
    )
    symbol_inds = _index_or_all(
        symbols, '"symbols" argument', size=phy.FFT_PER_SLOT, xp=xp
    )

    # first build each grid axis separately
    grid = []

    # axis 0: symbol number within each slot
    grid.append(phy.cp_start_idx[symbol_inds])

    # axis 1: slot number
    grid.append(phy.contiguous_size * slot_inds)

    # axis 2: frame number
    grid.append(frames * frame_size)

    # axis 3: cp index
    grid.append(xp.ogrid[0 : phy.cp_sizes[1]])

    grid = [x.squeeze() for x in grid if x.size > 1]
    # pad the axis dimensions so they can be broadcast together
    inds, *offsets = xp.meshgrid(*grid, indexing='ij', copy=False)

    # sum all of the index offsets
    inds = inds.copy()
    for offset in offsets:
        inds += offset

    return inds


class Phy3GPP(PhyOFDM):
    """Sampling and index parameters and lookup tables for 3GPP 5G-NR.

    These are equivalent to LTE if subcarrier_spacing is fixed to 15 kHz
    and slot length is redefined to match the period of 14 symbols including
    cyclic prefix.

    References:
        3GPP TS 38.211.
    """

    # the remaining 1 "slot" worth of samples per slot are for cyclic prefixes
    FFT_PER_SLOT = 14
    SUBFRAMES_PER_PRB = 12

    FFT_SIZE_TO_SUBCARRIERS = {
        128: 73,
        256: 181,
        512: 301,
        1024: 601,
        1536: 901,
        2048: 1201,
    }

    # "default" sample rates from LTE
    BW_TO_SAMPLE_RATE = {
        1.4e6: 1.92e6,
        3e6: 3.84e6,
        5e6: 7.68e6,
        10e6: 15.36e6,
        15e6: 23.04e6,
        20e6: 30.72e6,
        25e6: 38.40e6,
        30e6: 46.08e6,
        40e6: 61.44e6,
        60e6: 92.16e6,
        80e6: 122.88e6,
        100e6: 153.6e6,
    }

    # Slot structure including cyclic prefix (CP) indices are specified in
    # 3GPP TS 38.211, Section 5.3.1. Below are the sizes of all CPs (in samples)
    # in 1 slot for FFT size 128 at 15 kHz SCS. CP size then scales proportionally
    # to FFT size. 1 slot is the minimum number of contiguous symbols in a sequence.
    LTE_MIN_CP_SIZES = (10, 9, 9, 9, 9, 9, 9, 10, 9, 9, 9, 9, 9, 9)

    SCS_TO_SLOTS_PER_FRAME = {15e3: 10, 30e3: 20, 60e3: 40}

    # TODO: add 5G FR2 SCS values
    SUBCARRIER_SPACINGS = {15e3, 30e3, 60e3}

    def __init__(
        self,
        channel_bandwidth: float,
        subcarrier_spacing=15e3,
        generation: typing.Literal['4G', '5G'] = '4G',
        sample_rate=None,
        xp=None,
    ):
        if xp is None:
            xp = np

        if subcarrier_spacing not in self.SUBCARRIER_SPACINGS:
            raise ValueError(
                f'subcarrier_spacing must be one of {self.SUBCARRIER_SPACINGS}'
            )

        if sample_rate is None:
            sample_rate = self.BW_TO_SAMPLE_RATE[channel_bandwidth]
        else:
            sample_rate = sample_rate

        if isroundmod(sample_rate, subcarrier_spacing):
            nfft = round(sample_rate / subcarrier_spacing)
        else:
            raise ValueError('sample_rate / subcarrier_spacing must be counting number')

        if nfft in self.FFT_SIZE_TO_SUBCARRIERS:
            self.subcarriers = self.FFT_SIZE_TO_SUBCARRIERS[nfft]

        if generation.upper() == '4G':
            # assert subcarrier_spacing == 15e3, '4G LTE only supports 15 kHz subcarrier spacing'
            cp_sizes = nfft * xp.array(self.LTE_MIN_CP_SIZES, dtype=int) // 128
        elif generation.upper() == '5G':
            Tc = Fraction(1, 4096 * 480) * 1000  # microsec
            kappa = 64
            fs_MHz = Fraction(round(sample_rate), 1_000_000)

            scs_norm = Fraction(round(subcarrier_spacing), 15000)
            Tcp_us = [kappa * Tc * (144 / scs_norm) + 16 * kappa * Tc] + 13 * [
                kappa * Tc * (144 / scs_norm)
            ]
            cp_fractions = [T * fs_MHz for T in Tcp_us]
            if any(cp.denominator != 1 for cp in cp_fractions):
                raise ValueError(
                    'this {sample rate, subcarrier spacing} produces '
                    'non-integer cyclic prefixes'
                )
            cp_sizes = xp.array([cp.numerator for cp in cp_fractions], dtype=int)
        else:
            raise ValueError('generation must be "4G" or "5G"')

        super().__init__(
            channel_bandwidth=channel_bandwidth,
            nfft=nfft,
            sample_rate=sample_rate,
            frame_duration=10e-3,
            cp_sizes=cp_sizes,
        )

    def index_cyclic_prefix(
        self,
        *,
        frames: tuple[int, ...] = (0,),
        symbols: tuple[int, ...] | typing.Literal['all'] = 'all',
        slots: tuple[int, ...] | typing.Literal['all'] = 'all',
    ):
        """build an indexing tensor for performing cyclic prefix correlation across various axes"""
        return _get_3gpp_index_cyclic_prefix(
            self, frames=frames, symbols=symbols, slots=slots
        )


@util.lru_cache(4)
def get_802_16_phy(
    channel_bandwidth: float,
    *,
    alt_sample_rate: float | None = None,
    frame_duration: float = 5e-3,
    nfft: float = 2048,
    cp_ratio: float = 1 / 8,
    xp=None,
) -> Phy802_16:
    return Phy802_16(**locals())


class Phy802_16(PhyOFDM):
    """Sampling and index parameters and lookup tables for IEEE 802.16-2017 OFDMA"""

    VALID_CP_RATIOS = {1 / 32, 1 / 16, 1 / 8, 1 / 4}
    VALID_FFT_SIZES = {128, 512, 1024, 2048}
    VALID_FRAME_DURATIONS = {
        2e-3,
        2.5e-3,
        4e-3,
        5e-3,
        8e-3,
        10e-3,
        12.5e-3,
        20e-3,
        25e-3,
        40e-3,
        50e-3,
    }

    SAMPLING_FACTOR_BY_FREQUENCY_DIV = {
        1.25: 28 / 25,
        1.5: 28 / 25,
        1.75e6: 8 / 7,
        2: 28 / 25,
        2.75: 28 / 25,
    }

    def __init__(
        self,
        channel_bandwidth: float,
        *,
        alt_sample_rate: float | None = None,
        frame_duration: float = 5e-3,
        nfft: float = 2048,
        cp_ratio: float = 1 / 8,
        xp=None,
    ):
        """
        Args:
            channel_bandwidth: Channel bandwidth as defined by 802.16-2017
            alt_sample_rate (_type_, optional): If specified, overrides the 802.16-2017 value with sample rate of recorded data.
            frame_duration: _description_. Defaults to 5e-3.
            nfft: the fft size corresponding to the length of the useful portion of each symbol. Defaults to 2048.
            cp_ratio: the size of the cyclic prefix as a fraction of nfft. Defaults to 1/8.
        """
        if xp is None:
            xp = np

        if not isinstance(channel_bandwidth, Number):
            raise TypeError('expected numeric value for channel_bandwidth')
        elif channel_bandwidth < 1.25e6:
            raise ValueError(
                'standardized values for channel_bandwidth not supported yet'
            )
        elif not isclose(channel_bandwidth % 125e3, 0, abs_tol=1e-6):
            raise ValueError('channel bandwidth must be set in increments of 125 kHz')

        if nfft not in self.VALID_FFT_SIZES:
            raise ValueError(f'nfft must be one of {self.VALID_FFT_SIZES}')

        if cp_ratio in self.VALID_CP_RATIOS:
            self.cp_ratio = cp_ratio
        else:
            raise ValueError(f'cp_ratio must be one of {self.VALID_CP_RATIOS}')

        if frame_duration not in self.VALID_FRAME_DURATIONS:
            raise ValueError(
                f'frame_duration must be one of {self.VALID_FRAME_DURATIONS}'
            )

        for freq_divisor, n in self.SAMPLING_FACTOR_BY_FREQUENCY_DIV.items():
            if isclose(channel_bandwidth % freq_divisor, 0, abs_tol=1e-6):
                sampling_factor = self.sampling_factor = n
                break
        else:
            # no match with the table - standardized default
            sampling_factor = self.sampling_factor = 8 / 7

        std_sample_rate = np.floor(sampling_factor * channel_bandwidth / 8000) * 8000
        cp_size = int(np.rint(cp_ratio * nfft))
        self.total_symbol_duration = (
            int(np.rint((1 + cp_ratio) * nfft)) / std_sample_rate
        )
        self.symbols_per_frame = int(
            np.floor(frame_duration / self.total_symbol_duration)
        )

        if alt_sample_rate is None:
            sample_rate = std_sample_rate
        else:
            scale = alt_sample_rate / std_sample_rate

            if not (_isclosetoint(scale) or _isclosetoint(1 / scale)):
                raise ValueError(
                    'alt_sample_rate must be integer multiple or divisor of ofdm sample_rate'
                )
            if not _isclosetoint(cp_size * scale):
                raise ValueError(
                    'alt_sample_rate is too small to capture any cyclic prefixes'
                )

            nfft = round(nfft * scale)
            cp_size = round(cp_size * scale)
            sample_rate = alt_sample_rate

        super().__init__(
            channel_bandwidth=channel_bandwidth,
            nfft=nfft,
            sample_rate=sample_rate,
            frame_duration=frame_duration,
            cp_sizes=xp.full(self.symbols_per_frame, cp_size),
            contiguous_size=round(frame_duration * sample_rate),
        )

    def index_cyclic_prefix(
        self, *, frames=(0,), symbols: tuple[int, ...] | typing.Literal['all'] = 'all'
    ) -> Array:
        """build an indexing tensor for performing cyclic prefix correlation across various axes"""
        return _802_16_index_cyclic_prefix(self, frames=frames, symbols=symbols)


@util.lru_cache(4)
def _802_16_index_cyclic_prefix(
    phy: Phy802_16,
    *,
    frames=(0,),
    symbols: tuple[int, ...] | typing.Literal['all'] = 'all',
) -> Array:
    """build an indexing tensor for performing cyclic prefix correlation across various axes"""

    xp = array_namespace(phy.cp_sizes)
    frames = xp.array(frames)

    symbol_inds = _index_or_all(
        symbols, '"symbols" argument', size=phy.symbols_per_frame, xp=xp
    )

    # first build each grid axis separately
    grid = []

    # axis 0: symbol number in each frame
    grid.append(phy.cp_start_idx[symbol_inds])

    # axis 1: frame number
    grid.append(frames * phy.frame_size)

    # axis 2: cp index
    grid.append(xp.ogrid[0 : phy.cp_sizes[1]])

    # pad the axis dimensions so they can be broadcast together
    a = xp.meshgrid(*grid, indexing='ij', copy=False)

    # sum all of the index offsets
    inds = a[0].copy()
    for sub in a[1:]:
        inds += sub

    return inds
