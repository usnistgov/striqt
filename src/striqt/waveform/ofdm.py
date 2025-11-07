from __future__ import annotations
from datetime import datetime
from math import ceil
from numbers import Number
import typing

from . import fourier
from .util import lru_cache, array_namespace, isroundmod, pad_along_axis
from ._typing import ArrayType

import array_api_compat
import numpy as np
import methodtools


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


def _index_or_all(x, name, size, xp=np):
    if isinstance(x, str) and x == 'all':
        if size is None:
            raise ValueError('must set max to allow "all" value')
        x = xp.arange(size)
    elif xp.ndim(x) in (0, 1):
        x = xp.array(x)
    else:
        raise ValueError(f'{name} argument must be a flat array of indices or "all"')

    if xp.max(x) > size:
        raise ValueError(f'{name} value {x} exceeds the maximum {size}')
    if xp.max(-x) > size:
        raise ValueError(f'{name} value {x} is below the minimum {-size}')

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
        from ._jit.cpu import _corr_at_indices
    else:
        from ._jit.cuda import _corr_at_indices

        tpb = 32
        bpg = max((x.size + (tpb - 1)) // tpb, 1)

        _corr_at_indices = _corr_at_indices[bpg, tpb]

    _corr_at_indices(flat_inds, x, int(nfft), int(ncp), bool(norm), out)

    return out


class SyncParams(typing.NamedTuple):
    cp_samples: int
    frame_size: int
    slot_count: int
    corr_size: int
    frames_per_sync: int
    duration: float
    symbol_indexes: list[int]


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
    center_frequency=0,
    pad_cp=True,
    *,
    xp=np,
    dtype='complex64',
):
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

    if center_frequency == 0:
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
    m_seqs *= fourier.get_window(('dpss', 0.9), m_seqs.shape[1], xp=xp)[xp.newaxis]
    norm *= xp.sqrt(xp.mean(xp.abs(m_seqs) ** 2))

    seq_freq = pad_along_axis(m_seqs / norm, [(pad_lo, pad_hi)], axis=1)

    seq_freq = xp.fft.fftshift(seq_freq, axes=1)
    seq_time = fourier.ifft(seq_freq, axis=1, out=seq_freq)

    # prepend the cyclic prefix
    if pad_cp:
        cp_size = round(9 * sample_rate / subcarrier_spacing / 128)
        # seq_time = xp.concatenate([seq_time[:, -cp_size:], seq_time], axis=1)
        # seq_time = iqwaveform.util.pad_along_axis(seq_time, [[cp_size, 0]], axis=1)
        seq_time = xp.concatenate(
            [xp.zeros_like(seq_time[:, -cp_size:]), seq_time], axis=1
        )

    return xp.array(seq_time)


@lru_cache()
def pss_5g_nr(
    sample_rate: float,
    subcarrier_spacing: float,
    center_frequency=0,
    pad_cp=True,
    *,
    xp=np,
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

    return _generate_5g_nr_sync_sequence(
        seq_func=_pss_m_sequence,
        max_id=2,
        sample_rate=sample_rate,
        subcarrier_spacing=subcarrier_spacing,
        center_frequency=center_frequency,
        pad_cp=pad_cp,
        xp=xp,
        dtype=dtype,
    )


@lru_cache()
def sss_5g_nr(
    sample_rate: float,
    subcarrier_spacing: float,
    center_frequency=0,
    pad_cp=True,
    *,
    xp=np,
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
        pad_cp=pad_cp,
        xp=xp,
        dtype=dtype,
    )


@lru_cache()
def pss_params(
    *,
    sample_rate: float = 2 * 7.68e6,
    subcarrier_spacing: float,
    discovery_periodicity: float = 20e-3,
    shared_spectrum: bool = False,
) -> SyncParams:
    if not isroundmod(subcarrier_spacing, 15e3):
        raise ValueError('subcarrier_spacing must be multiple of 15000')

    if isroundmod(sample_rate, 128 * subcarrier_spacing):
        frame_size = round(10e-3 * sample_rate)
    else:
        raise ValueError(
            f'sample_rate must be a multiple of {128 * subcarrier_spacing}'
        )

    # if duration is None:
    #     duration = 2 * slot_duration
    # elif not iqwaveform.util.isroundmod(duration, slot_duration / 2):
    #     raise ValueError(
    #         f'duration must be a multiple of 1/2 slot duration, {slot_duration / 2}'
    #     )

    # The following cases are defined in 3GPP TS 138 213: Section 4.1
    if np.isclose(subcarrier_spacing, 15e3):
        # Case A
        offsets = [2, 8]
        mult = 14
        if shared_spectrum:
            nrange = range(5)
        else:
            # for center frequencies < 3 GHz (or 1.88 GHz in unpaired operation)
            # the upper 2 can be ignored
            nrange = range(4)
    # TODO: Implement Case B
    # elif np.isclose(subcarrier_spacing, 30e3):
    #     # Case B
    #     offsets = [2,8]
    #     if shared_spectrum:
    #         n = np.arange(10)
    #     else:
    #         # for center frequencies < 3 GHz, the upper 2 can be ignored
    #         n = np.arange(4)
    elif np.isclose(subcarrier_spacing, 30e3):
        # For now, all 30 kHz SCS is assumed to be "Case C"
        offsets = [2, 8]
        mult = 14
        if shared_spectrum:
            nrange = range(10)
        else:
            # for center frequencies < 3 GHz (or 1.88 GHz in unpaired
            # operation) the upper 2 can be ignored
            nrange = range(4)
    else:
        raise ValueError(
            'only 15 kHz and 30 kHz SCS (Case A, C) are currently supported (Case A,B,C)'
        )

    symbol_indexes = []
    for n in nrange:
        for offset in offsets:
            symbol_indexes.append(offset + mult * n)

    slot_count = ceil(symbol_indexes[-1] / 14)
    slot_duration = 10e-3 / (10 * subcarrier_spacing / 15e3)
    duration = slot_count * slot_duration
    corr_size = round(duration * sample_rate)

    if isroundmod(discovery_periodicity, 10e-3):
        frames_per_sync = round(discovery_periodicity / 10e-3)
    else:
        raise ValueError('discovery_periodicity must be a multiple of 10e-3')

    cp_samples = round(9 / 128 * sample_rate / subcarrier_spacing)

    return SyncParams(
        cp_samples=cp_samples,
        frame_size=frame_size,
        slot_count=slot_count,
        corr_size=corr_size,
        frames_per_sync=frames_per_sync,
        symbol_indexes=symbol_indexes,
        duration=duration,
    )


@lru_cache()
def sss_params(
    *,
    sample_rate: float = 2 * 7.68e6,
    subcarrier_spacing: float,
    discovery_periodicity: float = 20e-3,
    shared_spectrum: bool = False,
) -> SyncParams:
    # Match PSS except that the symbol indexes are incremented by 2

    template = pss_params(
        sample_rate=sample_rate,
        subcarrier_spacing=subcarrier_spacing,
        discovery_periodicity=discovery_periodicity,
        shared_spectrum=shared_spectrum,
    )

    indexes = [i + 2 for i in template.symbol_indexes]

    return SyncParams(
        cp_samples=template.cp_samples,
        frame_size=template.frame_size,
        slot_count=template.slot_count,
        corr_size=template.corr_size,
        frames_per_sync=template.frames_per_sync,
        symbol_indexes=indexes,
        duration=template.duration,
    )


class PhyOFDM:
    def __init__(
        self,
        *,
        channel_bandwidth: float,
        sample_rate: float,
        nfft: float,
        cp_sizes: ArrayType,
        frame_duration: float | None = None,
        contiguous_size: float | None = None,
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

        if cp_sizes is None:
            self.contiguous_size = contiguous_size
            self.cp_start_idx = None
            self.cp_idx = None
            self.symbol_idx = None

        else:
            if contiguous_size is not None:
                self.contiguous_size = contiguous_size
            else:
                # if not specified, assume no padding is needed to complete a contiguos block of symbols
                self.contiguous_size = np.sum(cp_sizes) + len(cp_sizes) * nfft

            # build a (start_idx, size) pair for each CP
            pair_sizes = xp.concatenate((xp.array((0,)), self.cp_sizes + self.nfft))
            self.cp_start_idx = (pair_sizes.cumsum()).astype(int)[:-1]
            start_and_size = zip(self.cp_start_idx, self.cp_sizes)

            idx_range = xp.arange(self.contiguous_size).astype(int)

            # indices in the contiguous range that are CP
            self.cp_idx = xp.concatenate(
                [idx_range[start : start + size] for start, size in start_and_size]
            )

            # indices in the contiguous range that are not CP
            self.symbol_idx = np.setdiff1d(idx_range, self.cp_idx)

    def index_cyclic_prefix(self) -> ArrayType:
        raise NotImplementedError


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
    MIN_CP_SIZES = np.array((10, 9, 9, 9, 9, 9, 9, 10, 9, 9, 9, 9, 9, 9), dtype=int)

    SCS_TO_SLOTS_PER_FRAME = {15e3: 10, 30e3: 20, 60e3: 40}

    # TODO: add 5G FR2 SCS values
    SUBCARRIER_SPACINGS = {15e3, 30e3, 60e3}

    def __init__(
        self, channel_bandwidth, subcarrier_spacing=15e3, sample_rate=None, xp=np
    ):
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

        cp_sizes = xp.array((nfft * self.MIN_CP_SIZES) // 128)

        super().__init__(
            channel_bandwidth=channel_bandwidth,
            nfft=nfft,
            sample_rate=sample_rate,
            frame_duration=10e-3,
            cp_sizes=cp_sizes,
        )

    @methodtools.lru_cache(4)
    def index_cyclic_prefix(
        self,
        *,
        frames=(0,),
        symbols='all',
        slots='all',
    ):
        """build an indexing tensor for performing cyclic prefix correlation across various axes"""
        xp = array_namespace(self.cp_sizes)

        frames = xp.array(frames)
        frame_size = round(self.sample_rate * 10e-3)

        slots = _index_or_all(
            slots,
            '"slots" argument',
            size=self.SCS_TO_SLOTS_PER_FRAME[self.subcarrier_spacing],
            xp=xp,
        )
        symbols = _index_or_all(
            symbols, '"symbols" argument', size=self.FFT_PER_SLOT, xp=xp
        )

        # first build each grid axis separately
        grid = []

        # axis 0: symbol number within each slot
        grid.append(self.cp_start_idx[symbols])

        # axis 1: slot number
        grid.append(self.contiguous_size * slots)

        # axis 2: frame number
        grid.append(frames * frame_size)

        # axis 3: cp index
        grid.append(xp.ogrid[0 : self.cp_sizes[1]])

        grid = [x.squeeze() for x in grid if x.size > 1]
        # pad the axis dimensions so they can be broadcast together
        inds, *offsets = xp.meshgrid(*grid, indexing='ij', copy=False)

        # sum all of the index offsets
        inds = inds.copy()
        for offset in offsets:
            inds += offset

        return inds


def isclosetoint(v, atol=1e-6):
    xp = array_namespace(v)
    return xp.isclose(v % 1, (0, 1), atol=atol).any()


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
        alt_sample_rate: float = None,
        frame_duration: float = 5e-3,
        nfft: float = 2048,
        cp_ratio: float = 1 / 8,
        xp=np,
    ):
        """
        Args:
            channel_bandwidth: Channel bandwidth as defined by 802.16-2017
            alt_sample_rate (_type_, optional): If specified, overrides the 802.16-2017 value with sample rate of recorded data.
            frame_duration: _description_. Defaults to 5e-3.
            nfft: the fft size corresponding to the length of the useful portion of each symbol. Defaults to 2048.
            cp_ratio: the size of the cyclic prefix as a fraction of nfft. Defaults to 1/8.
        """
        if not isinstance(channel_bandwidth, Number):
            raise TypeError('expected numeric value for channel_bandwidth')
        elif channel_bandwidth < 1.25e6:
            raise ValueError(
                'standardized values for channel_bandwidth not supported yet'
            )
        elif not np.isclose(channel_bandwidth % 125e3, 0, atol=1e-6):
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
            if np.isclose(channel_bandwidth % freq_divisor, 0, atol=1e-6):
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

            if not (isclosetoint(scale) or isclosetoint(1 / scale)):
                raise ValueError(
                    'alt_sample_rate must be integer multiple or divisor of ofdm sample_rate'
                )
            if not isclosetoint(cp_size * scale):
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

    @methodtools.lru_cache(4)
    def index_cyclic_prefix(
        self,
        *,
        frames=(0,),
        symbols='all',
    ) -> ArrayType:
        """build an indexing tensor for performing cyclic prefix correlation across various axes"""

        xp = array_namespace(self.cp_sizes)
        frames = xp.array(frames)

        symbols = _index_or_all(
            symbols, '"symbols" argument', size=self.symbols_per_frame, xp=xp
        )

        # first build each grid axis separately
        grid = []

        # axis 0: symbol number in each frame
        grid.append(self.cp_start_idx[symbols])

        # axis 1: frame number
        grid.append(frames * self.frame_size)

        # axis 2: cp index
        grid.append(xp.ogrid[0 : self.cp_sizes[1]])

        # pad the axis dimensions so they can be broadcast together
        a = xp.meshgrid(*grid, indexing='ij', copy=False)

        # sum all of the index offsets
        inds = a[0].copy()
        for sub in a[1:]:
            inds += sub

        return inds


empty_complex64 = np.zeros(0, dtype=np.complex64)


class BasebandClockSynchronizer:  # other base classes are basic_block, decim_block, interp_block
    """Use the cyclic prefix (CP) in the LTE PHY layer to
    (1) resample to correct clock mismatch relative to the transmitter, and
    (2) align LTE signal to the start of a CP (resolution of nfft*(1+9/128) samples)

    Usage:

        sync = BasebandClockSynchronizer(channel_bandwidth=channel_bandwidth)
        y = sync(x, 0.1)

    Notes:

        This has not been optimized for low SNR, so it it best used in laboratory settings
        with strong signal.

    """

    COARSE_CP0_STEP = (
        1.0 / 6
    )  # coarse search step, as fraction of the length of the first cyclic prefix

    def __init__(
        self,
        channel_bandwidth: float,  # the channel bandwidth in Hz: 1.4 MHz, 5 MHz, 10 MHz, 20 MHz, etc
        correlation_subframes: int = 20,  # correlation window size, in subframes (20 = 1 frame)
        sync_window_count: int = 2,  # how many correlation windows to synchronize at a time (suggest >= 2)
        which_cp: str = 'all',  # 'all', 'special', or 'normal'
        subcarrier_spacing=15e3,
        xp=np,
    ):
        self.phy = Phy3GPP(channel_bandwidth, subcarrier_spacing=subcarrier_spacing)
        self.correlation_subframes = correlation_subframes
        self.sync_size = (
            sync_window_count * correlation_subframes * self.phy.contiguous_size
        )

        # index array of cyclic prefix samples
        cp_gate = self.phy.cp_idx  # 1 single slot
        i_slot_starts = self.phy.contiguous_size * xp.arange(correlation_subframes)
        cp_gate = indexsum2d(
            i_slot_starts, cp_gate
        ).flatten()  # duplicate across slot_count slots

        # Define a coarse sample grid to get into the ballpark of the correlation peak.
        # self.COARSE_CP0_STEP defines the resolution of the search, in fractions of the
        # standard size (i.e., not the first) CP window.
        #
        # This grid spans a total length of a single slot.
        coarse_step = int(self.phy.cp_sizes[1] * self.COARSE_CP0_STEP)
        self.cp_offsets_coarse = xp.arange(
            0, self.phy.nfft + self.phy.cp_sizes[1], coarse_step, dtype=int
        )

        # 2-D search grid
        self.cp_indices_coarse = indexsum2d(self.cp_offsets_coarse, cp_gate)

        # Define the fine search sample grid, which is applied as an offset relative to the
        # result of the coarse search
        self.cp_offsets_fine = xp.arange(
            -np.ceil(coarse_step / 2), np.ceil(coarse_step / 2) + 1, 1, dtype=int
        )
        self.cp_indices_fine = indexsum2d(self.cp_offsets_fine, cp_gate)

    def _cp_correlate(self, x, cp_inds):
        """
        :cp_inds: 2D index offsets array of shape (M,N), where:
                  dimension 0 indices are trial index locations for the start of the slot
                  dimension 1 gives corresponding index offsets of CP samples relative to the start of the slot

        """
        return correlate_along_axis(x[cp_inds], x[self.phy.nfft :][cp_inds], axis=1)

    def _find_slot_start_offset(self, x):
        """Estimate the offset required to align the start of a slot to
        index 0 in the complex sample vector `x`
        """

        xp = array_namespace(x)

        # Coarse estimate of alignment offset to within coarse_step samples
        coarse_corr = xp.abs(self._cp_correlate(x, self.cp_indices_coarse))
        coarse_offset = self.cp_offsets_coarse[xp.argmax(coarse_corr)]

        # Fine calculation for the offset (near the coarse result)
        fine_corr = xp.abs(self._cp_correlate(x, self.cp_indices_fine + coarse_offset))
        n_fine = xp.argmax(fine_corr)
        fine_offset = coarse_offset + self.cp_offsets_fine[n_fine]

        noise_est = xp.nanmedian(xp.abs(xp.sort(coarse_corr)[:-3]))

        return fine_offset, fine_corr[n_fine], noise_est

    def _offset_by_sync_period(self, x):
        """Given the LTE baseband signal in complex array `x`, enforce synchronization
        to the start of a slot every `sync_size` samples.

        To find the offset required in each sync period, cyclic prefix correlation is
        applied to the full window between sync periods.
        """
        xp = array_namespace(x)

        ret = []
        input_chunks = xp.split(x, xp.mgrid[: x.size : self.sync_size][1:])

        if len(input_chunks[-1]) != len(input_chunks[0]):
            input_chunks = input_chunks[:-1]

        ret = [self._find_slot_start_offset(chunk) for chunk in input_chunks]

        return xp.array(ret)

    def _estimate_clock_mismatch(self, x, snr_min=3):
        """Phase-unwrapped linear regression to estimate the discrepancy
        between the transmit and receive baseband clock frequencies across
        all synchronization windows.
        """

        from sklearn import linear_model

        xp = array_namespace(x)

        offsets, weights, noise = self._offset_by_sync_period(x).T
        t_sync = (self.sync_size / self.phy.sample_rate) * xp.arange(offsets.size)

        self.snr = weights / noise

        # Require a minimum SNR to be included in the estimate. (otherwise,
        # noisy values are a potential problem for np.unwrap - this is still possibly
        # a lingering problem, even with this in place)
        select = self.snr > snr_min  # "good" indices

        print(
            f'  {select.sum()} sync windows had well-correlated cyclic prefix ({select.sum() / select.size * 100:0.1f}%)'
        )
        offsets = offsets[select]
        t_sync = t_sync[select]
        weights = weights[select]

        # the offsets wrap back to zero when they pass nfft+length of the first CP block.
        # unwrap here to keep linear regression from breaking
        offsets = self._unwrap_offsets(offsets)

        print('LinearRegression.fit()')
        print('tsync shape:' + str(t_sync.shape))
        print('offsets shape:' + str(offsets.shape))
        print('weights shape:' + str(weights.shape))
        fit = linear_model.LinearRegression().fit(
            t_sync.reshape(-1, 1), offsets.reshape(-1, 1), weights
        )
        print('LinearRegression.fit() finished')
        slipped_samples = xp.round(
            fit.coef_[0, 0] * x.size / self.phy.sample_rate
        ).astype(int)

        self._regression_info = dict(
            inputs=(t_sync, offsets, weights), fit=fit, slipped_samples=slipped_samples
        )

        return slipped_samples, fit.intercept_[0]

    def _unwrap_offsets(self, offsets):
        xp = array_namespace(offsets)

        scale_rad = 2 * np.pi / self.phy.nfft
        return (xp.unwrap(offsets * scale_rad) / scale_rad).astype(int)

    def plot_offset_with_fit(self, x):
        # TODO: this belongs in figures.py
        from matplotlib import pylab

        slope, intercept = self._estimate_clock_mismatch(x)
        t, offsets, weights = self._regression_info['inputs']
        pylab.plot(t, offsets, '.')
        pylab.plot(
            t, t * self._regression_info['slope'] + self._regression_info['intercept']
        )

    def __call__(
        self, x, subsample_offset_correction=True, max_passes=10, on_fail='except'
    ):
        """Resample to correct for baseband clock mismatch.

        :subsample_offset_correction:
            * True to perform subsampling
            * False round to the nearest clock offset for speed
        """

        # if there are enough sample slips, we might slip across 1 or more CP windows.
        # biasing the estimate of the number of slipped samples. work through this iteratively by successive
        # attempts at resampling.
        total_sample_slip = 0
        for i in range(max_passes + 1):
            print(f'baseband clock correction pass {i + 1}')
            sample_slip, offset = self._estimate_clock_mismatch(x)
            total_sample_slip += sample_slip

            if sample_slip == 0:
                break
            else:
                # resample to correct sample slipping
                print('resampling')
                print('sample slip iiiiiiis: ' + str(sample_slip))
                print('total samples is: ' + str(x.size - sample_slip))
                #                print('forcing sample slip to be 8.4Hz')
                #                sample_slip = 42
                now = datetime.now()
                print(
                    'start resample with sample_slip: '
                    + str(sample_slip)
                    + ' '
                    + str(now)
                )
                x = fourier.resample(x, x.size - sample_slip)
                elapsed = datetime.now() - now
                print('done resampling ' + str(sample_slip) + ' ' + str(elapsed))

        else:
            if on_fail == 'except':
                raise ValueError(
                    f'failed to converge on clock mismatch within {max_passes} passes'
                )

        print(
            f'corrected baseband clock slip by {total_sample_slip} samples({total_sample_slip / x.size * self.phy.sample_rate:0.2f} Hz clock mismatch)'
        )

        # last, correct the fixed offset at the beginning in an attempt to align
        if subsample_offset_correction:
            print(f'subsample shift to correct offset of {offset:0.3f} samples')
            x = subsample_shift(x, -offset)
        else:
            int_offset = int(offset.round())
            print(
                f'shift to correct offset of {int_offset} (out of {offset:0.3f}) samples'
            )
            x = x[int_offset % self.phy.contiguous_size :]

            # keep only an integer number of slots
        spare_samples = x.size % (2 * self.phy.contiguous_size)
        if spare_samples > 0:
            x = x[:-spare_samples]

        # if there are enough sample slips, we might slip backward by 1 CP duration (9/128 * self.phy.nfft)
        return x


class SymbolDecoder:
    """Decode symbols from a clock-synchronized received waveform. This uses simple LTE PHY numerology,
    and an edge detection scheme to synchronize symbols relative to slots.

    Usage:

        decode = SymbolDecoder(channel_bandwidth=channel_bandwidth)
        y = decode(x)

    Notes:
        This has not been optimized for low SNR, so it it best used in laboratory settings
        with strong signal.

    """

    def __init__(self, channel_bandwidth):
        self.phy = Phy3GPP(channel_bandwidth)

    @staticmethod
    def prb_power(symbols):
        """Return the total power in the PRB"""
        xp = array_namespace(symbols)
        by_prb = xp.abs(to_blocks(symbols, Phy3GPP.SUBFRAMES_PER_PRB)) ** 2
        return by_prb.sum(axis=-1)

    def _decode_symbols(self, x, only_3gpp_subcarriers=True):
        xp = array_namespace(x)

        # first, select symbol indices (== remove cyclic prefixes)
        x = to_blocks(x, 2 * self.phy.contiguous_size)[:, self.phy.symbol_idx].flatten()

        # break up the waveform into windows of length nfft
        blocks = to_blocks(x, self.phy.nfft)

        #  decode with the fft
        X = xp.fft.fftshift(xp.fft.fft(blocks, axis=-1), axes=(-1,))

        X /= xp.sqrt(2 * self.phy.nfft)

        if only_3gpp_subcarriers:
            # return only the FFT bins meant to contain data
            sc_start = X.shape[-1] // 2 - self.phy.subcarriers // 2
            sc_stop = X.shape[-1] // 2 + self.phy.subcarriers // 2
            X = X[:, sc_start:sc_stop]

        return X

    def _align_symbols_to_tti(self, symbols):
        xp = array_namespace(symbols)

        # determine the power change that is strongest across all PRBs in each FFT window
        power = self.prb_power(symbols)
        power_diff = xp.diff(power, axis=0, append=0) / power
        diff_peaks = xp.abs(power_diff).max(axis=1)
        diff_peak_by_symbol = to_blocks(diff_peaks, Phy3GPP.FFT_PER_SLOT)
        self._diff_peak_by_symbol = diff_peak_by_symbol
        self._diff_peaks = diff_peaks
        self._power_diff = power_diff

        # where do the maxima occur in each tti
        tti_offset = diff_peak_by_symbol.max(axis=0).argmax() + 1

        return symbols[tti_offset:]

    def __call__(self, x):
        """Ringlead the decoding process"""

        symbols = self._decode_symbols(x)
        symbols = self._align_symbols_to_tti(symbols)
        return symbols
