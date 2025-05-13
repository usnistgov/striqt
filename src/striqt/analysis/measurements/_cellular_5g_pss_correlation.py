from __future__ import annotations
import functools
import typing
import dataclasses
from math import ceil

from ..lib import specs, util

from xarray_dataclasses import AsDataArray, Coordof, Data, Attr, Name

if typing.TYPE_CHECKING:
    import iqwaveform
    import numpy as np
    import pandas as pd
else:
    iqwaveform = util.lazy_import('iqwaveform')
    np = util.lazy_import('numpy')
    pd = util.lazy_import('pandas')

from ..lib import registry, util


###
CellularCellID2Axis = typing.Literal['cellular_cell_id2']


@dataclasses.dataclass
class CellularCellID2Coords:
    data: Data[CellularCellID2Axis, np.uint16]
    standard_name: Attr[str] = r'Cell Identity 2 ($N_{ID}^{(2)}$)'

    @staticmethod
    @functools.lru_cache
    def factory(capture: specs.Capture, **_):
        values = np.array([0, 1, 2], dtype='uint16')
        return values, {}


# TODO: one day move to this approach
# @registry.coordinate(dtype='uint16', standard_name=r'Cell Identity 2 ($N_{ID}^{(2)}$)')
# @functools.lru_cache
# def cellular_id2(capture: specs.Capture, **_):
#     values = np.array([0, 1, 2], dtype='uint16')
#     return values, {}


### Subcarrier spacing label axis
CellularSSBStartTimeElapsedAxis = typing.Literal['cellular_ssb_start_time']


@dataclasses.dataclass
class CellularSSBStartTimeElapsedCoords:
    data: Data[CellularSSBStartTimeElapsedAxis, np.float32]
    standard_name: Attr[str] = 'Time elapsed at synchronization block start'
    units: Attr[str] = 's'

    @staticmethod
    @functools.lru_cache
    def factory(
        capture: specs.Capture,
        max_block_count: typing.Optional[int] = 1,
        **kws,
    ):
        params = _pss_params(capture, **kws)
        total_blocks = round(params['duration'] / params['discovery_periodicity'])
        if max_block_count is None:
            count = total_blocks
        else:
            count = min(max_block_count, total_blocks)

        return np.arange(max(count, 1)) * params['discovery_periodicity']


### Subcarrier spacing label axis
CellularSSBSymbolIndexAxis = typing.Literal['cellular_ssb_symbol_index']


@dataclasses.dataclass
class CellularSSBSymbolIndexCoords:
    data: Data[CellularSSBSymbolIndexAxis, np.uint8]
    standard_name: Attr[str] = 'SSB symbol index'

    @staticmethod
    @functools.lru_cache
    def factory(
        capture: specs.Capture, max_block_count: typing.Optional[int] = 1, **kws
    ):
        params = _pss_params(capture, **kws)

        return list(params['symbol_indexes'])


### Time elapsed dimension and coordinates
CellularPSSLagAxis = typing.Literal['cellular_pss_lag']


@dataclasses.dataclass
class CellularPSSLagCoords:
    data: Data[CellularPSSLagAxis, np.float32]
    standard_name: Attr[str] = 'PSS synchronization lag'
    units: Attr[str] = 's'

    @staticmethod
    @functools.lru_cache
    def factory(
        capture: specs.Capture, max_block_count: typing.Optional[int] = 1, **kws
    ) -> dict[str, np.ndarray]:
        params = _pss_params(capture, **kws)

        max_len = 2 * round(
            params['sample_rate'] / params['subcarrier_spacing'] + params['cp_samples']
        )

        if params['trim_cp']:
            max_len = max_len - round(0.5 * params['cp_samples'])

        axis_name = typing.get_args(CellularPSSLagAxis)[0]
        return pd.RangeIndex(0, max_len, name=axis_name) / params['sample_rate']


### Dataarray definition
@dataclasses.dataclass
class Cellular5GNRPSSCorrelation(AsDataArray):
    power_time_series: Data[
        tuple[
            CellularCellID2Axis,
            CellularSSBStartTimeElapsedAxis,
            CellularSSBSymbolIndexAxis,
            CellularPSSLagAxis,
        ],
        np.complex64,
    ]

    cellular_cell_id2: Coordof[CellularCellID2Coords]
    cellular_ssb_start_time: Coordof[CellularSSBStartTimeElapsedCoords]
    cellular_ssb_symbol_index: Coordof[CellularSSBSymbolIndexCoords]
    cellular_pss_lag: Coordof[CellularPSSLagCoords]

    standard_name: Attr[str] = 'PSS Correlation'
    name: Name[str] = 'cellular_5g_pss_correlation'


@functools.lru_cache()
def _m_sequence(N_id2: int) -> list[int]:
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


@functools.lru_cache()
def _pss_5g_nr(
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

    # number of occupied subcarriers in the PSS
    SC_COUNT = 127

    if not iqwaveform.util.isroundmod(subcarrier_spacing, 15e3):
        raise ValueError('subcarrier_spacing must be a multiple of 15000')

    min_sample_rate = SC_COUNT * subcarrier_spacing
    if sample_rate < min_sample_rate:
        raise ValueError(f'sample_rate must be at least {min_sample_rate} S/s')

    if iqwaveform.util.isroundmod(sample_rate, subcarrier_spacing):
        size_out = round(sample_rate / subcarrier_spacing)
    else:
        raise ValueError('sample_rate must be a multiple of subcarrier spacing')

    if center_frequency == 0:
        frequency_offset = 0
    elif iqwaveform.util.isroundmod(center_frequency, subcarrier_spacing):
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

    from scipy import signal

    norm = np.float32(np.sqrt(SC_COUNT))
    m_seqs = np.array([_m_sequence(i) for i in range(3)], dtype=dtype)
    m_seqs *= signal.get_window(('dpss', 0.9), m_seqs.shape[1])[np.newaxis]
    norm *= np.sqrt(np.mean(np.abs(m_seqs) ** 2))

    pss_freq = iqwaveform.util.pad_along_axis(m_seqs / norm, [(pad_lo, pad_hi)], axis=1)
    pss_time = np.fft.ifft(np.fft.fftshift(pss_freq, axes=1), axis=1)

    # prepend the cyclic prefix
    if pad_cp:
        cp_size = round(9 * sample_rate / subcarrier_spacing / 128)
        # pss_time = np.concatenate([pss_time[:, -cp_size:], pss_time], axis=1)
        # pss_time = iqwaveform.util.pad_along_axis(pss_time, [[cp_size, 0]], axis=1)
        pss_time = np.concatenate(
            [np.zeros_like(pss_time[:, -cp_size:]), pss_time], axis=1
        )

    return xp.array(pss_time)


@functools.lru_cache()
def _pss_params(
    capture: specs.Capture,
    *,
    sample_rate: float = 2 * 7.68e6,
    subcarrier_spacing: float,
    discovery_periodicity: float = 20e-3,
    frequency_offset: float = 0,
    trim_cp: bool = True,
    shared_spectrum: bool = False,
) -> dict:
    capture = capture.replace(sample_rate=sample_rate)

    if not iqwaveform.util.isroundmod(subcarrier_spacing, 15e3):
        raise ValueError('subcarrier_spacing must be multiple of 15000')

    if iqwaveform.util.isroundmod(sample_rate, 128 * subcarrier_spacing):
        frame_size = round(10e-3 * sample_rate)
    else:
        raise ValueError(
            f'capture.sample_rate must be a multiple of {128 * subcarrier_spacing}'
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

    if iqwaveform.util.isroundmod(discovery_periodicity, 10e-3):
        frames_per_sync = round(discovery_periodicity / 10e-3)
    else:
        raise ValueError('discovery_periodicity must be a multiple of 10e-3')

    cp_samples = round(9 / 128 * sample_rate / subcarrier_spacing)

    return locals()


@registry.measurement(Cellular5GNRPSSCorrelation)
def cellular_5g_pss_correlation(
    iq,
    capture: specs.Capture,
    *,
    subcarrier_spacing: float,
    sample_rate: float = 15.36e6,
    discovery_periodicity: float = 20e-3,
    frequency_offset: typing.Union[float, dict[float, float]] = 0,
    shared_spectrum: bool = False,
    max_block_count: typing.Optional[int] = 1,
):
    """correlate each channel of the IQ against the cellular primary synchronization signal (PSS) waveform.

    Returns a DataArray containing the time-lag for each combination of NID2, symbol, and SSB start time.

    Args:
        iq: the vector of size (N, M) for N channels and M IQ waveform samples
        capture: capture structure that describes the iq acquisition parameters
        sample_rate (samples/s): downsample to this rate before analysis (or None to follow capture.sample_rate)
        subcarrier_spacing (Hz): OFDM subcarrier spacing
        discovery_periodicity (s): interval between synchronization blocks
        frequency_offset (Hz): baseband center frequency of the synchronization block,
            (or a mapping to look up frequency_offset[capture.center_frequency])
        shared_spectrum: whether to assume "shared_spectrum" symbol layout in the SSB
            according to 3GPP TS 138 213: Section 4.1)
        max_block_count: if not None, the number of synchronization blocks to analyze
        as_xarray: if True (default), return an xarray.DataArray, otherwise a ChannelAnalysisResult object

    References:
        3GPP TS 138 211: Table 7.4.3.1-1, Section 7.4.2.2
        3GPP TS 138 213: Section 4.1
    """

    # TODO: make this part more explicit
    trim_cp = True

    if isinstance(frequency_offset, dict):
        if not hasattr(capture, 'center_frequency'):
            raise ValueError(
                'frequency_offset must be a float unless capture has a "center_frequency" attribute'
            )
        frequency_offset = frequency_offset[capture.center_frequency]  # noqa

    metadata = dict(locals())
    del metadata['iq'], metadata['capture']

    params = _pss_params(
        capture,
        subcarrier_spacing=subcarrier_spacing,
        sample_rate=sample_rate,
        discovery_periodicity=discovery_periodicity,
        frequency_offset=frequency_offset,
        shared_spectrum=shared_spectrum,
    )

    xp = iqwaveform.util.array_namespace(iq)

    # * 3 makes it compatible with the blackman window overlap of 2/3
    down = round(capture.sample_rate / subcarrier_spacing / 8 * 3)
    up = round(down * (sample_rate / capture.sample_rate))

    if max_block_count is not None:
        duration = round(max_block_count * discovery_periodicity * capture.sample_rate)
        iq = iq[..., :duration]

    iq = iqwaveform.fourier.oaresample(
        iq,
        fs=capture.sample_rate,
        up=up,
        down=down,
        axis=1,
        window='blackman',
        frequency_shift=frequency_offset,
    )
    capture = capture.replace(sample_rate=sample_rate)

    if isinstance(frequency_offset, dict):
        if not hasattr(capture, 'center_frequency'):
            raise ValueError(
                'frequency_offset must be a float unless capture has a "center_frequency" attribute'
            )
        frequency_offset = frequency_offset[capture.center_frequency]  # noqa

    frame_size = params['frame_size']
    slot_count = params['slot_count']
    corr_size = params['corr_size']
    frames_per_sync = params['frames_per_sync']

    pss = _pss_5g_nr(capture.sample_rate, subcarrier_spacing, xp=xp)

    # set up broadcasting to new dimensions:
    # (port index, cell Nid2, sync block index, IQ sample index)
    iq_bcast = iq.reshape((iq.shape[0], -1, frame_size))
    iq_bcast = iq_bcast[:, xp.newaxis, ::frames_per_sync, :corr_size]
    pss_bcast = pss[xp.newaxis, :, xp.newaxis, :]

    R = iqwaveform.oaconvolve(iq_bcast, pss_bcast, axes=3, mode='full')

    # shift correlation peaks to the symbol start
    cp_samples = round(9 / 128 * capture.sample_rate / subcarrier_spacing)
    offs = round(capture.sample_rate / subcarrier_spacing + 2 * cp_samples)
    R = xp.roll(R, -offs, axis=-1)[..., :corr_size]

    # -> (port index, cell Nid2, sync block index, slot index, IQ sample index)
    excess_cp = round(capture.sample_rate / subcarrier_spacing * 1 / 128)
    R = R.reshape(R.shape[:-1] + (slot_count, -1))[..., 2 * excess_cp :]

    # -> (port index, cell Nid2, sync block index, symbol pair index, IQ sample index)
    paired_symbol_shape = R.shape[:-2] + (7 * slot_count, -1)
    paired_symbol_indexes = xp.array(params['symbol_indexes'], dtype='uint32') // 2
    R = R.reshape(paired_symbol_shape)[..., paired_symbol_indexes, :]

    if trim_cp:
        R = R[..., : -cp_samples // 2]

    R = iqwaveform.envtopow(R)
    exp = xp.multiply(1j, xp.angle(R), dtype='complex64')
    R *= xp.exp(exp, out=exp)

    metadata = metadata | {'units': 'mW', 'standard_name': 'PSS Covariance'}

    return R.copy(), metadata
