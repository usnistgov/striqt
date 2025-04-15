import functools
import typing
import dataclasses

from ..api import util

from xarray_dataclasses import AsDataArray, Coordof, Data, Attr, Name

if typing.TYPE_CHECKING:
    import iqwaveform
    import numpy as np
    import pandas as pd
else:
    iqwaveform = util.lazy_import('iqwaveform')
    np = util.lazy_import('numpy')
    pd = util.lazy_import('pandas')

from ..api.registry import register_xarray_measurement
from ..api import structs, util


###
CellularSectorIDAxis = typing.Literal['cellular_sector_id']


@dataclasses.dataclass
class CellularSectorIDCoords:
    data: Data[CellularSectorIDAxis, str]
    standard_name: Attr[str] = r'Cellular Sector ID ($N_{ID}^{(2)}$)'

    @staticmethod
    @functools.lru_cache
    def factory(capture: structs.Capture, **_):
        values = np.array([0, 1, 2], dtype='uint8')
        return values, {}


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
        capture: structs.Capture,
        **kws,
    ):
        params = _pss_params(capture, **kws)
        count = round(params['duration'] / params['sync_period'])
        return np.arange(max(count, 1)) * params['sync_period']


### Subcarrier spacing label axis
CellularSSBSymbolIndexAxis = typing.Literal['cellular_ssb_symbol_index']


@dataclasses.dataclass
class CellularSSBSymbolIndexCoords:
    data: Data[CellularSSBSymbolIndexAxis, np.float32]
    standard_name: Attr[str] = 'SSB symbol index'

    @staticmethod
    @functools.lru_cache
    def factory(capture: structs.Capture, **kws):
        params = _pss_params(capture, **kws)

        slot_count = round(params['duration'] / params['slot_duration'])
        symbol_count = 14 * slot_count

        return list(range(0, symbol_count, 2))


### Time elapsed dimension and coordinates
CellularPSSLagAxis = typing.Literal['cellular_pss_lag']


@dataclasses.dataclass
class CellularPSSLagCoords:
    data: Data[CellularPSSLagAxis, np.float32]
    standard_name: Attr[str] = 'PSS synchronization lag'
    units: Attr[str] = 's'

    @staticmethod
    @functools.lru_cache
    def factory(capture: structs.Capture, **kws) -> dict[str, np.ndarray]:
        params = _pss_params(capture, **kws)

        max_len = 2 * round(
            capture.sample_rate / params['subcarrier_spacing'] + params['cp_samples']
        )

        if params['trim_cp']:
            max_len = max_len - round(0.5 * params['cp_samples'])

        axis_name = typing.get_args(CellularPSSLagAxis)[0]
        return pd.RangeIndex(0, max_len, name=axis_name) / capture.sample_rate


### Dataarray definition
# -> (port index, cell Nid2, sync block index, symbol pair index, IQ sample index)


@dataclasses.dataclass
class Cellular5GNRPSSCorrelation(AsDataArray):
    power_time_series: Data[
        tuple[
            CellularSectorIDAxis,
            CellularSSBStartTimeElapsedAxis,
            CellularSSBSymbolIndexAxis,
            CellularPSSLagAxis,
        ],
        np.complex64,
    ]

    cellular_sector_id: Coordof[CellularSectorIDCoords]
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

    norm = np.float32(np.sqrt(SC_COUNT))
    m_seqs = np.array([_m_sequence(i) for i in range(3)], dtype=dtype)
    pss_freq = iqwaveform.util.pad_along_axis(m_seqs / norm, [(pad_lo, pad_hi)], axis=1)
    pss_time = np.fft.ifft(np.fft.fftshift(pss_freq, axes=1), axis=1)

    # prepend the cyclic prefix
    cp_size = round(9 * sample_rate / subcarrier_spacing / 128)
    pss_time = np.concatenate([pss_time[:, -cp_size:], pss_time], axis=1)

    return xp.array(pss_time)


@functools.lru_cache()
def _pss_params(
    capture,
    *,
    subcarrier_spacing: float,
    sync_period: float = 10e-3,
    block_frequency_offset: float = 0,
    duration: typing.Optional[float] = None,
    trim_cp: bool = True,
) -> dict:
    if not iqwaveform.util.isroundmod(subcarrier_spacing, 15e3):
        raise ValueError('subcarrier_spacing must be multiple of 15000')

    if iqwaveform.util.isroundmod(capture.sample_rate, 128 * subcarrier_spacing):
        frame_size = round(10e-3 * capture.sample_rate)
    else:
        raise ValueError(
            f'capture.sample_rate must be a multiple of {128 * subcarrier_spacing}'
        )

    slot_duration = 10e-3 / (10 * subcarrier_spacing / 15e3)

    if duration is None:
        duration = 2 * slot_duration
    elif not iqwaveform.util.isroundmod(duration, slot_duration / 2):
        raise ValueError(
            f'duration must be a multiple of 1/2 slot duration, {slot_duration / 2}'
        )
    slot_count = round(duration / slot_duration)
    corr_size = round(duration * capture.sample_rate)

    if iqwaveform.util.isroundmod(sync_period, 10e-3):
        frames_per_sync = round(sync_period / 10e-3)
    else:
        raise ValueError('sync_period must be a multiple of 10e-3')

    cp_samples = round(9 / 128 * capture.sample_rate / subcarrier_spacing)

    return locals()


@register_xarray_measurement(Cellular5GNRPSSCorrelation)
def cellular_5g_pss_correlation(
    iq,
    capture: structs.Capture,
    *,
    subcarrier_spacing: float,
    sync_period: float = 10e-3,
    block_frequency_offset: float = 0,
    duration: typing.Optional[float] = None,
    trim_cp: bool = True,
):
    metadata = dict(locals())
    del metadata['iq'], metadata['capture']

    params = _pss_params(capture, **metadata)
    frame_size = params['frame_size']
    slot_count = params['slot_count']
    corr_size = params['corr_size']
    frames_per_sync = params['frames_per_sync']

    pss = _pss_5g_nr(
        capture.sample_rate, subcarrier_spacing, center_frequency=block_frequency_offset
    )

    # set up broadcasting to new dimensions:
    # (port index, cell Nid2, sync block index, IQ sample index)
    iq_bcast = iq.reshape((1, -1, frame_size))
    iq_bcast = iq_bcast[:, np.newaxis, ::frames_per_sync, :corr_size]
    pss_bcast = pss[np.newaxis, :, np.newaxis, :]

    R = iqwaveform.oaconvolve(iq_bcast, pss_bcast, axes=3, mode='full')

    # shift correlation peaks to the symbol start
    cp_samples = round(9 / 128 * capture.sample_rate / subcarrier_spacing)
    offs = round(capture.sample_rate / subcarrier_spacing + 2 * cp_samples)
    R = np.roll(R, -offs, axis=-1)[..., :corr_size]

    # -> (port index, cell Nid2, sync block index, slot index, IQ sample index)
    excess_cp = round(capture.sample_rate / subcarrier_spacing * 1 / 128)
    R = R.reshape(R.shape[:-1] + (slot_count, -1))[..., 2 * excess_cp :]

    # -> (port index, cell Nid2, sync block index, symbol pair index, IQ sample index)
    paired_symbol_shape = R.shape[:-2] + (7 * slot_count, -1)
    R = R.reshape(paired_symbol_shape)

    if trim_cp:
        R = R[..., : -cp_samples // 2]

    R = iqwaveform.envtopow(R) * np.exp(1j * np.angle(R))

    return R, metadata | {'units': 'mW'}
