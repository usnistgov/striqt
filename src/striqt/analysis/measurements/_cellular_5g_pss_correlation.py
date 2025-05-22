from __future__ import annotations
from math import ceil
import typing

from ..lib import registry, specs, util

if typing.TYPE_CHECKING:
    import iqwaveform
    import numpy as np
    import pandas as pd
else:
    iqwaveform = util.lazy_import('iqwaveform')
    np = util.lazy_import('numpy')
    pd = util.lazy_import('pandas')


class Cellular5GNRPSSCorrelationSpec(
    specs.Measurement,
    forbid_unknown_fields=True,
    cache_hash=True,
    kw_only=True,
    frozen=True,
):
    subcarrier_spacing: float
    sample_rate: float = 15.36e6
    discovery_periodicity: float = 20e-3
    frequency_offset: typing.Union[float, dict[float, float]] = 0
    shared_spectrum: bool = False
    max_block_count: typing.Optional[int] = 1


class Cellular5GNRPSSCorrelationKeywords(specs.AnalysisKeywords, total=False):
    subcarrier_spacing: float
    sample_rate: float
    discovery_periodicity: float
    frequency_offset: typing.Union[float, dict[float, float]]
    shared_spectrum: bool
    max_block_count: typing.Optional[int]


@registry.coordinate_factory(
    dtype='uint16', attrs={'standard_name': r'Cell Identity 2 ($N_{ID}^{(2)}$)'}
)
@util.lru_cache()
def cellular_cell_id2(capture: specs.Capture, spec: typing.Any):
    values = np.array([0, 1, 2], dtype='uint16')
    return values, {}


### Subcarrier spacing label axis
CellularSSBStartTimeElapsedAxis = typing.Literal['cellular_ssb_start_time']


@registry.coordinate_factory(
    dtype='float32', attrs={'standard_name': 'Time Elapsed', 'units': 's'}
)
@util.lru_cache()
def cellular_ssb_start_time(
    capture: specs.Capture, spec: Cellular5GNRPSSCorrelationSpec
):
    params = _pss_params(capture, spec)
    total_blocks = round(params['duration'] / spec.discovery_periodicity)
    if spec.max_block_count is None:
        count = total_blocks
    else:
        count = min(spec.max_block_count, total_blocks)

    return np.arange(max(count, 1)) * spec.discovery_periodicity


@registry.coordinate_factory(
    dtype='uint16', attrs={'standard_name': 'SSB symbol index'}
)
@util.lru_cache()
def cellular_ssb_symbol_index(
    capture: specs.Capture, spec: Cellular5GNRPSSCorrelationSpec
):
    params = _pss_params(capture, spec)

    return list(params['symbol_indexes'])


@registry.coordinate_factory(
    dtype='float32', attrs={'standard_name': 'Symbol lag', 'units': 's'}
)
@util.lru_cache()
def cellular_pss_lag(capture: specs.Capture, spec: Cellular5GNRPSSCorrelationSpec):
    params = _pss_params(capture, spec)

    max_len = 2 * round(
        spec.sample_rate / spec.subcarrier_spacing + params['cp_samples']
    )

    if params['trim_cp']:
        max_len = max_len - round(0.5 * params['cp_samples'])

    name = cellular_pss_lag.__name__
    return pd.RangeIndex(0, max_len, name=name) / spec.sample_rate


class SyncParams(typing.NamedTuple):
    cp_samples: int
    frame_size: int
    slot_count: int
    corr_size: int
    frames_per_sync: int
    trim_cp: bool
    symbol_indexes: list[int]


@util.lru_cache()
def _pss_params(capture: specs.Capture, spec: Cellular5GNRPSSCorrelationSpec, trim_cp=True) -> SyncParams:
    capture = capture.replace(sample_rate=spec.sample_rate)

    if not iqwaveform.util.isroundmod(spec.subcarrier_spacing, 15e3):
        raise ValueError('subcarrier_spacing must be multiple of 15000')

    if iqwaveform.util.isroundmod(spec.sample_rate, 128 * spec.subcarrier_spacing):
        frame_size = round(10e-3 * spec.sample_rate)
    else:
        raise ValueError(
            f'capture.sample_rate must be a multiple of {128 * spec.subcarrier_spacing}'
        )

    # if duration is None:
    #     duration = 2 * slot_duration
    # elif not iqwaveform.util.isroundmod(duration, slot_duration / 2):
    #     raise ValueError(
    #         f'duration must be a multiple of 1/2 slot duration, {slot_duration / 2}'
    #     )

    # The following cases are defined in 3GPP TS 138 213: Section 4.1
    if np.isclose(spec.subcarrier_spacing, 15e3):
        # Case A
        offsets = [2, 8]
        mult = 14
        if spec.shared_spectrum:
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
    elif np.isclose(spec.subcarrier_spacing, 30e3):
        # For now, all 30 kHz SCS is assumed to be "Case C"
        offsets = [2, 8]
        mult = 14
        if spec.shared_spectrum:
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
    slot_duration = 10e-3 / (10 * spec.subcarrier_spacing / 15e3)
    duration = slot_count * slot_duration
    corr_size = round(duration * spec.sample_rate)

    if iqwaveform.util.isroundmod(spec.discovery_periodicity, 10e-3):
        frames_per_sync = round(spec.discovery_periodicity / 10e-3)
    else:
        raise ValueError('discovery_periodicity must be a multiple of 10e-3')

    cp_samples = round(9 / 128 * spec.sample_rate / spec.subcarrier_spacing)

    return SyncParams(
        cp_samples=cp_samples,
        frame_size=frame_size,
        slot_count=slot_count,
        corr_size=corr_size,
        frames_per_sync=frames_per_sync,
        trim_cp=trim_cp,
        symbol_indexes=symbol_indexes
    )


@registry.measurement(
    coord_funcs=[
        cellular_cell_id2,
        cellular_ssb_start_time,
        cellular_ssb_symbol_index,
        cellular_pss_lag,
    ],
    dtype='complex64',
    spec_type=Cellular5GNRPSSCorrelationSpec,
    attrs={'standard_name': 'PSS Correlation'},
)
def cellular_5g_pss_correlation(
    iq,
    capture: specs.Capture,
    **kwargs: typing.Unpack[Cellular5GNRPSSCorrelationKeywords],
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

    spec = Cellular5GNRPSSCorrelationSpec.fromdict(kwargs).validate()

    if isinstance(spec.frequency_offset, dict):
        if not hasattr(capture, 'center_frequency'):
            raise ValueError(
                'frequency_offset must be a float unless capture has a "center_frequency" attribute'
            )
        lookup = dict(spec.frequency_offset)
        frequency_offset = lookup[capture.center_frequency]  # noqa

    params = _pss_params(capture, spec)

    xp = iqwaveform.util.array_namespace(iq)

    # * 3 makes it compatible with the blackman window overlap of 2/3
    down = round(capture.sample_rate / spec.subcarrier_spacing / 8 * 3)
    up = round(down * (spec.sample_rate / capture.sample_rate))

    if spec.max_block_count is not None:
        duration = round(
            spec.max_block_count * spec.discovery_periodicity * capture.sample_rate
        )
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
    capture = capture.replace(sample_rate=spec.sample_rate)

    if isinstance(frequency_offset, dict):
        if not hasattr(capture, 'center_frequency'):
            raise ValueError(
                'frequency_offset must be a float unless capture has a "center_frequency" attribute'
            )
        frequency_offset = frequency_offset[capture.center_frequency]  # noqa

    slot_count = params.slot_count
    corr_size = params.corr_size
    frames_per_sync = params.frames_per_sync

    pss = iqwaveform.ofdm.pss_5g_nr(capture.sample_rate, spec.subcarrier_spacing, xp=xp)

    # set up broadcasting to new dimensions:
    # (port index, cell Nid2, sync block index, IQ sample index)
    iq_bcast = iq.reshape((iq.shape[0], -1, params.frame_size))
    iq_bcast = iq_bcast[:, xp.newaxis, ::frames_per_sync, :corr_size]
    pss_bcast = pss[xp.newaxis, :, xp.newaxis, :]

    R = iqwaveform.oaconvolve(iq_bcast, pss_bcast, axes=3, mode='full')

    # shift correlation peaks to the symbol start
    cp_samples = round(9 / 128 * capture.sample_rate / spec.subcarrier_spacing)
    offs = round(capture.sample_rate / spec.subcarrier_spacing + 2 * cp_samples)
    R = xp.roll(R, -offs, axis=-1)[..., :corr_size]

    # -> (port index, cell Nid2, sync block index, slot index, IQ sample index)
    excess_cp = round(capture.sample_rate / spec.subcarrier_spacing * 1 / 128)
    R = R.reshape(R.shape[:-1] + (slot_count, -1))[..., 2 * excess_cp :]

    # -> (port index, cell Nid2, sync block index, symbol pair index, IQ sample index)
    paired_symbol_shape = R.shape[:-2] + (7 * slot_count, -1)
    paired_symbol_indexes = xp.array(params.symbol_indexes, dtype='uint32') // 2
    R = R.reshape(paired_symbol_shape)[..., paired_symbol_indexes, :]

    if params.trim_cp:
        R = R[..., : -cp_samples // 2]

    R = iqwaveform.envtopow(R)
    phase = xp.multiply(1j, xp.angle(R), dtype='complex64')
    R = R * xp.exp(phase)

    enbw = spec.subcarrier_spacing * 127
    metadata = {
        'units': f'mW/{enbw / 1e6:0.2f} MHz',
        'standard_name': 'PSS Covariance',
    }

    return R, metadata
