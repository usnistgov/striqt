from __future__ import annotations as __

import typing

from .. import specs

from ..lib import register, util
from ..lib.dataarrays import CAPTURE_DIM
from . import shared
from .shared import registry

import striqt.waveform as sw

if typing.TYPE_CHECKING:
    import numpy as np
    from ..lib.typing import Array
else:
    np = util.lazy_import('numpy')


@util.lru_cache()
def _spec_to_params(
    capture: specs.Capture,
    spec: specs.Cellular5GNPSSSync | specs.Cellular5GNRPSSCorrelator,
):
    return sw.ofdm.pss_params(
        sample_rate=spec.sample_rate,
        subcarrier_spacing=spec.subcarrier_spacing,
        discovery_periodicity=spec.discovery_periodicity,
        shared_spectrum=spec.shared_spectrum,
        max_lag_symbols=spec.max_lag_symbols,
        symbol_indexes=spec.symbol_indexes,
        center_frequency=getattr(capture, 'center_frequency', None),
    )


@registry.coordinates(dtype='float32', attrs={'standard_name': 'Lag', 'units': 's'})
@util.lru_cache()
def cellular_ssb_lag(capture: specs.Capture, spec: specs.Cellular5GNRPSSCorrelator):
    # TODO: this now needs to account for PSS vs SSS
    params = _spec_to_params(capture, spec)
    offs = round(spec.sample_rate * spec.delay)
    return np.arange(offs, offs + params.lag_count) / spec.sample_rate


_coord_factories = [
    shared.cellular_cell_id2,
    shared.cellular_ssb_start_time,
    shared.cellular_ssb_beam_index,
    cellular_ssb_lag,
]
dtype = 'complex64'


correlator_cache = register.KwArgCache([CAPTURE_DIM, 'spec'])


@correlator_cache.apply
def correlate_5g_pss(
    iq: 'Array',
    capture: specs.Capture,
    spec: specs.Cellular5GNRPSSCorrelator,
) -> 'Array':
    xp = sw.array_namespace(iq)

    ssb_iq = shared.get_5g_ssb_iq(iq, capture=capture, spec=spec)

    if ssb_iq is None:
        return shared.empty_5g_ssb_correlation(
            iq, capture=capture, spec=spec, coord_factories=_coord_factories
        )

    params = _spec_to_params(capture, spec)
    pss_seq = sw.ofdm.pss_5g_nr(spec.sample_rate, spec.subcarrier_spacing, xp=xp)

    return sw.ofdm.correlate_sync_sequence(
        ssb_iq, pss_seq, params=params, cell_id_split=1
    )


sync_cache = register.KwArgCache([CAPTURE_DIM, 'spec'])


@sync_cache.apply
def choose_sync_offsets(
    iq: Array,
    capture: specs.Capture,
    *,
    spec: specs.Cellular5GNPSSSync,
) -> Array:
    # R.shape -> (..., port index, cell Nid2, SSB index, symbol start index, IQ sample index)

    corr_spec = specs.Cellular5GNRPSSCorrelator.from_spec(spec).validate()

    r = correlate_5g_pss(iq, capture=capture, spec=corr_spec)
    params = _spec_to_params(capture, spec)
    return sw.ofdm.choose_ssb_offset(
        r,
        params,
        max_beams=spec.max_beams,
        per_port=spec.per_port,
        window_fill=spec.window_fill,
    )


@shared.hint_keywords(specs.Cellular5GNPSSSync)
@registry.signal_trigger(specs.Cellular5GNPSSSync, lag_coord_func=cellular_ssb_lag)
@registry.measurement(
    specs.Cellular5GNPSSSync,
    coord_factories=[],
    dtype='float32',
    caches=(correlator_cache, shared.ssb_iq_cache, sync_cache),
    prefer_iq_source='pre_align',
    store_compressed=False,
    attrs={'standard_name': 'PSS Synchronization Delay', 'units': 's'},
)
def cellular_5g_pss_sync(iq, capture: specs.Capture, **kwargs):
    """compute sync index offsets based on correlate_5g_pss"""

    spec = specs.Cellular5GNPSSSync.from_dict(kwargs).validate()
    offs = choose_sync_offsets(iq, capture=capture, spec=spec)
    delay = round(spec.delay * spec.sample_rate) / spec.sample_rate
    return delay + offs / spec.sample_rate


@shared.hint_keywords(specs.Cellular5GNRPSSCorrelator)
@registry.measurement(
    specs.Cellular5GNRPSSCorrelator,
    coord_factories=_coord_factories,
    dtype=dtype,
    caches=(correlator_cache, shared.ssb_iq_cache),
    prefer_iq_source='pre_align',
    store_compressed=False,
    attrs={'standard_name': 'PSS Cross-Covariance'},
)
def cellular_5g_pss_correlation(
    iq, capture: specs.Capture, **kwargs
) -> tuple[Array, dict]:
    """correlate each channel of the IQ against the cellular primary synchronization signal (PSS) waveform.

    Returns a DataArray containing the time-lag for each combination of NID2, symbol, and SSB start time.

    Args:
        iq: the vector of size (N, M) for N channels and M IQ waveform samples
        capture: capture structure that describes the iq acquisition parameters
        sample_rate (samples/s): downsample to this rate before analysis (or None to follow capture.sample_rate)
        subcarrier_spacing (Hz): OFDM subcarrier spacing
        discovery_periodicity (s): interval between synchronization blocks
        frequency_offset (Hz): baseband center frequency of the synchronization block
        shared_spectrum: whether to assume "shared_spectrum" symbol layout in the SSB
            according to 3GPP TS 138 213: Section 4.1)
        max_block_count: if not None, the number of synchronization blocks to analyze
        as_xarray: if True (default), return an xarray.DataArray, otherwise a ChannelAnalysisResult object

    References:
        3GPP TS 138 211: Table 7.4.3.1-1, Section 7.4.2.2
        3GPP TS 138 213: Section 4.1
    """

    spec = specs.Cellular5GNRPSSCorrelator.from_dict(kwargs).validate()

    R = correlate_5g_pss(iq, capture=capture, spec=spec)

    if spec.max_block_count is not None:
        R = sw.arrays.axis_slice(R, 0, spec.max_block_count, axis=-3)

    enbw = spec.sample_rate
    metadata = {'units': f'√mW/{enbw / 1e6:0.2f} MHz', 'noise_bandwidth': enbw}

    return R, metadata
