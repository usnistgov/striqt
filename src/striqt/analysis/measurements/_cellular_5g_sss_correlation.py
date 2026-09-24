from __future__ import annotations as __

import typing

from .. import specs

from ..lib import register
from ..lib.dataarrays import CAPTURE_DIM
from . import shared
from .shared import registry

import striqt.waveform as sw

if typing.TYPE_CHECKING:
    from ..lib.typing import Array


_coord_factories = [
    shared.cellular_cell_id2,
    shared.cellular_ssb_start_time,
    shared.cellular_ssb_beam_index,
    shared.cellular_ssb_lag,
]
dtype = 'complex64'


correlator_cache = register.KwArgCache([CAPTURE_DIM, 'spec'])


@correlator_cache.apply
def correlate_5g_sss(
    iq: 'Array',
    capture: specs.Capture,
    spec: specs.Cellular5GNRSSSCorrelator,
) -> 'Array':
    xp = sw.array_namespace(iq)

    ssb_iq = shared.get_5g_ssb_iq(iq, capture=capture, spec=spec)

    params = shared.sync_params(capture, spec, 'sss')
    sss_seq = sw.ofdm.sss_5g_nr(spec.sample_rate, spec.subcarrier_spacing, xp=xp)

    return sw.ofdm.correlate_sync_sequence(
        ssb_iq, sss_seq, params=params, cell_id_split=None
    )


sync_cache = register.KwArgCache([CAPTURE_DIM, 'spec'])


@sync_cache.apply
def choose_sync_offsets(
    iq: Array,
    capture: specs.Capture,
    *,
    spec: specs.Cellular5GNSSSSync,
) -> Array:
    # R.shape -> (..., port index, cell Nid2, SSB index, symbol start index, IQ sample index)

    corr_spec = specs.Cellular5GNRSSSCorrelator.from_spec(spec).validate()

    r = correlate_5g_sss(iq, capture=capture, spec=corr_spec)
    params = shared.sync_params(capture, spec, 'sss')
    return sw.ofdm.choose_ssb_offset(
        r,
        params,
        max_beams=spec.max_beams,
        per_port=spec.per_port,
        window_fill=spec.window_fill,
    )


@shared.hint_keywords(specs.Cellular5GNSSSSync)
@registry.signal_trigger(
    specs.Cellular5GNSSSSync, lag_coord_func=shared.cellular_ssb_lag
)
@registry.measurement(
    specs.Cellular5GNSSSSync,
    coord_factories=[],
    dtype='float32',
    caches=(correlator_cache, shared.ssb_iq_cache, sync_cache),
    prefer_iq_source='pre_align',
    store_compressed=False,
    attrs={'standard_name': 'SSS Synchronization Delay', 'units': 's'},
    validate=shared.validated_5g_ssb_sync_params,
)
def cellular_5g_sss_sync(iq, capture: specs.Capture, **kwargs):
    """compute sync index offsets based on correlate_5g_sss"""

    spec = specs.Cellular5GNSSSSync.from_dict(kwargs).validate()
    offs = choose_sync_offsets(iq, capture=capture, spec=spec)
    delay = round(spec.delay * spec.sample_rate) / spec.sample_rate
    return delay + offs / spec.sample_rate


@shared.hint_keywords(specs.Cellular5GNRSSSCorrelator)
@registry.measurement(
    specs.Cellular5GNRSSSCorrelator,
    coord_factories=_coord_factories,
    dtype=dtype,
    caches=(correlator_cache, shared.ssb_iq_cache),
    prefer_iq_source='pre_align',
    store_compressed=False,
    attrs={'standard_name': 'SSS Cross-Covariance'},
    validate=shared.validated_5g_ssb_sync_params,
)
def cellular_5g_sss_correlation(
    iq, capture: specs.Capture, **kwargs
) -> tuple[Array, dict]:
    """correlate each channel of the IQ against the cellular secondary synchronization signal (SSS) waveform.

    Returns a DataArray containing the time-lag for each combination of NID2, symbol, and SSB start time.

    Args:
    {args}

    References:
        3GPP TS 138 211: Table 7.4.3.1-1, Section 7.4.2.2
        3GPP TS 138 213: Section 4.1
    """

    spec = specs.Cellular5GNRSSSCorrelator.from_dict(kwargs).validate()

    R = correlate_5g_sss(iq, capture=capture, spec=spec)

    if spec.max_block_count is not None:
        R = sw.arrays.axis_slice(R, 0, spec.max_block_count, axis=-3)

    enbw = spec.sample_rate
    metadata = {'units': f'√mW/{enbw / 1e6:0.2f} MHz', 'noise_bandwidth': enbw}

    return R, metadata
