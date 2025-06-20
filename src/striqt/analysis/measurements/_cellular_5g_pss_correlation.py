from __future__ import annotations
import typing

from . import shared
from ..lib import dataarrays, register, specs, util

import array_api_compat

if typing.TYPE_CHECKING:
    import iqwaveform
    import numpy as np
    import pandas as pd
    import iqwaveform.type_stubs
else:
    iqwaveform = util.lazy_import('iqwaveform')
    np = util.lazy_import('numpy')
    pd = util.lazy_import('pandas')


class Cellular5GNRPSSCorrelationSpec(
    shared.Cellular5GNRSyncCorrelationSpec,
    forbid_unknown_fields=True,
    cache_hash=True,
    kw_only=True,
    frozen=True,
    dict=True,
):
    pass


_coord_factories = [
    shared.cellular_cell_id2,
    shared.cellular_ssb_start_time,
    shared.cellular_ssb_beam_index,
    shared.cellular_ssb_lag,
]
dtype = 'complex64'


pss_alignment_cache = register.KeywordArgumentCache([dataarrays, 'spec'])


@pss_alignment_cache.apply
def correlate_5g_pss(
    iq: 'iqwaveform.type_stubs.ArrayType',
    *,
    capture: specs.Capture,
    spec: Cellular5GNRPSSCorrelationSpec,
) -> 'iqwaveform.type_stubs.ArrayType':
    xp = iqwaveform.util.array_namespace(iq)

    ssb_iq = shared.get_5g_ssb_iq(iq, capture=capture, spec=spec)

    if ssb_iq is None:
        return shared.empty_5g_sync_measurement(
            iq, capture=capture, spec=spec, coord_factories=_coord_factories
        )

    params = iqwaveform.ofdm.pss_params(
        sample_rate=spec.sample_rate,
        subcarrier_spacing=spec.subcarrier_spacing,
        discovery_periodicity=spec.discovery_periodicity,
        shared_spectrum=spec.shared_spectrum,
    )

    pss_seq = iqwaveform.ofdm.pss_5g_nr(
        spec.sample_rate, spec.subcarrier_spacing, xp=xp
    )

    return shared.correlate_sync_sequence(
        ssb_iq, pss_seq, spec=spec, params=params, cell_id_split=1
    )


@register.alignment_source(
    Cellular5GNRPSSCorrelationSpec, lag_coord_func=shared.cellular_ssb_lag
)
def sync_aggregate_5g_pss(
    iq,
    capture: specs.Capture,
    window_fill=0.5,
    **kwargs: typing.Unpack[shared.Cellular5GNRSyncCorrelationKeywords],
):
    """compute alignment index offsets based on correlate_5g_pss.

    This approach is meant to account for a weighted average of nearby peaks
    to reduce mis-alignment errors in measurements of aggregate interference.

    The underlying heuristic is a triangular weighting function to include energy
    within +/- 1/4 symbol of each peak. Outside of this range, spectrogram errors
    due to "ISI" begin to increase quickly.
    """

    spec = Cellular5GNRPSSCorrelationSpec.fromdict(kwargs).validate()

    xp = iqwaveform.util.array_namespace(iq)

    kwargs['as_xarray'] = False

    R, _ = cellular_5g_pss_correlation(iq, capture, **kwargs)

    # start dimensions: (..., port index, cell Nid2, sync block index, symbol pair index, IQ sample index)
    Ragg = iqwaveform.envtopow(R.sum(axis=(-4, -2)))

    # reduce port index, etc in power space
    Ragg = Ragg.mean(axis=tuple(range(Ragg.ndim - 1)))
    Ragg = Ragg - xp.median(Ragg)
    assert Ragg.ndim == 1

    weights = iqwaveform.get_window(
        'triang',
        nwindow=round(window_fill * Ragg.size),
        nzero=round((1 - window_fill) * Ragg.size),
        norm=False,
        xp=xp,
    )
    weights = xp.roll(weights, round((1 - window_fill) * Ragg.size / 2))

    if array_api_compat.is_cupy_array(Ragg):
        from cupyx.scipy import ndimage
    else:
        from scipy import ndimage

    est = ndimage.correlate1d(Ragg, weights, mode='wrap')
    i = int(est.argmax())

    return shared.cellular_ssb_lag(capture, spec)[i]


@register.measurement(
    Cellular5GNRPSSCorrelationSpec,
    coord_factories=_coord_factories,
    dtype=dtype,
    caches=(pss_alignment_cache, shared.ssb_iq_cache),
    prefer_unaligned_input=True,
    attrs={'standard_name': 'PSS Cross-Covariance', 'units': '√mW'},
)
def cellular_5g_pss_correlation(
    iq,
    capture: specs.Capture,
    **kwargs: typing.Unpack[shared.Cellular5GNRSyncCorrelationKeywords],
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

    R = correlate_5g_pss(iq, capture=capture, spec=spec)

    enbw = spec.subcarrier_spacing * 127
    metadata = {'units': f'mW/{enbw / 1e6:0.2f} MHz'}

    return R, metadata
