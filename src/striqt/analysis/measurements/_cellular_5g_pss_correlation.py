from __future__ import annotations as __

import typing

from .. import specs

from ..lib import register, util
from ..lib.dataarrays import CAPTURE_DIM
from . import shared
from .shared import registry

if typing.TYPE_CHECKING:
    import array_api_compat
    import numpy as np

    import striqt.waveform as waveform
    from striqt.waveform._typing import ArrayType

else:
    waveform = util.lazy_import('striqt.waveform')
    np = util.lazy_import('numpy')
    array_api_compat = util.lazy_import('array_api_compat')


_coord_factories = [
    shared.cellular_cell_id2,
    shared.cellular_ssb_start_time,
    shared.cellular_ssb_beam_index,
    shared.cellular_ssb_lag,
]
dtype = 'complex64'


pss_cache = register.KeywordArgumentCache([CAPTURE_DIM, 'spec'])


@pss_cache.apply
def correlate_5g_pss(
    iq: ArrayType,
    capture: specs.Capture,
    spec: specs.Cellular5GNRPSSCorrelator,
) -> ArrayType:
    xp = waveform.util.array_namespace(iq)

    ssb_iq = shared.get_5g_ssb_iq(iq, capture=capture, spec=spec)

    if ssb_iq is None:
        return shared.empty_5g_ssb_correlation(
            iq, capture=capture, spec=spec, coord_factories=_coord_factories
        )

    params = waveform.ofdm.pss_params(
        sample_rate=spec.sample_rate,
        subcarrier_spacing=spec.subcarrier_spacing,
        discovery_periodicity=spec.discovery_periodicity,
        shared_spectrum=spec.shared_spectrum,
    )

    pss_seq = waveform.ofdm.pss_5g_nr(
        spec.sample_rate, spec.subcarrier_spacing, xp=xp
    )

    return shared.correlate_sync_sequence(
        ssb_iq, pss_seq, spec=spec, params=params, cell_id_split=1
    )


pss_weighted_cache = register.KeywordArgumentCache(
    [CAPTURE_DIM, 'spec', 'window_fill', 'snr_window_fill']
)


def weight_correlation_locally(R, spec: specs.Cellular5GNPSSSync):
    xp = waveform.util.array_namespace(R)

    if R.ndim == 4:
        R = R[np.newaxis, ...]

    # R.shape -> (..., port index, cell Nid2, symbol start index, IQ sample index)
    R = R.mean(axis=-3)

    if util.is_cupy_array(R):
        from cupyx.scipy import ndimage  # type: ignore
    else:
        from scipy import ndimage

    Rpow = waveform.envtopow(R)

    # estimate an SNR
    window_size = round(spec.snr_window_fill * R.shape[-1])
    Rpow_median = ndimage.median_filter(
        Rpow, size=(Rpow.ndim - 1) * (1,) + (window_size,)
    )
    Rsnr = Rpow / Rpow_median

    # scale by the 
    ipeak = xp.argmax(Rsnr, axis=-1, keepdims=True)

    Rpow_corr = Rsnr * (
        np.take_along_axis(Rpow, ipeak, axis=-1)
        / np.take_along_axis(Rsnr, ipeak, axis=-1)
    )

    # Ragg.shape: (IQ sample index,)
    if not spec.per_port:
        Rpow_corr = Rpow_corr.mean(axis=-4, keepdims=True)
    Ragg = Rpow_corr.max(axis=(-3, -2))
    assert Ragg.ndim == 2

    fill_count = round(spec.window_fill * Ragg.shape[1])

    weights = waveform.get_window(
        'triang',
        nwindow=fill_count,
        nzero=Ragg.shape[1] - fill_count,
        norm=False,
        xp=xp,
    )

    weight_shift = Ragg.shape[1] // 2 - round((Ragg.shape[1] - fill_count) / 2)
    weights = xp.roll(weights, weight_shift)

    if util.is_cupy_array(Ragg):
        from cupyx.scipy import ndimage  # type: ignore
    else:
        from scipy import ndimage

    offs = ndimage.correlate1d(Ragg, weights, mode='wrap', axis=1)
    return offs


@pss_weighted_cache.apply
def pss_local_weighted_correlator(
    iq: ArrayType,
    capture: specs.Capture,
    *,
    spec: specs.Cellular5GNPSSSync,
) -> ArrayType:
    # R.shape -> (..., port index, cell Nid2, SSB index, symbol start index, IQ sample index)

    corr_spec = specs.Cellular5GNRPSSCorrelator.from_spec(spec).validate()

    R = correlate_5g_pss(iq, capture=capture, spec=corr_spec)

    return weight_correlation_locally(R, spec)


@shared.hint_keywords(specs.Cellular5GNPSSSync)
@registry.signal_trigger(
    specs.Cellular5GNPSSSync, lag_coord_func=shared.cellular_ssb_lag
)
@registry.measurement(
    specs.Cellular5GNPSSSync,
    coord_factories=[],
    dtype='float32',
    caches=(pss_cache, shared.ssb_iq_cache, pss_weighted_cache),
    prefer_unaligned_input=True,
    store_compressed=False,
    attrs={'standard_name': 'PSS Synchronization Delay', 'units': 's'},
)
def cellular_5g_pss_sync(iq, capture: specs.Capture, **kwargs):
    """compute sync index offsets based on correlate_5g_pss.

    This approach is meant to account for a weighted average of nearby peaks
    to reduce mis-alignment errors in measurements of aggregate interference.

    The underlying heuristic is a triangular weighting function to include energy
    within +/- 1/4 symbol of each peak. Outside of this range, spectrogram errors
    due to "ISI" begin to increase quickly.
    """

    spec = specs.Cellular5GNPSSSync.from_dict(kwargs).validate()

    est = pss_local_weighted_correlator(
        iq,
        capture=capture,
        spec=spec,
    )

    i = est.argmax(axis=1)

    return i/spec.sample_rate#shared.cellular_ssb_lag(capture, spec)[i]


@shared.hint_keywords(specs.Cellular5GNRPSSCorrelator)
@registry.measurement(
    specs.Cellular5GNRPSSCorrelator,
    coord_factories=_coord_factories,
    dtype=dtype,
    caches=(pss_cache, shared.ssb_iq_cache),
    prefer_unaligned_input=True,
    store_compressed=False,
    attrs={'standard_name': 'PSS Cross-Covariance'},
)
def cellular_5g_pss_correlation(iq, capture: specs.Capture, **kwargs):
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

    spec = specs.Cellular5GNRPSSCorrelator.from_dict(kwargs).validate()

    R = correlate_5g_pss(iq, capture=capture, spec=spec)

    enbw = spec.sample_rate
    metadata = {'units': f'√mW/{enbw / 1e6:0.2f} MHz', 'noise_bandwidth': enbw}

    return R, metadata
