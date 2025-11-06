from __future__ import annotations
import typing

from . import shared
from .shared import registry
from ..lib import register, specs, util
from ..lib.dataarrays import CAPTURE_DIM


if typing.TYPE_CHECKING:
    import striqt.waveform as iqwaveform
    import numpy as np
    from striqt.waveform._typing import ArrayType
    import array_api_compat

else:
    iqwaveform = util.lazy_import('striqt.waveform')
    np = util.lazy_import('numpy')
    array_api_compat = util.lazy_import('array_api_compat')


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


pss_correlator_cache = register.KeywordArgumentCache([CAPTURE_DIM, 'spec'])


@pss_correlator_cache.apply
def correlate_5g_pss(
    iq: ArrayType,
    capture: specs.CaptureBase,
    spec: Cellular5GNRPSSCorrelationSpec,
) -> ArrayType:
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


pss_local_weighted_correlator_cache = register.KeywordArgumentCache(
    [CAPTURE_DIM, 'spec', 'window_fill', 'snr_window_fill']
)


def weight_correlation_locally(R, window_fill=0.5, snr_window_fill=0.08):
    xp = iqwaveform.util.array_namespace(R)

    if R.ndim == 4:
        R = R[np.newaxis, ...]

    # R.shape -> (..., port index, cell Nid2, symbol start index, IQ sample index)
    R = R.mean(axis=-3)

    if util.is_cupy_array(R):
        from cupyx.scipy import ndimage
    else:
        from scipy import ndimage

    global Rpow, Rsnr, ipeak, Rpow_corr
    Rpow = iqwaveform.envtopow(R)

    # estimate an SNR
    window_size = round(snr_window_fill * R.shape[-1])
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
    Ragg = Rpow_corr.mean(axis=-4).max(axis=(-3, -2))
    assert Ragg.ndim == 1

    weights = iqwaveform.get_window(
        'triang',
        nwindow=round(window_fill * Ragg.size),
        nzero=round((1 - window_fill) * Ragg.size),
        norm=False,
        xp=xp,
    )
    weight_shift = Ragg.size // 2 - round((1 - window_fill) * Ragg.size / 2)
    weights = xp.roll(weights, weight_shift)

    if util.is_cupy_array(Ragg):
        from cupyx.scipy import ndimage
    else:
        from scipy import ndimage

    return ndimage.correlate1d(Ragg, weights, mode='wrap')


class Cellular5GNRWeightedCorrelationSpec(Cellular5GNRPSSCorrelationSpec):
    window_fill: float = 0.5
    snr_window_fill: float = 0.08


class Cellular5GNRWeightedCorrelationKeywords(
    shared.Cellular5GNRSyncCorrelationKeywords
):
    window_fill: typing.NotRequired[float]
    snr_window_fill: typing.NotRequired[float]


@pss_local_weighted_correlator_cache.apply
def pss_local_weighted_correlator(
    iq: ArrayType,
    capture: specs.CaptureBase,
    *,
    spec: Cellular5GNRPSSCorrelationSpec,
    window_fill=0.5,
    snr_window_fill=0.08,
):
    # R.shape -> (..., port index, cell Nid2, SSB index, symbol start index, IQ sample index)
    R = correlate_5g_pss(iq, capture=capture, spec=spec)

    return weight_correlation_locally(
        R, window_fill=window_fill, snr_window_fill=snr_window_fill
    )


@registry.channel_sync_source(
    Cellular5GNRPSSCorrelationSpec, lag_coord_func=shared.cellular_ssb_lag
)
def cellular_5g_pss_sync(
    iq,
    capture: specs.CaptureBase,
    **kwargs: typing.Unpack[Cellular5GNRWeightedCorrelationKeywords],
):
    """compute sync index offsets based on correlate_5g_pss.

    This approach is meant to account for a weighted average of nearby peaks
    to reduce mis-alignment errors in measurements of aggregate interference.

    The underlying heuristic is a triangular weighting function to include energy
    within +/- 1/4 symbol of each peak. Outside of this range, spectrogram errors
    due to "ISI" begin to increase quickly.
    """

    weighted_spec = Cellular5GNRWeightedCorrelationSpec.fromdict(kwargs).validate()
    spec = Cellular5GNRPSSCorrelationSpec.fromspec(weighted_spec).validate()

    est = pss_local_weighted_correlator(
        iq,
        capture=capture,
        spec=spec,
        window_fill=weighted_spec.window_fill,
        snr_window_fill=weighted_spec.snr_window_fill,
    )

    i = int(est.argmax())

    return shared.cellular_ssb_lag(capture, spec)[i]


@registry.measurement(
    Cellular5GNRWeightedCorrelationSpec,
    coord_factories=[],
    dtype='float32',
    caches=(
        pss_correlator_cache,
        shared.ssb_iq_cache,
        pss_local_weighted_correlator_cache,
    ),
    prefer_unaligned_input=True,
    store_compressed=False,
    attrs={'standard_name': 'PSS Synchronization Delay', 'units': 's'},
)
def cellular_5g_pss_sync_offset(
    iq,
    capture: specs.CaptureBase,
    **kwargs: typing.Unpack[Cellular5GNRWeightedCorrelationKeywords],
):
    return cellular_5g_pss_sync(iq, capture, **kwargs)


@registry.measurement(
    Cellular5GNRPSSCorrelationSpec,
    coord_factories=_coord_factories,
    dtype=dtype,
    caches=(pss_correlator_cache, shared.ssb_iq_cache),
    prefer_unaligned_input=True,
    store_compressed=False,
    attrs={'standard_name': 'PSS Cross-Covariance'},
)
def cellular_5g_pss_correlation(
    iq,
    capture: specs.CaptureBase,
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

    enbw = spec.sample_rate
    metadata = {'units': f'√mW/{enbw / 1e6:0.2f} MHz', 'noise_bandwidth': enbw}

    return R, metadata
