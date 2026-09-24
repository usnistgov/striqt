from __future__ import annotations as __

import typing
from fractions import Fraction
from math import ceil
from typing import Any, Literal

from .. import specs

from ..lib import register, util
from ..lib.dataarrays import CAPTURE_DIM
from ..lib.util import np, pd
from . import power, shared, spectrum
from .shared import registry, hint_keywords

import striqt.waveform as sw

if typing.TYPE_CHECKING:
    from ..lib.typing import Array
    from ..specs.structs import _Cellular5GNRSSBCorrelator, _Cellular5GNRSSBSync


# %% 5G NR synchronization block helpers
@specs.helpers.lru_cache_on_converted(specs.AnalysisCapture)
def sync_params(
    capture: specs.AnalysisCapture,
    spec: _Cellular5GNRSSBCorrelator,
    kind: Literal['pss', 'sss'],
) -> sw.ofdm.SyncParams:
    """the 3GPP synchronization layout implied by `spec`, from `sw.ofdm.pss_params`
    or `sw.ofdm.sss_params` according to `kind`"""
    if kind == 'pss':
        build = sw.ofdm.pss_params
    else:
        build = sw.ofdm.sss_params

    return build(
        sample_rate=spec.sample_rate,
        subcarrier_spacing=spec.subcarrier_spacing,
        discovery_periodicity=spec.discovery_periodicity,
        shared_spectrum=spec.shared_spectrum,
        max_lag_symbols=spec.max_lag_symbols,
        symbol_indexes=spec.symbol_indexes,
        center_frequency=capture.center_frequency,
    )


def ssb_block_count(
    duration: float,
    spec: specs.Cellular5GNRPSSCorrelator
    | specs.Cellular5GNRSSSCorrelator
    | specs.Cellular5GNRSSBSpectrogram,
) -> int:
    """the number of synchronization blocks in `duration`, limited to
    `spec.max_block_count` and never fewer than 1"""
    total_blocks = round(duration / spec.discovery_periodicity)
    if spec.max_block_count is None:
        count = total_blocks
    else:
        count = min(spec.max_block_count, total_blocks)

    return max(count, 1)


@registry.coordinates(
    dtype='uint16', attrs={'standard_name': r'Cell Sector ID ($N_{ID}^\text{(2)}$)'}
)
@specs.helpers.lru_cache_on_converted(specs.Capture)
def cellular_cell_id2(capture: specs.Capture, spec: Any):
    values = np.array([0, 1, 2], dtype='uint16')
    return values


@registry.coordinates(dtype='uint16', attrs={'standard_name': 'SSB beam index'})
@specs.helpers.lru_cache_on_converted(specs.AnalysisCapture)
def cellular_ssb_beam_index(capture: specs.AnalysisCapture, spec: _Cellular5GNRSSBSync):
    # pss_params and sss_params return the same number of symbol indexes
    params = sync_params(capture, spec, 'sss')

    return list(range(len(params.symbol_indexes)))


@registry.coordinates(
    dtype='float32', attrs={'standard_name': 'Time Elapsed', 'units': 's'}
)
@specs.helpers.lru_cache_on_converted(specs.Capture)
def cellular_ssb_start_time(
    capture: specs.Capture,
    spec: specs.Cellular5GNRPSSCorrelator | specs.Cellular5GNRSSSCorrelator,
):
    # the bare Capture projection carries no center_frequency, so the cell search
    # case is resolved here as for an unknown band
    params = sync_params(capture, spec, 'pss')
    count = ssb_block_count(params.duration, spec)

    return np.arange(count).astype('float32') * spec.discovery_periodicity


@registry.coordinates(dtype='float32', attrs={'standard_name': 'Lag', 'units': 's'})
@specs.helpers.lru_cache_on_converted(specs.AnalysisCapture)
def cellular_ssb_lag(capture: specs.AnalysisCapture, spec: _Cellular5GNRSSBCorrelator):
    # pss_params and sss_params agree on lag_count
    params = sync_params(capture, spec, 'pss')
    offs = round(spec.sample_rate * spec.delay)
    return np.arange(offs, offs + params.lag_count) / spec.sample_rate


_SSB_CORRELATION_COORDS = [
    cellular_cell_id2,
    cellular_ssb_start_time,
    cellular_ssb_beam_index,
    cellular_ssb_lag,
]


@specs.helpers.lru_cache_on_converted(
    specs.AnalysisCapture, specs.Cellular5GNRSSSCorrelator
)
def validated_5g_ssb_sync_params(
    capture: specs.AnalysisCapture, spec: _Cellular5GNRSSBCorrelator
) -> sw.ofdm.SyncParams:
    """check the 3GPP sync layout implied by a (capture, SSB correlator spec) pair,
    returning the sync parameters it resolves to.

    `sss_params` is the stricter of the two parameter builders and reaches every check
    in `pss_params`, so one validator covers the PSS and SSS measurements alike. It
    also warms the `get_3gpp_phy` design, which the correlators would otherwise pay for
    after the acquisition.

    The resampling arithmetic mirrors `get_5g_ssb_iq` on the whole capture, the way
    `get_5g_ssb_iq` here calls it.
    """
    params = sync_params(capture, spec, 'sss')

    size_in = shared.capture_sample_count(capture)
    frequency_step = capture.sample_rate / size_in
    if not sw.isroundmod(spec.frequency_offset, frequency_step):
        raise ValueError(
            'frequency_offset must be a counting-number multiple of '
            'sample_rate/duration, the frequency step of the resampler '
            f'(frequency_offset: {spec.frequency_offset}, '
            f'sample_rate: {capture.sample_rate}, duration: {capture.duration}, '
            f'frequency step: {frequency_step})'
        )

    size_out = round(size_in * spec.sample_rate / capture.sample_rate)
    shift = round(size_in * spec.frequency_offset / capture.sample_rate)
    try:
        sw.fourier.resample_edges(size_in, size_out, shift)
    except ValueError as ex:
        raise ValueError(
            f'the capture cannot be resampled to the synchronization block: {ex} '
            f'(duration: {capture.duration}, capture sample_rate: '
            f'{capture.sample_rate}, sample_rate: {spec.sample_rate}, '
            f'frequency_offset: {spec.frequency_offset}, '
            f'capture samples: {size_in}, block samples: {size_out})'
        ) from ex

    try:
        sw.ofdm.sync_frame_count(size_out, params)
    except ValueError as ex:
        raise ValueError(
            f'duration must hold whole 10 ms frames for the correlator: {ex} '
            f'(duration: {capture.duration}, sample_rate: {spec.sample_rate}, '
            f'subcarrier_spacing: {spec.subcarrier_spacing}, '
            f'symbol_indexes: {spec.symbol_indexes!r}, '
            f'max_lag_symbols: {spec.max_lag_symbols})'
        ) from ex

    return params


ssb_iq_cache = register.KwArgCache([CAPTURE_DIM, 'spec'])


@ssb_iq_cache.apply
def get_5g_ssb_iq(
    iq: Array,
    capture: specs.Capture,
    spec: _Cellular5GNRSSBCorrelator,
    oaresample=False,
) -> Array:
    """return a sync block waveform, which returns IQ that is recentered
    at baseband frequency spec.frequency_offset and downsampled to spec.sample_rate."""

    return sw.ofdm.get_5g_ssb_iq(
        iq,
        discovery_periodicity=spec.discovery_periodicity,
        fs_out=spec.sample_rate,
        fs_in=capture.sample_rate,
        subcarrier_spacing=spec.subcarrier_spacing,
        frequency_offset=spec.frequency_offset,
        delay=spec.delay,
        # max_block_count=spec.max_block_count,
        oaresample=oaresample,
    )


def cellular_stft_window_fractions(
    cyclic_prefix: Literal['normal', 'extended'],
) -> tuple[Fraction, Fraction]:
    """the (fractional_overlap, window_fill) that hop one STFT window per OFDM symbol.

    They hold only at a frequency_resolution of half the subcarrier spacing, where one
    FFT window spans 2 subcarrier periods and a slot spans 15 of them (3GPP TS 38.211
    Section 5.3.1), so a `window_fill` of 15/28 is the mean share of the window taken by
    one of the 14 normal-cyclic-prefix symbols in a slot.
    """
    if cyclic_prefix == 'normal':
        return Fraction(13, 28), Fraction(15, 28)
    else:
        # cyclic_prefix is a Literal, so 'extended' is the only other value
        return Fraction(11, 24), Fraction(13, 24)


# %% cellular_5g_pss_correlation and cellular_5g_pss_sync
_pss_correlator_cache = register.KwArgCache([CAPTURE_DIM, 'spec'])


@_pss_correlator_cache.apply
def correlate_5g_pss(
    iq: 'Array',
    capture: specs.Capture,
    spec: specs.Cellular5GNRPSSCorrelator,
) -> 'Array':
    xp = sw.array_namespace(iq)

    ssb_iq = get_5g_ssb_iq(iq, capture=capture, spec=spec)

    params = sync_params(capture, spec, 'pss')
    pss_seq = sw.ofdm.pss_5g_nr(spec.sample_rate, spec.subcarrier_spacing, xp=xp)

    return sw.ofdm.correlate_sync_sequence(
        ssb_iq, pss_seq, params=params, cell_id_split=1
    )


_pss_sync_cache = register.KwArgCache([CAPTURE_DIM, 'spec'])


@_pss_sync_cache.apply
def choose_pss_sync_offsets(
    iq: Array,
    capture: specs.Capture,
    *,
    spec: specs.Cellular5GNPSSSync,
) -> Array:
    # R.shape -> (..., port index, cell Nid2, SSB index, symbol start index, IQ sample index)

    corr_spec = specs.Cellular5GNRPSSCorrelator.from_spec(spec).validate()

    r = correlate_5g_pss(iq, capture=capture, spec=corr_spec)
    params = sync_params(capture, spec, 'pss')
    return sw.ofdm.choose_ssb_offset(
        r,
        params,
        max_beams=spec.max_beams,
        per_port=spec.per_port,
        window_fill=spec.window_fill,
    )


@hint_keywords(specs.Cellular5GNPSSSync)
@registry.signal_trigger(specs.Cellular5GNPSSSync, lag_coord_func=cellular_ssb_lag)
@registry.measurement(
    specs.Cellular5GNPSSSync,
    coord_factories=[],
    dtype='float32',
    caches=(_pss_correlator_cache, ssb_iq_cache, _pss_sync_cache),
    prefer_iq_source='pre_align',
    store_compressed=False,
    attrs={'standard_name': 'PSS Synchronization Delay', 'units': 's'},
    validate=validated_5g_ssb_sync_params,
)
def cellular_5g_pss_sync(iq, capture: specs.Capture, **kwargs):
    """compute sync index offsets based on correlate_5g_pss"""

    spec = specs.Cellular5GNPSSSync.from_dict(kwargs).validate()
    offs = choose_pss_sync_offsets(iq, capture=capture, spec=spec)
    delay = round(spec.delay * spec.sample_rate) / spec.sample_rate
    return delay + offs / spec.sample_rate


@hint_keywords(specs.Cellular5GNRPSSCorrelator)
@registry.measurement(
    specs.Cellular5GNRPSSCorrelator,
    coord_factories=_SSB_CORRELATION_COORDS,
    dtype='complex64',
    caches=(_pss_correlator_cache, ssb_iq_cache),
    prefer_iq_source='pre_align',
    store_compressed=False,
    attrs={'standard_name': 'PSS Cross-Covariance'},
    validate=validated_5g_ssb_sync_params,
)
def cellular_5g_pss_correlation(
    iq, capture: specs.Capture, **kwargs
) -> tuple[Array, dict]:
    """correlate each channel of the IQ against the cellular primary synchronization signal (PSS) waveform.

    Returns a DataArray containing the time-lag for each combination of NID2, symbol, and SSB start time.

    Args:
    {args}

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


# %% cellular_5g_sss_correlation and cellular_5g_sss_sync
_sss_correlator_cache = register.KwArgCache([CAPTURE_DIM, 'spec'])


@_sss_correlator_cache.apply
def correlate_5g_sss(
    iq: 'Array',
    capture: specs.Capture,
    spec: specs.Cellular5GNRSSSCorrelator,
) -> 'Array':
    xp = sw.array_namespace(iq)

    ssb_iq = get_5g_ssb_iq(iq, capture=capture, spec=spec)

    params = sync_params(capture, spec, 'sss')
    sss_seq = sw.ofdm.sss_5g_nr(spec.sample_rate, spec.subcarrier_spacing, xp=xp)

    return sw.ofdm.correlate_sync_sequence(
        ssb_iq, sss_seq, params=params, cell_id_split=None
    )


_sss_sync_cache = register.KwArgCache([CAPTURE_DIM, 'spec'])


@_sss_sync_cache.apply
def choose_sss_sync_offsets(
    iq: Array,
    capture: specs.Capture,
    *,
    spec: specs.Cellular5GNSSSSync,
) -> Array:
    # R.shape -> (..., port index, cell Nid2, SSB index, symbol start index, IQ sample index)

    corr_spec = specs.Cellular5GNRSSSCorrelator.from_spec(spec).validate()

    r = correlate_5g_sss(iq, capture=capture, spec=corr_spec)
    params = sync_params(capture, spec, 'sss')
    return sw.ofdm.choose_ssb_offset(
        r,
        params,
        max_beams=spec.max_beams,
        per_port=spec.per_port,
        window_fill=spec.window_fill,
    )


@hint_keywords(specs.Cellular5GNSSSSync)
@registry.signal_trigger(specs.Cellular5GNSSSSync, lag_coord_func=cellular_ssb_lag)
@registry.measurement(
    specs.Cellular5GNSSSSync,
    coord_factories=[],
    dtype='float32',
    caches=(_sss_correlator_cache, ssb_iq_cache, _sss_sync_cache),
    prefer_iq_source='pre_align',
    store_compressed=False,
    attrs={'standard_name': 'SSS Synchronization Delay', 'units': 's'},
    validate=validated_5g_ssb_sync_params,
)
def cellular_5g_sss_sync(iq, capture: specs.Capture, **kwargs):
    """compute sync index offsets based on correlate_5g_sss"""

    spec = specs.Cellular5GNSSSSync.from_dict(kwargs).validate()
    offs = choose_sss_sync_offsets(iq, capture=capture, spec=spec)
    delay = round(spec.delay * spec.sample_rate) / spec.sample_rate
    return delay + offs / spec.sample_rate


@hint_keywords(specs.Cellular5GNRSSSCorrelator)
@registry.measurement(
    specs.Cellular5GNRSSSCorrelator,
    coord_factories=_SSB_CORRELATION_COORDS,
    dtype='complex64',
    caches=(_sss_correlator_cache, ssb_iq_cache),
    prefer_iq_source='pre_align',
    store_compressed=False,
    attrs={'standard_name': 'SSS Cross-Covariance'},
    validate=validated_5g_ssb_sync_params,
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


# %% cellular_5g_ssb_spectrogram
@registry.coordinates(dtype='uint16', attrs={'standard_name': 'Symbols elapsed'})
@specs.helpers.lru_cache_on_converted(specs.Capture)
def cellular_ssb_symbol_index(
    capture: specs.Capture, spec: specs.Cellular5GNRSSBSpectrogram
):
    symbol_count = round(28 * spec.subcarrier_spacing / 15e3)
    return np.arange(symbol_count, dtype='uint16')


@registry.coordinates(
    dtype='float64', attrs={'standard_name': 'SSB Baseband Frequency', 'units': 'Hz'}
)
@specs.helpers.lru_cache_on_converted(specs.Capture)
def cellular_ssb_baseband_frequency(
    capture: specs.Capture, spec: specs.Cellular5GNRSSBSpectrogram, xp=np
) -> np.ndarray:
    nfft = round(2 * capture.sample_rate / spec.subcarrier_spacing)
    bb_freqs = sw.fftfreq(nfft, capture.sample_rate)
    bb_freqs = sw.binned_mean(bb_freqs, count=2, axis=0, fft=True)
    bb_freqs = sw.fourier.truncate_freqs(
        bb_freqs,
        nfft // 2,
        capture.sample_rate,
        spec.sample_rate,
        offset=spec.frequency_offset,
        axis=0,
    )

    return bb_freqs - spec.frequency_offset


@registry.coordinates(dtype='uint16', attrs={'standard_name': 'Capture SSB index'})
@specs.helpers.lru_cache_on_converted(specs.Capture)
def cellular_ssb_index(capture: specs.Capture, spec: specs.Cellular5GNRSSBSpectrogram):
    count = ssb_block_count(capture.duration, spec)
    return np.arange(count, dtype='uint16')


_SSB_SPECTROGRAM_COORDS = [
    cellular_ssb_index,
    cellular_ssb_symbol_index,
    cellular_ssb_baseband_frequency,
]


@util.lru_cache()
def _ssb_spectrogram_spec(spec: specs.Cellular5GNRSSBSpectrogram) -> specs.Spectrogram:
    """the STFT that lands one bin on each half-subcarrier and one hop on each symbol"""
    fractional_overlap, window_fill = cellular_stft_window_fractions('normal')

    return specs.Spectrogram(
        frequency_resolution=spec.subcarrier_spacing / 2,
        fractional_overlap=fractional_overlap,
        window_fill=window_fill,
        window=spec.window,
        lo_bandstop=spec.lo_bandstop,
        integration_bandwidth=spec.subcarrier_spacing,
        trim_stopband=False,
    )


def validated_ssb_spectrogram_sizing(
    capture: specs.Capture, spec: specs.Cellular5GNRSSBSpectrogram
) -> spectrum.SpectrogramSizing:
    """check the STFT sizing of the symbol-resolved SSB spectrogram.

    The 3GPP cell-search parameters are deliberately not consulted: this measurement
    lays symbols out from `subcarrier_spacing` and `discovery_periodicity` alone, and
    unlike the correlators it has no `symbol_indexes` field to pick a search case with.
    """
    sizing = spectrum.validated_spectrogram_sizing(capture, _ssb_spectrogram_spec(spec))

    # the frequency axis is binned to one bin per subcarrier before the band is cut
    shared.check_frequency_band(
        'sample_rate',
        round(capture.sample_rate / spec.subcarrier_spacing),
        capture.sample_rate,
        spec.sample_rate,
        offset=spec.frequency_offset,
    )

    symbol_count = _burst_symbol_count(spec)
    discovery_symbols = _discovery_symbol_count(spec)
    window_count = spectrum.spectrogram_window_count(capture, sizing)
    full_periods, trailing = divmod(window_count, discovery_symbols)
    kept = full_periods * min(discovery_symbols, symbol_count)
    kept += min(trailing, symbol_count)
    if kept == 0 or kept % symbol_count != 0:
        raise ValueError(
            'duration must end on a whole discovery_periodicity or after a complete '
            f'burst set of {symbol_count} symbols '
            f'(duration: {capture.duration}, '
            f'discovery_periodicity: {spec.discovery_periodicity}, '
            f'subcarrier_spacing: {spec.subcarrier_spacing}, '
            f'symbols in capture: {window_count}, '
            f'symbols per discovery period: {discovery_symbols}, '
            f'burst symbols kept: {kept})'
        )

    return sizing


def _burst_symbol_count(spec: specs.Cellular5GNRSSBSpectrogram) -> int:
    """the symbols in the first two slots of a frame, which hold the SSB burst set"""
    return round(28 * spec.subcarrier_spacing / 15e3)


def _discovery_symbol_count(spec: specs.Cellular5GNRSSBSpectrogram) -> int:
    # TODO: this is normal CP; support extended CP?
    symbol_period = sw.ofdm.slot_period(spec.subcarrier_spacing) / 14
    discovery_symbols = round(spec.discovery_periodicity / symbol_period)
    if discovery_symbols < 1:
        raise ValueError(
            'discovery_periodicity must span at least one OFDM symbol '
            f'(discovery_periodicity: {spec.discovery_periodicity}, '
            f'symbol_period: {symbol_period})'
        )
    return discovery_symbols


def ssb_spectrogram_tolerance(
    capture: specs.Capture, spec: specs.Cellular5GNRSSBSpectrogram, **kwargs
) -> specs.Tolerance:
    return spectrum.spectrogram_level_tolerance(
        capture, _ssb_spectrogram_spec(spec), dtype='float16', limit_digits=3, **kwargs
    )


@hint_keywords(specs.Cellular5GNRSSBSpectrogram)
@registry.measurement(
    specs.Cellular5GNRSSBSpectrogram,
    coord_factories=_SSB_SPECTROGRAM_COORDS,
    dtype='float16',
    caches=(spectrum.spectrogram_cache,),
    prefer_iq_source='pre_filter',
    attrs={'standard_name': 'SSB Spectrogram'},
    validate=validated_ssb_spectrogram_sizing,
    tolerance=ssb_spectrogram_tolerance,
)
def cellular_5g_ssb_spectrogram(iq, capture: specs.Capture, **kwargs):
    """spectrogram of each 5G NR synchronization signal block (SSB) burst set, resolved to OFDM symbol and subcarrier.

    The STFT hops once per OFDM symbol, and its half-subcarrier bins are integrated
    pairwise to one bin per subcarrier of the given `subcarrier_spacing` before the
    frequency axis is trimmed to the `sample_rate` band about `frequency_offset`. Only
    the symbols of the burst set at the start of each `discovery_periodicity` are kept.

    Returns a DataArray of power spectral density indexed by SSB burst set within the
    capture, OFDM symbol within the burst set, and baseband frequency.

    Args:
    {args}

    References:
        3GPP TS 138 213: Section 4.1
    """

    spec = specs.Cellular5GNRSSBSpectrogram.from_dict(kwargs).validate()

    symbol_count = _burst_symbol_count(spec)
    discovery_symbols = _discovery_symbol_count(spec)

    spg, attrs = spectrum.evaluate_spectrogram(
        iq,
        capture=capture,
        spec=_ssb_spectrogram_spec(spec),
        limit_digits=3,
        dtype='float16',
    )

    # keep only the first two slots in the frame
    symbol_index = np.arange(spg.shape[1])
    mask_time = (symbol_index % discovery_symbols) < symbol_count
    spg = spg[:, mask_time]

    # select frequency
    nfft = round(capture.sample_rate / spec.subcarrier_spacing)
    spg = sw.fourier.truncate_freqs(
        spg,
        nfft,
        capture.sample_rate,
        spec.sample_rate,
        offset=spec.frequency_offset,
        axis=2,
    )

    # split by synchronization block
    spg = spg.reshape((spg.shape[0], -1, symbol_count, spg.shape[2]))

    return spg, attrs


# %% cellular_cyclic_autocorrelation
class SlotBySymbol(typing.TypedDict):
    d: str
    u: str
    s: str | None


class NormalizedTDDSlotConfig(typing.NamedTuple):
    frame_slots: str
    special_symbols: str | None
    code_maps: dict[str, dict[str, float]]
    slot_by_symbol: SlotBySymbol
    downlink_slot_indexes: tuple[int, ...]
    uplink_slot_indexes: tuple[int, ...]
    frame_by_symbol: str


@util.lru_cache()
def tdd_config_from_str(
    subcarrier_spacing: float,
    frame_slots: str | None,
    special_symbols: str | None = None,
    *,
    normal_cp=True,
    flex_as=None,
) -> NormalizedTDDSlotConfig:
    """generate a symbol-by-symbol sequence of masking arrays for uplink and downlink.

    The number of slots given in the frame match the appropriate number for a given
    5G NR or LTE subcarrier spacing.

    Arguments:
        frame_slots: a string composed of the characters {'d', 'u', 's'} that
            indicate the sequence of slots in 1 cellular frame (or None for all downlink)
        special_symbols: the a string composed of the characters {'d', 'u', 'f'} that
            indicate the sequence of symbol types in the special slot (or None for all downlink)

    Returns:
        a NormalizedTDDSlotConfig object containing more detailed parameters
    """

    expect_slot_count = round(10 * subcarrier_spacing / 15e3)

    if flex_as is not None:
        flex_as = flex_as.lower()

    if frame_slots is None:
        frame_slots = expect_slot_count * 'd'
    elif len(frame_slots) == 1:
        frame_slots = expect_slot_count * frame_slots
    elif len(frame_slots) != expect_slot_count:
        raise ValueError(
            f'frame_slots must have length {expect_slot_count} to match the slot count at {round(subcarrier_spacing / 1e3)} kHz '
            f'(frame_slots: {frame_slots}, '
            f'subcarrier_spacing: {subcarrier_spacing})'
        )
    else:
        frame_slots = frame_slots.lower()

    if special_symbols is not None:
        special_symbols = special_symbols.lower()

    if len(frame_slots.strip('dus')) > 0:
        allowed = set('dus')
        raise ValueError(
            f'frame_slots string may only contain {allowed} '
            f'(frame_slots: {frame_slots})'
        )

    if special_symbols is None:
        pass
    elif len(special_symbols.strip('duf')) > 0:
        allowed = set('duf')
        raise ValueError(
            f'special_symbols string may only contain {allowed} '
            f'(special_symbols: {special_symbols})'
        )

    if normal_cp:
        symbols_per_slot = 14
    else:
        symbols_per_slot = 12

    downlink_code_to_value = {
        'd': 1.0,
        'u': float('nan'),
        'f': 1.0 if flex_as == 'd' else float('nan'),
    }
    uplink_code_to_value = {
        'd': float('nan'),
        'u': 1.0,
        'f': 1.0 if flex_as == 'u' else float('nan'),
    }

    code_mapping = {'downlink': downlink_code_to_value, 'uplink': uplink_code_to_value}

    slot_by_symbol = SlotBySymbol(
        d=symbols_per_slot * 'd',
        u=symbols_per_slot * 'u',
        s=special_symbols,
    )

    downlink_slots = [i for i, s in enumerate(frame_slots) if s == 'd']
    uplink_slots = [i for i, s in enumerate(frame_slots) if s == 'u']

    if 's' not in frame_slots or special_symbols is not None:
        frame_by_symbol = ''.join([slot_by_symbol[k] for k in frame_slots])  # ty: ignore
    else:
        frame_by_symbol = 'd' * len(frame_slots)

    return NormalizedTDDSlotConfig(
        frame_slots=frame_slots,
        special_symbols=special_symbols,
        code_maps=code_mapping,
        slot_by_symbol=slot_by_symbol,
        uplink_slot_indexes=tuple(uplink_slots),
        downlink_slot_indexes=tuple(downlink_slots),
        frame_by_symbol=frame_by_symbol,
    )


@registry.coordinates(
    dtype='float32', attrs={'standard_name': 'Cyclic sample lag', 'units': 's'}
)
@specs.helpers.lru_cache_on_converted(specs.Capture)
def cyclic_sample_lag(
    capture: specs.Capture, spec: specs.CellularCyclicAutocorrelator
) -> 'pd.Index':
    max_len = _get_max_corr_size(capture, subcarrier_spacings=spec.subcarrier_spacings)
    name = cyclic_sample_lag.__name__
    return pd.RangeIndex(0, max_len, name=name) / capture.sample_rate


@registry.coordinates(
    dtype='float32', attrs={'standard_name': 'Subcarrier spacing', 'units': 'Hz'}
)
@specs.helpers.lru_cache_on_converted(specs.Capture)
def subcarrier_spacing(
    capture: specs.Capture, spec: specs.CellularCyclicAutocorrelator
):
    if isinstance(spec.subcarrier_spacings, tuple):
        return list(spec.subcarrier_spacings)
    else:
        return [spec.subcarrier_spacings]


@registry.coordinates(dtype='str', attrs={'standard_name': 'Link direction'})
@specs.helpers.lru_cache_on_converted(specs.Capture)
def link_direction(capture: specs.Capture, spec: specs.CellularCyclicAutocorrelator):
    values = np.array(['downlink', 'uplink'], dtype='U8')
    return values, {}


@util.lru_cache()
def _get_phy_mapping(
    channel_bandwidth: float,
    sample_rate: float,
    subcarrier_spacings: float | tuple[float, ...],
    generation: typing.Literal['4G', '5G'] = '4G',
    xp=np,
) -> dict[float, sw.ofdm.Phy3GPP]:
    seq = (
        subcarrier_spacings
        if isinstance(subcarrier_spacings, tuple)
        else [subcarrier_spacings]
    )

    phy = {}

    for scs in seq:
        phy[scs] = sw.ofdm.get_3gpp_phy(
            subcarrier_spacing=scs,
            channel_bandwidth=channel_bandwidth,
            generation=generation,
            sample_rate=sample_rate,
            xp=xp,
        )

    return phy


@specs.helpers.lru_cache_on_converted(specs.Capture)
def _get_max_corr_size(
    capture: specs.Capture,
    *,
    subcarrier_spacings: float | tuple[float, ...],
    generation: typing.Literal['4G', '5G'] = '4G',
):
    phy_scs = _get_phy_mapping(
        capture.analysis_bandwidth,
        capture.sample_rate,
        subcarrier_spacings,
        generation=generation,
    )
    sizes = [np.diff(phy.cp_start_idx).min() for phy in phy_scs.values()]
    return max(sizes)


@typing.overload
def _get_spec_range(field_range: tuple[int, None], name) -> typing.Literal['all']:
    pass


@typing.overload
def _get_spec_range(
    field_range: typing.Union[int, tuple[int, int]], name
) -> tuple[int, ...]:
    pass


def _get_spec_range(
    field_range: typing.Union[int, tuple[int, None], tuple[int, int]], name
) -> tuple[int, ...] | typing.Literal['all']:
    if field_range in ((0,), (None, None), (0, None)):
        return 'all'

    elif not isinstance(field_range, tuple):
        return (field_range,)

    start, stop = field_range

    if stop is None:
        raise TypeError(
            f'{name!r} field [start, stop] indices must have two integers unless start is 0'
        )

    return tuple(range(start, stop))


def _subcarrier_spacing_tuple(
    spec: specs.CellularCyclicAutocorrelator,
) -> tuple[float, ...]:
    if isinstance(spec.subcarrier_spacings, tuple):
        return spec.subcarrier_spacings
    return (spec.subcarrier_spacings,)


def validated_autocorrelation_lag_count(
    capture: specs.Capture, spec: specs.CellularCyclicAutocorrelator
) -> int:
    """check the frame configuration and the index ranges, returning the lag axis length.

    `tdd_config_from_str` owns the `frame_slots` rules and `_get_spec_range` the
    open-ended-range rule; `_get_max_corr_size` warms the `get_3gpp_phy` design for
    every requested subcarrier spacing. The index bounds are those of
    `Phy3GPP.index_cyclic_prefix`, whose frame axis is not bounds-checked against the
    waveform: a frame past the end of the capture reads zeros into the correlation.
    """
    scs = _subcarrier_spacing_tuple(spec)

    for one_scs in scs:
        tdd_config = tdd_config_from_str(
            subcarrier_spacing=one_scs, frame_slots=spec.frame_slots
        )
        if len(tdd_config.downlink_slot_indexes) == 0:
            raise ValueError(
                "frame_slots must include at least one downlink slot 'd' "
                f'(frame_slots: {spec.frame_slots!r}, subcarrier_spacing: {one_scs})'
            )

    frame_range = _get_spec_range(spec.frame_range, 'frame_range')
    if frame_range == 'all' or len(frame_range) == 0:
        raise ValueError(
            'frame_range must select at least one frame '
            f'(frame_range: {spec.frame_range!r})'
        )
    frame_size = round(capture.sample_rate * 10e-3)
    samples = shared.capture_sample_count(capture)
    if min(frame_range) < 0 or (max(frame_range) + 1) * frame_size > samples:
        raise ValueError(
            'frame_range must index whole 10 ms frames inside the capture '
            f'(frame_range: {spec.frame_range!r}, duration: {capture.duration}, '
            f'frames in capture: {samples / frame_size})'
        )

    symbol_range = _get_spec_range(spec.symbol_range, 'symbol_range')  # ty: ignore
    if symbol_range != 'all':
        symbols_per_slot = sw.ofdm.Phy3GPP.FFT_PER_SLOT
        if len(symbol_range) == 0:
            raise ValueError(
                'symbol_range must select at least one symbol '
                f'(symbol_range: {spec.symbol_range!r})'
            )
        if (
            min(symbol_range) < -symbols_per_slot
            or max(symbol_range) >= symbols_per_slot
        ):
            raise ValueError(
                f'symbol_range must index the {symbols_per_slot} symbols of a slot, '
                f'within [{-symbols_per_slot}, {symbols_per_slot - 1}] '
                f'(symbol_range: {spec.symbol_range!r})'
            )

    return int(
        _get_max_corr_size(capture, subcarrier_spacings=scs, generation=spec.generation)
    )


@hint_keywords(specs.CellularCyclicAutocorrelator)
@registry.measurement(
    coord_factories=[link_direction, subcarrier_spacing, cyclic_sample_lag],
    dtype='float32',
    prefer_iq_source='pre_align',
    spec_type=specs.CellularCyclicAutocorrelator,
    attrs={'units': 'mW', 'standard_name': 'Cyclic Autocovariance'},
    validate=validated_autocorrelation_lag_count,
)
def cellular_cyclic_autocorrelation(iq: 'Array', capture: specs.Capture, **kwargs):
    """evaluate the cyclic autocorrelation of the IQ sequence based on 4G or 5G cellular
    cyclic prefix sample lag offsets.

    The correlation can be configured to evaluate across specified ranges of frame
    indices, slot indices (across the frames), and symbol indices (across the slots).
    Each range may be specified as a single number ("first $N$ indices") or as a
    tuple that is passed to the python builtin `range`.

    Args:
    {args}

    Returns:
        an float32-valued array with matching the array type of `iq`
    """

    spec = specs.CellularCyclicAutocorrelator.from_dict(kwargs)

    xp = sw.array_namespace(iq)
    scs = _subcarrier_spacing_tuple(spec)

    phy_scs = _get_phy_mapping(
        capture.analysis_bandwidth,
        capture.sample_rate,
        scs,
        generation=spec.generation,
        xp=xp,
    )
    metadata = {}

    metadata['frames'] = spec.frame_range
    metadata['symbols'] = spec.symbol_range

    frame_range = _get_spec_range(spec.frame_range, 'frame_range')
    symbol_range = _get_spec_range(spec.symbol_range, 'symbol_range')  # ty: ignore

    def corr_for_slots(phy, x, slots):
        cp_inds = phy.index_cyclic_prefix(
            frames=frame_range, symbols=symbol_range, slots=slots
        )
        # the kernel zero-fills reads past the end rather than raising
        if int(cp_inds.max()) >= x.shape[-1]:
            raise ValueError(
                f'cyclic prefix index {int(cp_inds.max())} lies past the '
                f'{x.shape[-1]} samples of the waveform'
            )
        return sw.ofdm.corr_at_indices(cp_inds, x, phy.nfft, norm=False)

    max_len = _get_max_corr_size(
        capture, subcarrier_spacings=scs, generation=spec.generation
    )

    result = xp.full((iq.shape[0], 2, len(scs), max_len), np.nan, dtype=np.float32)
    for chan in range(iq.shape[0]):
        for iscs, phy in enumerate(phy_scs.values()):
            tdd_config = tdd_config_from_str(
                subcarrier_spacing=phy.subcarrier_spacing, frame_slots=spec.frame_slots
            )

            R = corr_for_slots(phy, iq[chan], tdd_config.downlink_slot_indexes)
            result[chan][0][iscs][: R.size] = xp.abs(R)

            if len(tdd_config.uplink_slot_indexes) == 0:
                continue

            R = corr_for_slots(phy, iq[chan], tdd_config.uplink_slot_indexes)
            result[chan][1][iscs][: R.size] = xp.abs(R)

    return result, metadata


# %% cellular_resource_power_histogram
@registry.coordinates(
    dtype='float32',
    attrs={'standard_name': 'Cellular resource grid bin power', 'units': 'dBm'},
)
@specs.helpers.lru_cache_on_converted(specs.Capture)
def cellular_resource_power_bin(
    capture: specs.Capture, spec: specs.CellularResourcePowerHistogram
) -> tuple[np.ndarray, dict[str, typing.Any]]:
    """returns a dictionary of coordinate values, keyed by axis dimension name"""

    bins = power.make_power_bins(
        power_low=spec.power_low,
        power_high=spec.power_high,
        power_resolution=spec.power_resolution,
    )

    enbw = validated_resource_grid_sizing(capture, spec).enbw
    metadata = {
        'noise_bandwidth': float(enbw),
        'units': f'dBm/{enbw / 1e3:0.0f} kHz',
    }
    return bins, metadata


def apply_mask(
    spectrogram,
    freqs,
    *,
    channel_bandwidth,
    subcarrier_spacing,
    frame_slots: str,
    special_symbols: typing.Optional[str],
    guard_left=None,
    guard_right=None,
    link_direction=('downlink', 'uplink'),
    flex_as=None,
    normal_cp=True,
    xp=np,
) -> 'Array':
    """splits the spectrogram into TDD downlink and uplink components that are masked
    with `float('nan')`.

    See also:
        `build_tdd_link_symbol_masks`
    """

    if isinstance(link_direction, str):
        # ensure link_direction is a tuple
        link_direction = (link_direction,)

    if (
        len(link_direction)
        - link_direction.count('downlink')
        - link_direction.count('uplink')
        > 0
    ):
        raise ValueError(
            'only "downlink" or "uplink" are valid values for link_direction tuple'
        )

    # null frequencies in the guard interval
    eps = 1e-6
    ilo = xp.searchsorted(freqs, xp.asarray(-channel_bandwidth / 2 + guard_left + eps))
    ihi = xp.searchsorted(freqs, xp.asarray(channel_bandwidth / 2 - guard_right - eps))
    spg_left = sw.axis_slice(spectrogram, 0, ilo, axis=-1)
    spg_right = sw.axis_slice(spectrogram, ihi, None, axis=-1)
    xp.copyto(spg_left, float('nan'))
    xp.copyto(spg_right, float('nan'))

    # null in time to select each of the {down,up} links
    masks = build_tdd_link_symbol_masks(
        subcarrier_spacing=subcarrier_spacing,
        frame_slots=frame_slots,
        special_symbols=special_symbols,
        link_direction=link_direction,
        count=spectrogram.shape[-2],
        xp=xp,
        flex_as=flex_as,
        normal_cp=normal_cp,
    )

    # broadcast into dimensions (input channel, link direction, symbols elapsed, frequency)
    return masks[np.newaxis, :, :, np.newaxis] * spectrogram[:, np.newaxis, :, :]


@util.lru_cache()
def build_tdd_link_symbol_masks(
    subcarrier_spacing: float,
    frame_slots: str,
    special_symbols: typing.Optional[str] = None,
    *,
    link_direction: tuple[str, ...] = ('downlink', 'uplink'),
    count: int | None = None,
    normal_cp=True,
    flex_as=None,
    xp=np,
) -> 'Array':
    """generate a symbol-by-symbol sequence of masking arrays for uplink and downlink.

    The number of slots given in the frame match the appropriate number for a given
    5G NR or LTE subcarrier spacing.

    Arguments:
        frame_slots: a string composed of the characters {'d', 'u', 's'} that
            indicate the sequence of slots in 1 cellular frame
        special_symbols: the a string composed of the characters {'d', 'u', 'f'} that
            indicate the sequence of symbol types in the special slot.
    """

    tdd_config = tdd_config_from_str(
        subcarrier_spacing=subcarrier_spacing,
        frame_slots=frame_slots,
        special_symbols=special_symbols,
        normal_cp=normal_cp,
        flex_as=flex_as,
    )

    out_shape = (len(link_direction), count or 1)
    out = xp.empty(out_shape, dtype='float32')
    for i, direction in enumerate(link_direction):
        single_mask = [
            tdd_config.code_maps[direction][k] for k in tdd_config.frame_by_symbol
        ]

        if count is None:
            frame_count = 1
        else:
            frame_count = ceil(count / len(single_mask))

        mask = (single_mask * frame_count)[:count]
        out[i] = xp.asarray(mask)

    return out


def _get_integration_bandwidth(
    spec: specs.CellularResourcePowerHistogram,
) -> float:
    if spec.average_rbs == 'half':
        return 6 * spec.subcarrier_spacing
    elif spec.average_rbs:
        return 12 * spec.subcarrier_spacing
    else:
        return spec.subcarrier_spacing


@util.lru_cache()
def _resource_grid_spectrogram_spec(
    spec: specs.CellularResourcePowerHistogram,
) -> specs.Spectrogram:
    """the STFT that lands one bin on each half-subcarrier and one hop on each symbol"""
    fractional_overlap, window_fill = cellular_stft_window_fractions(spec.cyclic_prefix)

    return specs.Spectrogram(
        window=spec.window,
        frequency_resolution=spec.subcarrier_spacing / 2,
        fractional_overlap=fractional_overlap,
        window_fill=window_fill,
        integration_bandwidth=_get_integration_bandwidth(spec),
        lo_bandstop=spec.lo_bandstop,
    )


class ResourceGridSizing(typing.NamedTuple):
    """the frame layout and STFT sizing implied by a (capture, resource grid spec) pair"""

    frame_slots: str
    spectrogram: specs.Spectrogram
    time_bin_averaging: typing.Optional[int]
    enbw: float


def validated_resource_grid_sizing(
    capture: specs.Capture, spec: specs.CellularResourcePowerHistogram
) -> ResourceGridSizing:
    """check the frame configuration and the STFT sizing of the resource grid.

    `tdd_config_from_str` owns the `frame_slots` and `special_symbols` rules. The slot
    averaging has to land on a whole number of STFT hops, which the spectrogram sizing
    cannot check for us because `time_aperture` is derived here rather than given.
    """
    if (
        spec.frame_slots is not None
        and 's' in spec.frame_slots.lower()
        and spec.special_symbols is None
    ):
        raise ValueError(
            'specify special_symbols that implement the requested "s" special slot '
            f'(frame_slots: {spec.frame_slots})'
        )

    tdd_config = tdd_config_from_str(
        subcarrier_spacing=spec.subcarrier_spacing,
        frame_slots=spec.frame_slots,
        special_symbols=spec.special_symbols,
    )

    spg_spec = _resource_grid_spectrogram_spec(spec)
    sizing = spectrum.validated_spectrogram_sizing(capture, spg_spec)

    slot_period = sw.ofdm.slot_period(spec.subcarrier_spacing)
    if not spec.average_slots:
        time_bin_averaging = None
    elif sw.isroundmod(slot_period, sizing.hop_period):
        time_bin_averaging = round(slot_period / sizing.hop_period)
        window_count = spectrum.spectrogram_window_count(capture, sizing)
        if time_bin_averaging > window_count:
            raise ValueError(
                'duration must span at least one slot to average across slots '
                f'(duration: {capture.duration}, slot_period: {slot_period}, '
                f'symbols in capture: {window_count})'
            )
    else:
        raise ValueError(
            'a slot must span a counting number of STFT hops to average across slots '
            f'(subcarrier_spacing: {spec.subcarrier_spacing}, '
            f'sample_rate: {capture.sample_rate}, '
            f'slot_period: {slot_period}, '
            f'hop_period: {sizing.hop_period})'
        )

    return ResourceGridSizing(
        frame_slots=tdd_config.frame_slots,
        spectrogram=spg_spec,
        time_bin_averaging=time_bin_averaging,
        enbw=sizing.enbw,
    )


@hint_keywords(specs.CellularResourcePowerHistogram)
@registry.measurement(
    coord_factories=[link_direction, cellular_resource_power_bin],
    dtype='float32',
    depends=spectrum.spectrogram,
    spec_type=specs.CellularResourcePowerHistogram,
    prefer_iq_source='pre_filter',
    attrs={'standard_name': 'Fraction of resource grid'},
    validate=validated_resource_grid_sizing,
)
def cellular_resource_power_histogram(iq: 'Array', capture: specs.Capture, **kwargs):
    """Evaluate the spectrograms of a cellular resource grid on each port, and
    return a flattened histogram of its power levels.

    Args:
    {args}

    Returns:
        `xarray.DataArray` or `(array, dict)` based on `as_xarray`
    """
    spec = specs.CellularResourcePowerHistogram.from_dict(kwargs)

    xp = sw.array_namespace(iq)

    link_direction = 'downlink', 'uplink'

    sizing = validated_resource_grid_sizing(capture, spec)

    spg, metadata = spectrum.evaluate_spectrogram(
        iq, capture, sizing.spectrogram, dtype='float32', dB=False
    )
    del metadata['units']

    freqs = spectrum.spectrogram_freqs(capture, sizing.spectrogram)
    freqs = xp.asarray(freqs)

    masked_spgs = apply_mask(
        spg,
        freqs,
        subcarrier_spacing=spec.subcarrier_spacing,
        link_direction=link_direction,
        channel_bandwidth=capture.analysis_bandwidth,
        frame_slots=sizing.frame_slots,
        special_symbols=spec.special_symbols,
        guard_left=spec.guard_bandwidths[0],
        guard_right=spec.guard_bandwidths[1],
        xp=xp,
    )

    # apply the time binning only now, to allow for averaging
    # across mask boundaries
    if sizing.time_bin_averaging is not None:
        masked_spgs = sw.binned_mean(
            masked_spgs, sizing.time_bin_averaging, axis=2, fft=False
        )

    masked_spgs = sw.powtodB(masked_spgs, overwrite_x=True)
    bin_edges = power.make_power_histogram_bin_edges(
        power_low=spec.power_low,
        power_high=spec.power_high,
        power_resolution=spec.power_resolution,
        xp=xp,
    )

    flat_shape = masked_spgs.shape[:2] + (-1,)
    counts, _ = sw.histogram_last_axis(masked_spgs.reshape(flat_shape), bin_edges)

    norm = xp.sum(counts, axis=(1, 2), keepdims=True)
    norm[norm == 0] = 1
    data = counts / norm

    return data, metadata
