from __future__ import annotations as __

from .. import specs

from ..lib import util
from ..lib.util import np
from . import shared, spectrum
from .shared import registry, hint_keywords

import striqt.waveform as sw


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
    count = shared.ssb_block_count(capture.duration, spec)
    return np.arange(count, dtype='uint16')


_coord_factories = [
    cellular_ssb_index,
    cellular_ssb_symbol_index,
    cellular_ssb_baseband_frequency,
]


@util.lru_cache()
def _spectrogram_spec(spec: specs.Cellular5GNRSSBSpectrogram) -> specs.Spectrogram:
    """the STFT that lands one bin on each half-subcarrier and one hop on each symbol"""
    fractional_overlap, window_fill = shared.cellular_stft_window_fractions('normal')

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
    sizing = spectrum.validated_spectrogram_sizing(capture, _spectrogram_spec(spec))

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
        capture, _spectrogram_spec(spec), dtype='float16', limit_digits=3, **kwargs
    )


@hint_keywords(specs.Cellular5GNRSSBSpectrogram)
@registry.measurement(
    specs.Cellular5GNRSSBSpectrogram,
    coord_factories=_coord_factories,
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
        spec=_spectrogram_spec(spec),
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
