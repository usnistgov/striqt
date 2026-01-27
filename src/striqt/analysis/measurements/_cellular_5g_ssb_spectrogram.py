from __future__ import annotations as __

import typing
from fractions import Fraction

from .. import specs

from ..lib import util
from . import shared
from .shared import registry, hint_keywords

if typing.TYPE_CHECKING:
    import numpy as np

    import striqt.waveform as waveform
else:
    np = util.lazy_import('numpy')
    waveform = util.lazy_import('striqt.waveform')


@registry.coordinates(dtype='uint16', attrs={'standard_name': 'Symbols elapsed'})
@util.lru_cache()
def cellular_ssb_symbol_index(
    capture: specs.Capture, spec: specs.Cellular5GNRSSBSpectrogram
):
    symbol_count = round(28 * spec.subcarrier_spacing / 15e3)
    return np.arange(symbol_count, dtype='uint16')


@registry.coordinates(
    dtype='float64', attrs={'standard_name': 'SSB Baseband Frequency', 'units': 'Hz'}
)
@util.lru_cache()
def cellular_ssb_baseband_frequency(
    capture: specs.Capture, spec: specs.Cellular5GNRSSBSpectrogram, xp=np
) -> np.ndarray:
    nfft = round(2 * capture.sample_rate / spec.subcarrier_spacing)
    bb_freqs = shared.fftfreq(nfft, capture.sample_rate)

    frequency_offset = specs.helpers.maybe_lookup_with_capture_key(
        capture,
        spec.frequency_offset,
        capture_attr='center_frequency',
        error_label='frequency_offset',
        default=None,
    )

    if frequency_offset is None:
        raise KeyError(
            'center_frequency did not match any keys given for frequency_offset'
        )

    bb_freqs = waveform.util.binned_mean(bb_freqs, count=2, axis=0, fft=True)
    bb_freqs = shared.truncate_spectrogram_bandwidth(
        bb_freqs,
        nfft // 2,
        capture.sample_rate,
        spec.sample_rate,
        offset=frequency_offset,
        axis=0,
    )

    return bb_freqs - frequency_offset


@registry.coordinates(dtype='uint16', attrs={'standard_name': 'Capture SSB index'})
@util.lru_cache()
def cellular_ssb_index(capture: specs.Capture, spec: specs.Cellular5GNRSSBSpectrogram):
    # pss_params and sss_params return the same number of symbol indexes
    # params  = iqwaveform.ofdm.pss_params(
    #     sample_rate=spec.sample_rate,
    #     subcarrier_spacing=spec.subcarrier_spacing,
    #     discovery_periodicity=spec.discovery_periodicity,
    #     shared_spectrum=spec.shared_spectrum,
    # )
    total_blocks = round(capture.duration / spec.discovery_periodicity)
    if spec.max_block_count is None:
        count = total_blocks
    else:
        count = min(spec.max_block_count, total_blocks)

    return np.arange(max(count, 1), dtype='uint16')


_coord_factories = [
    cellular_ssb_index,
    cellular_ssb_symbol_index,
    cellular_ssb_baseband_frequency,
]


@hint_keywords(specs.Cellular5GNRSSBSpectrogram)
@registry.measurement(
    specs.Cellular5GNRSSBSpectrogram,
    coord_factories=_coord_factories,
    dtype='float16',
    caches=(shared.spectrogram_cache,),
    prefer_unaligned_input=False,
    attrs={'standard_name': 'SSB Spectrogram'},
)
def cellular_5g_ssb_spectrogram(iq, capture: specs.Capture, **kwargs):
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
        max_block_count: if not None, the number of synchronization blocks to analyze
        as_xarray: if True (default), return an xarray.DataArray, otherwise a ChannelAnalysisResult object

    References:
        3GPP TS 138 211: Table 7.4.3.1-1, Section 7.4.2.2
        3GPP TS 138 213: Section 4.1
    """

    spec = specs.Cellular5GNRSSBSpectrogram.from_dict(kwargs).validate()

    # TODO: compute this with the striqt.waveform.ofdm
    symbol_count = round(28 * spec.subcarrier_spacing / 15e3)  # per burst set

    frequency_offset = specs.helpers.maybe_lookup_with_capture_key(
        capture,
        spec.frequency_offset,
        capture_attr='center_frequency',
        error_label='frequency_offset',
        default=None,
    )

    if frequency_offset is None:
        raise TypeError(
            'no key found for center_frequency in frequency_offset lookup dictionary'
        )

    spg_spec = specs.Spectrogram(
        frequency_resolution=spec.subcarrier_spacing / 2,
        fractional_overlap=Fraction(13, 28),
        window_fill=Fraction(15, 28),
        window=spec.window,
        lo_bandstop=spec.lo_bandstop,
        integration_bandwidth=spec.subcarrier_spacing,
        trim_stopband=False,
    )

    spg, attrs = shared.evaluate_spectrogram(
        iq, capture=capture, spec=spg_spec, limit_digits=3, dtype='float16'
    )

    slot_period = 1e-3 * (15e3 / spec.subcarrier_spacing)
    symbol_period = slot_period / 14  # TODO: this is normal CP; support extended CP?
    discovery_symbols = round(spec.discovery_periodicity / symbol_period)

    # keep only the first two slots in the frame
    symbol_index = np.arange(spg.shape[1])
    mask_time = (symbol_index % discovery_symbols) < symbol_count
    spg = spg[:, mask_time]

    # select frequency
    nfft = round(capture.sample_rate / spec.subcarrier_spacing)
    spg = shared.truncate_spectrogram_bandwidth(
        spg,
        nfft,
        capture.sample_rate,
        spec.sample_rate,
        offset=frequency_offset,
        axis=2,
    )

    # split by synchronization block
    spg = spg.reshape((spg.shape[0], -1, symbol_count, spg.shape[2]))

    return spg, attrs
