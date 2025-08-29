from __future__ import annotations
import typing

from . import shared
from ..lib import register, specs, util

if typing.TYPE_CHECKING:
    import numpy as np
    import iqwaveform
else:
    np = util.lazy_import('numpy')
    iqwaveform = util.lazy_import('iqwaveform')


# %% Cellular 5G NR synchronizatino
class Cellular5GNRSSBSpectrogramSpec(
    specs.Measurement,
    forbid_unknown_fields=True,
    cache_hash=True,
    kw_only=True,
    frozen=True,
    dict=True,
):
    """
    subcarrier_spacing (float): 3GPP channel subcarrier spacing (Hz)
    sample_rate (float): output sample rate for the resampled synchronization waveform (samples/s)
    discovery_periodicity (float): time period between synchronization blocks (s)
    frequency_offset (float or dict[float, float]):
        center frequency offset (see notes)
    shared_spectrum:
        whether to follow the 3GPP "shared spectrum" synchronizatio block layout
    max_block_count: number of synchronization blocks to evaluate
    trim_cp: whether to trim the cyclic prefix duration from the output
    """

    subcarrier_spacing: float

    # ssb parameters
    sample_rate: float = 15.36e6 / 2
    discovery_periodicity: float = 20e-3
    frequency_offset: typing.Union[float, dict[float, float]] = 0
    max_block_count: typing.Optional[int] = None

    # spectrogram info
    window: specs.WindowType = ('kaiser_by_enbw', 2)
    lo_bandstop: typing.Optional[float] = None

    # hard-coded for re-use of PSS/SSS functions
    shared_spectrum = False
    trim_cp = False


class Cellular5GNRSSBSpectrogramKeywords(specs.AnalysisKeywords, total=False):
    subcarrier_spacing: float
    sample_rate: float
    discovery_periodicity: float
    frequency_offset: typing.Union[float, dict[float, float]]
    max_block_count: typing.Optional[int]
    window: typing.Optional[specs.WindowType]
    lo_bandstop: typing.Optional[float]


@register.coordinate_factory(dtype='uint16', attrs={'standard_name': 'Symbols elapsed'})
@util.lru_cache()
def cellular_ssb_symbol_index(_: specs.Capture, spec: Cellular5GNRSSBSpectrogramSpec):
    symbol_count = round(14 * spec.subcarrier_spacing / 15e3)
    return np.arange(symbol_count, dtype='uint16')


@register.coordinate_factory(
    dtype='float64', attrs={'standard_name': 'SSB Baseband Frequency', 'units': 'Hz'}
)
@util.lru_cache()
def cellular_ssb_baseband_frequency(
    capture: specs.Capture, spec: Cellular5GNRSSBSpectrogramSpec, xp=np
) -> np.ndarray:
    nfft = round(2 * capture.sample_rate / spec.subcarrier_spacing)
    freqs = shared.fftfreq(nfft, capture.sample_rate)
    freqs = shared.truncate_spectrogram_bandwidth(
        freqs,
        nfft,
        capture.sample_rate,
        spec.sample_rate,
        offset=spec.frequency_offset,
        axis=0,
    )

    # due to integration_bandwidth=2*subcarrier_spacing
    return iqwaveform.util.binned_mean(freqs, count=2, axis=0, fft=True)


@register.coordinate_factory(
    dtype='uint16', attrs={'standard_name': 'Capture SSB index'}
)
@util.lru_cache()
def cellular_ssb_index(capture: specs.Capture, spec: Cellular5GNRSSBSpectrogramSpec):
    # pss_params and sss_params return the same number of symbol indexes
    # params = iqwaveform.ofdm.pss_params(
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


@register.measurement(
    Cellular5GNRSSBSpectrogramSpec,
    coord_factories=_coord_factories,
    dtype='float16',
    caches=(shared.spectrogram_cache,),
    prefer_unaligned_input=False,
    attrs={'standard_name': 'SSB Spectrogram'},
)
def cellular_5g_ssb_spectrogram(
    iq,
    capture: specs.Capture,
    **kwargs: typing.Unpack[Cellular5GNRSSBSpectrogramKeywords],
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
        max_block_count: if not None, the number of synchronization blocks to analyze
        as_xarray: if True (default), return an xarray.DataArray, otherwise a ChannelAnalysisResult object

    References:
        3GPP TS 138 211: Table 7.4.3.1-1, Section 7.4.2.2
        3GPP TS 138 213: Section 4.1
    """

    spec = Cellular5GNRSSBSpectrogramSpec.fromdict(kwargs).validate()

    spg_spec = shared.SpectrogramSpec(
        frequency_resolution=spec.subcarrier_spacing / 2,
        fractional_overlap=13 / 28,
        window_fill=15 / 28,
        window=spec.window,
        lo_bandstop=spec.lo_bandstop,
        integration_bandwidth=spec.subcarrier_spacing,
        trim_stopband=True,
    )

    spg_capture = capture.replace(analysis_bandwidth=spec.sample_rate)

    spg, attrs = shared.evaluate_spectrogram(
        iq, capture=spg_capture, spec=spg_spec, limit_digits=3, dtype='float16'
    )

    slot_period = 1e-3 * (15e3 / spec.subcarrier_spacing)
    symbol_period = slot_period / 14
    discovery_symbols = round(spec.discovery_periodicity / symbol_period)

    # keep only the first two slots in the frame
    symbol_index = np.arange(spg.shape[1])
    mask_time = (symbol_index % discovery_symbols) < 28
    spg = spg[:, mask_time]

    # split by synchronization block
    spg = spg.reshape((spg.shape[0], -1, 28, spg.shape[2]))

    return spg, attrs
