from __future__ import annotations as __

from fractions import Fraction
from typing import Any, Callable, Literal, NamedTuple, Optional, TYPE_CHECKING, Union

from .. import specs

from ..lib import dataarrays, register, util
from ..lib.register import registry

import striqt.waveform as sw

if TYPE_CHECKING:
    from ..specs.structs import _Cellular5GNRSSBSync, _Cellular5GNRSSBCorrelator
    import numpy as np
    from ..lib.typing import Array, CoordFunc, P, R, WrappedAnalysis, WrappedCoord
    from typing import Sequence

else:
    np = util.lazy_import('numpy')
    array_api_compat = util.lazy_import('array_api_compat')


def hint_keywords(
    func: Callable[P, Any],
) -> Callable[[WrappedAnalysis[..., R]], WrappedAnalysis[P, R]]:
    """fill in type hints for the analysis parameters"""
    return lambda f: f  # pyright: ignore


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
    params = sw.ofdm.sss_params(
        sample_rate=spec.sample_rate,
        subcarrier_spacing=spec.subcarrier_spacing,
        discovery_periodicity=spec.discovery_periodicity,
        shared_spectrum=spec.shared_spectrum,
        max_lag_symbols=spec.max_lag_symbols,
        symbol_indexes=spec.symbol_indexes,
        center_frequency=capture.center_frequency,
    )

    return list(range(len(params.symbol_indexes)))


@registry.coordinates(
    dtype='float32', attrs={'standard_name': 'Time Elapsed', 'units': 's'}
)
@specs.helpers.lru_cache_on_converted(specs.Capture)
def cellular_ssb_start_time(
    capture: specs.Capture,
    spec: specs.Cellular5GNRPSSCorrelator | specs.Cellular5GNRSSSCorrelator,
):
    # pss_params and sss_params return the same number of symbol indexes
    params = sw.ofdm.pss_params(
        sample_rate=spec.sample_rate,
        subcarrier_spacing=spec.subcarrier_spacing,
        discovery_periodicity=spec.discovery_periodicity,
        shared_spectrum=spec.shared_spectrum,
        max_lag_symbols=spec.max_lag_symbols,
        symbol_indexes=spec.symbol_indexes,
    )
    total_blocks = round(params.duration / spec.discovery_periodicity)
    if spec.max_block_count is None:
        count = total_blocks
    else:
        count = min(spec.max_block_count, total_blocks)

    return np.arange(max(count, 1)).astype('float32') * spec.discovery_periodicity


def empty_5g_ssb_correlation(
    iq,
    *,
    capture: specs.Capture,
    spec: _Cellular5GNRSSBCorrelator,
    coord_factories: Sequence[CoordFunc | WrappedCoord],
    dtype='complex64',
):
    xp = sw.array_namespace(iq)
    meas_ax_shape = [len(f(capture, spec)) for f in coord_factories]
    new_shape = iq.shape[:-1] + tuple(meas_ax_shape)
    return xp.full(new_shape, 0, dtype=dtype)


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
    """
    return sw.ofdm.sss_params(
        sample_rate=spec.sample_rate,
        subcarrier_spacing=spec.subcarrier_spacing,
        discovery_periodicity=spec.discovery_periodicity,
        shared_spectrum=spec.shared_spectrum,
        max_lag_symbols=spec.max_lag_symbols,
        symbol_indexes=spec.symbol_indexes,
        center_frequency=capture.center_frequency,
    )


ssb_iq_cache = register.KwArgCache([dataarrays.CAPTURE_DIM, 'spec'])


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


class SpectrogramSizing(NamedTuple):
    """the STFT sizing implied by a (capture, spectrogram spec) combination"""

    nfft: int
    noverlap: int
    nzero: int
    hop_size: int
    hop_period: float
    frequency_bin_averaging: Optional[int]
    time_bin_averaging: Optional[int]
    enbw: float


@specs.helpers.lru_cache_on_converted(specs.Capture, specs.Spectrogram)
def validated_spectrogram_sizing(
    capture: specs.Capture, spec: specs.Spectrogram
) -> SpectrogramSizing:
    """check that `spec` divides `capture` evenly, returning the derived STFT sizing.

    Callers may pass any `FrequencyAnalysisSpecBase`, since both arguments are projected
    onto the fields that determine the sizing, so the result is shared across captures
    and spectrogram-derived measurements that differ only in fields it does not read.
    The capture projection is the base `Capture`, not `AnalysisCapture`: nothing here
    reads `center_frequency`, so one entry covers a whole frequency sweep.
    """
    if not sw.isroundmod(capture.sample_rate, spec.frequency_resolution):
        raise ValueError(
            'sample_rate/resolution must be a counting number '
            f'(sample_rate: {capture.sample_rate}, '
            f'frequency_resolution: {spec.frequency_resolution})'
        )
    nfft = round(capture.sample_rate / spec.frequency_resolution)

    noverlap = round(spec.fractional_overlap * nfft)

    nzero = (1 - spec.window_fill) * nfft
    if nzero.denominator != 1:
        raise ValueError(
            '(1-window_fill) * sample_rate must be a counting-number multiple of frequency_resolution '
            f'(window_fill: {spec.window_fill}, '
            f'sample_rate: {capture.sample_rate}, '
            f'frequency_resolution: {spec.frequency_resolution}, '
            f'(1-window_fill)*nfft: {nzero})'
        )
    nzero = nzero.numerator

    if spec.integration_bandwidth is None:
        frequency_bin_averaging = None
    elif sw.isroundmod(spec.integration_bandwidth, spec.frequency_resolution):
        frequency_bin_averaging = round(
            spec.integration_bandwidth / spec.frequency_resolution
        )
    else:
        raise ValueError(
            'when specified, integration_bandwidth must be a multiple of frequency_resolution '
            f'(integration_bandwidth: {spec.integration_bandwidth}, '
            f'frequency_resolution: {spec.frequency_resolution})'
        )

    hop_size = nfft - noverlap
    hop_period = hop_size / capture.sample_rate
    if spec.time_aperture is None:
        time_bin_averaging = None
    elif sw.isroundmod(spec.time_aperture, hop_period):
        time_bin_averaging = round(spec.time_aperture / hop_period)
    else:
        raise ValueError(
            'when specified, time_aperture must be a multiple of (1-fractional_overlap)/frequency_resolution '
            f'(time_aperture: {spec.time_aperture}, '
            f'fractional_overlap: {spec.fractional_overlap}, '
            f'frequency_resolution: {spec.frequency_resolution}, '
            f'hop_period: {hop_period})'
        )

    # the design that `sw.stft` will build for complex64 IQ, so a bad window name or
    # parameter fails here and the disk-cached search is paid before acquisition
    sw.get_window(
        spec.window, nfft - nzero, nzero=nzero, dtype='complex64', fftshift=True
    )

    if spec.integration_bandwidth is None:
        enbw = spec.frequency_resolution
    else:
        enbw = spec.integration_bandwidth

    return SpectrogramSizing(
        nfft=nfft,
        noverlap=noverlap,
        nzero=nzero,
        hop_size=hop_size,
        hop_period=hop_period,
        frequency_bin_averaging=frequency_bin_averaging,
        time_bin_averaging=time_bin_averaging,
        enbw=enbw,
    )


def evaluate_spectrogram(
    iq: Array,
    capture: specs.Capture,
    spec: specs.Spectrogram,
    *,
    dtype: Union[Literal['float16'], Literal['float32']] = 'float32',
    limit_digits: Optional[int] = None,
    dB=True,
) -> tuple[Array, dict]:
    spg, attrs = _cached_spectrogram(iq, capture, spec=spec)
    xp = sw.array_namespace(iq)

    copied = False
    if dB:
        spg = sw.powtodB(spg, eps=1e-25)
        copied = True

    if limit_digits is not None:
        spg = xp.round(spg, limit_digits, out=spg if copied else None)
        copied = True

    if dtype == 'float16':
        spg = spg.astype(dtype, copy=not copied)

    attrs = attrs | {'limit_digits': limit_digits}

    return spg, attrs


spectrogram_cache = register.KwArgCache([dataarrays.CAPTURE_DIM, 'spec'])


@spectrogram_cache.apply
def _cached_spectrogram(
    iq: Array,
    capture: specs.Capture,
    spec: specs.Spectrogram,
) -> tuple[Array, dict]:
    spec = spec.validate()
    sizing = validated_spectrogram_sizing(capture, spec)

    spg = sw.spectrogram(
        iq,
        window=spec.window,
        fs=capture.sample_rate,
        nperseg=sizing.nfft,
        noverlap=sizing.noverlap,
        nzero=sizing.nzero,
        axis=1,
        return_axis_arrays=False,
    )

    if spec.lo_bandstop is not None:
        sw.fourier.null_lo(
            spg, sizing.nfft, capture.sample_rate, spec.lo_bandstop, axis=2
        )

    # truncate to the analysis bandwidth
    if spec.trim_stopband and np.isfinite(capture.analysis_bandwidth):
        # stick with python arithmetic to ensure consistency with axis bounds calculations
        spg = sw.fourier.truncate_freqs(
            spg,
            sizing.nfft,
            capture.sample_rate,
            bandwidth=capture.analysis_bandwidth,
            axis=2,
        )

    if sizing.frequency_bin_averaging is not None:
        spg = sw.binned_mean(spg, sizing.frequency_bin_averaging, axis=2, fft=True)

        # mean -> sum
        spg *= sizing.frequency_bin_averaging

    if sizing.time_bin_averaging is not None:
        spg = sw.binned_mean(spg, sizing.time_bin_averaging, axis=1, fft=False)

    attrs = {
        'noise_bandwidth': float(sizing.enbw),
        'units': f'dBm/{sizing.enbw / 1e3:0.0f} kHz',
    }

    return spg, attrs


@specs.helpers.lru_cache_on_converted(specs.Capture, specs.Spectrogram)
def spectrogram_freqs(capture: specs.Capture, spec: specs.Spectrogram) -> np.ndarray:
    sizing = validated_spectrogram_sizing(capture, spec)
    nfft = sizing.nfft

    # use the striqt.waveform.fourier fftfreq for higher precision, which avoids
    # headaches when merging spectra with different sampling parameters due
    # to rounding errors.
    freqs = sw.fftfreq(nfft, capture.sample_rate)

    if spec.trim_stopband and np.isfinite(capture.analysis_bandwidth):
        # stick with python arithmetic here for numpy/cupy consistency
        freqs = sw.fourier.truncate_freqs(
            freqs, nfft, capture.sample_rate, capture.analysis_bandwidth, axis=0
        )

    if spec.integration_bandwidth is not None:
        freqs = sw.binned_mean(freqs, sizing.frequency_bin_averaging, fft=True)

    # only now downconvert. round to a still-large number of digits
    return freqs.astype('float64').round(16)


@registry.coordinates(
    dtype='float64', attrs={'standard_name': 'Baseband Frequency', 'units': 'Hz'}
)
def spectrogram_baseband_frequency(
    capture: specs.Capture, spec: specs.Spectrogram
) -> np.ndarray:
    return spectrogram_freqs(capture, spec)
