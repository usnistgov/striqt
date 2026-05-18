from __future__ import annotations as __

from typing import Any, Callable, Literal, Optional, TYPE_CHECKING, Union

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
@util.lru_cache()
def cellular_cell_id2(capture: specs.Capture, spec: Any):
    values = np.array([0, 1, 2], dtype='uint16')
    return values


@registry.coordinates(dtype='uint16', attrs={'standard_name': 'SSB beam index'})
@util.lru_cache()
def cellular_ssb_beam_index(capture: specs.Capture, spec: _Cellular5GNRSSBSync):
    # pss_params and sss_params return the same number of symbol indexes
    params = sw.ofdm.sss_params(
        sample_rate=spec.sample_rate,
        subcarrier_spacing=spec.subcarrier_spacing,
        discovery_periodicity=spec.discovery_periodicity,
        shared_spectrum=spec.shared_spectrum,
        max_lag_symbols=spec.max_lag_symbols,
        symbol_indexes=spec.symbol_indexes,
        center_frequency=getattr(capture, 'center_frequency', None),
    )

    return list(range(len(params.symbol_indexes)))


@registry.coordinates(
    dtype='float32', attrs={'standard_name': 'Time Elapsed', 'units': 's'}
)
@util.lru_cache()
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

    if sw.isroundmod(capture.sample_rate, spec.frequency_resolution):
        nfft = round(capture.sample_rate / spec.frequency_resolution)
    else:
        raise ValueError('sample_rate/resolution must be a counting number')

    if sw.isroundmod(capture.sample_rate, spec.frequency_resolution):
        noverlap = round(spec.fractional_overlap * nfft)
    else:
        raise ValueError('sample_rate_Hz/resolution must be a counting number')

    nzero = (1 - spec.window_fill) * nfft
    if nzero.denominator == 1:
        nzero = nzero.numerator
    else:
        raise ValueError(
            '(1-window_fill) * sample_rate must be a counting-number multiple of frequency_resolution'
        )

    if spec.integration_bandwidth is None:
        frequency_bin_averaging = None
    elif sw.isroundmod(spec.integration_bandwidth, spec.frequency_resolution):
        frequency_bin_averaging = round(
            spec.integration_bandwidth / spec.frequency_resolution
        )
    else:
        raise ValueError(
            'when specified, integration_bandwidth must be a multiple of frequency_resolution'
        )

    hop_size = nfft - noverlap
    hop_period = hop_size / capture.sample_rate
    if spec.time_aperture is None:
        time_bin_averaging = None
    elif sw.isroundmod(spec.time_aperture, hop_period):
        time_bin_averaging = round(spec.time_aperture / hop_period)
    else:
        raise ValueError(
            'when specified, time_aperture must be a multiple of (1-fractional_overlap)/frequency_resolution'
        )

    spg = sw.spectrogram(
        iq,
        window=spec.window,
        fs=capture.sample_rate,
        nperseg=nfft,
        noverlap=noverlap,
        nzero=nzero,
        axis=1,
        return_axis_arrays=False,
    )

    if spec.lo_bandstop is not None:
        sw.fourier.null_lo(spg, nfft, capture.sample_rate, spec.lo_bandstop, axis=2)

    # truncate to the analysis bandwidth
    if spec.trim_stopband and np.isfinite(capture.analysis_bandwidth):
        # stick with python arithmetic to ensure consistency with axis bounds calculations
        spg = sw.fourier.truncate_freqs(
            spg, nfft, capture.sample_rate, bandwidth=capture.analysis_bandwidth, axis=2
        )

    if frequency_bin_averaging is not None:
        spg = sw.binned_mean(spg, frequency_bin_averaging, axis=2, fft=True)

        # mean -> sum
        spg *= frequency_bin_averaging

    if time_bin_averaging is not None:
        spg = sw.binned_mean(spg, time_bin_averaging, axis=1, fft=False)

    if spec.integration_bandwidth is None:
        enbw = spec.frequency_resolution
    else:
        enbw = spec.integration_bandwidth

    attrs = {
        'noise_bandwidth': float(enbw),
        'units': f'dBm/{enbw / 1e3:0.0f} kHz',
    }

    return spg, attrs


@util.lru_cache()
def spectrogram_freqs(capture: specs.Capture, spec: specs.Spectrogram) -> np.ndarray:
    if sw.isroundmod(capture.sample_rate, spec.frequency_resolution):
        nfft = round(capture.sample_rate / spec.frequency_resolution)
    else:
        raise ValueError('sample_rate/resolution must be a counting number')

    if spec.integration_bandwidth is None:
        frequency_bin_averaging = None
    elif sw.isroundmod(spec.integration_bandwidth, spec.frequency_resolution):
        frequency_bin_averaging = round(
            spec.integration_bandwidth / spec.frequency_resolution
        )
    else:
        raise ValueError(
            'when specified, integration_bandwidth must be a multiple of frequency_resolution'
        )

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
        freqs = sw.binned_mean(freqs, frequency_bin_averaging, fft=True)

    # only now downconvert. round to a still-large number of digits
    return freqs.astype('float64').round(16)


@registry.coordinates(
    dtype='float64', attrs={'standard_name': 'Baseband Frequency', 'units': 'Hz'}
)
def spectrogram_baseband_frequency(
    capture: specs.Capture, spec: specs.Spectrogram
) -> np.ndarray:
    return spectrogram_freqs(capture, spec)
