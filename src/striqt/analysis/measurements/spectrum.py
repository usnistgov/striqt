from __future__ import annotations as __

import math
import typing
from typing import Any, Literal, NamedTuple, Optional

import striqt.waveform as sw

from .. import specs
from ..lib import dataarrays, register
from ..lib.util import np
from . import power, shared
from .shared import hint_keywords, registry

if typing.TYPE_CHECKING:
    from typing_extensions import Unpack

    from ..lib.typing import Array, Measurement, ToleranceKws


# %% STFT sizing and the shared spectrogram cache
class SpectrogramSizing(NamedTuple):
    """the STFT sizing of one capture under a spectrogram spec."""

    nfft: int
    """samples per FFT window, including the zero padding"""
    noverlap: int
    """samples shared by consecutive windows"""
    nzero: int
    """zero-padded samples in each window"""
    hop_size: int
    """samples between consecutive window starts"""
    hop_period: float
    """time (in s) between consecutive window starts"""
    frequency_bin_averaging: Optional[int]
    """bins summed per `integration_bandwidth`, or None when that field is unset"""
    time_bin_averaging: Optional[int]
    """windows averaged per `time_aperture`, or None when that field is unset"""
    noise_bandwidth: float
    """bandwidth B (in Hz) in the mean power ``N0 * B`` of one output bin for white
    noise with power spectral density N0: `frequency_resolution`, or
    `integration_bandwidth` when bins are summed"""
    resolution_bandwidth: float
    """equivalent noise bandwidth (in Hz) of the nonzero samples of the analysis
    window"""


@specs.helpers.lru_cache_on_converted(specs.Capture, specs.Spectrogram)
def validated_spectrogram_sizing(
    capture: specs.Capture, spec: specs.Spectrogram
) -> SpectrogramSizing:
    """return the STFT sizing of `capture` under `spec`.

    Any `FrequencyAnalysisSpecBase` is accepted as `spec`. The result is cached on
    the fields of `capture` and `spec` that set the sizing. `center_frequency` is not
    read. The `window` design for complex64 IQ is evaluated and written to the
    disk-backed window cache.

    Raises:
        ValueError: if `sample_rate` is not a whole multiple of `frequency_resolution`,
            the capture is shorter than one FFT window, `lo_bandstop` or (with
            `trim_stopband`) `analysis_bandwidth` is off the frequency grid, the
            zero-padded part of the window under `window_fill` is not a whole number
            of samples, `integration_bandwidth` is not a multiple of
            `frequency_resolution` or exceeds the analyzed bandwidth, `time_aperture`
            is not a multiple of the hop period or exceeds the capture, or
            `sw.get_window` rejects `window`
    """
    if not sw.isroundmod(capture.sample_rate, spec.frequency_resolution):
        raise ValueError(
            'sample_rate/resolution must be a counting number '
            f'(sample_rate: {capture.sample_rate}, '
            f'frequency_resolution: {spec.frequency_resolution})'
        )
    nfft = round(capture.sample_rate / spec.frequency_resolution)

    samples = shared.capture_sample_count(capture)
    if samples < nfft:
        raise ValueError(
            'duration must span at least one FFT window of 1/frequency_resolution '
            f'(duration: {capture.duration}, sample_rate: {capture.sample_rate}, '
            f'frequency_resolution: {spec.frequency_resolution}, '
            f'samples: {samples}, nfft: {nfft})'
        )

    if spec.lo_bandstop is not None:
        shared.check_frequency_band(
            'lo_bandstop', nfft, capture.sample_rate, spec.lo_bandstop
        )

    if spec.trim_stopband and math.isfinite(capture.analysis_bandwidth):
        kept_bins = shared.check_frequency_band(
            'analysis_bandwidth', nfft, capture.sample_rate, capture.analysis_bandwidth
        )
    else:
        kept_bins = nfft

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

    if frequency_bin_averaging is not None and frequency_bin_averaging > kept_bins:
        raise ValueError(
            'integration_bandwidth must not exceed the analyzed bandwidth '
            f'(integration_bandwidth: {spec.integration_bandwidth}, '
            f'sample_rate: {capture.sample_rate}, '
            f'analysis_bandwidth: {capture.analysis_bandwidth}, '
            f'trim_stopband: {spec.trim_stopband}, '
            f'frequency bins: {kept_bins})'
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

    window_count = (samples - nfft) // hop_size + 1
    if time_bin_averaging is not None and time_bin_averaging > window_count:
        raise ValueError(
            'duration must span at least one time_aperture of STFT windows '
            f'(duration: {capture.duration}, time_aperture: {spec.time_aperture}, '
            f'hop_period: {hop_period}, STFT windows: {window_count})'
        )

    # `get_window` is called for 2 side effects: ValueError on an invalid window name
    # during validation rather than at the first analysis, and the window design
    # written to the disk-backed cache read by `sw.spectrogram`.
    sw.get_window(
        spec.window,
        nfft - nzero,
        nzero=nzero,
        dtype=np.dtype('complex64'),
        norm=True,
        fftshift=True,
    )

    if spec.integration_bandwidth is None:
        noise_bandwidth = spec.frequency_resolution
    else:
        noise_bandwidth = spec.integration_bandwidth

    # the window is normalized to unit mean power over the padded length. The mean
    # noise power in a bin is then N0 * frequency_resolution for every window shape.
    # The ENBW over the nonzero samples sets the power of a tone.
    nwindow = nfft - nzero
    enbw_bins = float(sw.equivalent_noise_bandwidth(spec.window, nwindow))
    resolution_bandwidth = enbw_bins * capture.sample_rate / nwindow

    return SpectrogramSizing(
        nfft=nfft,
        noverlap=noverlap,
        nzero=nzero,
        hop_size=hop_size,
        hop_period=hop_period,
        frequency_bin_averaging=frequency_bin_averaging,
        time_bin_averaging=time_bin_averaging,
        noise_bandwidth=noise_bandwidth,
        resolution_bandwidth=resolution_bandwidth,
    )


def evaluate_spectrogram(
    iq: Array,
    capture: specs.Capture,
    spec: specs.Spectrogram,
    *,
    dtype: Literal['float16', 'float32'] = 'float32',
    limit_digits: Optional[int] = None,
    dB=True,
) -> tuple[Array, dict]:
    spg, attrs = _cached_spectrogram(iq, capture, spec=spec)
    xp = sw.array_namespace(iq)

    copied = False
    if dB:
        spg = sw.powtodB(spg, bounded=True)
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
        sw.fourier.null_lo(spg, capture.sample_rate, spec.lo_bandstop, axis=2)

    if spec.trim_stopband and np.isfinite(capture.analysis_bandwidth):
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

    attrs = shared.bandwidth_attrs(
        sizing.noise_bandwidth,
        sw.units.DB_POWER,
        resolution_bandwidth=sizing.resolution_bandwidth,
    )

    return spg, attrs


@specs.helpers.lru_cache_on_converted(specs.Capture, specs.Spectrogram)
def spectrogram_freqs(capture: specs.Capture, spec: specs.Spectrogram) -> np.ndarray:
    sizing = validated_spectrogram_sizing(capture, spec)
    nfft = sizing.nfft

    # sw.fftfreq is exact. Spectra with different sampling parameters then merge
    # without rounding mismatches in the frequency coordinates.
    freqs = sw.fftfreq(nfft, capture.sample_rate)

    if spec.trim_stopband and np.isfinite(capture.analysis_bandwidth):
        freqs = sw.fourier.truncate_freqs(
            freqs, nfft, capture.sample_rate, capture.analysis_bandwidth, axis=0
        )

    if spec.integration_bandwidth is not None:
        freqs = sw.binned_mean(freqs, sizing.frequency_bin_averaging, fft=True)

    return freqs.astype('float64').round(16)


@registry.coordinates(dtype='float64', standard_name='Baseband frequency', units='Hz')
def spectrogram_baseband_frequency(
    capture: specs.Capture, spec: specs.Spectrogram
) -> np.ndarray:
    """baseband frequency of each bin, in steps of `frequency_resolution` about DC
    or of `integration_bandwidth` when set, truncated to a finite
    `analysis_bandwidth` when `trim_stopband` is set"""
    return spectrogram_freqs(capture, spec)


def spectrogram_window_count(capture: specs.Capture, sizing: SpectrogramSizing) -> int:
    """return the STFT window count of `capture` before `time_aperture` averaging"""
    samples = round(capture.duration * capture.sample_rate)
    return (samples - sizing.nfft) // sizing.hop_size + 1


@specs.helpers.lru_cache_on_converted(specs.Capture, specs.Spectrogram)
def spectrogram_level_tolerance(
    capture: specs.Capture,
    spec: specs.Spectrogram,
    *,
    array_backend: sw.typing.ArrayBackend = 'numpy',
    input_error: float = 0.0,
    dtype: Literal['float16', 'float32'] = 'float32',
    limit_digits: int | None = None,
    statistic_count: int = 1,
) -> specs.Tolerance:
    """bound the error (in dB) of a dB spectrogram evaluated by `evaluate_spectrogram`.

    Any `FrequencyAnalysisSpecBase` is accepted as `spec`.

    Args:
        array_backend: selects the FFT roundoff model of `sw.fourier.fft_tolerance_rms`
        input_error: relative rms error of the IQ amplitude (linear, unitless)
        dtype: the storage dtype of the output
        limit_digits: the number of decimals kept in the output; None keeps all
        statistic_count: number of windows averaged after the spectrogram, or 1
            when no windows are averaged

    Returns:
        the `specs.Tolerance` in dB over every output element (frequency bin by
        window) that remains after any averaging
    """
    sizing = validated_spectrogram_sizing(capture, spec)
    n_windows = spectrogram_window_count(capture, sizing)
    bins = sizing.nfft

    power_rtol = sw.power_analysis.envelope_power_rtol(np.float32, complex_input=True)
    power_rms = 0.0
    if sizing.frequency_bin_averaging is not None:
        power_rms = sw.arrays.accum_rms(np.float32, sizing.frequency_bin_averaging)
        bins //= sizing.frequency_bin_averaging
    # the frequency bins are the contiguous axis, and the window axis is strided
    if sizing.time_bin_averaging is not None:
        power_rtol += sw.arrays.accum_rtol(np.float32, sizing.time_bin_averaging)
        n_windows //= sizing.time_bin_averaging
    if statistic_count > 1:
        power_rtol += sw.arrays.accum_rtol(np.float32, statistic_count)

    fft_error = sw.fourier.fft_tolerance_rms(
        np.complex64, [sizing.nfft], array_backend=array_backend
    )

    return shared.level_tolerance(
        amplitude_rms=math.hypot(input_error, fft_error),
        size=max(bins * n_windows, 1),
        power_rtol=power_rtol,
        power_rms=power_rms,
        log_tol=sw.power_analysis.log_conversion_tol(
            np.float32, 10, complex_input=True
        ),
        quantization=shared.quantization_dB(dtype, limit_digits),
    )


# %% spectrogram
@registry.coordinates(dtype='float32', standard_name='Time elapsed', units='s')
@specs.helpers.lru_cache_on_converted(specs.Capture)
def spectrogram_time(capture: specs.Capture, spec: specs.Spectrogram) -> np.ndarray:
    """start time of each STFT window from the start of the capture, in steps of the
    hop set by `frequency_resolution` and `fractional_overlap`, or of `time_aperture`
    when set"""
    sizing = validated_spectrogram_sizing(capture, spec)

    scale = sizing.nfft / sizing.hop_size
    size = int(scale * (capture.sample_rate * capture.duration / sizing.nfft - 1) + 1)
    hop_period = sizing.hop_period

    if sizing.time_bin_averaging is not None:
        size = size // sizing.time_bin_averaging
        hop_period = hop_period * sizing.time_bin_averaging

    return np.arange(size) * hop_period


def spectrogram_tolerance(
    capture: specs.Capture, spec: specs.Spectrogram, **kwargs: Unpack[ToleranceKws]
) -> specs.Tolerance:
    return spectrogram_level_tolerance(
        capture, spec, dtype='float16', limit_digits=2, **kwargs
    )


@hint_keywords(specs.Spectrogram)
@registry.measurement(
    coord_factories=[spectrogram_time, spectrogram_baseband_frequency],
    spec_type=specs.Spectrogram,
    dtype='float16',
    caches=spectrogram_cache,
    standard_name='PSD',
    long_name='Spectrogram',
    validate=validated_spectrogram_sizing,
    tolerance=spectrogram_tolerance,
    prefer_filtered=False,
)
def spectrogram(x: Array, capture: specs.Capture, **kwargs: Any) -> Measurement:
    """evaluate a power spectrogram of the IQ by STFT.

    The input is the IQ before the `analysis_bandwidth` filter and after trigger
    alignment. Each value is the power in one frequency bin of one STFT window.
    White noise with power spectral density N0 has a mean of N0 times the noise
    bandwidth in every bin for every `window`.

    Output:
        - Array dtype: {dtype}
        - Dimensions: {dims}
        - Coordinates: {coords}
        - Attributes: {attrs}
          * `units`: power units based on acquisition (dBm when calibrated)
          * `noise_bandwidth`: the bandwidth (in Hz) that the values are referred
            to: `frequency_resolution`, or `integration_bandwidth` when set
          * `resolution_bandwidth`: the equivalent noise bandwidth (in Hz) of the
            nonzero samples of `window`; a CW tone has
            ``frequency_resolution / resolution_bandwidth`` of the tone power in one
            bin
          * `limit_digits`: 2, the decimal places of the rounding

    Args:
    {args}
    """
    spec = specs.Spectrogram.from_dict(kwargs).validate()
    spg, attrs = evaluate_spectrogram(
        x, capture, spec, dB=True, limit_digits=2, dtype='float16'
    )

    return spg, attrs


# %% power_spectral_density
@registry.coordinates(dtype='str', standard_name='Time statistic')
@specs.helpers.lru_cache_on_converted(specs.Capture)
def time_statistic(
    capture: specs.Capture, spec: specs.PowerSpectralDensity
) -> np.ndarray:
    """name of each statistic, in the order of `time_statistic`"""
    time_statistic = [str(s) for s in spec.time_statistic]
    return np.asarray(time_statistic, dtype=object)


@registry.coordinates(dtype='float64', standard_name='Baseband frequency', units='Hz')
@specs.helpers.lru_cache_on_converted(specs.Capture, specs.Spectrogram)
def baseband_frequency(capture: specs.Capture, spec: specs.Spectrogram) -> np.ndarray:
    """baseband frequency of each bin, in steps of `frequency_resolution` about DC
    or of `integration_bandwidth` when set, truncated to a finite
    `analysis_bandwidth` when `trim_stopband` is set"""
    return spectrogram_baseband_frequency(capture, spec)


def validated_psd_sizing(
    capture: specs.Capture, spec: specs.PowerSpectralDensity
) -> SpectrogramSizing:
    """return the STFT sizing of `capture` under `spec`.

    Raises:
        ValueError: for the conditions of `validated_spectrogram_sizing`, or if a
            `time_statistic` entry is not a supported statistic
    """
    sizing = validated_spectrogram_sizing(capture, spec)
    shared.check_statistics('time_statistic', spec.time_statistic)
    return sizing


def power_spectral_density_tolerance(
    capture: specs.Capture,
    spec: specs.PowerSpectralDensity,
    **kwargs: Unpack[ToleranceKws],
) -> specs.Tolerance:
    spg_spec = specs.Spectrogram.from_spec(spec)
    sizing = validated_spectrogram_sizing(capture, spg_spec)
    if any(s in ('mean', 'rms') for s in spec.time_statistic):
        statistic_count = spectrogram_window_count(capture, sizing)
    else:
        statistic_count = 1
    return spectrogram_level_tolerance(
        capture, spg_spec, dtype='float32', statistic_count=statistic_count, **kwargs
    )


@hint_keywords(specs.PowerSpectralDensity)
@registry.measurement(
    depends=spectrogram,
    coord_factories=[time_statistic, baseband_frequency],
    spec_type=specs.PowerSpectralDensity,
    dtype='float32',
    standard_name='Power spectral density',
    long_name='Power spectral density',
    validate=validated_psd_sizing,
    tolerance=power_spectral_density_tolerance,
    prefer_filtered=False,
)
def power_spectral_density(
    x: Array, capture: specs.Capture, **kwargs: Any
) -> Measurement:
    """estimate the power spectral density from time statistics of the spectrogram.

    The input is the IQ before the `analysis_bandwidth` filter and after trigger
    alignment. Each `time_statistic` reduces the unrounded linear-power
    `spectrogram` output over the STFT windows at each frequency bin. ``'mean'`` is
    the Welch estimate (Welch, 1967). A float in ``[0, 1]`` is a quantile.

    Output:
        - Array dtype: {dtype}
        - Dimensions: {dims}
        - Coordinates: {coords}
        - Attributes: {attrs}
          * `units`: power units based on acquisition (dBm when calibrated)
          * `noise_bandwidth`: the bandwidth (in Hz) that the values are referred
            to: `frequency_resolution`, or `integration_bandwidth` when set; the
            output minus ``10 * log10(noise_bandwidth)`` is the density per Hz
          * `resolution_bandwidth`: the equivalent noise bandwidth (in Hz) of the
            nonzero samples of `window`

    References:
        P. D. Welch, "The use of fast Fourier transform for the estimation of power
        spectra: A method based on time averaging over short, modified
        periodograms," IEEE Trans. Audio Electroacoust., vol. 15, no. 2,
        pp. 70-73, 1967. doi:10.1109/TAU.1967.1161901

    Args:
    {args}
    """

    spec = specs.PowerSpectralDensity.from_dict(kwargs)
    spg_spec = specs.Spectrogram.from_spec(spec)

    from striqt.waveform.lib.power import stat_ufunc_from_shorthand

    working_dtype = 'float32'

    xp = sw.array_namespace(x)
    axis = 1

    spg, metadata = evaluate_spectrogram(
        x,
        capture,
        spg_spec,
        dB=False,
        dtype=working_dtype,
    )

    findquantile = sw.util.find_float_inds(tuple(spec.time_statistic))

    newshape = list(spg.shape)
    newshape[axis] = len(spec.time_statistic)
    psd = xp.empty(newshape, dtype=working_dtype)

    q = [spec.time_statistic[i] for i, flag in enumerate(findquantile) if flag]
    psd[:, findquantile] = (
        xp
        .quantile(spg, q, axis=axis)
        .swapaxes(0, axis)  # quantile moves the statistic axis to axis 0
        .astype(working_dtype)
    )

    i_isnt_quantile = np.where(~np.array(findquantile))[0]
    for i in i_isnt_quantile:
        ufunc = stat_ufunc_from_shorthand(spec.time_statistic[i], xp=xp)
        sw.axis_index(psd, i, axis=axis)[:] = ufunc(spg, axis=axis)

    psd = sw.powtodB(psd)

    sizing = validated_spectrogram_sizing(capture, spg_spec)
    metadata = shared.bandwidth_attrs(
        sizing.noise_bandwidth,
        sw.units.DB_POWER,
        resolution_bandwidth=sizing.resolution_bandwidth,
    )

    return psd, metadata


# %% spectrogram_histogram
@registry.coordinates(
    dtype='float32',
    standard_name='Spectrogram bin power',
    units=sw.units.DB_POWER,
)
@specs.helpers.lru_cache_on_converted(specs.Capture, specs.SpectrogramHistogram)
def spectrogram_power_bin(
    capture: specs.Capture, spec: specs.SpectrogramHistogram
) -> tuple[np.ndarray, dict[str, typing.Any]]:
    """center of each power bin, from `power_low` to `power_high` in steps of
    `power_resolution`, with ``-inf`` and ``inf`` catch-all bins at each end

    Returns:
        the bin centers, and attrs with ``'units'`` and the ``'noise_bandwidth'``
        and ``'resolution_bandwidth'`` (in Hz) of the spectrogram
    """
    bins = power.make_power_bins(
        power_low=spec.power_low,
        power_high=spec.power_high,
        power_resolution=spec.power_resolution,
    )

    sizing = validated_spectrogram_sizing(capture, spec)
    attrs = shared.bandwidth_attrs(
        sizing.noise_bandwidth,
        sw.units.DB_POWER,
        resolution_bandwidth=sizing.resolution_bandwidth,
    )

    return bins, attrs


@hint_keywords(specs.SpectrogramHistogram)
@registry.measurement(
    depends=spectrogram,
    coord_factories=[spectrogram_power_bin],
    spec_type=specs.SpectrogramHistogram,
    dtype='float32',
    standard_name='Fraction of counts',
    long_name='Spectrogram histogram',
    validate=validated_spectrogram_sizing,
    prefer_filtered=False,
)
def spectrogram_histogram(
    x: Array, capture: specs.Capture, **kwargs: Any
) -> Measurement:
    """evaluate the fraction of spectrogram pixels that fall in each power bin.

    The input is the IQ before the `analysis_bandwidth` filter and after trigger
    alignment. The pixels are the unrounded dB spectrogram of the same STFT fields
    as `spectrogram`, flattened over time and frequency. Values sum to 1 per port.
    The ``-inf`` and ``inf`` entries collect the pixels more than half a bin below
    `power_low` or above `power_high`.

    Output:
        - Array dtype: {dtype}
        - Dimensions: {dims}
        - Coordinates: {coords}
        - Attributes: {attrs}
          * `noise_bandwidth`: the noise bandwidth (in Hz) of the spectrogram pixels,
            `frequency_resolution` or `integration_bandwidth` when set
          * `resolution_bandwidth`: the equivalent noise bandwidth (in Hz) of the
            nonzero samples of `window`
          * `limit_digits`: None; the pixels are not rounded

    Args:
    {args}
    """
    spec = specs.SpectrogramHistogram.from_dict(kwargs)
    spg_spec = specs.Spectrogram.from_spec(spec)

    spg, metadata = evaluate_spectrogram(
        x,
        capture,
        spec=spg_spec,
        dtype='float32',
    )

    metadata = dict(metadata)
    metadata.pop('units')

    xp = sw.array_namespace(x)
    bin_edges = power.make_power_histogram_bin_edges(
        power_low=spec.power_low,
        power_high=spec.power_high,
        power_resolution=spec.power_resolution,
        xp=xp,
    )

    count_dtype = xp.finfo(x.dtype).dtype
    counts = xp.asarray(
        [xp.histogram(spg[i].ravel(), bin_edges)[0] for i in range(spg.shape[0])],
        dtype=count_dtype,
    )

    data = counts / xp.sum(counts[0])

    return data, metadata


# %% spectrogram_ratio_histogram
@registry.coordinates(
    dtype='float32',
    standard_name='Spectrogram cross-channel power ratio',
    units='dB',
)
@specs.helpers.lru_cache_on_converted(specs.Capture)
def spectrogram_ratio_power_bin(
    capture: specs.Capture, spec: specs.SpectrogramHistogramRatio
) -> tuple[np.ndarray, dict[str, typing.Any]]:
    """center of each power-ratio bin, from `power_low` to `power_high` in steps of
    `power_resolution`, with ``-inf`` and ``inf`` catch-all bins at each end

    Returns:
        the bin centers, and attrs with ``'units'`` of ``'dB'``
    """

    bins, _ = spectrogram_power_bin(capture, spec)

    return bins, {'units': 'dB'}


@hint_keywords(specs.SpectrogramHistogramRatio)
@registry.measurement(
    depends=spectrogram,
    coord_factories=[spectrogram_ratio_power_bin],
    spec_type=specs.SpectrogramHistogramRatio,
    dtype='float32',
    standard_name='Fraction of counts',
    long_name='Spectrogram ratio histogram',
    validate=validated_spectrogram_sizing,
    prefer_filtered=False,
)
def spectrogram_ratio_histogram(
    x: Array, capture: specs.Capture, **kwargs: Any
) -> Measurement:
    """evaluate a histogram of the spectrogram power ratio between 2 ports.

    The input is the IQ before the `analysis_bandwidth` filter and after trigger
    alignment. The spectrograms are the unrounded dB spectrograms of the same STFT
    fields as `spectrogram`. The ratio is the pixel-by-pixel difference of the 2
    spectrograms in dB. The first output port is the ratio of the first input port
    to the second. The second output port is the negative of the first. Values sum
    to 1 per port. The ``-inf`` and ``inf`` entries collect the ratios more than half
    a bin below `power_low` or above `power_high`.

    Output:
        - Array dtype: {dtype}
        - Dimensions: {dims}
        - Coordinates: {coords}
        - Attributes: {attrs}
          * `limit_digits`: None; the pixels are not rounded

    Raises:
        ValueError: if the capture has a port count other than 2

    Args:
    {args}
    """
    spec = specs.SpectrogramHistogramRatio.from_dict(kwargs)
    spg_spec = specs.Spectrogram.from_spec(spec)

    spg, metadata = evaluate_spectrogram(
        x,
        capture,
        spg_spec,
        dtype='float32',
    )

    if spg.shape[0] != 2:
        raise ValueError(
            'ratio histograms are only supported for 2-channel measurements'
        )

    xp = sw.array_namespace(x)
    xp.subtract(spg[0], spg[1], out=spg[0])
    xp.negative(spg[0], out=spg[1])

    # a ratio of powers has no unit or bandwidth
    metadata = dict(metadata)
    for key in ('units', 'noise_bandwidth', 'resolution_bandwidth'):
        metadata.pop(key)

    bin_edges = power.make_power_histogram_bin_edges(
        power_low=spec.power_low,
        power_high=spec.power_high,
        power_resolution=spec.power_resolution,
        xp=xp,
    )

    count_dtype = xp.finfo(x.dtype).dtype
    counts = xp.asarray(
        [xp.histogram(spg[i].ravel(), bin_edges)[0] for i in range(spg.shape[0])],
        dtype=count_dtype,
    )

    data = counts / xp.sum(counts[0])

    return data, metadata
