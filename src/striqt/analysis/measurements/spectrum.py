from __future__ import annotations as __

import math
import typing
from typing import Any, Literal, NamedTuple, Optional, Union

from .. import specs

from ..lib import dataarrays, register
from ..lib.util import np
from . import power, shared
from .shared import registry, hint_keywords

import striqt.waveform as sw

if typing.TYPE_CHECKING:
    from typing_extensions import Unpack

    from ..lib.typing import Array, Measurement, ToleranceKws


# %% STFT sizing and the shared spectrogram cache
class SpectrogramSizing(NamedTuple):
    """the STFT sizing of one capture under a spectrogram spec.

    Instances come from `validated_spectrogram_sizing`. `nfft`, `noverlap`, `nzero` and `hop_size` are in samples and `hop_period` in s;
    `frequency_bin_averaging` and `time_bin_averaging` are the bins summed per
    `integration_bandwidth` and the windows averaged per `time_aperture`, each `None`
    when its field is unset; `enbw` is the noise bandwidth of one output bin in Hz.
    """

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

    The window that `sw.spectrogram` builds for complex64 IQ is designed here, so a bad
    window name or parameter fails before acquisition and the disk-cached design
    search is paid once.

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

    # the design that `sw.spectrogram` will build for complex64 IQ, so a bad window
    # name or parameter fails here and the disk-cached search is paid before
    # acquisition
    sw.get_window(
        spec.window,
        nfft - nzero,
        nzero=nzero,
        dtype=np.dtype('complex64'),
        norm=True,
        fftshift=True,
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

    The argument projections match `validated_spectrogram_sizing`, so any
    `FrequencyAnalysisSpecBase` may be passed as `spec`.

    Args:
        array_backend: selects the FFT roundoff model of `sw.fourier.fft_tolerance_rms`
        input_error: relative rms error of the IQ amplitude (linear, unitless)
        dtype: the storage dtype the measurement hands to `evaluate_spectrogram`
        limit_digits: the decimal rounding the measurement hands to
            `evaluate_spectrogram`; `None` for none
        statistic_count: number of windows a derived measurement averages afterward
            (the PSD's mean statistic); 1 when it keeps them or only selects among them

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
    # the frequency bins are the contiguous axis; the window axis is strided
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
@registry.coordinates(
    dtype='float32', attrs={'standard_name': 'Time Elapsed', 'units': 's'}
)
@specs.helpers.lru_cache_on_converted(specs.Capture)
def spectrogram_time(capture: specs.Capture, spec: specs.Spectrogram) -> np.ndarray:
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
    prefer_iq_source='pre_filter',
    attrs={'standard_name': 'PSD', 'long_name': 'Power Spectral Density'},
    validate=validated_spectrogram_sizing,
    tolerance=spectrogram_tolerance,
)
def spectrogram(iq: Array, capture: specs.Capture, **kwargs: Any) -> Measurement:
    """evaluate a power spectrogram of the IQ by STFT.

    The input is the resampled IQ before the `analysis_bandwidth` low-pass filter and
    any trigger alignment; the output array namespace (numpy or cupy) follows it. Per
    capture, the result has dimensions
    ``(spectrogram_time, spectrogram_baseband_frequency)``: `spectrogram_time` in s
    runs from the start of the capture at the STFT hop, which `fractional_overlap`
    and `frequency_resolution` set, and `spectrogram_baseband_frequency` in Hz is the
    FFT grid at `frequency_resolution`, truncated to the `analysis_bandwidth` about
    DC when `trim_stopband` is set and the bandwidth is finite. Values are power in
    dBm per noise bandwidth, which is `frequency_resolution` or, when set,
    `integration_bandwidth` (adjacent bins summed); the `units` and `noise_bandwidth`
    attrs record it. `time_aperture` averages consecutive windows, and `lo_bandstop`
    sets the bins within that bandwidth of DC to ``nan``. Values are rounded to 2
    decimals and stored as float16. `striqt.waveform.spectrogram` evaluates the STFT.

    Args:
    {args}
    """
    spec = specs.Spectrogram.from_dict(kwargs).validate()
    spg, attrs = evaluate_spectrogram(
        iq, capture, spec, dB=True, limit_digits=2, dtype='float16'
    )

    return spg, attrs


# %% power_spectral_density
@registry.coordinates(dtype='str', attrs={'standard_name': 'Time statistic'})
@specs.helpers.lru_cache_on_converted(specs.Capture)
def time_statistic(
    capture: specs.Capture, spec: specs.PowerSpectralDensity
) -> np.ndarray:
    time_statistic = [str(s) for s in spec.time_statistic]
    return np.asarray(time_statistic, dtype=object)


@registry.coordinates(
    dtype='float64', attrs={'standard_name': 'Baseband frequency', 'units': 'Hz'}
)
@specs.helpers.lru_cache_on_converted(specs.Capture, specs.Spectrogram)
def baseband_frequency(capture: specs.Capture, spec: specs.Spectrogram) -> np.ndarray:
    return spectrogram_baseband_frequency(capture, spec)


def validated_psd_sizing(
    capture: specs.Capture, spec: specs.PowerSpectralDensity
) -> SpectrogramSizing:
    """check the STFT sizing and the `time_statistic` names, returning the sizing.

    Raises:
        ValueError: for the conditions of `validated_spectrogram_sizing`, or a
            `time_statistic` entry that is not a supported statistic
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
    prefer_iq_source='pre_filter',
    dtype='float32',
    attrs={'standard_name': 'Power spectral density'},
    validate=validated_psd_sizing,
    tolerance=power_spectral_density_tolerance,
)
def power_spectral_density(
    iq: Array, capture: specs.Capture, **kwargs: Any
) -> Measurement:
    """estimate the power spectral density by the Welch method.

    Each `time_statistic` reduces the `spectrogram` output with the same STFT fields
    (in linear power, unrounded) across its windows at each frequency bin; 'mean' is
    the Welch estimate (P. D. Welch, IEEE Trans. Audio Electroacoust. 15(2), 1967)
    and a float in ``[0, 1]`` is a quantile. Per capture, the result has dimensions
    ``(time_statistic, baseband_frequency)`` in dBm per noise bandwidth, as recorded
    in the `units` and `noise_bandwidth` attrs, stored as float32.

    Args:
    {args}
    """

    spec = specs.PowerSpectralDensity.from_dict(kwargs)
    spg_spec = specs.Spectrogram.from_spec(spec)

    from striqt.waveform.lib.power_analysis import stat_ufunc_from_shorthand

    working_dtype = 'float32'

    xp = sw.array_namespace(iq)
    axis = 1

    spg, metadata = evaluate_spectrogram(
        iq,
        capture,
        spg_spec,
        dB=False,
        dtype=working_dtype,
    )

    findquantile = sw.util.find_float_inds(tuple(spec.time_statistic))

    newshape = list(spg.shape)
    newshape[axis] = len(spec.time_statistic)
    psd = xp.empty(newshape, dtype=working_dtype)

    # all of the quantiles, evaluated together
    q = [spec.time_statistic[i] for i, flag in enumerate(findquantile) if flag]
    psd[:, findquantile] = (
        xp
        .quantile(spg, q, axis=axis)
        .swapaxes(0, axis)  # quantile bumps the output result to axis 0
        .astype(working_dtype)  #
    )

    # everything else
    i_isnt_quantile = np.where(~np.array(findquantile))[0]
    for i in i_isnt_quantile:
        ufunc = stat_ufunc_from_shorthand(spec.time_statistic[i], xp=xp)
        sw.axis_index(psd, i, axis=axis)[:] = ufunc(spg, axis=axis)

    psd = sw.powtodB(psd)

    return psd, metadata


# %% spectrogram_histogram
@registry.coordinates(
    dtype='float32', attrs={'standard_name': 'Spectrogram bin power', 'units': 'dBm'}
)
@specs.helpers.lru_cache_on_converted(specs.Capture, specs.SpectrogramHistogram)
def spectrogram_power_bin(
    capture: specs.Capture, spec: specs.SpectrogramHistogram
) -> tuple[np.ndarray, dict[str, typing.Any]]:
    """return the `spectrogram_power_bin` coordinate values and attrs.

    Returns:
        the bin centers of `power.make_power_bins` (with ``-inf`` and ``inf`` catch-all
        bins), and a dict whose ``'units'`` is dBm per the spectrogram noise bandwidth
    """
    bins = power.make_power_bins(
        power_low=spec.power_low,
        power_high=spec.power_high,
        power_resolution=spec.power_resolution,
    )

    enbw = validated_spectrogram_sizing(capture, spec).enbw

    return bins, {'units': f'dBm/{enbw / 1e3:0.0f} kHz'}


@hint_keywords(specs.SpectrogramHistogram)
@registry.measurement(
    depends=spectrogram,
    coord_factories=[spectrogram_power_bin],
    spec_type=specs.SpectrogramHistogram,
    dtype='float32',
    prefer_iq_source='pre_filter',
    attrs={'standard_name': 'Fraction of counts'},
    validate=validated_spectrogram_sizing,
)
def spectrogram_histogram(
    iq: Array, capture: specs.Capture, **kwargs: Any
) -> Measurement:
    """evaluate the fraction of spectrogram pixels that fall in each power bin.

    The pixels are those of the `spectrogram` output with the same STFT fields
    (float32, unrounded), flattened over time and frequency. Per capture, the result
    has the dimension ``(spectrogram_power_bin,)`` and sums to 1.
    `spectrogram_power_bin` holds the bin centers in dBm per noise bandwidth; its
    ``-inf`` and ``inf`` entries collect the pixels more than half a bin below
    `power_low` or above `power_high`. The `noise_bandwidth` attr is in Hz.

    Args:
    {args}
    """
    spec = specs.SpectrogramHistogram.from_dict(kwargs)
    spg_spec = specs.Spectrogram.from_spec(spec)

    spg, metadata = evaluate_spectrogram(
        iq,
        capture,
        spec=spg_spec,
        dtype='float32',
    )

    metadata = dict(metadata)
    metadata.pop('units')

    xp = sw.array_namespace(iq)
    bin_edges = power.make_power_histogram_bin_edges(
        power_low=spec.power_low,
        power_high=spec.power_high,
        power_resolution=spec.power_resolution,
        xp=xp,
    )

    count_dtype = xp.finfo(iq.dtype).dtype
    counts = xp.asarray(
        [xp.histogram(spg[i].flatten(), bin_edges)[0] for i in range(spg.shape[0])],
        dtype=count_dtype,
    )

    data = counts / xp.sum(counts[0])

    return data, metadata


# %% spectrogram_ratio_histogram
@registry.coordinates(
    dtype='float32',
    attrs={'standard_name': 'Spectrogram cross-channel power ratio', 'units': 'dB'},
)
@specs.helpers.lru_cache_on_converted(specs.Capture)
def spectrogram_ratio_power_bin(
    capture: specs.Capture, spec: specs.SpectrogramHistogramRatio
) -> tuple[np.ndarray, dict[str, typing.Any]]:
    """return the `spectrogram_ratio_power_bin` coordinate values and attrs.

    Returns:
        the bin centers of `spectrogram_power_bin`, and a copy of its attrs with
        ``'units'`` relabeled from dBm to dB
    """

    bins, attrs = spectrogram_power_bin(capture, spec)

    # `attrs` is held in a cache that `spectrogram_histogram` shares whenever the two
    # measurements agree on the spectrogram and power bin fields, so relabeling it in
    # place would relabel that measurement's absolute powers as ratios
    return bins, attrs | {'units': attrs['units'].replace('dBm', 'dB')}


@hint_keywords(specs.SpectrogramHistogramRatio)
@registry.measurement(
    depends=spectrogram,
    coord_factories=[spectrogram_ratio_power_bin],
    spec_type=specs.SpectrogramHistogramRatio,
    dtype='float32',
    prefer_iq_source='pre_filter',
    attrs={'standard_name': 'Fraction of counts'},
    validate=validated_spectrogram_sizing,
)
def spectrogram_ratio_histogram(
    iq: Array, capture: specs.Capture, **kwargs: Any
) -> Measurement:
    """evaluate a histogram of the spectrogram power ratio between two channels.

    The spectrograms are the `spectrogram` output with the same STFT fields (float32,
    unrounded), and the ratio is their pixel-by-pixel difference in dB. The capture
    must hold exactly 2 channels; otherwise `ValueError` is raised. Per capture, the
    result has the dimension ``(spectrogram_ratio_power_bin,)`` with bin centers in
    dB; the first channel holds the ratio of the first to the second, and the second
    channel its inverse. Normalization and the catch-all bins follow
    `spectrogram_histogram`.

    Args:
    {args}
    """
    spec = specs.SpectrogramHistogramRatio.from_dict(kwargs)
    spg_spec = specs.Spectrogram.from_spec(spec)

    spg, metadata = evaluate_spectrogram(
        iq,
        capture,
        spg_spec,
        dtype='float32',
    )

    if spg.shape[0] != 2:
        raise ValueError(
            'ratio histograms are only supported for 2-channel measurements'
        )

    spg[0], spg[1] = spg[0] - spg[1], spg[1] - spg[0]

    metadata = dict(metadata)
    metadata.pop('units')

    xp = sw.array_namespace(iq)
    bin_edges = power.make_power_histogram_bin_edges(
        power_low=spec.power_low,
        power_high=spec.power_high,
        power_resolution=spec.power_resolution,
        xp=xp,
    )

    count_dtype = xp.finfo(iq.dtype).dtype
    counts = xp.asarray(
        [xp.histogram(spg[i].flatten(), bin_edges)[0] for i in range(spg.shape[0])],
        dtype=count_dtype,
    )

    data = counts / xp.sum(counts[0])

    return data, metadata
