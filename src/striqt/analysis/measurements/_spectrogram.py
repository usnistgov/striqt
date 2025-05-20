from __future__ import annotations
import contextlib
import decimal
import functools
import typing
import warnings

from ..lib import registry, specs, util

if typing.TYPE_CHECKING:
    import iqwaveform
    import numpy as np
else:
    iqwaveform = util.lazy_import('iqwaveform')
    np = util.lazy_import('numpy')

warnings.filterwarnings(
    'ignore', '.*Mean of empty slice.*', category=RuntimeWarning, module=__name__
)


class FrequencyAnalysisBase(
    specs.Measurement,
    forbid_unknown_fields=True,
    cache_hash=True,
    kw_only=True,
    frozen=True,
):
    window: typing.Union[str, tuple[str, float]]
    frequency_resolution: float
    fractional_overlap: float = 0
    window_fill: float = 1
    frequency_bin_averaging: typing.Optional[int] = None


class FrequencyAnalysisKeywords(specs.AnalysisKeywords):
    window: typing.Union[str, tuple[str, float]]
    frequency_resolution: float
    fractional_overlap: typing.NotRequired[float]
    window_fill: typing.NotRequired[float]
    frequency_bin_averaging: typing.NotRequired[typing.Optional[int]]


class SpectrogramSpec(
    FrequencyAnalysisBase,
    forbid_unknown_fields=True,
    cache_hash=True,
    kw_only=True,
    frozen=True,
):
    time_bin_averaging: typing.Optional[int] = None


class SpectrogramKeywords(FrequencyAnalysisKeywords):
    time_bin_averaging: typing.NotRequired[typing.Optional[int]]


@util.lru_cache()
def equivalent_noise_bandwidth(window: typing.Union[str, tuple[str, float]], nfft: int):
    """return the equivalent noise bandwidth (ENBW) of a window, in bins"""
    w = iqwaveform.fourier.get_window(window, nfft)
    return len(w) * np.sum(w**2) / np.sum(w) ** 2


def truncate_spectrogram_bandwidth(x, nfft, fs, bandwidth, axis=0):
    """trim an array outside of the specified bandwidth on a frequency axis"""
    edges = iqwaveform.fourier._freq_band_edges(
        nfft, 1.0 / fs, cutoff_low=-bandwidth / 2, cutoff_hi=bandwidth / 2
    )
    return iqwaveform.util.axis_slice(x, *edges, axis=axis)


def fftfreq(nfft, fs, dtype='float64') -> 'np.ndarray':
    """compute fftfreq for a specified sample rate.

    This is meant to produce higher-precision results based on
    rational sample rates in order to avoid rounding errors
    when merging captures with different sample rates.
    """
    # high resolution rational representation of frequency resolution
    fres = decimal.Decimal(fs) / nfft
    span = range(-nfft // 2, -nfft // 2 + nfft)
    if nfft % 2 == 0:
        values = [fres * n for n in span]
    else:
        values = [fres * (n + 1) for n in span]
    return np.array(values, dtype=dtype)


def freq_axis_values(
    capture: specs.RadioCapture, fres: int, navg: int = None, trim_stopband=False
):
    if iqwaveform.isroundmod(capture.sample_rate, fres):
        nfft = round(capture.sample_rate / fres)
    else:
        raise ValueError('sample_rate/resolution must be a counting number')

    if navg is None:
        pass
    elif iqwaveform.util.isroundmod(navg, 1):
        navg = round(navg)
    else:
        raise ValueError('frequency_bin_averaging must be an integer bin count')

    # use the iqwaveform.fourier fftfreq for higher precision, which avoids
    # headaches when merging spectra with different sampling parameters due
    # to rounding errors.
    freqs = fftfreq(nfft, capture.sample_rate)

    if trim_stopband and np.isfinite(capture.analysis_bandwidth):
        # stick with python arithmetic here for numpy/cupy consistency
        freqs = truncate_spectrogram_bandwidth(
            freqs, nfft, capture.sample_rate, capture.analysis_bandwidth, axis=0
        )

    if navg is not None:
        freqs = iqwaveform.util.binned_mean(freqs, navg)
        freqs -= freqs[freqs.size // 2]

    # only now downconvert. round to a still-large number of digits
    return freqs.astype('float64').round(16)


class _spectrogram_cache:
    """A single-element cache keyed on arguments to _evaluate"""

    _key: frozenset = None
    _value = None
    enabled = False

    @staticmethod
    def kw_key(kws):
        if kws is None:
            return None

        kws = dict(kws)
        del kws['iq']
        return frozenset(kws.items())

    @classmethod
    def clear(cls):
        cls.update(None, None)

    @classmethod
    def lookup(cls, kws: dict):
        if cls._key is None or not cls.enabled:
            return None

        if cls.kw_key(kws) == cls._key:
            return cls._value
        else:
            return None

    @classmethod
    def update(cls, kws: dict, value):
        if not cls.enabled:
            return
        cls._key = cls.kw_key(kws)
        cls._value = value

    @classmethod
    def cached_calls(cls, func):
        @functools.wraps(func)
        def wrapped(**kws):
            match = cls.lookup(kws)
            if match is not None:
                return match

            ret = func(**kws)
            cls.update(kws, ret)
            return ret

        return wrapped


def evaluate_spectrogram(
    iq: 'iqwaveform.util.Array',
    capture: specs.Capture,
    spec: SpectrogramSpec,
    *,
    dtype: typing.Union[
        typing.Literal['float16'], typing.Literal['float32']
    ] = 'float16',
    limit_digits: typing.Optional[int] = None,
    trim_stopband: bool = True,
):
    spg, metadata = _cached_spectrogram(iq, capture, spec, trim_stopband=trim_stopband)

    xp = iqwaveform.util.array_namespace(iq)

    copied = False
    if spec.dB:
        spg = iqwaveform.powtodB(spg, eps=1e-25)
        copied = True

    spg = spg.astype(dtype, copy=not copied)

    if limit_digits is not None:
        xp.round(spg, spec.limit_digits, out=spg)

    metadata = metadata | {'limit_digits': spec.limit_digits}

    return spg, metadata


@contextlib.contextmanager
def cached_spectrograms():
    global _spectrogram_cache
    _spectrogram_cache.enabled = True
    yield
    print()
    _spectrogram_cache.clear()
    _spectrogram_cache.enabled = False


@_spectrogram_cache.cached_calls
def _cached_spectrogram(
    iq: 'iqwaveform.util.Array',
    capture: specs.Capture,
    spec: SpectrogramSpec,
    *,
    trim_stopband: bool = True,
):
    # TODO: integrate this back into iqwaveform
    spec = spec.validate()

    if iqwaveform.isroundmod(capture.sample_rate, spec.frequency_resolution):
        nfft = round(capture.sample_rate / spec.frequency_resolution)
    else:
        raise ValueError('sample_rate/resolution must be a counting number')

    if iqwaveform.isroundmod(capture.sample_rate, spec.frequency_resolution):
        noverlap = round(spec.fractional_overlap * nfft)
    else:
        raise ValueError('sample_rate_Hz/resolution must be a counting number')

    if iqwaveform.isroundmod((1 - spec.window_fill) * nfft, 1):
        nzero = round((1 - spec.window_fill) * nfft)
    else:
        raise ValueError(
            '(1-window_fill) * (sample_rate/frequency_resolution) must be a counting number'
        )

    spg = iqwaveform.fourier.spectrogram(
        iq,
        window=spec.window,
        fs=capture.sample_rate,
        nperseg=nfft,
        noverlap=noverlap,
        nzero=nzero,
        axis=1,
        return_axis_arrays=False,
        iter_axes=0,
    )

    # truncate to the analysis bandwidth
    if trim_stopband and np.isfinite(capture.analysis_bandwidth):
        # stick with python arithmetic to ensure consistency with axis bounds calculations
        spg = truncate_spectrogram_bandwidth(
            spg, nfft, capture.sample_rate, bandwidth=capture.analysis_bandwidth, axis=2
        )

    if spec.frequency_bin_averaging is not None:
        spg = iqwaveform.util.binned_mean(spg, spec.frequency_bin_averaging, axis=2)

    if spec.time_bin_averaging is not None:
        spg = iqwaveform.util.binned_mean(
            spg, spec.time_bin_averaging, axis=1, fft=False
        )

    if iqwaveform.util.is_cupy_array(iq):
        import cupy

        stream = cupy.cuda.get_current_stream()
        stream.synchronize()

    return spg, get_metadata(spec, nfft)


def get_metadata(spec: SpectrogramSpec, nfft):
    enbw = spec.frequency_resolution * equivalent_noise_bandwidth(spec.window, nfft)

    return {
        'window': spec.window,
        'frequency_resolution': spec.frequency_resolution,
        'fractional_overlap': spec.fractional_overlap,
        'noise_bandwidth': enbw,
        'units': f'dBm/{enbw / 1e3:0.0f} kHz',
        'time_bin_averaging': spec.time_bin_averaging,
        'frequency_bin_averaging': spec.frequency_bin_averaging,
    }


@registry.coordinate_factory(
    dtype='float32', attrs={'standard_name': 'Time elapsed', 'units': 's'}
)
@util.lru_cache()
def spectrogram_time(
    capture: specs.Capture, spec: SpectrogramSpec
) -> dict[str, np.ndarray]:
    import pandas as pd

    # validation of these is handled inside iqwaveform
    nfft = round(capture.sample_rate / spec.frequency_resolution)
    hop_size = nfft - round(spec.fractional_overlap * nfft)
    scale = nfft / hop_size
    size = int(scale * (capture.sample_rate * capture.duration / nfft - 1) + 1)

    if spec.time_bin_averaging:
        size = size // spec.time_bin_averaging
        hop_size = hop_size * spec.time_bin_averaging

    return pd.RangeIndex(size) * hop_size / capture.sample_rate


@registry.coordinate_factory(dtype='float64', attrs={'units': 'Hz'})
@util.lru_cache()
def spectrogram_baseband_frequency(
    capture: specs.Capture, spec: SpectrogramSpec
) -> dict[str, np.ndarray]:
    return freq_axis_values(
        capture,
        fres=spec.frequency_resolution,
        navg=spec.frequency_bin_averaging,
        trim_stopband=spec.truncate,
    )


@registry.measurement(
    coord_funcs=[spectrogram_time, spectrogram_baseband_frequency],
    spec_type=SpectrogramSpec,
    dtype='float16',
    attrs={'standard_name': 'PSD', 'long_name': 'Power Spectral Density'},
)
def spectrogram(
    iq: 'iqwaveform.util.Array',
    capture: specs.Capture,
    **kwargs: typing.Unpack[SpectrogramKeywords],
):
    spec = SpectrogramSpec.fromdict(kwargs).validate()
    return evaluate_spectrogram(iq, capture, spec, limit_digits=3, dtype='float16')
