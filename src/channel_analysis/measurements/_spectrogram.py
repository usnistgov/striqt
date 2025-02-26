from __future__ import annotations
import contextlib
import dataclasses
import decimal
import functools
import typing

from xarray_dataclasses import AsDataArray, Coordof, Data, Attr

from ..api.registry import register_xarray_measurement
from ..api import structs, util

if typing.TYPE_CHECKING:
    import iqwaveform
    from scipy import signal
    import numpy as np
else:
    iqwaveform = util.lazy_import('iqwaveform')
    signal = util.lazy_import('scipy.signal')
    np = util.lazy_import('numpy')


@functools.lru_cache
def equivalent_noise_bandwidth(window: typing.Union[str, tuple[str, float]], nfft: int):
    """return the equivalent noise bandwidth (ENBW) of a window, in bins"""
    w = iqwaveform.fourier._get_window(window, nfft)
    return len(w) * np.sum(w**2) / np.sum(w) ** 2


def truncate_spectrogram_bandwidth(x, nfft, fs, bandwidth, axis=0):
    """trim an array outside of the specified bandwidth on a frequency axis"""
    edges = iqwaveform.fourier._freq_band_edges(
        nfft, 1.0 / fs, cutoff_low=-bandwidth / 2, cutoff_hi=bandwidth / 2
    )
    return iqwaveform.util.axis_slice(x, *edges, axis=axis)


def binned_mean(x, count, *, axis=0, truncate=True):
    """reduce an array by averaging into bins on the specified axis"""

    if truncate:
        trim = x.shape[axis] % (count)
        dimsize = (x.shape[axis] // count) * count
        if trim > 0:
            x = iqwaveform.util.axis_slice(x, trim // 2, trim // 2 + dimsize, axis=axis)
    x = iqwaveform.fourier.to_blocks(x, count, axis=axis)
    ret = x.mean(axis=axis + 1)
    return ret


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
    capture: structs.RadioCapture, fres: int, navg: int = None, trim_stopband=False
):
    if iqwaveform.isroundmod(capture.sample_rate, fres):
        nfft = round(capture.sample_rate / fres)
    else:
        raise ValueError('sample_rate/resolution must be a counting number')

    # otherwise negligible rounding errors lead to headaches when merging
    # spectra with different sampling parameters. start here with long floats
    # to minimize this problem
    # fs = np.longdouble(capture.sample_rate)
    freqs = fftfreq(nfft, capture.sample_rate)
    # freqs = iqwaveform.fourier.fftfreq(nfft, 1.0 / fs, dtype='longdouble')

    if trim_stopband and np.isfinite(capture.analysis_bandwidth):
        # stick with python arithmetic here for numpy/cupy consistency
        freqs = truncate_spectrogram_bandwidth(
            freqs, nfft, capture.sample_rate, capture.analysis_bandwidth, axis=0
        )

    if navg is not None:
        freqs = binned_mean(freqs, navg)
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


@contextlib.contextmanager
def cached_spectrograms():
    global _spectrogram_cache
    _spectrogram_cache.enabled = True
    yield
    _spectrogram_cache.clear()
    _spectrogram_cache.enabled = False


@_spectrogram_cache.cached_calls
def _evaluate(
    iq: 'iqwaveform.util.Array',
    capture: structs.Capture,
    *,
    window: typing.Union[str, tuple[str, float]],
    frequency_resolution: float,
    fractional_overlap: float = 0,
    window_scale: float = 1,
    trim_stopband: bool = True,
):
    # TODO: integrate this back into iqwaveform
    if iqwaveform.isroundmod(capture.sample_rate, frequency_resolution):
        nfft = round(capture.sample_rate / frequency_resolution)
    else:
        raise ValueError('sample_rate/resolution must be a counting number')
    enbw = frequency_resolution * equivalent_noise_bandwidth(window, nfft)

    if isinstance(window, list):
        # lists break lru_cache
        window = tuple(window)

    if iqwaveform.isroundmod(capture.sample_rate, frequency_resolution):
        noverlap = round(fractional_overlap * nfft)
    else:
        raise ValueError('sample_rate_Hz/resolution must be a counting number')

    if iqwaveform.isroundmod((1 - window_scale) * nfft, 1):
        nzero = round((1 - window_scale) * nfft)
    else:
        raise ValueError(
            '(1-window_scale) * (sample_rate/frequency_resolution) must be a counting number'
        )

    spg = iqwaveform.fourier.spectrogram(
        iq,
        window=window,
        fs=capture.sample_rate,
        nperseg=nfft,
        noverlap=noverlap,
        nzero=nzero,
        axis=1,
        return_axis_arrays=False,
    )

    # truncate to the analysis bandwidth
    if trim_stopband and np.isfinite(capture.analysis_bandwidth):
        # stick with python arithmetic to ensure consistency with axis bounds calculations
        spg = truncate_spectrogram_bandwidth(
            spg, nfft, capture.sample_rate, bandwidth=capture.analysis_bandwidth, axis=2
        )

    metadata = {
        'window': window,
        'frequency_resolution': frequency_resolution,
        'fractional_overlap': fractional_overlap,
        'noise_bandwidth': enbw,
        'units': f'dBm/{enbw / 1e3:0.0f} kHz',
    }

    return spg, metadata


def compute_spectrogram(
    iq: 'iqwaveform.util.Array',
    capture: structs.Capture,
    *,
    window: typing.Union[str, tuple[str, float]],
    frequency_resolution: float,
    fractional_overlap: float = 0,
    window_scale: float = 1,
    frequency_bin_averaging: typing.Optional[int] = None,
    time_bin_averaging: typing.Optional[int] = None,
    limit_digits: int = None,
    trim_stopband: bool = True,
    dB: bool = True,
    dtype='float16',
):
    eval_kws = dict(
        window=window,
        frequency_resolution=frequency_resolution,
        fractional_overlap=fractional_overlap,
        trim_stopband=trim_stopband,
        window_scale=window_scale,
    )
    spg, metadata = _evaluate(iq=iq, capture=capture, **eval_kws)

    xp = iqwaveform.util.array_namespace(iq)

    if frequency_bin_averaging is not None:
        spg = binned_mean(spg, frequency_bin_averaging, axis=2)

    if time_bin_averaging is not None:
        spg = binned_mean(spg, time_bin_averaging, axis=1)

    if dB:
        spg = iqwaveform.powtodB(spg, eps=1e-25)

    if limit_digits is not None:
        xp.round(spg, limit_digits, spg)

    spg = spg.astype(dtype)

    metadata = dict(
        metadata,
        time_bin_averaging=time_bin_averaging,
        limit_digits=limit_digits,
        frequency_bin_averaging=frequency_bin_averaging,
    )

    return spg, metadata


# Axis and coordinates
SpectrogramTimeAxis = typing.Literal['spectrogram_time']


@dataclasses.dataclass
class SpectrogramTimeCoords:
    data: Data[SpectrogramTimeAxis, np.float32]
    standard_name: Attr[str] = 'Time elapsed'
    units: Attr[str] = 's'

    @staticmethod
    @functools.lru_cache
    def factory(
        capture: structs.Capture,
        *,
        frequency_resolution: float,
        fractional_overlap: float = 0,
        time_bin_averaging: typing.Optional[int] = None,
        **_,
    ) -> dict[str, np.ndarray]:
        import pandas as pd

        # validation of these is handled inside iqwaveform
        nfft = round(capture.sample_rate / frequency_resolution)
        hop_size = nfft - round(fractional_overlap * nfft)
        scale = nfft / hop_size
        size = int(scale * (capture.sample_rate * capture.duration / nfft - 1) + 1)

        if not time_bin_averaging:
            pass
        elif size % time_bin_averaging == 0:
            size = size // time_bin_averaging
            hop_size = hop_size * time_bin_averaging
        else:
            raise ValueError(
                'spectrogram time bin count must be a whole multiple of time_bin_averaging'
            )

        return pd.RangeIndex(size) * hop_size / capture.sample_rate


### Baseband frequency axis and coordinates
SpectrogramBasebandFrequencyAxis = typing.Literal['spectrogram_baseband_frequency']


@dataclasses.dataclass
class SpectrogramBasebandFrequencyCoords:
    data: Data[SpectrogramBasebandFrequencyAxis, np.float64]
    standard_name: Attr[str] = 'Baseband frequency'
    units: Attr[str] = 'Hz'

    @staticmethod
    @functools.lru_cache
    def factory(
        capture: structs.Capture,
        *,
        frequency_resolution: float,
        fractional_overlap: float = 0,
        frequency_bin_averaging: typing.Optional[int] = None,
        truncate: bool = True,
        **_,
    ) -> dict[str, np.ndarray]:
        return freq_axis_values(
            capture,
            fres=frequency_resolution,
            navg=frequency_bin_averaging,
            trim_stopband=truncate,
        )


@dataclasses.dataclass
class Spectrogram(AsDataArray):
    spectrogram: Data[
        tuple[SpectrogramTimeAxis, SpectrogramBasebandFrequencyAxis], np.float16
    ]
    spectrogram_time: Coordof[SpectrogramTimeCoords]
    spectrogram_baseband_frequency: Coordof[SpectrogramBasebandFrequencyCoords]
    standard_name: Attr[str] = 'PSD'
    long_name: Attr[str] = 'Power spectral density'


@register_xarray_measurement(Spectrogram)
def spectrogram(
    iq: 'iqwaveform.util.Array',
    capture: structs.Capture,
    *,
    window: typing.Union[str, tuple[str, float]],
    frequency_resolution: float,
    fractional_overlap: float = 0,
    window_scale: float = 1,
    frequency_bin_averaging: typing.Optional[int] = None,
    time_bin_averaging: typing.Optional[int] = None,
):
    return compute_spectrogram(**locals(), limit_digits=3, dtype='float16')
