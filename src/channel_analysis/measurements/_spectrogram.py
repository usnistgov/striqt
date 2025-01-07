from __future__ import annotations
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
def equivalent_noise_bandwidth(window: typing.Union[str, tuple[str, float]], N: int):
    """return the equivalent noise bandwidth (ENBW) of a window, in bins"""
    w = iqwaveform.fourier._get_window(window, N)
    return len(w) * np.sum(w**2) / np.sum(w) ** 2


def truncate_spectrogram_bandwidth(x, nfft, fs, bandwidth, axis=0):
    """trim an array outside of the specified bandwidth on a frequency axis"""
    edges = iqwaveform.fourier._freq_band_edges(
        nfft, 1.0 / fs, cutoff_low=-bandwidth / 2, cutoff_hi=bandwidth / 2
    )
    return iqwaveform.util.axis_slice(x, *edges, axis=axis)


def _binned_mean(x, count, *, axis=0, truncate=True):
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
    # high resolution period
    fres = decimal.Decimal(fs) / nfft
    # if fs_digits is not None:
    #     fs_fixed =
    # fres = round(, fs_digits)/nfft
    span = range(-nfft // 2, -nfft // 2 + nfft)
    if nfft % 2 == 0:
        values = [fres * n for n in span]
    else:
        values = [fres * (n + 1) for n in span]
    return np.array(values, dtype=dtype)


def freq_axis_values(
    capture: structs.RadioCapture, fres: int, navg: int = None, truncate=False
):
    if iqwaveform.isroundmod(capture.sample_rate, fres):
        nfft = round(capture.sample_rate / fres)
    else:
        raise ValueError('sample_rate/resolution must be a counting number')

    # otherwise negligible rounding errors lead to h~eadaches when merging
    # spectra with different sampling parameters. start with long floats
    # to minimize this problem
    # fs = np.longdouble(capture.sample_rate)
    freqs = fftfreq(nfft, capture.sample_rate)
    # freqs = iqwaveform.fourier.fftfreq(nfft, 1.0 / fs, dtype='longdouble')

    if truncate and np.isfinite(capture.analysis_bandwidth):
        # stick with python arithmetic here for numpy/cupy consistency
        freqs = truncate_spectrogram_bandwidth(
            freqs, nfft, capture.sample_rate, capture.analysis_bandwidth, axis=0
        )

    if navg is not None:
        freqs = _binned_mean(freqs, navg)
        freqs -= freqs[freqs.size // 2]

    # only now downconvert. round to a still large number of digits
    return freqs.astype('float64').round(16)


def _do_spectrogram(
    iq: 'iqwaveform.util.Array',
    capture: structs.Capture,
    *,
    window: typing.Union[str, tuple[str, float]],
    frequency_resolution: float,
    fractional_overlap: float = 0,
    frequency_bin_averaging: int = None,
    limit_digits: int = None,
    truncate_to_bandwidth: bool = True,
    dtype='float16',
):
    # TODO: integrate this back into iqwaveform
    if iqwaveform.isroundmod(capture.sample_rate, frequency_resolution):
        nfft = round(capture.sample_rate / frequency_resolution)
    else:
        raise ValueError('sample_rate/resolution must be a counting number')

    xp = iqwaveform.util.array_namespace(iq)

    if isinstance(window, list):
        # lists break lru_cache
        window = tuple(window)

    enbw = frequency_resolution * equivalent_noise_bandwidth(window, nfft)

    if iqwaveform.isroundmod(capture.sample_rate, frequency_resolution):
        noverlap = round(fractional_overlap * nfft)
    else:
        raise ValueError('sample_rate_Hz/resolution must be a counting number')

    _, _, spg = iqwaveform.fourier.spectrogram(
        iq,
        window=window,
        fs=capture.sample_rate,
        nperseg=nfft,
        noverlap=noverlap,
        axis=1,
    )

    # truncate to the analysis bandwidth
    if truncate_to_bandwidth and np.isfinite(capture.analysis_bandwidth):
        # stick with python arithmetic to ensure consistency with axis bounds calculations
        spg = truncate_spectrogram_bandwidth(
            spg, nfft, capture.sample_rate, bandwidth=capture.analysis_bandwidth, axis=2
        )

    if frequency_bin_averaging is not None:
        spg = _binned_mean(spg, frequency_bin_averaging, axis=2)

    # if iqwaveform.util.is_cupy_array(spg):
    #     import cupyx
    #     out = cupyx.empty_pinned(spg.shape, dtype='float32')
    #     out = util.pinned_array_as_cupy(out)
    # else:
    #     out = None

    spg = iqwaveform.powtodB(spg, eps=1e-25, out=spg)

    if limit_digits is not None:
        xp.round(spg, limit_digits, out=spg)

    spg = spg.astype(dtype)

    metadata = {
        'window': window,
        'frequency_resolution': frequency_resolution,
        'fractional_overlap': fractional_overlap,
        'noise_bandwidth': enbw,
        'units': f'dBm/{enbw/1e3:0.0f} kHz',
    }

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
        fractional_overlap: float,
        **_,
    ) -> dict[str, np.ndarray]:
        import pandas as pd

        # validation of these is handled inside iqwaveform
        nfft = round(capture.sample_rate / frequency_resolution)
        hop_size = nfft - round(fractional_overlap * nfft)
        scale = nfft / hop_size
        size = int(scale * (capture.sample_rate * capture.duration / nfft - 1) + 1)
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
        frequency_bin_averaging: float = None,
        truncate: bool = True,
        **_,
    ) -> dict[str, np.ndarray]:
        return freq_axis_values(
            capture,
            fres=frequency_resolution,
            navg=frequency_bin_averaging,
            truncate=truncate,
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
    frequency_bin_averaging: int = None,
):
    return _do_spectrogram(**locals(), limit_digits=3, dtype='float16')
