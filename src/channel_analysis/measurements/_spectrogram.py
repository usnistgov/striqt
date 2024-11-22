from __future__ import annotations
import dataclasses
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


def _centered_trim(x, freqs, bandwidth, axis=0):
    """trim an array outside of the specified bandwidth on a frequency axis"""
    axis_slice = signal._arraytools.axis_slice

    edges = iqwaveform.fourier._freq_band_edges(
        freqs[0],
        freqs[1] - freqs[0],
        freqs.size,
        cutoff_low=-bandwidth / 2,
        cutoff_hi=+bandwidth / 2,
    )

    return axis_slice(x, *edges, axis=axis), freqs[edges[0] : edges[1]]


def _binned_mean(x, count, *, axis=0, truncate=True):
    """reduce an array by averaging into bins on the specified axis"""
    axis_slice = signal._arraytools.axis_slice

    if truncate:
        trim = x.shape[axis] % (2 * count)
        if trim > 0:
            x = axis_slice(x, trim // 2, -trim // 2, axis=axis)
    x = iqwaveform.fourier.to_blocks(x, count, axis=axis)
    return x.mean(axis=axis + 1)


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
        nfft = round(capture.sample_rate / frequency_resolution)

        freqs, _ = iqwaveform.fourier._get_stft_axes(
            fs=capture.sample_rate,
            nfft=nfft,
            time_size=1,
            overlap_frac=fractional_overlap,
            xp=np,
        )

        if capture.analysis_bandwidth is not None and truncate:
            freqs, _ = _centered_trim(freqs, freqs, capture.analysis_bandwidth, axis=0)

        if frequency_bin_averaging is not None:
            freqs = _binned_mean(freqs, frequency_bin_averaging, axis=0)

        return freqs


@dataclasses.dataclass
class Spectrogram(AsDataArray):
    spectrogram: Data[
        tuple[SpectrogramTimeAxis, SpectrogramBasebandFrequencyAxis], np.float16
    ]
    spectrogram_time: Coordof[SpectrogramTimeCoords]
    spectrogram_baseband_frequency: Coordof[SpectrogramBasebandFrequencyCoords]
    standard_name: Attr[str] = 'PSD'
    long_name: Attr[str] = 'Power spectral density'


def _do_spectrogram(
    iq: 'iqwaveform.util.Array',
    capture: structs.Capture,
    *,
    window: typing.Union[str, tuple[str, float]],
    frequency_resolution: float,
    fractional_overlap: float = 0,
    frequency_bin_averaging: int = None,
    dtype='float16',
):
    # TODO: integrate this back into iqwaveform
    if iqwaveform.isroundmod(capture.sample_rate, frequency_resolution):
        nfft = round(capture.sample_rate / frequency_resolution)
    else:
        raise ValueError('sample_rate/resolution must be a counting number')

    if isinstance(window, list):
        # lists break lru_cache
        window = tuple(window)

    enbw = frequency_resolution * equivalent_noise_bandwidth(window, nfft)

    if iqwaveform.isroundmod(capture.sample_rate, frequency_resolution):
        nfft = round(capture.sample_rate / frequency_resolution)
        noverlap = round(fractional_overlap * nfft)
    else:
        # need sample_rate_Hz/resolution to give us a counting number
        raise ValueError('sample_rate_Hz/resolution must be a counting number')

    freqs, _, spg = iqwaveform.fourier.spectrogram(
        iq,
        window=window,
        fs=capture.sample_rate,
        nperseg=nfft,
        noverlap=noverlap,
        axis=0,
    )

    # truncate to the analysis bandwidth
    if capture.analysis_bandwidth is not None:
        spg, freqs = _centered_trim(
            spg, freqs, bandwidth=capture.analysis_bandwidth, axis=1
        )

    if frequency_bin_averaging is not None:
        spg = _binned_mean(spg, frequency_bin_averaging, axis=1)

    spg = iqwaveform.powtodB(spg, eps=1e-25, out=spg)
    spg = spg.astype(dtype)

    metadata = {
        'window': window,
        'frequency_resolution': frequency_resolution,
        'fractional_overlap': fractional_overlap,
        # 'noise_bandwidth': enbw,
        'units': f'dBm/{enbw/1e3:0.0f} kHz',
    }

    return spg, metadata


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
    return _do_spectrogram(**locals(), dtype='float16')
