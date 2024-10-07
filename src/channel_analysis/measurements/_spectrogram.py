from __future__ import annotations
import dataclasses
import functools
import typing

import numpy as np

from xarray_dataclasses import AsDataArray, Coordof, Data, Attr
import iqwaveform

from .._api import structs, type_stubs
from ._common import as_registered_channel_analysis
from ._persistence_spectrum import equivalent_noise_bandwidth, BasebandFrequencyCoords, BasebandFrequencyAxis
from ._channel_power_ccdf import make_power_bins, ChannelPowerCoords


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
        capture: structs.Capture, *, frequency_resolution: float, fractional_overlap: float, **_
    ) -> dict[str, np.ndarray]:
        import pandas as pd

        # validation of these is handled inside iqwaveform
        nfft = round(capture.sample_rate / frequency_resolution)
        hop_size = nfft-round(fractional_overlap*nfft)
        scale = nfft/hop_size
        size = int(scale * (capture.sample_rate * capture.duration / nfft - 1) + 1)
        print('times: ', size)
        return pd.RangeIndex(size) * hop_size

### Baseband frequency axis and coordinates
SpectrogramFrequencyAxis = typing.Literal['spectrogram_baseband_frequency']


@dataclasses.dataclass
class SpectrogramFrequencyCoords:
    data: Data[SpectrogramFrequencyAxis, np.float64]
    standard_name: Attr[str] = 'Baseband frequency'
    units: Attr[str] = 'Hz'

    @staticmethod
    @functools.lru_cache
    def factory(
        capture: structs.Capture,
        *,
        frequency_resolution: float,
        fractional_overlap: float = 0,
        frequency_bin_averaging: float = 1,
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

        if truncate and capture.analysis_bandwidth is not None:
            which_freqs = np.abs(freqs) <= capture.analysis_bandwidth / 2
            freqs = freqs[which_freqs]

        freq_step = round(freqs.size/frequency_bin_averaging/2)
        freqs = freqs[freq_step::frequency_bin_averaging]

        print('freqs: ', freqs.size)
        return freqs


@dataclasses.dataclass
class Spectrogram(AsDataArray):
    spectrogram: Data[tuple[SpectrogramTimeAxis, SpectrogramFrequencyAxis], np.float16]
    spectrogram_time: Coordof[SpectrogramTimeCoords]
    spectrogram_baseband_frequency: Coordof[SpectrogramFrequencyCoords]
    standard_name: Attr[str] = 'Spectrogram'


@as_registered_channel_analysis(Spectrogram)
def spectrogram(
    iq: type_stubs.ArrayType,
    capture: structs.Capture,
    *,
    window: typing.Union[str, tuple[str, float]],
    frequency_resolution: float,
    fractional_overlap: float = 0,
    frequency_bin_averaging: int = None,
):
    xp = iqwaveform.util.array_namespace(iq)

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
        bw_args = (-capture.analysis_bandwidth / 2, +capture.analysis_bandwidth / 2)
        ilo, ihi = iqwaveform.fourier._freq_band_edges(
            freqs[0], freqs[-1], freqs.size, *bw_args
        )
        spg = spg[:, ilo:ihi]
        print(nfft, ihi-ilo, spg.shape)

    if frequency_bin_averaging is not None:
        trim = spg.shape[1] % (2 * frequency_bin_averaging)
        if trim > 0:
            spg = spg[:, trim // 2 : -trim // 2 :]
        spg = iqwaveform.fourier.to_blocks(spg, frequency_bin_averaging, axis=1)
        spg = spg.mean(axis=2)

    spg = iqwaveform.powtodB(spg, eps=1e-25, out=spg)

    spg = spg.astype('float16')

    metadata = {
        'window': window,
        'frequency_resolution': frequency_resolution,
        'fractional_overlap': fractional_overlap,
        'noise_bandwidth': enbw,
        'fft_size': nfft,
        'units': f'dBm/{enbw/1e3:0.3f} kHz',
    }

    return spg, metadata
