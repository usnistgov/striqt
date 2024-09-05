from __future__ import annotations
from functools import lru_cache
from dataclasses import dataclass

import numpy as np
from xarray_dataclasses import AsDataArray, Coordof, Data, Attr, Name
import iqwaveform

from ..dataarrays import expose_in_yaml, ChannelAnalysisResult, select_parameter_kws
from ..structs import Capture
from .. import type_stubs
import typing



@lru_cache
def equivalent_noise_bandwidth(window: typing.Union[str, tuple[str, float]], N: int):
    """return the equivalent noise bandwidth (ENBW) of a window, in bins"""
    w = iqwaveform.fourier._get_window(window, N)
    return len(w) * np.sum(w**2) / np.sum(w) ** 2


@expose_in_yaml
def persistence_spectrum(
    iq: type_stubs.ArrayType,
    capture: Capture,
    *,
    window: typing.Union[str, tuple[str, float]],
    frequency_resolution: float,
    persistence_statistics: tuple[typing.Union[str,float], ...],
    fractional_overlap: float = 0,
    truncate: bool = True,
) -> ChannelAnalysisResult:
    params = select_parameter_kws(locals())

    # TODO: support other persistence statistics, such as mean
    if iqwaveform.isroundmod(capture.sample_rate, frequency_resolution):
        # need capture.sample_rate/resolution to give us a counting number
        nfft = round(capture.sample_rate / frequency_resolution)
    else:
        raise ValueError('sample_rate/resolution must be a counting number')

    if isinstance(window, list):
        # lists break lru_cache
        window = tuple(window)

    enbw = frequency_resolution * equivalent_noise_bandwidth(window, nfft)

    data = iqwaveform.fourier.persistence_spectrum(
        iq,
        fs=capture.sample_rate,
        bandwidth=capture.analysis_bandwidth,
        window=window,
        resolution=frequency_resolution,
        fractional_overlap=fractional_overlap,
        statistics=persistence_statistics,
        truncate=truncate,
        dB=True,
    )

    metadata = {
        'window': window,
        'frequency_resolution': frequency_resolution,
        'fractional_overlap': fractional_overlap,
        'noise_bandwidth': enbw,
        'fft_size': nfft,
        'units': f'dBm/{round(enbw/1e3, 1):0.0f} kHz',
    }

    return ChannelAnalysisResult(
        PersistenceSpectrum, data, capture, parameters=params, attrs=metadata
    )


### Persistence statistics dimension and coordinates
PersistenceStatisticAxis = typing.Literal['persistence_statistic']


@lru_cache
def persistence_statistic_coord_factory(
    capture: Capture, *, persistence_statistics: tuple[typing.Union[str, float], ...], **kws
) -> np.ndarray:
    return np.asarray(persistence_statistics, dtype=object)


@dataclass
class PersistenceStatisticCoords:
    data: Data[PersistenceStatisticAxis, str]
    standard_name: Attr[str] = 'Persistence statistic'

    factory: callable = persistence_statistic_coord_factory


### Baseband frequency axis and coordinates
BasebandFrequencyAxis = typing.Literal['baseband_frequency']


@lru_cache
def baseband_frequency_coord_factory(
    capture: Capture,
    *,
    frequency_resolution: float,
    fractional_overlap: float = 0,
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

    return freqs


@dataclass
class BasebandFrequencyCoords:
    data: Data[BasebandFrequencyAxis, np.float64]
    standard_name: Attr[str] = 'Baseband frequency'
    units: Attr[str] = 'Hz'

    factory: callable = baseband_frequency_coord_factory


### Dataarray
@dataclass
class PersistenceSpectrum(AsDataArray):
    power_time_series: Data[
        tuple[PersistenceStatisticAxis, BasebandFrequencyAxis], np.float32
    ]

    persistence_statistic: Coordof[PersistenceStatisticCoords]
    baseband_frequency: Coordof[BasebandFrequencyCoords]

    standard_name: Attr[str] = 'Power spectral density'