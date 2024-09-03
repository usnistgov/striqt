from __future__ import annotations
from typing import Literal, Any, Union, get_args
from functools import lru_cache

from dataclasses import dataclass
import numpy as np

from xarray_dataclasses import AsDataArray, Coordof, Data, Attr, Name
from iqwaveform import powtodB, iq_to_bin_power, fourier
from iqwaveform.power_analysis import isroundmod

from ..dataarrays import expose_in_yaml, ChannelAnalysisResult, select_parameter_kws
from ..structs import Capture
from .. import type_stubs


@lru_cache
def equivalent_noise_bandwidth(window: str | tuple[str, float], N: int):
    """return the equivalent noise bandwidth (ENBW) of a window, in bins"""
    w = fourier._get_window(window, N)
    return len(w) * np.sum(w**2) / np.sum(w) ** 2


@expose_in_yaml
def persistence_spectrum(
    x: type_stubs.ArrayType,
    capture: Capture,
    *,
    window: Any,
    frequency_resolution: float,
    persistence_statistics: tuple[Union[str, float], ...],
    fractional_overlap: float = 0,
    truncate: bool = True,
) -> ChannelAnalysisResult:
    params = select_parameter_kws(locals())

    # TODO: support other persistence statistics, such as mean
    if isroundmod(capture.sample_rate, frequency_resolution):
        # need capture.sample_rate/resolution to give us a counting number
        nfft = round(capture.sample_rate / frequency_resolution)
    else:
        raise ValueError('sample_rate/resolution must be a counting number')

    if isinstance(window, list):
        # lists break lru_cache
        window = tuple(window)

    enbw = frequency_resolution * equivalent_noise_bandwidth(window, nfft)

    metadata = {
        'window': window,
        'frequency_resolution': frequency_resolution,
        'fractional_overlap': fractional_overlap,
        'noise_bandwidth': enbw,
        'fft_size': nfft,
        'units': f'dBm/{enbw/1e3:0.3f} kHz',
    }

    data = fourier.persistence_spectrum(
        x,
        fs=capture.sample_rate,
        bandwidth=capture.analysis_bandwidth,
        window=window,
        resolution=frequency_resolution,
        fractional_overlap=fractional_overlap,
        statistics=persistence_statistics,
        truncate=truncate,
        dB=True,
    )

    ChannelAnalysisResult(
        PersistenceSpectrum, data, capture, parameters=params, attrs=metadata
    )

#     kws = {'iq': iq, 'Ts': 1 / capture.sample_rate, 'Tbin': detector_period}

#     data = [
#         powtodB(iq_to_bin_power(kind=detector, **kws).astype('float32'))
#         for detector in power_detectors
#     ]

#     metadata = {
#         'detector_period': detector_period,
#         'standard_name': 'Channel power',
#         'units': f'dBm/{(capture.analysis_bandwidth or capture.sample_rate)/1e6} MHz',
#     }

#     return ChannelAnalysisResult(
#         ChannelPowerTimeSeries, data, capture, parameters=params, attrs=metadata
#     )


# @lru_cache
# def _persistence_spectrum_coords(
#     capture: Capture,
#     nfft: int,
#     stat_names: tuple[str],
#     overlap_frac: float = 0,
#     truncate: bool = True,
#     xp=np,
# ):
#     freqs, _ = fourier._get_stft_axes(
#         fs=capture.sample_rate,
#         nfft=nfft,
#         time_size=1,
#         overlap_frac=overlap_frac,
#         xp=np,
#     )

#     if truncate and capture.analysis_bandwidth is not None:
#         which_freqs = np.abs(freqs) <= capture.analysis_bandwidth / 2
#         freqs = freqs[which_freqs]

#     freqs = xr.DataArray(
#         freqs,
#         dims='baseband_frequency',
#         attrs={'standard_name': 'Baseband frequency', 'units': 'Hz'},
#     )

#     stats = xr.DataArray(
#         [str(n) for n in stat_names],
#         dims='persistence_statistic',
#         attrs={'standard_name': 'Persistence statistic'},
#     ).astype('object')

#     return xr.Coordinates({stats.dims[0]: stats, freqs.dims[0]: freqs})


### Persistence statistics dimension and coordinates
PersistenceStatisticAxis = Literal['persistence_statistic']


@lru_cache
def persistence_statistic_coord_factory(
    capture: Capture, *,
    persistence_statistics: tuple[Union[str, float], ...],
    **kws
) -> np.ndarray:

    return np.ndarray(persistence_statistics, dtype=object)


@dataclass
class PersistenceStatisticCoords:
    data: Data[{PersistenceStatisticAxis}, object]
    standard_name: Attr[str] = 'Persistence statistic'

    factory: callable = persistence_statistic_coord_factory


### Baseband frequency axis and coordinates
BasebandFrequencyAxis = Literal['baseband_frequency']

@lru_cache
def baseband_frequency_coord_factory(
    capture: Capture, *,
    frequency_resolution: float,
    fractional_overlap: float = 0,
    truncate: bool = True,   
    **kws
) -> dict[str, np.ndarray]:

    nfft = round(capture.sample_rate / frequency_resolution)

    freqs, _ = fourier._get_stft_axes(
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
    data: Data[BasebandFrequencyAxis, object]
    standard_name: Attr[str] = 'Baseband frequency'
    units: Attr[str] = 'Hz'

    factory: callable = baseband_frequency_coord_factory


### Dataarray
@dataclass
class PersistenceSpectrum(AsDataArray):
    power_time_series: Data[tuple[PersistenceStatisticAxis, BasebandFrequencyAxis], np.float32]

    persistence_statistic: Coordof[PersistenceStatisticCoords]
    baseband_frequency: Coordof[PersistenceStatisticCoords]

    standard_name: Attr[str] = 'Power spectral density'
    # units set dynamically on return
