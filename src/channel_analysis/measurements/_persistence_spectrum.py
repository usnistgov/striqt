from __future__ import annotations
import dataclasses
import functools
import typing

from xarray_dataclasses import AsDataArray, Coordof, Data, Attr

from ..api.registry import register_xarray_measurement
from ._spectrogram import freq_axis_values, equivalent_noise_bandwidth
from ..api import structs, util

if typing.TYPE_CHECKING:
    import iqwaveform
    import numpy as np
else:
    iqwaveform = util.lazy_import('iqwaveform')
    np = util.lazy_import('numpy')


### Persistence statistics dimension and coordinates
PersistenceStatisticAxis = typing.Literal['persistence_statistic']


@dataclasses.dataclass
class PersistenceStatisticCoords:
    data: Data[PersistenceStatisticAxis, str]
    standard_name: Attr[str] = 'Persistence statistic'

    @staticmethod
    @functools.lru_cache
    def factory(
        capture: structs.Capture,
        *,
        persistence_statistics: tuple[typing.Union[str, float], ...],
        **_,
    ) -> np.ndarray:
        persistence_statistics = [str(s) for s in persistence_statistics]
        return np.asarray(persistence_statistics, dtype=object)


### Baseband frequency axis and coordinates
BasebandFrequencyAxis = typing.Literal['baseband_frequency']


@dataclasses.dataclass
class BasebandFrequencyCoords:
    data: Data[BasebandFrequencyAxis, np.float64]
    standard_name: Attr[str] = 'Baseband frequency'
    units: Attr[str] = 'Hz'

    @staticmethod
    @functools.lru_cache
    def factory(
        capture: structs.Capture,
        *,
        frequency_resolution: float,
        fractional_overlap: float = 0,
        truncate: bool = True,
        **_,
    ) -> dict[str, np.ndarray]:
        return freq_axis_values(capture, fres=frequency_resolution, truncate=truncate)


### Dataarray
@dataclasses.dataclass
class PersistenceSpectrum(AsDataArray):
    power_time_series: Data[
        tuple[PersistenceStatisticAxis, BasebandFrequencyAxis], np.float32
    ]

    persistence_statistic: Coordof[PersistenceStatisticCoords]
    baseband_frequency: Coordof[BasebandFrequencyCoords]

    standard_name: Attr[str] = 'Power spectral density'


@register_xarray_measurement(PersistenceSpectrum)
def persistence_spectrum(
    iq: 'iqwaveform.util.Array',
    capture: structs.Capture,
    *,
    window: typing.Union[str, tuple[str, float]],
    frequency_resolution: float,
    persistence_statistics: tuple[typing.Union[str, float], ...],
    fractional_overlap: float = 0,
    truncate: bool = True,
):
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
        axis=1,
    )

    metadata = {
        'window': window,
        'frequency_resolution': frequency_resolution,
        'fractional_overlap': fractional_overlap,
        'noise_bandwidth': enbw,
        'units': f'dBm/{round(enbw / 1e3, 1):0.0f} kHz',
    }

    return data, metadata
