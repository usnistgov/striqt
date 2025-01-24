from __future__ import annotations
import dataclasses
import functools
import typing

from xarray_dataclasses import AsDataArray, Coordof, Data, Attr

from ..api.registry import register_xarray_measurement
from ._spectrogram import freq_axis_values, compute_spectrogram
from ..api import structs, util

if typing.TYPE_CHECKING:
    import iqwaveform
    import numpy as np
else:
    iqwaveform = util.lazy_import('iqwaveform')
    np = util.lazy_import('numpy')


### Persistence statistics dimension and coordinates
PersistenceStatisticAxis = typing.Literal['frequency_statistic']


@dataclasses.dataclass
class PeriodogramStatisticCoords:
    data: Data[PersistenceStatisticAxis, str]
    standard_name: Attr[str] = 'Frequency statistic'

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
class PowerSpectralDensity(AsDataArray):
    power_time_series: Data[
        tuple[PersistenceStatisticAxis, BasebandFrequencyAxis], np.float32
    ]

    persistence_statistic: Coordof[PeriodogramStatisticCoords]
    baseband_frequency: Coordof[BasebandFrequencyCoords]

    standard_name: Attr[str] = 'Power spectral density'


@register_xarray_measurement(PowerSpectralDensity)
def power_spectral_density(
    iq: 'iqwaveform.util.Array',
    capture: structs.Capture,
    *,
    window: typing.Union[str, tuple[str, float]],
    frequency_resolution: float,
    frequency_statistics: tuple[typing.Union[str, float], ...] = ('mean',),
    fractional_overlap: float = 0,
    frequency_bin_averaging: typing.Optional[float]=None,
    truncate: bool = True,
    axis=1,
):
    """estimate power spectral density using the Welch method.

    A list of statistics can be supplied to evaluate across the frequency axis,
    including 'mean' as applied in the original method.
    """

    spg, metadata = compute_spectrogram(
        iq,
        capture,
        window=window,
        frequency_resolution=frequency_resolution,
        frequency_bin_averaging=frequency_bin_averaging,
        truncate_to_bandwidth=truncate,
        fractional_overlap=fractional_overlap,
    )

    xp = iqwaveform.fourier.array_namespace(iq)
    axis_index = iqwaveform.util.axis_index
    isquantile = iqwaveform.util.find_float_inds(tuple(frequency_statistics))

    newshape = list(spg.shape)
    newshape[axis] = len(frequency_statistics)
    psd = xp.empty(tuple(newshape), dtype='float32')

    quantiles = list(np.asarray(frequency_statistics)[isquantile].astype('float32'))

    out_quantiles = axis_index(psd, isquantile, axis=axis).swapaxes(0, 1)
    out_quantiles[:] = xp.quantile(spg, xp.array(quantiles), axis=axis)

    for i, isquantile in enumerate(isquantile):
        if not isquantile:
            ufunc = iqwaveform.fourier.stat_ufunc_from_shorthand(
                frequency_statistics[i], xp=xp
            )
            axis_index(psd, i, axis=axis)[...] = ufunc(spg, axis=axis)

    # data = iqwaveform.fourier.power_spectral_density(
    #     iq,
    #     fs=capture.sample_rate,
    #     bandwidth=capture.analysis_bandwidth,
    #     window=window,
    #     resolution=frequency_resolution,
    #     fractional_overlap=fractional_overlap,
    #     statistics=frequency_statistics,
    #     truncate=truncate,
    #     dB=True,
    #     axis=1,
    # )

    return psd, metadata
