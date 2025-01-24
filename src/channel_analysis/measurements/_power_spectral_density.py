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
class FrequencyStatisticCoords:
    data: Data[PersistenceStatisticAxis, str]
    standard_name: Attr[str] = 'Frequency statistic'

    @staticmethod
    @functools.lru_cache
    def factory(
        capture: structs.Capture,
        *,
        frequency_statistic: tuple[typing.Union[str, float], ...],
        **_,
    ) -> np.ndarray:
        frequency_statistic = [str(s) for s in frequency_statistic]
        return np.asarray(frequency_statistic, dtype=object)


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

    frequency_statistic: Coordof[FrequencyStatisticCoords]
    baseband_frequency: Coordof[BasebandFrequencyCoords]

    standard_name: Attr[str] = 'Power spectral density'

total = 0

@register_xarray_measurement(PowerSpectralDensity)
def power_spectral_density(
    iq: 'iqwaveform.util.Array',
    capture: structs.Capture,
    *,
    window: typing.Union[str, tuple[str, float]],
    frequency_resolution: float,
    frequency_statistic: tuple[typing.Union[str, float], ...] = ('mean',),
    fractional_overlap: float = 0,
    frequency_bin_averaging: typing.Optional[float] = None,
    truncate: bool = True,
):
    """estimate power spectral density using the Welch method.

    A list of statistics can be supplied to evaluate across the frequency axis,
    including 'mean' as applied in the original method.
    """

    from iqwaveform.util import axis_index, array_namespace
    from iqwaveform.fourier import stat_ufunc_from_shorthand

    xp = array_namespace(iq)
    axis = 1
    dtype = 'float32'

    spg, metadata = compute_spectrogram(
        iq,
        capture,
        window=window,
        frequency_resolution=frequency_resolution,
        frequency_bin_averaging=frequency_bin_averaging,
        truncate_to_bandwidth=truncate,
        fractional_overlap=fractional_overlap,
        dB=False,
        dtype=dtype,
    )

    findquantile = iqwaveform.util.find_float_inds(tuple(frequency_statistic))

    newshape = list(spg.shape)
    newshape[axis] = len(frequency_statistic)
    psd = xp.empty(newshape, dtype=dtype)

    # all of the quantiles, evaluated together
    q = np.array(frequency_statistic)[findquantile].astype(dtype)
    q_out = axis_index(psd, findquantile, axis=axis)
    q_out[:] = (
        xp.quantile(spg, list(q), axis=axis)
        .swapaxes(0, axis)  # quantile bumps the output result to axis 0
        .astype(dtype)  #
    )
    print(q_out.max(axis=-1))
    print(psd.max(axis=-1))

    global total
    if total == 1:
        1//0
    # everything else
    i_isnt_quantile = np.where(~np.array(findquantile))[0]
    for i in i_isnt_quantile:
        ufunc = stat_ufunc_from_shorthand(frequency_statistic[i], xp=xp)
        axis_index(psd, i, axis=axis)[:] = ufunc(spg, axis=axis)

    print(psd.max(axis=-1))

    psd = iqwaveform.powtodB(psd).copy()

    print(psd.max(axis=-1))

    total += 1

    return psd, metadata
