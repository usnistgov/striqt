from __future__ import annotations
import dataclasses
import typing

from xarray_dataclasses import AsDataArray, Coordof, Data, Attr, Name

from ..lib.registry import measurement
from . import _spectrogram
from ..lib import specs, util

if typing.TYPE_CHECKING:
    import iqwaveform
    import numpy as np
else:
    iqwaveform = util.lazy_import('iqwaveform')
    np = util.lazy_import('numpy')


### Persistence statistics dimension and coordinates
TimeStatisticAxis = typing.Literal['time_statistic']


@dataclasses.dataclass
class TimeStatisticCoords:
    data: Data[TimeStatisticAxis, str]
    standard_name: Attr[str] = 'Frequency statistic'

    @staticmethod
    @util.lru_cache()
    def factory(
        capture: specs.Capture, spec: PowerSpectralDensityAnalysis
    ) -> np.ndarray:
        time_statistic = [str(s) for s in spec.time_statistic]
        return np.asarray(time_statistic, dtype=object)


### Baseband frequency axis and coordinates
BasebandFrequencyAxis = typing.Literal['baseband_frequency']


@dataclasses.dataclass
class BasebandFrequencyCoords:
    data: Data[BasebandFrequencyAxis, np.float64]
    standard_name: Attr[str] = 'Baseband frequency'
    units: Attr[str] = 'Hz'

    @staticmethod
    @util.lru_cache()
    def factory(
        capture: specs.Capture, spec: PowerSpectralDensityAnalysis
    ) -> dict[str, np.ndarray]:
        return _spectrogram.freq_axis_values(
            capture,
            fres=spec.frequency_resolution,
            trim_stopband=spec.trim_stopband,
            navg=spec.frequency_bin_averaging,
        )


### Dataarray
@dataclasses.dataclass
class PowerSpectralDensity(AsDataArray):
    power_time_series: Data[tuple[TimeStatisticAxis, BasebandFrequencyAxis], np.float32]

    time_statistic: Coordof[TimeStatisticCoords]
    baseband_frequency: Coordof[BasebandFrequencyCoords]

    standard_name: Attr[str] = 'Power spectral density'
    name: Name[str] = 'power_spectral_density'


class PowerSpectralDensityAnalysis(
    _spectrogram.FrequencyAnalysisBase, kw_only=True, frozen=True
):
    time_statistic: tuple[typing.Union[str, float], ...] = (('mean',),)


@measurement(
    PowerSpectralDensity, basis='spectrogram', spec_type=PowerSpectralDensityAnalysis
)
def power_spectral_density(
    iq: 'iqwaveform.util.Array',
    capture: specs.Capture,
    **kwargs: typing.Unpack[PowerSpectralDensityAnalysis],
):
    """estimate power spectral density using the Welch method.

    A list of statistics can be supplied to evaluate across the frequency axis,
    including 'mean' as applied in the original method.
    """

    spec = PowerSpectralDensityAnalysis.fromdict(kwargs)
    spg_spec = _spectrogram.SpectrogramAnalysis.fromspec(spec)

    from iqwaveform.util import axis_index, array_namespace
    from iqwaveform.fourier import stat_ufunc_from_shorthand

    working_dtype = 'float32'

    xp = array_namespace(iq)
    axis = 1

    spg, metadata = _spectrogram.evaluate_spectrogram(
        iq,
        capture,
        spg_spec,
        dB=False,
        dtype=working_dtype,
    )

    findquantile = iqwaveform.util.find_float_inds(tuple(spec.time_statistic))

    newshape = list(spg.shape)
    newshape[axis] = len(spec.time_statistic)
    psd = xp.empty(newshape, dtype=working_dtype)

    # all of the quantiles, evaluated together
    q = [spec.time_statistic[i] for i, flag in enumerate(findquantile) if flag]
    psd[:, findquantile] = (
        xp.quantile(spg, q, axis=axis)
        .swapaxes(0, axis)  # quantile bumps the output result to axis 0
        .astype(working_dtype)  #
    )

    # everything else
    i_isnt_quantile = np.where(~np.array(findquantile))[0]
    for i in i_isnt_quantile:
        ufunc = stat_ufunc_from_shorthand(spec.time_statistic[i], xp=xp)
        axis_index(psd, i, axis=axis)[:] = ufunc(spg, axis=axis)

    psd = iqwaveform.powtodB(psd).astype('float16')

    return psd, metadata
