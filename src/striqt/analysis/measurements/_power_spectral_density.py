from __future__ import annotations
import typing

from . import _spectrogram, shared
from ..lib import register, specs, util

if typing.TYPE_CHECKING:
    import iqwaveform
    import numpy as np
else:
    iqwaveform = util.lazy_import('iqwaveform')
    np = util.lazy_import('numpy')


class PowerSpectralDensitySpec(
    shared.FrequencyAnalysisSpecBase,
    forbid_unknown_fields=True,
    cache_hash=True,
    kw_only=True,
    frozen=True,
):
    time_statistic: tuple[typing.Union[str, float], ...] = (('mean',),)


class PowerSpectralDensityKeywords(shared.FrequencyAnalysisKeywords, total=False):
    time_statistic: tuple[typing.Union[str, float], ...]


@register.coordinate_factory(dtype='str', attrs={'standard_name': 'Time statistic'})
@util.lru_cache()
def time_statistic(
    capture: specs.Capture, spec: PowerSpectralDensitySpec
) -> np.ndarray:
    time_statistic = [str(s) for s in spec.time_statistic]
    return np.asarray(time_statistic, dtype=object)


@register.coordinate_factory(
    dtype='float64', attrs={'standard_name': 'Baseband frequency', 'units': 'Hz'}
)
@util.lru_cache()
def baseband_frequency(
    capture: specs.Capture, spec: PowerSpectralDensitySpec
) -> dict[str, np.ndarray]:
    spg_spec = shared.SpectrogramSpec.fromspec(spec)
    return shared.spectrogram_baseband_frequency(capture, spg_spec)


@register.measurement(
    depends=_spectrogram.spectrogram,
    coord_factories=[time_statistic, baseband_frequency],
    spec_type=PowerSpectralDensitySpec,
    dtype='float32',
    attrs={'standard_name': 'Power spectral density'},
)
def power_spectral_density(
    iq: 'iqwaveform.util.Array',
    capture: specs.Capture,
    **kwargs: typing.Unpack[PowerSpectralDensityKeywords],
):
    """estimate power spectral density using the Welch method.

    A list of statistics can be supplied to evaluate across the frequency axis,
    including 'mean' as applied in the original method.
    """

    spec = PowerSpectralDensitySpec.fromdict(kwargs)
    spg_spec = shared.SpectrogramSpec.fromspec(spec)

    from iqwaveform.util import axis_index, array_namespace
    from iqwaveform.fourier import stat_ufunc_from_shorthand

    working_dtype = 'float32'

    xp = array_namespace(iq)
    axis = 1

    spg, metadata = shared.evaluate_spectrogram(
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
