from __future__ import annotations as __

import typing

from .. import specs

from ..lib import util
from . import _spectrogram, shared
from .shared import registry, hint_keywords

import striqt.waveform as sw

if typing.TYPE_CHECKING:
    import numpy as np
else:
    np = util.lazy_import('numpy')


@registry.coordinates(dtype='str', attrs={'standard_name': 'Time statistic'})
@specs.helpers.lru_cache_on_converted(specs.Capture)
def time_statistic(
    capture: specs.Capture, spec: specs.PowerSpectralDensity
) -> np.ndarray:
    time_statistic = [str(s) for s in spec.time_statistic]
    return np.asarray(time_statistic, dtype=object)


@registry.coordinates(
    dtype='float64', attrs={'standard_name': 'Baseband frequency', 'units': 'Hz'}
)
@specs.helpers.lru_cache_on_converted(specs.Capture, specs.Spectrogram)
def baseband_frequency(capture: specs.Capture, spec: specs.Spectrogram) -> np.ndarray:
    return shared.spectrogram_baseband_frequency(capture, spec)


def power_spectral_density_tolerance(
    capture: specs.Capture, spec: specs.PowerSpectralDensity, **kwargs
) -> specs.Tolerance:
    spg_spec = specs.Spectrogram.from_spec(spec)
    sizing = shared.validated_spectrogram_sizing(capture, spg_spec)
    n_windows = shared.spectrogram_window_count(capture, sizing)
    return shared.spectrogram_tolerance(
        capture, spg_spec, dtype='float32', statistic_count=n_windows, **kwargs
    )


@hint_keywords(specs.PowerSpectralDensity)
@registry.measurement(
    depends=_spectrogram.spectrogram,
    coord_factories=[time_statistic, baseband_frequency],
    spec_type=specs.PowerSpectralDensity,
    prefer_iq_source='pre_filter',
    dtype='float32',
    attrs={'standard_name': 'Power spectral density'},
    validate=shared.validated_spectrogram_sizing,
    tolerance=power_spectral_density_tolerance,
)
def power_spectral_density(iq, capture, **kwargs):
    """estimate power spectral density using the Welch method.

    A list of statistics can be supplied to evaluate across the frequency axis,
    including 'mean' as applied in the original method.

    Args:
    {args}
    """

    spec = specs.PowerSpectralDensity.from_dict(kwargs)
    spg_spec = specs.Spectrogram.from_spec(spec)

    from striqt.waveform.lib.power_analysis import stat_ufunc_from_shorthand

    working_dtype = 'float32'

    xp = sw.array_namespace(iq)
    axis = 1

    spg, metadata = shared.evaluate_spectrogram(
        iq,
        capture,
        spg_spec,
        dB=False,
        dtype=working_dtype,
    )

    findquantile = sw.util.find_float_inds(tuple(spec.time_statistic))

    newshape = list(spg.shape)
    newshape[axis] = len(spec.time_statistic)
    psd = xp.empty(newshape, dtype=working_dtype)

    # all of the quantiles, evaluated together
    q = [spec.time_statistic[i] for i, flag in enumerate(findquantile) if flag]
    psd[:, findquantile] = (
        xp
        .quantile(spg, q, axis=axis)
        .swapaxes(0, axis)  # quantile bumps the output result to axis 0
        .astype(working_dtype)  #
    )

    # everything else
    i_isnt_quantile = np.where(~np.array(findquantile))[0]
    for i in i_isnt_quantile:
        ufunc = stat_ufunc_from_shorthand(spec.time_statistic[i], xp=xp)
        sw.axis_index(psd, i, axis=axis)[:] = ufunc(spg, axis=axis)

    psd = sw.powtodB(psd)

    return psd, metadata
