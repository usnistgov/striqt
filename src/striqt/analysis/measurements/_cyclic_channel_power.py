from __future__ import annotations
import fractions
import typing

from ._channel_power_time_series import power_detector

from ..lib import register, specs, util

if typing.TYPE_CHECKING:
    import iqwaveform
    import numpy as np
else:
    iqwaveform = util.lazy_import('iqwaveform')
    np = util.lazy_import('numpy')


class CyclicChannelPowerSpec(
    specs.Measurement,
    forbid_unknown_fields=True,
    cache_hash=True,
    kw_only=True,
    frozen=True,
):
    cyclic_period: float
    detector_period: fractions.Fraction
    power_detectors: tuple[str, ...] = ('rms', 'peak')
    cyclic_statistics: tuple[typing.Union[str, float], ...] = ('min', 'mean', 'max')


class CyclicChannelPowerKeywords(specs.AnalysisKeywords):
    cyclic_period: float
    detector_period: fractions.Fraction
    power_detectors: typing.Optional[tuple[str, ...]]
    cyclic_statistics: typing.Optional[tuple[typing.Union[str, float], ...]]


@register.coordinate_factory(dtype=object, attrs={'standard_name': 'Cyclic statistic'})
@util.lru_cache()
def cyclic_statistic(capture: specs.Capture, spec: CyclicChannelPowerSpec):
    return list(spec.cyclic_statistics)


@register.coordinate_factory(
    dtype='float32', attrs={'standard_name': 'Cyclic lag', 'units': 's'}
)
@util.lru_cache()
def cyclic_lag(capture: specs.Capture, spec: CyclicChannelPowerSpec):
    lag_count = int(np.rint(spec.cyclic_period / spec.detector_period))

    return np.arange(lag_count) * spec.detector_period


@register.measurement(
    coord_factories=[power_detector, cyclic_statistic, cyclic_lag],
    spec_type=CyclicChannelPowerSpec,
    dtype='float32',
    attrs={'standard_name': 'Cyclic channel power', 'units': 'dBm'},
)
def cyclic_channel_power(
    iq, capture: specs.Capture, **kwargs: typing.Unpack[CyclicChannelPowerSpec]
):
    spec = CyclicChannelPowerSpec.fromdict(kwargs)

    xp = iqwaveform.util.array_namespace(iq)

    nested_ret = iqwaveform.iq_to_cyclic_power(
        iq,
        1 / capture.sample_rate,
        cyclic_period=spec.cyclic_period,
        detector_period=spec.detector_period,
        detectors=spec.power_detectors,
        cycle_stats=spec.cyclic_statistics,
        axis=1,
    )

    # pull arrays from the returned nested dict and combine into one ndarray
    x = xp.array([list(d.values()) for d in nested_ret.values()])

    # move the capture axis to the front
    x = xp.moveaxis(x, -2, 0)

    return iqwaveform.powtodB(x).astype('float32')
