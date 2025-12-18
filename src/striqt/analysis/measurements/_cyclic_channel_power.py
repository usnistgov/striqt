from __future__ import annotations as __

import fractions
import typing

from .. import specs

from ..lib import util
from ._channel_power_time_series import power_detector
from .shared import registry, hint_keywords

if typing.TYPE_CHECKING:
    import numpy as np

    import striqt.waveform as iqwaveform
else:
    iqwaveform = util.lazy_import('striqt.waveform')
    np = util.lazy_import('numpy')


@registry.coordinates(dtype=object, attrs={'standard_name': 'Cyclic statistic'})
@util.lru_cache()
def cyclic_statistic(capture: specs.Capture, spec: specs.CyclicChannelPower):
    return list(spec.cyclic_statistics)


@registry.coordinates(
    dtype='float32', attrs={'standard_name': 'Cyclic lag', 'units': 's'}
)
@util.lru_cache()
def cyclic_lag(capture: specs.Capture, spec: specs.CyclicChannelPower):
    lag_count = int(np.rint(spec.cyclic_period / spec.detector_period))

    return np.arange(lag_count) * float(spec.detector_period)


@hint_keywords(specs.CyclicChannelPower)
@registry.measurement(
    coord_factories=[power_detector, cyclic_statistic, cyclic_lag],
    spec_type=specs.CyclicChannelPower,
    dtype='float32',
    attrs={'standard_name': 'Cyclic channel power', 'units': 'dBm'},
)
def cyclic_channel_power(iq, capture: specs.Capture, **kwargs):
    spec = specs.CyclicChannelPower.from_dict(kwargs)

    xp = iqwaveform.util.array_namespace(iq)

    nested_ret = iqwaveform.iq_to_cyclic_power(
        iq,
        1 / capture.sample_rate,
        cyclic_period=spec.cyclic_period,
        detector_period=float(spec.detector_period),
        detectors=spec.power_detectors,
        cycle_stats=spec.cyclic_statistics,
        axis=1,
    )

    # pull arrays from the returned nested dict and combine into one ndarray
    x = xp.array([list(d.values()) for d in nested_ret.values()])

    # move the capture axis to the front
    x = xp.moveaxis(x, -2, 0)

    return iqwaveform.powtodB(x).astype('float32')
