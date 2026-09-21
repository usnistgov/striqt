from __future__ import annotations as __

import fractions
import typing

from .. import specs

from ..lib import util
from ._channel_power_time_series import power_detector
from .shared import registry, hint_keywords
import striqt.waveform as sw

if typing.TYPE_CHECKING:
    import numpy as np
else:
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
    prefer_iq_source='aligned',
    attrs={'standard_name': 'Cyclic channel power', 'units': 'dBm'},
)
def cyclic_channel_power(iq, capture: specs.Capture, **kwargs):
    """Evaluate cyclic statistics of channel power across the cycles in a capture.

    Each power detector bins the capture on `detector_period`; the binned series is
    folded into cycles of `cyclic_period`, and each cyclic statistic reduces across
    the cycles, following D.G. Kuester et al., "Cyclic Analysis of Power in Radio
    Channels". Per capture, the result has
    dimensions ``(power_detector, cyclic_statistic, cyclic_lag)``. The `cyclic_lag`
    coordinate runs from 0 to ``cyclic_period - detector_period`` in steps of
    `detector_period`, so every trace holds ``cyclic_period / detector_period``
    samples regardless of the capture `duration`.

    Signals whose period divides `cyclic_period` (e.g., 10 ms covers TDD cellular frames,
    5 ms WiMAX frames and the 1 ms CBRS test ``bin 1'' pulse repetition interval) resolve
    at fixed lags with little spread between the 'min' and 'max' statistics, so the
    uplink and downlink levels of a TDD network can be read from disjoint lag
    windows, while occupancy with an incommensurate period smears across all lags.

    Statistics are evaluated in linear power and converted to dBm afterwards. The
    capture `duration` must be a whole number of cycles, `cyclic_period` a whole
    number of detector periods, and `detector_period` a whole number of samples at
    `sample_rate`; write it as a fraction (``1/28000`` s is one OFDM symbol at
    30 kHz subcarrier spacing) to keep that exact.

    Args:
    {args}
    """
    spec = specs.CyclicChannelPower.from_dict(kwargs)

    xp = sw.array_namespace(iq)

    nested_ret = sw.iq_to_cyclic_power(
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

    return sw.powtodB(x).astype('float32')
