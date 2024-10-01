from __future__ import annotations
import functools
import dataclasses
import typing

import numpy as np
from xarray_dataclasses import AsDataArray, Coordof, Data, Attr
import iqwaveform

from ._channel_power_time_series import PowerDetectorCoords, PowerDetectorAxis
from ._channel_power_ccdf import (
    ChannelPowerCoords,
    ChannelPowerBinAxis,
    make_power_bins,
)
from ._common import as_registered_channel_analysis
from .._api import structs


@functools.lru_cache
def make_power_histogram_bin_edges(power_low, power_high, power_resolution, xp=np):
    """generate the list of power bins"""

    bin_centers = (
        make_power_bins(
            power_low=power_low,
            power_high=power_high,
            power_resolution=power_resolution,
            xp=xp,
        )
        + power_resolution / 2
    )

    top_edge = xp.array([power_high + power_resolution / 2])
    return xp.concatenate((bin_centers - power_resolution, top_edge))


@dataclasses.dataclass
class ChannelPowerHistogram(AsDataArray):
    histogram: Data[tuple[PowerDetectorAxis, ChannelPowerBinAxis], np.float32]

    # Coordinates: matching the number and order of axes in the data
    power_detector: Coordof[PowerDetectorCoords]
    channel_power_bin: Coordof[ChannelPowerCoords]

    standard_name: Attr[str] = 'Fraction of counts'


@as_registered_channel_analysis(ChannelPowerHistogram)
def channel_power_histogram(
    iq,
    capture: structs.Capture,
    *,
    detector_period: typing.Optional[float] = None,
    power_detectors: tuple[str, ...] = ('rms',),
    power_low: float,
    power_high: float,
    power_resolution: float,
):
    """evaluate the fraction of channel power readings binned on a uniform grid spacing.

    The outputs correspond to bin centers.
    """

    if typing.TYPE_CHECKING:
        import array_api_compat.numpy as xp
    else:
        xp = iqwaveform.util.array_namespace(iq)

    dtype = xp.finfo(iq.dtype).dtype
    bin_edges = make_power_histogram_bin_edges(
        power_low=power_low,
        power_high=power_high,
        power_resolution=power_resolution,
        xp=xp,
    )

    kws = {'iq': iq, 'Ts': 1 / capture.sample_rate, 'Tbin': detector_period}

    data = []
    for detector in power_detectors:
        if detector_period is None:
            power_dB = iqwaveform.envtodB(iq)
        else:
            power = iqwaveform.iq_to_bin_power(kind=detector, **kws).astype('float32')
            power_dB = iqwaveform.powtodB(power, out=power)
        counts, _ = xp.histogram(power_dB, bin_edges)
        data.append(counts.astype(dtype) / xp.sum(counts))

    metadata = {
        'detector_period': detector_period,
    }

    return data, metadata
