from __future__ import annotations
import functools
import dataclasses
import typing

import numpy as np
from xarray_dataclasses import AsDataArray, Coordof, Data, Attr
import iqwaveform

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

    return xp.concatenate(
        [bin_centers - power_resolution, [power_high + power_resolution / 2]]
    )


@dataclasses.dataclass
class ChannelPowerHistogram(AsDataArray):
    ccdf: Data[ChannelPowerBinAxis, np.float32]
    channel_power_bin: Coordof[ChannelPowerCoords]
    standard_name: Attr[str] = 'Fraction of counts'


@as_registered_channel_analysis(ChannelPowerHistogram)
def channel_power_histogram(
    iq,
    capture: structs.Capture,
    *,
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

    power = iqwaveform.envtodB(iq)
    bin_edges = make_power_histogram_bin_edges(
        power_low=power_low,
        power_high=power_high,
        power_resolution=power_resolution,
        xp=xp,
    )

    counts, _ = xp.histogram(power, bin_edges)
    return counts.astype(dtype) / xp.sum(counts)
