from __future__ import annotations
import functools
import dataclasses
import typing

import numpy as np
from xarray_dataclasses import AsDataArray, Coordof, Data, Attr
import iqwaveform

from ._common import as_registered_channel_analysis
from .._api import structs


@functools.lru_cache
def make_power_bins(power_low, power_high, power_resolution, xp=np):
    """generate the list of power bins"""
    ret = xp.arange(power_low, power_high, power_resolution)
    if power_high - ret[-1] > power_resolution / 2:
        ret = xp.pad(ret, [[0, 1]], mode='constant', constant_values=power_high).copy()
    return ret


ChannelPowerBinAxis = typing.Literal['channel_power_bin']


@dataclasses.dataclass
class ChannelPowerCoords:
    data: Data[ChannelPowerBinAxis, np.float32]
    standard_name: Attr[str] = 'Channel power'
    units: Attr[str] = 'dBm'

    @staticmethod
    @functools.lru_cache
    def factory(
        capture: structs.Capture,
        *,
        power_low: float,
        power_high: float,
        power_resolution: float,
        **_,
    ) -> dict[str, np.ndarray]:
        """returns a dictionary of coordinate values, keyed by axis dimension name"""
        return make_power_bins(power_low, power_high, power_resolution)


@dataclasses.dataclass
class ChannelPowerCCDF(AsDataArray):
    ccdf: Data[ChannelPowerBinAxis, np.float32]
    channel_power_bin: Coordof[ChannelPowerCoords]
    standard_name: Attr[str] = 'CCDF'
    long_name: Attr[str] = r'Exceedance fraction'


@as_registered_channel_analysis(ChannelPowerCCDF)
def channel_power_ccdf(
    iq,
    capture: structs.Capture,
    *,
    power_low: float,
    power_high: float,
    power_resolution: float,
):
    """analyze the channel, and return an array response"""

    xp = iqwaveform.util.array_namespace(iq)
    dtype = xp.finfo(iq.dtype).dtype

    power = iqwaveform.envtodB(iq)
    bins = make_power_bins(
        power_low=power_low,
        power_high=power_high,
        power_resolution=power_resolution,
        xp=xp,
    )

    return iqwaveform.sample_ccdf(power, bins).astype(dtype)
