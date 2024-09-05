from __future__ import annotations
import typing
from functools import lru_cache

from dataclasses import dataclass
import numpy as np

from xarray_dataclasses import AsDataArray, Coordof, Data, Attr
import iqwaveform

from ..dataarrays import expose_in_yaml, ChannelAnalysisResult, select_parameter_kws
from ..structs import Capture


@expose_in_yaml
def channel_power_ccdf(
    iq,
    capture: Capture,
    *,
    power_low: float,
    power_high: float,
    power_resolution: float,
) -> ChannelAnalysisResult:
    """analyze the channel, and return an array response"""

    params = select_parameter_kws(locals())
    xp = iqwaveform.util.array_namespace(iq)
    dtype = xp.finfo(iq.dtype).dtype

    power = iqwaveform.envtodB(iq)
    bins = make_power_bins(**params, xp=xp)
    result = iqwaveform.sample_ccdf(power, bins).astype(dtype)

    return ChannelAnalysisResult(
        datacls=ChannelPowerCCDF,
        data=result,
        capture=capture,
        parameters=params,
    )


@lru_cache
def make_power_bins(power_low, power_high, power_resolution, xp=np):
    """generate the list of power bins"""
    ret = xp.arange(power_low, power_high, power_resolution)
    if power_high - ret[-1] > power_resolution / 2:
        ret = xp.pad(ret, [[0, 1]], mode='constant', constant_values=power_high).copy()
    return ret


ChannelPowerBinAxis = typing.Literal['channel_power_bin']


@lru_cache
def power_bin_coord_factory(
    capture: Capture,
    *,
    power_low: float,
    power_high: float,
    power_resolution: float,
    **_
) -> dict[str, np.ndarray]:
    """returns a dictionary of coordinate values, keyed by axis dimension name"""
    return make_power_bins(power_low, power_high, power_resolution)


@dataclass
class ChannelPowerCoords:
    data: Data[ChannelPowerBinAxis, np.float32]
    standard_name: Attr[str] = 'Channel power'
    units: Attr[str] = 'dBm'

    factory: callable = power_bin_coord_factory


@dataclass
class ChannelPowerCCDF(AsDataArray):
    ccdf: Data[ChannelPowerBinAxis, np.float32]
    channel_power_bin: Coordof[ChannelPowerCoords]
    standard_name: Attr[str] = 'CCDF'
