from __future__ import annotations
import functools
import dataclasses
import typing

from xarray_dataclasses import AsDataArray, Coordof, Data, Attr

from ._channel_power_time_series import (
    PowerDetectorCoords,
    PowerDetectorAxis,
    channel_power_time_series,
)
from ..lib.registry import measurement
from ..lib import specs, util

if typing.TYPE_CHECKING:
    import iqwaveform
    import numpy as np
else:
    iqwaveform = util.lazy_import('iqwaveform')
    np = util.lazy_import('numpy')


@functools.lru_cache
def make_power_bins(power_low, power_high, power_resolution, xp=np):
    """generate the list of power bins"""
    ret = xp.arange(power_low, power_high, power_resolution)
    if power_high - ret[-1] > power_resolution / 2:
        ret = xp.pad(ret, [[0, 1]], mode='constant', constant_values=power_high).copy()

    # catch-all bins for counts outside of the specified range
    bottom_inf = xp.array([float('-inf')])
    top_inf = xp.array([float('inf')])
    ret = xp.concatenate((bottom_inf, ret, top_inf))
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
        capture: specs.Capture,
        *,
        power_low: float,
        power_high: float,
        power_resolution: float,
        **_,
    ) -> dict[str, np.ndarray]:
        """returns a dictionary of coordinate values, keyed by axis dimension name"""
        return make_power_bins(power_low, power_high, power_resolution)


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

    top_edge = xp.array(
        [float(power_high + power_resolution / 2), float(bin_centers[-1])]
    )
    return xp.concatenate((bin_centers[:-1] - power_resolution, top_edge))


@dataclasses.dataclass
class ChannelPowerHistogram(AsDataArray):
    histogram: Data[tuple[PowerDetectorAxis, ChannelPowerBinAxis], np.float32]

    # Coordinates: matching the number and order of axes in the data
    power_detector: Coordof[PowerDetectorCoords]
    channel_power_bin: Coordof[ChannelPowerCoords]

    standard_name: Attr[str] = 'Fraction of counts'


@measurement(ChannelPowerHistogram, basis='channel_power')
def channel_power_histogram(
    iq,
    capture: specs.Capture,
    *,
    power_low: float,
    power_high: float,
    power_resolution: float,
    detector_period: typing.Optional[float] = None,
    power_detectors: tuple[str, ...] = ('rms',),
):
    """evaluate the fraction of channel power readings binned on a uniform grid spacing.

    The outputs correspond to bin centers.
    """

    if typing.TYPE_CHECKING:
        import array_api_compat.numpy as xp
    else:
        xp = iqwaveform.util.array_namespace(iq)

    bin_edges = make_power_histogram_bin_edges(
        power_low=power_low,
        power_high=power_high,
        power_resolution=power_resolution,
        xp=xp,
    )

    power_dB, _ = channel_power_time_series(
        iq,
        capture,
        power_detectors=power_detectors,
        detector_period=detector_period,
        as_xarray=False,
    )

    count_dtype = xp.finfo(iq.dtype).dtype

    data = []
    for i_chan in range(power_dB.shape[0]):
        counts = []
        for i_detector in range(power_dB.shape[1]):
            hist = xp.histogram(power_dB[i_chan, i_detector], bin_edges)[0]
            counts.append(hist)
        counts = xp.asarray(counts, dtype=count_dtype)
        data.append(counts / xp.sum(counts))

    data = xp.asarray(data, dtype=count_dtype)

    metadata = {
        'detector_period': detector_period,
    }

    return data, metadata
