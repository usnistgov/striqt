from __future__ import annotations
import typing

from .shared import registry
from ..lib import specs, util
from . import _channel_power_time_series

if typing.TYPE_CHECKING:
    import striqt.waveform as iqwaveform
    import numpy as np
else:
    iqwaveform = util.lazy_import('striqt.waveform')
    np = util.lazy_import('numpy')


class ChannelPowerHistogramSpec(
    _channel_power_time_series.ChannelPowerTimeSeriesSpec,
    forbid_unknown_fields=True,
    cache_hash=True,
    kw_only=True,
    frozen=True,
):
    power_low: float
    power_high: float
    power_resolution: float


class ChannelPowerHistogramKeywords(
    _channel_power_time_series.ChannelPowerTimeSeriesKeywords
):
    power_low: float
    power_high: float
    power_resolution: float


@util.lru_cache()
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


@util.lru_cache()
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


@registry.coordinates(
    dtype='float32', attrs={'standard_name': 'Channel power', 'units': 'dBm'}
)
@util.lru_cache()
def channel_power_bin(
    capture: specs.CaptureBase, spec: ChannelPowerHistogramSpec
) -> dict[str, np.ndarray]:
    """returns a dictionary of coordinate values, keyed by axis dimension name"""
    return make_power_bins(spec.power_low, spec.power_high, spec.power_resolution)


@registry.measurement(
    coord_factories=[_channel_power_time_series.power_detector, channel_power_bin],
    depends=[_channel_power_time_series.channel_power_time_series],
    spec_type=ChannelPowerHistogramSpec,
    dtype='float32',
    attrs={'standard_name': 'Fraction of channel power readings'},
)
def channel_power_histogram(
    iq,
    capture: specs.CaptureBase,
    **kwargs: typing.Unpack[ChannelPowerHistogramKeywords],
):
    """evaluate the fraction of channel power readings binned on a uniform grid spacing.

    The outputs correspond to bin centers.
    """

    spec = ChannelPowerHistogramSpec.fromdict(kwargs)

    if typing.TYPE_CHECKING:
        import array_api_compat.numpy as xp
    else:
        xp = iqwaveform.util.array_namespace(iq)

    bin_edges = make_power_histogram_bin_edges(
        power_low=spec.power_low,
        power_high=spec.power_high,
        power_resolution=spec.power_resolution,
        xp=xp,
    )

    pvt_spec = _channel_power_time_series.ChannelPowerTimeSeriesSpec.fromspec(spec)
    power_dB = _channel_power_time_series.evaluate_channel_power_time_series(
        iq,
        capture=capture,
        spec=pvt_spec,
    )

    count_dtype = xp.finfo(iq.dtype).dtype

    data = []
    for i_chan in range(power_dB.shape[0]):
        counts = []
        for i_detector in range(power_dB.shape[1]):
            hist = xp.histogram(power_dB[i_chan, i_detector], bin_edges)[0]
            counts.append(hist)
        counts = xp.asarray(counts, dtype=count_dtype)
        data.append(counts / (xp.sum(counts) / power_dB.shape[1]))

    data = xp.asarray(data, dtype=count_dtype)

    metadata = {
        'detector_period': spec.detector_period,
    }

    return data, metadata
