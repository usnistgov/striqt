from __future__ import annotations
import dataclasses
import typing

from xarray_dataclasses import AsDataArray, Coordof, Data, Attr

from ..api.registry import register_xarray_measurement
from ._spectrogram import _do_spectrogram
from ._spectrogram_ccdf import SpectrogramPowerBinCoords, SpectrogramPowerBinAxis
from ._channel_power_histogram import make_power_histogram_bin_edges

from ..api import structs, util

if typing.TYPE_CHECKING:
    import iqwaveform
    import numpy as np
else:
    iqwaveform = util.lazy_import('iqwaveform')
    np = util.lazy_import('numpy')


@dataclasses.dataclass
class SpectrogramHistogram(AsDataArray):
    ccdf: Data[SpectrogramPowerBinAxis, np.float32]
    spectrogram_power_bin: Coordof[SpectrogramPowerBinCoords]
    standard_name: Attr[str] = 'Fraction of counts'


@register_xarray_measurement(SpectrogramHistogram)
def spectrogram_histogram(
    iq: 'iqwaveform.util.Array',
    capture: structs.Capture,
    *,
    window: typing.Union[str, tuple[str, float]],
    frequency_resolution: float,
    power_low: float,
    power_high: float,
    power_resolution: float,
    fractional_overlap: float = 0,
    frequency_bin_averaging: int = None,
):
    spg, metadata = _do_spectrogram(
        iq,
        capture,
        window=window,
        frequency_resolution=frequency_resolution,
        fractional_overlap=fractional_overlap,
        frequency_bin_averaging=frequency_bin_averaging,
        dtype='float32',
    )

    metadata = dict(metadata)
    metadata.pop('units')

    xp = iqwaveform.util.array_namespace(iq)
    bin_edges = make_power_histogram_bin_edges(
        power_low=power_low,
        power_high=power_high,
        power_resolution=power_resolution,
        xp=xp,
    )

    count_dtype = xp.finfo(iq.dtype).dtype
    counts, _ = xp.histogram(spg.flatten(), bin_edges)
    data = counts.astype(count_dtype) / xp.sum(counts)

    return data, metadata
