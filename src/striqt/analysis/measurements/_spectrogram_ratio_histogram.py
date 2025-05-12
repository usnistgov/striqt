from __future__ import annotations
import dataclasses
import functools
import typing

from xarray_dataclasses import AsDataArray, Coordof, Data, Attr

from ..lib.registry import measurement
from ._spectrogram import compute_spectrogram
from ._spectrogram_histogram import SpectrogramPowerBinCoords
from ._channel_power_histogram import make_power_histogram_bin_edges

from ..lib import specs, util

if typing.TYPE_CHECKING:
    import iqwaveform
    import numpy as np
else:
    iqwaveform = util.lazy_import('iqwaveform')
    np = util.lazy_import('numpy')


SpectrogramPowerRatioBinAxis = typing.Literal['spectrogram_power_ratio_bin']


@dataclasses.dataclass
class SpectrogramPowerRatioBinCoords:
    data: Data[SpectrogramPowerRatioBinAxis, np.float32]
    standard_name: Attr[str] = 'Spectrogram power ratio'
    units: Attr[str] = 'dB'

    @staticmethod
    @functools.lru_cache
    def factory(capture: specs.Capture, **kws):
        bins, attrs = SpectrogramPowerBinCoords.factory(capture, **kws)
        attrs['units'] = attrs['units'].replace('dBm', 'dB')
        return bins, attrs


@dataclasses.dataclass
class SpectrogramRatioHistogram(AsDataArray):
    counts: Data[SpectrogramPowerRatioBinAxis, np.float32]
    spectrogram_power_ratio_bin: Coordof[SpectrogramPowerRatioBinCoords]
    standard_name: Attr[str] = 'Fraction of counts'


@measurement(SpectrogramRatioHistogram)
def spectrogram_ratio_histogram(
    iq: 'iqwaveform.util.Array',
    capture: specs.Capture,
    *,
    window: typing.Union[str, tuple[str, float]],
    frequency_resolution: float,
    power_low: float,
    power_high: float,
    power_resolution: float,
    fractional_overlap: float = 0,
    window_fill: float = 1,
    frequency_bin_averaging: int = None,
    time_bin_averaging: int = None,
):
    spg, metadata = compute_spectrogram(
        iq,
        capture,
        window=window,
        frequency_resolution=frequency_resolution,
        fractional_overlap=fractional_overlap,
        window_fill=window_fill,
        frequency_bin_averaging=frequency_bin_averaging,
        time_bin_averaging=time_bin_averaging,
        dtype='float32',
    )

    if spg.shape[0] != 2:
        print(spg.shape)
        raise ValueError(
            'ratio histograms are only supported for 2-channel measurements'
        )

    spg[0], spg[1] = spg[0] - spg[1], spg[1] - spg[0]

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
    counts = xp.asarray(
        [xp.histogram(spg[i].flatten(), bin_edges)[0] for i in range(spg.shape[0])],
        dtype=count_dtype,
    )

    data = counts / xp.sum(counts[0])

    return data, metadata
