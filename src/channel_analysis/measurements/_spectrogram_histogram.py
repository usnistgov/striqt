from __future__ import annotations
import dataclasses
import functools
import typing

from xarray_dataclasses import AsDataArray, Coordof, Data, Attr, Name

from ..lib.registry import register_xarray_measurement
from ._spectrogram import compute_spectrogram, equivalent_noise_bandwidth
from ._channel_power_histogram import ChannelPowerCoords, make_power_histogram_bin_edges

from ..lib import specs, util

if typing.TYPE_CHECKING:
    import iqwaveform
    import numpy as np
else:
    iqwaveform = util.lazy_import('iqwaveform')
    np = util.lazy_import('numpy')


# Axis and coordinates
SpectrogramPowerBinAxis = typing.Literal['spectrogram_power_bin']


@dataclasses.dataclass
class SpectrogramPowerBinCoords:
    data: Data[SpectrogramPowerBinAxis, np.float32]
    standard_name: Attr[str] = 'Spectrogram bin power'
    units: Attr[str] = 'dBm'

    @staticmethod
    @functools.lru_cache
    def factory(
        capture: specs.Capture,
        *,
        window: typing.Union[str, tuple[str, float]],
        frequency_resolution: float,
        power_low: float,
        power_high: float,
        power_resolution: float,
        **_,
    ) -> dict[str, np.ndarray]:
        """returns a dictionary of coordinate values, keyed by axis dimension name"""
        bins = ChannelPowerCoords.factory(
            capture,
            power_low=power_low,
            power_high=power_high,
            power_resolution=power_resolution,
        )

        if iqwaveform.isroundmod(capture.sample_rate, frequency_resolution):
            # need capture.sample_rate/resolution to give us a counting number
            nfft = round(capture.sample_rate / frequency_resolution)
        else:
            raise ValueError('sample_rate/resolution must be a counting number')

        if isinstance(window, list):
            # lists break lru_cache
            window = tuple(window)

        enbw = frequency_resolution * equivalent_noise_bandwidth(window, nfft)

        return bins, {'units': f'dBm/{enbw / 1e3:0.0f} kHz'}


@dataclasses.dataclass
class SpectrogramHistogram(AsDataArray):
    counts: Data[SpectrogramPowerBinAxis, np.float32]
    spectrogram_power_bin: Coordof[SpectrogramPowerBinCoords]
    standard_name: Attr[str] = 'Fraction of counts'
    name: Name[str] = 'cellular_resource_power_histogram'


@register_xarray_measurement(SpectrogramHistogram)
def spectrogram_histogram(
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
