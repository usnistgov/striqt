from __future__ import annotations
import dataclasses
import typing

from xarray_dataclasses import AsDataArray, Coordof, Data, Attr, Name

from ..lib.registry import measurement
from . import _spectrogram
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
    @util.lru_cache()
    def factory(
        capture: specs.Capture, spec: SpectrogramHistogramAnalysis
    ) -> dict[str, np.ndarray]:
        """returns a dictionary of coordinate values, keyed by axis dimension name"""
        bins = ChannelPowerCoords.factory(
            capture,
            power_low=spec.power_low,
            power_high=spec.power_high,
            power_resolution=spec.power_resolution,
        )

        if iqwaveform.isroundmod(capture.sample_rate, spec.frequency_resolution):
            # need capture.sample_rate/resolution to give us a counting number
            nfft = round(capture.sample_rate / spec.frequency_resolution)
        else:
            raise ValueError('sample_rate/resolution must be a counting number')

        enbw = spec.frequency_resolution * _spectrogram.equivalent_noise_bandwidth(
            spec.window, nfft
        )

        return bins, {'units': f'dBm/{enbw / 1e3:0.0f} kHz'}


@dataclasses.dataclass
class SpectrogramHistogram(AsDataArray):
    counts: Data[SpectrogramPowerBinAxis, np.float32]
    spectrogram_power_bin: Coordof[SpectrogramPowerBinCoords]
    standard_name: Attr[str] = 'Fraction of counts'
    name: Name[str] = 'cellular_resource_power_histogram'


class SpectrogramHistogramAnalysis(
    _spectrogram.SpectrogramAnalysis, kw_only=True, frozen=True
):
    power_low: float
    power_high: float
    power_resolution: float


@measurement(
    SpectrogramHistogram, basis='spectrogram', spec_type=SpectrogramHistogramAnalysis
)
def spectrogram_histogram(
    iq: 'iqwaveform.util.Array',
    capture: specs.Capture,
    **kwargs: typing.Unpack[SpectrogramHistogramAnalysis],
):
    spec = SpectrogramHistogramAnalysis.fromdict(kwargs)
    spg_spec = _spectrogram.SpectrogramAnalysis.fromspec(spec)

    spg, metadata = _spectrogram.evaluate_spectrogram(
        iq,
        capture,
        spec=spg_spec,
        dtype='float32',
    )

    metadata = dict(metadata)
    metadata.pop('units')

    xp = iqwaveform.util.array_namespace(iq)
    bin_edges = make_power_histogram_bin_edges(
        power_low=spec.power_low,
        power_high=spec.power_high,
        power_resolution=spec.power_resolution,
        xp=xp,
    )

    count_dtype = xp.finfo(iq.dtype).dtype
    counts = xp.asarray(
        [xp.histogram(spg[i].flatten(), bin_edges)[0] for i in range(spg.shape[0])],
        dtype=count_dtype,
    )

    data = counts / xp.sum(counts[0])

    return data, metadata
