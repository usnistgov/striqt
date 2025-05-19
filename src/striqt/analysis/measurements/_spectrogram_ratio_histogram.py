from __future__ import annotations
import dataclasses
import typing

from xarray_dataclasses import AsDataArray, Coordof, Data, Attr

from ..lib.registry import measurement
from . import _spectrogram, _spectrogram_histogram
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
    @util.lru_cache()
    def factory(
        capture: specs.Capture, spec: SpectrogramHistogramRatioAnalysis
    ) -> dict[str, np.ndarray]:
        """returns a dictionary of coordinate values, keyed by axis dimension name"""

        abs_spec = _spectrogram_histogram.SpectrogramHistogramAnalysis.fromspec(spec)
        bins, attrs = _spectrogram_histogram.SpectrogramPowerBinCoords.factory(
            capture, abs_spec
        )
        attrs['units'] = attrs['units'].replace('dBm', 'dB')
        return bins, attrs


@dataclasses.dataclass
class SpectrogramRatioHistogram(AsDataArray):
    counts: Data[SpectrogramPowerRatioBinAxis, np.float32]
    spectrogram_power_ratio_bin: Coordof[SpectrogramPowerRatioBinCoords]
    standard_name: Attr[str] = 'Fraction of counts'


class SpectrogramHistogramRatioAnalysis(
    _spectrogram_histogram.SpectrogramHistogramAnalysis, kw_only=True, frozen=True
):
    pass


@measurement(
    SpectrogramRatioHistogram,
    basis='spectrogram',
    spec_type=SpectrogramHistogramRatioAnalysis,
)
def spectrogram_ratio_histogram(
    iq: 'iqwaveform.util.Array',
    capture: specs.Capture,
    **kwargs: typing.Unpack[SpectrogramHistogramRatioAnalysis],
):
    spec = SpectrogramHistogramRatioAnalysis.fromdict(kwargs)
    spg_spec = _spectrogram.SpectrogramAnalysis.fromspec(spec)

    spg, metadata = _spectrogram.evaluate_spectrogram(
        iq,
        capture,
        spg_spec,
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
