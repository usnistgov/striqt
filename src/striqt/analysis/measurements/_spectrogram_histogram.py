from __future__ import annotations
import typing

from . import _spectrogram, _channel_power_histogram
from ..lib import registry, specs, util

if typing.TYPE_CHECKING:
    import iqwaveform
    import numpy as np
else:
    iqwaveform = util.lazy_import('iqwaveform')
    np = util.lazy_import('numpy')


class SpectrogramHistogramSpec(
    _spectrogram.SpectrogramSpec,
    forbid_unknown_fields=True,
    cache_hash=True,
    kw_only=True,
    frozen=True,
):
    power_low: float
    power_high: float
    power_resolution: float


class SpectrogramHistogramKeywords(_spectrogram.SpectrogramKeywords):
    power_low: float
    power_high: float
    power_resolution: float


@registry.coordinate_factory(
    dtype='float32', attrs={'standard_name': 'Spectrogram bin power', 'units': 'dBm'}
)
@util.lru_cache()
def spectrogram_power_bin(
    capture: specs.Capture, spec: SpectrogramHistogramSpec
) -> dict[str, np.ndarray]:
    """returns a dictionary of coordinate values, keyed by axis dimension name"""
    bins = _channel_power_histogram.make_power_bins(
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


@registry.measurement(
    depends=_spectrogram.spectrogram,
    coord_funcs=[spectrogram_power_bin],
    spec_type=SpectrogramHistogramSpec,
    dtype='float32',
    attrs={'standard_name': 'Fraction of counts'},
)
def spectrogram_histogram(
    iq: 'iqwaveform.util.Array',
    capture: specs.Capture,
    **kwargs: typing.Unpack[SpectrogramHistogramKeywords],
):
    spec = SpectrogramHistogramSpec.fromdict(kwargs)
    spg_spec = _spectrogram.SpectrogramSpec.fromspec(spec)

    spg, metadata = _spectrogram.evaluate_spectrogram(
        iq,
        capture,
        spec=spg_spec,
        dtype='float32',
    )

    metadata = dict(metadata)
    metadata.pop('units')

    xp = iqwaveform.util.array_namespace(iq)
    bin_edges = _channel_power_histogram.make_power_histogram_bin_edges(
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
