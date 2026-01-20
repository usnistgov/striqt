from __future__ import annotations as __

import typing

from .. import specs

from ..lib import util
from . import _channel_power_histogram, _spectrogram, shared
from .shared import registry, hint_keywords

if typing.TYPE_CHECKING:
    import numpy as np

    import striqt.waveform as waveform
else:
    waveform = util.lazy_import('striqt.waveform')
    np = util.lazy_import('numpy')


@registry.coordinates(
    dtype='float32', attrs={'standard_name': 'Spectrogram bin power', 'units': 'dBm'}
)
@util.lru_cache()
def spectrogram_power_bin(
    capture: specs.Capture, spec: specs.SpectrogramHistogram
) -> dict[str, np.ndarray]:
    """returns a dictionary of coordinate values, keyed by axis dimension name"""
    bins = _channel_power_histogram.make_power_bins(
        power_low=spec.power_low,
        power_high=spec.power_high,
        power_resolution=spec.power_resolution,
    )

    if waveform.isroundmod(capture.sample_rate, spec.frequency_resolution):
        # need capture.sample_rate/resolution to give us a counting number
        nfft = round(capture.sample_rate / spec.frequency_resolution)
    else:
        raise ValueError('sample_rate/resolution must be a counting number')

    if spec.integration_bandwidth is None:
        enbw = spec.frequency_resolution
    else:
        enbw = spec.integration_bandwidth

    return bins, {'units': f'dBm/{enbw / 1e3:0.0f} kHz'}


@hint_keywords(specs.SpectrogramHistogram)
@registry.measurement(
    depends=_spectrogram.spectrogram,
    coord_factories=[spectrogram_power_bin],
    spec_type=specs.SpectrogramHistogram,
    dtype='float32',
    attrs={'standard_name': 'Fraction of counts'},
)
def spectrogram_histogram(
    iq: 'waveform.util.ArrayType', capture: specs.Capture, **kwargs
):
    spec = specs.SpectrogramHistogram.from_dict(kwargs)
    spg_spec = specs.Spectrogram.from_spec(spec)

    spg, metadata = shared.evaluate_spectrogram(
        iq,
        capture,
        spec=spg_spec,
        dtype='float32',
    )

    metadata = dict(metadata)
    metadata.pop('units')

    xp = waveform.util.array_namespace(iq)
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
