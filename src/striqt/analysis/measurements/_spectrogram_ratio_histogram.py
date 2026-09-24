from __future__ import annotations as __

import typing

from .. import specs

from ..lib.util import np
from . import _spectrogram, _spectrogram_histogram, power, shared
from .shared import registry, hint_keywords
import striqt.waveform as sw

if typing.TYPE_CHECKING:
    from ..lib.typing import Array


@registry.coordinates(
    dtype='float32',
    attrs={'standard_name': 'Spectrogram cross-channel power ratio', 'units': 'dB'},
)
@specs.helpers.lru_cache_on_converted(specs.Capture)
def spectrogram_ratio_power_bin(
    capture: specs.Capture, spec: specs.SpectrogramHistogramRatio
) -> tuple[np.ndarray, dict[str, typing.Any]]:
    """returns a dictionary of coordinate values, keyed by axis dimension name"""

    bins, attrs = _spectrogram_histogram.spectrogram_power_bin(capture, spec)

    # `attrs` is held in a cache that `spectrogram_histogram` shares whenever the two
    # measurements agree on the spectrogram and power bin fields, so relabeling it in
    # place would relabel that measurement's absolute powers as ratios
    return bins, attrs | {'units': attrs['units'].replace('dBm', 'dB')}


@hint_keywords(specs.SpectrogramHistogramRatio)
@registry.measurement(
    depends=_spectrogram.spectrogram,
    coord_factories=[spectrogram_ratio_power_bin],
    spec_type=specs.SpectrogramHistogramRatio,
    dtype='float32',
    prefer_iq_source='pre_filter',
    attrs={'standard_name': 'Fraction of counts'},
    validate=shared.validated_spectrogram_sizing,
)
def spectrogram_ratio_histogram(iq: 'Array', capture: specs.Capture, **kwargs):
    """Compute the ratio of spectrogram readings across two channels, and return its
    its histogram.

    Args:
    {args}
    """
    spec = specs.SpectrogramHistogramRatio.from_dict(kwargs)
    spg_spec = specs.Spectrogram.from_spec(spec)

    spg, metadata = shared.evaluate_spectrogram(
        iq,
        capture,
        spg_spec,
        dtype='float32',
    )

    if spg.shape[0] != 2:
        raise ValueError(
            'ratio histograms are only supported for 2-channel measurements'
        )

    spg[0], spg[1] = spg[0] - spg[1], spg[1] - spg[0]

    metadata = dict(metadata)
    metadata.pop('units')

    xp = sw.array_namespace(iq)
    bin_edges = power.make_power_histogram_bin_edges(
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
