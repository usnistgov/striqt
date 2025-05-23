from __future__ import annotations
import typing

from . import _channel_power_histogram, _spectrogram, _spectrogram_histogram

from ..lib import register, specs, util

if typing.TYPE_CHECKING:
    import iqwaveform
    import numpy as np
else:
    iqwaveform = util.lazy_import('iqwaveform')
    np = util.lazy_import('numpy')


class SpectrogramHistogramRatioSpec(
    _spectrogram_histogram.SpectrogramHistogramSpec,
    forbid_unknown_fields=True,
    cache_hash=True,
    kw_only=True,
    frozen=True,
):
    pass


class SpectrogramHistogramRatioKeywords(
    _spectrogram_histogram.SpectrogramHistogramKeywords
):
    pass


@register.coordinate_factory(
    dtype='float32',
    attrs={'standard_name': 'Spectrogram cross-channel power ratio', 'units': 'dB'},
)
@util.lru_cache()
def spectrogram_ratio_power_bin(
    capture: specs.Capture, spec: SpectrogramHistogramRatioSpec
) -> dict[str, np.ndarray]:
    """returns a dictionary of coordinate values, keyed by axis dimension name"""

    abs_spec = _spectrogram_histogram.SpectrogramHistogramSpec.fromspec(spec)
    bins, attrs = _spectrogram_histogram.spectrogram_power_bin(capture, abs_spec)
    attrs['units'] = attrs['units'].replace('dBm', 'dB')
    return bins, attrs


@register.measurement(
    depends=_spectrogram.spectrogram,
    coord_factories=[spectrogram_ratio_power_bin],
    spec_type=SpectrogramHistogramRatioSpec,
    dtype='float32',
    attrs={'standard_name': 'Fraction of counts'},
)
def spectrogram_ratio_histogram(
    iq: 'iqwaveform.util.Array',
    capture: specs.Capture,
    **kwargs: typing.Unpack[SpectrogramHistogramRatioKeywords],
):
    spec = SpectrogramHistogramRatioSpec.fromdict(kwargs)
    spg_spec = _spectrogram.SpectrogramSpec.fromspec(spec)

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
