from __future__ import annotations
import typing
import warnings

from . import shared
from ..lib import register, specs, util

if typing.TYPE_CHECKING:
    import iqwaveform
    import numpy as np
else:
    iqwaveform = util.lazy_import('iqwaveform')
    np = util.lazy_import('numpy')

warnings.filterwarnings(
    'ignore', '.*Mean of empty slice.*', category=RuntimeWarning, module=__name__
)


@register.coordinate_factory(
    dtype='float32', attrs={'standard_name': 'Time Elapsed', 'units': 's'}
)
@util.lru_cache()
def spectrogram_time(
    capture: specs.Capture, spec: shared.SpectrogramSpec
) -> dict[str, np.ndarray]:
    import pandas as pd

    # validation of these is handled inside iqwaveform
    nfft = round(capture.sample_rate / spec.frequency_resolution)
    hop_size = nfft - round(spec.fractional_overlap * nfft)
    scale = nfft / hop_size
    size = int(scale * (capture.sample_rate * capture.duration / nfft - 1) + 1)

    if spec.time_bin_averaging:
        size = size // spec.time_bin_averaging
        hop_size = hop_size * spec.time_bin_averaging

    return pd.RangeIndex(size) * hop_size / capture.sample_rate


@register.measurement(
    coord_factories=[spectrogram_time, shared.spectrogram_baseband_frequency],
    spec_type=shared.SpectrogramSpec,
    dtype='float16',
    caches=shared.spectrogram_cache,
    attrs={'standard_name': 'PSD', 'long_name': 'Power Spectral Density'},
)
def spectrogram(
    iq: 'iqwaveform.util.Array',
    capture: specs.Capture,
    **kwargs: typing.Unpack[shared.SpectrogramKeywords],
):
    spec = shared.SpectrogramSpec.fromdict(kwargs).validate()
    ret = shared.evaluate_spectrogram(
        iq, capture, spec, dB=True, limit_digits=2, dtype='float16'
    )
    return ret
