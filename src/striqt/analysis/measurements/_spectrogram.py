from __future__ import annotations as __

import typing
import warnings

from .. import specs

from ..lib import util
from . import shared
from .shared import registry, hint_keywords

if typing.TYPE_CHECKING:
    import numpy as np

    import striqt.waveform as sw
else:
    sw = util.lazy_import('striqt.waveform')
    np = util.lazy_import('numpy')

warnings.filterwarnings(
    'ignore', '.*Mean of empty slice.*', category=RuntimeWarning, module=__name__
)


@registry.coordinates(
    dtype='float32', attrs={'standard_name': 'Time Elapsed', 'units': 's'}
)
@util.lru_cache()
def spectrogram_time(capture: specs.Capture, spec: specs.Spectrogram) -> np.ndarray:
    sizing = shared.validate_spectrogram_sizing(capture, spec)

    scale = sizing.nfft / sizing.hop_size
    size = int(scale * (capture.sample_rate * capture.duration / sizing.nfft - 1) + 1)
    hop_period = sizing.hop_period

    if sizing.time_bin_averaging is not None:
        size = size // sizing.time_bin_averaging
        hop_period = hop_period * sizing.time_bin_averaging

    return np.arange(size) * hop_period


@hint_keywords(specs.Spectrogram)
@registry.measurement(
    coord_factories=[spectrogram_time, shared.spectrogram_baseband_frequency],
    spec_type=specs.Spectrogram,
    dtype='float16',
    caches=shared.spectrogram_cache,
    prefer_iq_source='pre_filter',
    # typed_kwargs=shared.SpectrogramKeywords,
    attrs={'standard_name': 'PSD', 'long_name': 'Power Spectral Density'},
    validate=shared.validate_spectrogram_sizing,
)
def spectrogram(iq: 'sw.util.Array', capture: specs.Capture, **kwargs):
    """Evaluate a spectrogram based on an STFT.

    The analysis parameters are in physical time and frequency units
    based on `capture.sample_rate`. The frequency axis is
    truncated to ±`capture.analysis_bandwidth`.

    The underlying implementation is `striqt.waveform.spectrogram`.
    As a result this accepts `cupy` or `numpy` arrays interchangably and
    implements speed optimizations specific to complex-valued IQ waveforms.

    Args:
    {args}

    See also:
        `striqt.waveform.spectrogram`
        `scipy.signal.spectrogram`
    """
    spec = specs.Spectrogram.from_dict(kwargs).validate()
    spg, attrs = shared.evaluate_spectrogram(
        iq, capture, spec, dB=True, limit_digits=2, dtype='float16'
    )

    return spg, attrs
