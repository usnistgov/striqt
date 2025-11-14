from __future__ import annotations

import typing
import warnings

from ..lib import specs, util
from . import shared
from .shared import registry

if typing.TYPE_CHECKING:
    import numpy as np

    import striqt.waveform as iqwaveform
else:
    iqwaveform = util.lazy_import('striqt.waveform')
    np = util.lazy_import('numpy')

warnings.filterwarnings(
    'ignore', '.*Mean of empty slice.*', category=RuntimeWarning, module=__name__
)


@registry.coordinates(
    dtype='float32', attrs={'standard_name': 'Time Elapsed', 'units': 's'}
)
@util.lru_cache()
def spectrogram_time(
    capture: specs.CaptureBase, spec: shared.SpectrogramSpec
) -> dict[str, np.ndarray]:
    import pandas as pd

    # validation of these is handled inside striqt.waveform
    nfft = round(capture.sample_rate / spec.frequency_resolution)
    hop_size = nfft - round(spec.fractional_overlap * nfft)
    hop_period = hop_period = hop_size / capture.sample_rate
    scale = nfft / hop_size
    size = int(scale * (capture.sample_rate * capture.duration / nfft - 1) + 1)

    if spec.time_aperture is None:
        pass
    elif iqwaveform.isroundmod(spec.time_aperture, hop_period):
        average_bins = round(spec.time_aperture / hop_period)
        size = size // average_bins
        hop_period = hop_period * average_bins
    else:
        raise ValueError(
            'when specified, time_aperture must be a multiple of (1-fractional_overlap)/frequency_resolution'
        )

    return pd.RangeIndex(size) * hop_period


@registry.measurement(
    coord_factories=[spectrogram_time, shared.spectrogram_baseband_frequency],
    spec_type=shared.SpectrogramSpec,
    dtype='float16',
    caches=shared.spectrogram_cache,
    # typed_kwargs=shared.SpectrogramKeywords,
    attrs={'standard_name': 'PSD', 'long_name': 'Power Spectral Density'},
)
def spectrogram(
    iq: 'striqt.waveform.type_stubs.ArrayType',
    capture: specs.CaptureBase,
    **kwargs: typing.Unpack[shared.SpectrogramKeywords],
):
    """Evaluate a spectrogram based on an STFT.

    The analysis parameters are in physical time and frequency units
    based on `capture.sample_rate`. The frequency axis is
    truncated to ±`capture.analysis_bandwidth`.

    The underlying implementation is `striqt.waveform.spectrogram`.
    As a result this accepts `cupy` or `numpy` arrays interchangably and
    implements speed optimizations specific to complex-valued IQ waveforms.

    See also:
        `striqt.waveform.spectrogram`
        `scipy.signal.spectrogram`
    """
    spec = shared.SpectrogramSpec.fromdict(kwargs).validate()
    spg, attrs = shared.evaluate_spectrogram(
        iq, capture, spec, dB=True, limit_digits=2, dtype='float16'
    )

    return spg, attrs
