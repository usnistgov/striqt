from __future__ import annotations as __

import typing

from .. import specs

from ..lib.util import np
from . import shared
from .shared import registry, hint_keywords

if typing.TYPE_CHECKING:
    from ..lib.typing import Array


@registry.coordinates(
    dtype='float32', attrs={'standard_name': 'Time Elapsed', 'units': 's'}
)
@specs.helpers.lru_cache_on_converted(specs.Capture)
def spectrogram_time(capture: specs.Capture, spec: specs.Spectrogram) -> np.ndarray:
    sizing = shared.validated_spectrogram_sizing(capture, spec)

    scale = sizing.nfft / sizing.hop_size
    size = int(scale * (capture.sample_rate * capture.duration / sizing.nfft - 1) + 1)
    hop_period = sizing.hop_period

    if sizing.time_bin_averaging is not None:
        size = size // sizing.time_bin_averaging
        hop_period = hop_period * sizing.time_bin_averaging

    return np.arange(size) * hop_period


def spectrogram_tolerance(
    capture: specs.Capture, spec: specs.Spectrogram, **kwargs
) -> specs.Tolerance:
    return shared.spectrogram_tolerance(
        capture, spec, dtype='float16', limit_digits=2, **kwargs
    )


@hint_keywords(specs.Spectrogram)
@registry.measurement(
    coord_factories=[spectrogram_time, shared.spectrogram_baseband_frequency],
    spec_type=specs.Spectrogram,
    dtype='float16',
    caches=shared.spectrogram_cache,
    prefer_iq_source='pre_filter',
    attrs={'standard_name': 'PSD', 'long_name': 'Power Spectral Density'},
    validate=shared.validated_spectrogram_sizing,
    tolerance=spectrogram_tolerance,
)
def spectrogram(iq: 'Array', capture: specs.Capture, **kwargs):
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
