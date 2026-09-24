from __future__ import annotations as __

import typing
from typing import Any, Literal

from .. import specs
from ..lib.util import pd
from . import shared
from .shared import hint_keywords, registry

if typing.TYPE_CHECKING:
    from striqt.waveform.lib.typing import ArrayBackend

    from ..lib.typing import Array, Measurement


# %% iq_waveform
@typing.overload
def _get_start_stop_index(
    capture: specs.Capture, spec: specs.IQWaveform, allow_none: Literal[False]
) -> tuple[int, int]: ...


@typing.overload
def _get_start_stop_index(
    capture: specs.Capture, spec: specs.IQWaveform, allow_none: Literal[True] = ...
) -> tuple[int | None, int | None]: ...


def _get_start_stop_index(
    capture: specs.Capture,
    spec: specs.IQWaveform,
    allow_none: bool = True,
) -> tuple[int | None, int | None]:
    # bounds are clamped into the capture so that the index range agrees with the
    # waveform slice, which numpy clips silently
    size = round(capture.duration * capture.sample_rate)

    if spec.start_time_sec is None:
        if allow_none:
            start = None
        else:
            start = 0
    else:
        start = min(max(int(spec.start_time_sec * capture.sample_rate), 0), size)

    if spec.stop_time_sec is None:
        if allow_none:
            stop = None
        else:
            stop = size
    else:
        stop = min(max(int(spec.stop_time_sec * capture.sample_rate), 0), size)

    return start, stop


def iq_waveform_tolerance(
    capture: specs.Capture,
    spec: specs.IQWaveform,
    *,
    array_backend: ArrayBackend = 'numpy',
    input_error: float = 0.0,
) -> specs.Tolerance:
    """the error budget on the IQ envelope level ``20*log10|iq|`` in dB, which is
    the pass-through of `input_error` since the slice itself is exact"""
    start, stop = _get_start_stop_index(capture, spec, allow_none=False)
    # the slice may be empty, where peak_factor's log is undefined
    size = max(stop - start, 1)
    return shared.level_tolerance(
        amplitude_rms=input_error,
        size=size,
        power_rtol=0.0,
        log_tol={'rtol': 0.0, 'atol': 0.0},
    )


@registry.coordinates(dtype='uint64', attrs={'standard_name': 'Sample Index'})
@specs.helpers.lru_cache_on_converted(specs.Capture)
def iq_index(capture: specs.Capture, spec: specs.IQWaveform) -> typing.Iterable[int]:
    start, stop = _get_start_stop_index(capture, spec, allow_none=False)
    return pd.RangeIndex(start, stop, name=iq_index.__name__)


@hint_keywords(specs.IQWaveform)
@registry.measurement(
    coord_factories=[iq_index],
    spec_type=specs.IQWaveform,
    dtype='complex64',
    attrs={'standard_name': 'IQ waveform', 'units': 'V/√Ω'},
    store_compressed=False,
    tolerance=iq_waveform_tolerance,
)
def iq_waveform(iq: Array, capture: specs.Capture, **kwargs: Any) -> Measurement:
    """package the IQ waveform as a measurement result.

    Args:
    {args}
    """

    spec = specs.IQWaveform.from_dict(kwargs)

    metadata = spec.to_dict()

    start, stop = _get_start_stop_index(capture, spec)

    return iq[:, start:stop], metadata
