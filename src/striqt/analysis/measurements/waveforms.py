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
    # numpy clips an out-of-range slice silently. Clamping keeps the index coordinate
    # the same length as the slice.
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
    """bound the error (in dB) of the IQ sample magnitudes.

    The slice adds no error. The bound is `input_error` alone.

    Args:
        input_error: relative rms error of the IQ amplitude (linear, unitless)
        array_backend: has no effect on this bound

    Returns:
        the `specs.Tolerance` in dB over the sliced samples
    """
    start, stop = _get_start_stop_index(capture, spec, allow_none=False)
    # peak_factor takes a log, which is undefined for an empty slice
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
    attrs={'standard_name': 'IQ waveform', 'units': '√mW'},
    store_compressed=False,
    tolerance=iq_waveform_tolerance,
)
def iq_waveform(x: Array, capture: specs.Capture, **kwargs: Any) -> Measurement:
    """return the IQ samples between `start_time_sec` and `stop_time_sec`.

    The input is the IQ after the `analysis_bandwidth` filter and trigger alignment.
    Per capture, the output has the dimension ``(iq_index,)`` of sample indexes into
    the capture. The values are complex64 in √mW. The squared magnitude ``|iq|**2``
    is power in mW. A None time bound selects the start or the end of the capture.
    A bound outside the capture is clamped to the capture. The output has no
    `noise_bandwidth` attr. The noise bandwidth of each capture is the
    `analysis_bandwidth` coordinate, or `sample_rate` where `analysis_bandwidth` is
    infinite.

    Args:
    {args}
    """

    spec = specs.IQWaveform.from_dict(kwargs)

    # no noise_bandwidth attr: the bandwidth is the analysis_bandwidth (or sample_rate)
    # of the capture, which varies between the captures of one sweep, and a variable
    # attr is written once per store
    metadata = spec.to_dict()

    start, stop = _get_start_stop_index(capture, spec)

    return x[:, start:stop], metadata
