from __future__ import annotations
import typing

from .shared import registry
from ..lib import specs, util

if typing.TYPE_CHECKING:
    import iqwaveform
    import pandas as pd
    import numpy as np

else:
    iqwaveform = util.lazy_import('iqwaveform')
    pd = util.lazy_import('pandas')
    np = util.lazy_import('numpy')


class IQWaveformSpec(
    specs.Measurement,
    forbid_unknown_fields=True,
    cache_hash=True,
    kw_only=True,
    frozen=True,
):
    start_time_sec: typing.Optional[float] = None
    stop_time_sec: typing.Optional[float] = None


class IQWaveformKeywords(specs.AnalysisKeywords):
    start_time_sec: typing.NotRequired[typing.Optional[float]]
    stop_time_sec: typing.NotRequired[typing.Optional[float]]


def _get_start_stop_index(
    capture: specs.Capture,
    spec: IQWaveformSpec,
    allow_none=True,
):
    if spec.start_time_sec is None:
        if allow_none:
            start = None
        else:
            start = 0
    else:
        start = int(spec.start_time_sec * capture.sample_rate)

    if spec.stop_time_sec is None:
        if allow_none:
            stop = None
        else:
            stop = int(capture.sample_rate * capture.duration)
    else:
        stop = int(spec.stop_time_sec * capture.sample_rate)

    return start, stop


@registry.coordinates(dtype='uint64', attrs={'standard_name': 'Sample Index'})
@util.lru_cache()
def iq_index(capture: specs.Capture, spec: IQWaveformSpec) -> typing.Iterable[int]:
    start, stop = _get_start_stop_index(capture, spec, allow_none=False)
    return pd.RangeIndex(start, stop, name=iq_index.__name__)


@registry.measurement(
    coord_factories=[iq_index],
    spec_type=IQWaveformSpec,
    dtype='complex64',
    attrs={'standard_name': 'IQ waveform', 'units': 'V/√Ω'},
    store_compressed=False,
)
def iq_waveform(
    iq: 'iqwaveform.util.Array',
    capture: specs.Capture,
    **kwargs: typing.Unpack[IQWaveformKeywords],
) -> 'iqwaveform.util.Array':
    """package a clipping of the IQ waveform"""

    spec = IQWaveformSpec.fromdict(kwargs)

    metadata = spec.todict()

    if spec.start_time_sec is None:
        start = None
    else:
        start = int(spec.start_time_sec * capture.sample_rate)

    if spec.stop_time_sec is None:
        stop = None
    else:
        stop = int(spec.stop_time_sec * capture.sample_rate)

    start, stop = _get_start_stop_index(capture, spec)

    return iq[:, start:stop], metadata
