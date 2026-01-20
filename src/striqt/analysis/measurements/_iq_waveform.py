from __future__ import annotations as __

import typing

from .. import specs

from ..lib import util
from .shared import registry, hint_keywords

if typing.TYPE_CHECKING:
    import numpy as np
    import pandas as pd

    import striqt.waveform as waveform

else:
    waveform = util.lazy_import('striqt.waveform')
    pd = util.lazy_import('pandas')
    np = util.lazy_import('numpy')


def _get_start_stop_index(
    capture: specs.Capture,
    spec: specs.IQWaveform,
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
)
def iq_waveform(iq, capture, **kwargs):
    """package a clipping of the IQ waveform"""

    spec = specs.IQWaveform.from_dict(kwargs)

    metadata = spec.to_dict()

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
