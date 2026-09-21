from __future__ import annotations as __

import typing

from .. import specs

from ..lib import util
from .shared import registry, hint_keywords

if typing.TYPE_CHECKING:
    import pandas as pd
else:
    pd = util.lazy_import('pandas')


def _get_start_stop_index(
    capture: specs.Capture,
    spec: specs.IQWaveform,
    allow_none=True,
):
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
    """package the IQ waveform as a measurement result.

    Args:
    {args}
    """

    spec = specs.IQWaveform.from_dict(kwargs)

    metadata = spec.to_dict()

    start, stop = _get_start_stop_index(capture, spec)

    return iq[:, start:stop], metadata
