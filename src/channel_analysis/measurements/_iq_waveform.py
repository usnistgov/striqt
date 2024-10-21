from __future__ import annotations
import dataclasses
import functools
import typing

import numpy as np
import iqwaveform
from xarray_dataclasses import AsDataArray, Coordof, Data, Attr

from .._api import structs
from ._common import as_registered_channel_analysis

if typing.TYPE_CHECKING:
    import pandas as pd
else:
    pd = iqwaveform.util.lazy_import('pandas')


def _get_start_stop_index(
    capture: structs.Capture,
    start_time_sec: typing.Optional[float] = None,
    stop_time_sec: typing.Optional[float] = None,
    allow_none=True,
):
    if start_time_sec is None:
        if allow_none:
            start = None
        else:
            start = 0
    else:
        start = int(start_time_sec * capture.sample_rate)

    if stop_time_sec is None:
        if allow_none:
            stop = None
        else:
            stop = int(capture.sample_rate * capture.duration)
    else:
        stop = int(stop_time_sec * capture.sample_rate)

    return start, stop


### IQ sample index dimension and coordinates
IQSampleIndexAxis = typing.Literal['iq_index']


@dataclasses.dataclass
class IQSampleIndexCoords:
    data: Data[IQSampleIndexAxis, int]
    standard_name: Attr[str] = 'Sample Index'

    @staticmethod
    @functools.lru_cache
    def factory(
        capture: structs.Capture,
        *,
        start_time_sec: typing.Optional[float] = None,
        stop_time_sec: typing.Optional[float] = None,
    ) -> typing.Iterable[int]:
        start, stop = _get_start_stop_index(
            capture,
            start_time_sec=start_time_sec,
            stop_time_sec=stop_time_sec,
            allow_none=False,
        )
        name = typing.get_args(IQSampleIndexAxis)[0]
        return pd.RangeIndex(start, stop, name=name)


### DataArray definition
@dataclasses.dataclass
class IQWaveform(AsDataArray):
    power_time_series: Data[IQSampleIndexAxis, np.complex64]
    iq_index: Coordof[IQSampleIndexCoords]

    standard_name: Attr[str] = 'IQ waveform'
    units: Attr[str] = 'V/√Ω'


@as_registered_channel_analysis(IQWaveform)
def iq_waveform(
    iq: 'iqwaveform.util.Array',
    capture: structs.Capture,
    *,
    start_time_sec: typing.Optional[float] = None,
    stop_time_sec: typing.Optional[float] = None,
) -> 'iqwaveform.util.Array':
    """package a clipping of the IQ waveform"""

    metadata = {
        'start_time_sec': start_time_sec,
        'stop_time_sec': stop_time_sec,
    }

    if start_time_sec is None:
        start = None
    else:
        start = int(start_time_sec * capture.sample_rate)

    if stop_time_sec is None:
        stop = None
    else:
        stop = int(stop_time_sec * capture.sample_rate)

    start, stop = _get_start_stop_index(
        capture, start_time_sec=start_time_sec, stop_time_sec=stop_time_sec
    )
    data = iq[start:stop]

    return data, metadata
