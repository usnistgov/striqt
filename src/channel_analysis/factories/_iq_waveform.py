from __future__ import annotations
from typing import Iterable, Literal, Optional, get_args
from functools import lru_cache
from dataclasses import dataclass

import numpy as np
from xarray_dataclasses import AsDataArray, Coordof, Data, Attr
import iqwaveform

from ..dataarrays import expose_in_yaml, ChannelAnalysisResult, select_parameter_kws
from ..structs import Capture
from ..type_stubs import ArrayType

pd = iqwaveform.util.lazy_import('pandas')


def get_start_stop_index(
    capture: Capture,
    start_time_sec: Optional[float] = None,
    stop_time_sec: Optional[float] = None,
    allow_none=True
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


@expose_in_yaml
def iq_waveform(
    iq: ArrayType,
    capture: Capture,
    *,
    start_time_sec: Optional[float] = None,
    stop_time_sec: Optional[float] = None,
) -> ChannelAnalysisResult:
    """package a clipping of the IQ waveform"""

    params = select_parameter_kws(locals())

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

    start, stop = get_start_stop_index(
        capture, start_time_sec=start_time_sec, stop_time_sec=stop_time_sec
    )
    data = iq[start:stop].copy()

    return ChannelAnalysisResult(
        IQWaveform, data, capture, parameters=params, attrs=metadata
    )


### IQ sample index dimension and coordinates
IQSampleIndexAxis = Literal['iq_index']


@lru_cache
def iq_sample_coord_factory(
    capture: Capture,
    *,
    start_time_sec: Optional[float] = None,
    stop_time_sec: Optional[float] = None,
) -> Iterable[int]:
    start, stop = get_start_stop_index(
        capture, start_time_sec=start_time_sec, stop_time_sec=stop_time_sec, allow_none=False
    )

    return pd.RangeIndex(start, stop, name=get_args(IQSampleIndexAxis)[0])


@dataclass
class IQSampleIndexCoords:
    data: Data[IQSampleIndexAxis, int]
    standard_name: Attr[str] = 'IQ waveform'
    units: Attr[str] = 'V/√Ω'

    factory: callable = iq_sample_coord_factory


### DataArray definition
@dataclass
class IQWaveform(AsDataArray):
    power_time_series: Data[IQSampleIndexAxis, np.complex64]
    iq_index: Coordof[IQSampleIndexCoords]

    standard_name: Attr[str] = 'IQ waveform'
    units: Attr[str] = 'V/√Ω'
