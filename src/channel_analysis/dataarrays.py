"""wrap lower-level iqwaveform DSP calls to accept physical inputs and return xarray.DataArray"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Any
from xarray_dataclasses import AsDataArray, Coordof, Data, Attr
import numpy as np
import functools
import xarray_dataclasses

import inspect
from xarray_dataclasses.dataarray import OptionedClass, TDataArray, DataClass, PInit
from xarray_dataclasses.datamodel import AnyEntry
from typing import Type, overload
import xarray as xr


from collections import UserDict
from functools import lru_cache
import typing

import msgspec
import labbench as lb
from array_api_compat import is_cupy_array, is_numpy_array, is_torch_array
from frozendict import frozendict

import iqwaveform
from iqwaveform import powtodB

from . import structs
from . import (
    type_stubs,
)  # import type_stubs.DataArrayType, type_stubs.DatasetType, type_stubs.ArrayType, CoordinatesType

if typing.TYPE_CHECKING:
    import numpy as np
    import scipy
    import pandas as pd
    import xarray as xr
else:
    np = lb.util.lazy_import('numpy')
    scipy = lb.util.lazy_import('scipy')
    pd = lb.util.lazy_import('pandas')
    xr = lb.util.lazy_import('xarray')


def _entry_stub(entry: AnyEntry):
    return np.empty(len(entry.dims) * (1,), dtype=entry.dtype)


@overload
def shaped(
    cls: Type[OptionedClass[PInit, TDataArray]],
) -> TDataArray: ...


@overload
@classmethod
def shaped(
    cls: Type[DataClass[PInit]],
) -> xr.DataArray: ...


@functools.lru_cache
def dataarray_stub(cls: Any) -> Any:
    """return an empty array of type `cls`"""

    entries = get_data_model(cls).entries
    params = inspect.signature(cls.new).parameters
    kws = {
        name: _entry_stub(entries[name])
        for name, param in params.items()
        if param.default is param.empty
    }

    stub = cls.new(**kws)
    slices = dict.fromkeys(stub.dims, slice(None, 0))
    return stub.isel(slices)


@functools.lru_cache
def get_data_model(dataclass: Any):
    return xarray_dataclasses.datamodel.DataModel.from_dataclass(dataclass)


def channel_dataarray(
    cls, data: np.ndarray, capture, parameters: dict[str, Any]
) -> xr.DataArray:
    """build an `xarray.DataArray` from an ndarray, capture information, and channel analysis keyword arguments"""
    template = dataarray_stub(cls)

    data = np.asarray(data)

    # to bypass initialization overhead, grow from the empty template
    da = template.pad({dim: [0, data.shape[i]] for i, dim in enumerate(template.dims)})
    da.values[:] = data

    for entry in get_data_model(cls).coords:
        arr = entry.base.factory(capture, **parameters)
        da[entry.name].indexes[entry.dims[0]].values[:] = arr

    return da


expose_in_yaml = structs.KeywordConfigRegistry(structs.ChannelAnalysis)


IQ_WAVEFORM_INDEX_NAME = 'iq_index'


@dataclass
class ChannelAnalysisResult(UserDict):
    """represents the return result from a channel analysis function.

    This includes a method to convert to `xarray.DataArray`, which is
    delayed to leave the GPU time to initiate the evaluation of multiple
    analyses before we materialize them on the CPU.
    """

    datacls: type
    data: typing.Union[type_stubs.ArrayType, dict]
    capture: structs.RadioCapture
    parameters: dict[str, Any]
    attrs: list[str] = frozendict()

    def to_xarray(self) -> type_stubs.DataArrayType:
        return channel_dataarray(
            cls=self.datacls,
            data=_to_maybe_nested_numpy(self.data),
            capture=self.capture,
            parameters=self.parameters,
        ).assign_attrs(self.attrs)


def _to_maybe_nested_numpy(obj: tuple | list | dict | type_stubs.ArrayType):
    """convert an array, or a container of arrays, into a numpy array (or container of numpy arrays)"""

    if isinstance(obj, (tuple, list)):
        return [_to_maybe_nested_numpy(item) for item in obj]
    elif isinstance(obj, dict):
        return [_to_maybe_nested_numpy(item) for item in obj.values()]
    elif is_torch_array(obj):
        return obj.cpu()
    elif is_cupy_array(obj):
        return obj.get()
    elif is_numpy_array(obj):
        return obj
    else:
        raise TypeError(f'obj type {type(obj)} is unrecognized')


def select_parameter_kws(locals_: dict, omit=('iq', 'capture', 'out')) -> dict:
    """return the analysis parameters from the locals() evaluated at the beginning of analysis function"""

    return {k: v for k, v in locals_.items() if k not in omit}


@expose_in_yaml
def iq_waveform(
    iq,
    capture: structs.Capture,
    *,
    start_time_sec: typing.Optional[float] = None,
    stop_time_sec: typing.Optional[float] = None,
) -> ChannelAnalysisResult:
    """package a clipping of the IQ waveform"""

    metadata = {
        'standard_name': 'IQ waveform',
        'units': 'V',
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

    coords = xr.Coordinates(
        {
            IQ_WAVEFORM_INDEX_NAME: pd.RangeIndex(
                start, stop, name=IQ_WAVEFORM_INDEX_NAME
            )
        }
    )

    return ChannelAnalysisResult(
        data=iq[start:stop].copy(),
        name='iq_waveform',
        coords=coords,
        attrs=metadata,
    )


@lru_cache(8)
def _generate_iir_lpf(
    capture: structs.Capture,
    *,
    passband_ripple: float | int,
    stopband_attenuation: float | int,
    transition_bandwidth: float | int,
) -> np.ndarray:
    """
    Generate an elliptic IIR low pass filter for complex-valued waveforms.

    Args:
        passband_ripple:
            Maximum amplitude ripple in the passband of the frequency-response below unity gain (dB)
        stopband_attenuation:
            Maximum amplitude ripple in the passband of the frequency-response below unity gain (dB).
        transition_bandwidth:
            Passband-to-stopband transition width (Hz)

    Returns:
        Second-order sections (sos) representation of the IIR filter.
    """

    order, wn = scipy.signal.ellipord(
        capture.analysis_bandwidth / 2,
        capture.analysis_bandwidth / 2 + transition_bandwidth,
        passband_ripple,
        stopband_attenuation,
        False,
        capture.sample_rate,
    )

    sos = scipy.signal.ellip(
        order,
        passband_ripple,
        stopband_attenuation,
        wn,
        'lowpass',
        False,
        'sos',
        capture.sample_rate,
    )

    return sos


def iir_filter(
    iq: type_stubs.ArrayType,
    capture: structs.Capture,
    *,
    passband_ripple: float | int,
    stopband_attenuation: float | int,
    transition_bandwidth: float | int,
    out=None,
):
    filter_kws = select_parameter_kws(locals())
    sos = _generate_iir_lpf(capture, **filter_kws)

    xp = iqwaveform.util.array_namespace(iq)

    if is_cupy_array(iq):
        from . import cuda_filter

        sos = xp.asarray(sos)
        return cuda_filter.sosfilt(sos.astype('float32'), iq)

    else:
        return scipy.signal.sosfilt(sos.astype('float32'), iq)


def ola_filter(
    iq: type_stubs.ArrayType,
    capture: structs.Capture,
    *,
    nfft: int,
    window: typing.Any = 'hamming',
    out=None,
    cache=None,
):
    kwargs = select_parameter_kws(locals())

    return iqwaveform.fourier.ola_filter(
        iq,
        fs=capture.sample_rate,
        passband=(-capture.analysis_bandwidth / 2, capture.analysis_bandwidth / 2),
        **kwargs,
    )


def _evaluate_raw_channel_analysis(
    iq: type_stubs.ArrayType,
    capture: structs.Capture,
    *,
    spec: str | dict | structs.ChannelAnalysis,
):
    # round-trip for type conversion and validation
    spec = msgspec.convert(spec, expose_in_yaml.spec_type())
    spec_dict = msgspec.to_builtins(spec)

    results = {}

    # evaluate each possible analysis function if specified
    for name, func_kws in spec_dict.items():
        func = expose_in_yaml[type(getattr(spec, name))]

        if func_kws:
            results[name] = func(iq, capture, **func_kws)

    return results


def _package_channel_analysis(
    capture: structs.Capture, results: dict[str, structs.ChannelAnalysis]
) -> type_stubs.DatasetType:
    # materialize as xarrays
    xarrays = {res.name: res.to_xarray() for res in results.values()}
    # capture.analysis_filter = dict(capture.analysis_filter)
    # capture = msgspec.convert(capture, type=type(capture))
    attrs = msgspec.to_builtins(capture, builtin_types=(frozendict,))
    if isinstance(capture, structs.FilteredCapture):
        attrs['analysis_filter'] = dict(capture.analysis_filter)
    return xr.Dataset(xarrays, attrs=attrs)


def analyze_by_spec(
    iq: type_stubs.ArrayType,
    capture: structs.Capture,
    *,
    spec: str | dict | structs.ChannelAnalysis,
) -> type_stubs.DatasetType:
    """evaluate a set of different channel analyses on the iq waveform as specified by spec"""

    results = _evaluate_raw_channel_analysis(iq, capture, spec=spec)
    return _package_channel_analysis(capture, results)
