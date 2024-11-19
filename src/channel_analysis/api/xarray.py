"""wrap lower-level iqwaveform DSP calls to accept physical inputs and return xarray.DataArray"""

from __future__ import annotations

import collections
import functools
import dataclasses
import inspect
import msgspec
import typing

from . import structs, util


if typing.TYPE_CHECKING:
    import numpy as np
    import xarray as xr
    import iqwaveform
    from xarray_dataclasses import datamodel, dataarray
else:
    np = util.lazy_import('numpy')
    xr = util.lazy_import('xarray')
    iqwaveform = util.lazy_import('iqwaveform')
    datamodel = util.lazy_import('xarray_dataclasses.datamodel')
    dataarray = util.lazy_import('xarray_dataclasses.dataarray')


TFunc = typing.Callable[..., typing.Any]


def _entry_stub(entry: 'datamodel.AnyEntry'):
    return np.empty(len(entry.dims) * (1,), dtype=entry.dtype)


@typing.overload
def shaped(
    cls: type['dataarray.OptionedClass[dataarray.PInit, dataarray.TDataArray]'],
) -> 'dataarray.TDataArray': ...
@functools.lru_cache
def dataarray_stub(cls: typing.Any) -> typing.Any:
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
def get_data_model(dataclass: typing.Any):
    return datamodel.DataModel.from_dataclass(dataclass)


def freezevalues(parameters: dict) -> dict:
    return {
        k: (tuple(v) if isinstance(v, (list, np.ndarray)) else v)
        for k, v in parameters.items()
    }


def channel_dataarray(
    cls, data: 'np.ndarray', capture, parameters: dict[str, typing.Any]
) -> 'xr.DataArray':
    """build an `xarray.DataArray` from an ndarray, capture information, and channel analysis keyword arguments"""
    template = dataarray_stub(cls)
    data = np.asarray(data)
    parameters = freezevalues(parameters)

    # to bypass initialization overhead, grow from the empty template
    da = template.pad({dim: [0, data.shape[i]] for i, dim in enumerate(template.dims)})
    da.values[:] = data

    for entry in get_data_model(cls).coords:
        ret = entry.base.factory(capture, **parameters)

        try:
            if isinstance(ret, tuple):
                arr, metadata = ret
            else:
                arr = ret
                metadata = {}

            da[entry.name].indexes[entry.dims[0]].values[:] = arr
            da[entry.name].attrs.update(metadata)

        except BaseException as ex:
            raise ValueError(
                f'error building xarray {cls.__qualname__}.{entry.name}'
            ) from ex

    return da


@dataclasses.dataclass
class ChannelAnalysisResult(collections.UserDict):
    """represents the return result from a channel analysis function.

    This includes a method to convert to `xarray.DataArray`, which is
    delayed to leave the GPU time to initiate the evaluation of multiple
    analyses before we materialize them on the CPU.
    """

    datacls: type
    data: typing.Union['np.ndarray', dict]
    capture: structs.RadioCapture
    parameters: dict[str, typing.Any]
    attrs: dict[str] = dataclasses.field(default_factory=dict)

    def to_xarray(self) -> 'xr.DataArray':
        return channel_dataarray(
            cls=self.datacls,
            data=self.data,
            capture=self.capture,
            parameters=self.parameters,
        ).assign_attrs(self.attrs)


def select_parameter_kws(locals_: dict, omit=('capture', 'out')) -> dict:
    """return the analysis parameters from the locals() evaluated at the beginning of analysis function"""

    items = list(locals_.items())
    return {k: v for k, v in items[1:] if k not in omit}


def evaluate_channel_analysis(
    iq: 'iqwaveform.util.Array',
    capture: structs.Capture,
    *,
    spec: str | dict | structs.ChannelAnalysis,
    registry,
):
    """evaluate the specified channel analysis for the given IQ waveform and
    its capture information"""
    # round-trip for type conversion and validation
    spec = structs.builtins_to_struct(spec, registry.spec_type())
    spec_dict = structs.struct_to_builtins(spec)

    results = {}

    # evaluate each possible analysis function if specified
    for name, func_kws in spec_dict.items():
        func = registry[type(getattr(spec, name))]

        if func_kws:
            results[name] = func(iq, capture, delay_xarray=True, **func_kws)

    return results


def package_channel_analysis(
    capture: structs.Capture, results: dict[str, structs.ChannelAnalysis]
) -> 'xr.Dataset':
    # materialize as xarrays
    xarrays = {name: res.to_xarray() for name, res in results.items()}
    # capture.analysis_filter = dict(capture.analysis_filter)
    # capture = structs.builtins_to_struct(capture, type=type(capture))
    attrs = structs.struct_to_builtins(capture)
    if isinstance(capture, structs.FilteredCapture):
        attrs['analysis_filter'] = msgspec.to_builtins(capture.analysis_filter)
    return xr.Dataset(xarrays, attrs=attrs)
