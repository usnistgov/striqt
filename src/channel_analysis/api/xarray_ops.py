"""wrap lower-level iqwaveform DSP calls to accept physical inputs and return xarray.DataArray"""

from __future__ import annotations

import collections
import functools
import dataclasses
import inspect
import math
import msgspec
import typing

from . import structs, util


if typing.TYPE_CHECKING:
    import numpy as np
    import xarray as xr
    import iqwaveform
    from xarray_dataclasses import datamodel, dataarray
    import labbench as lb
else:
    np = util.lazy_import('numpy')
    xr = util.lazy_import('xarray')
    iqwaveform = util.lazy_import('iqwaveform')
    datamodel = util.lazy_import('xarray_dataclasses.datamodel')
    dataarray = util.lazy_import('xarray_dataclasses.dataarray')
    lb = util.lazy_import('labbench')


TFunc = typing.Callable[..., typing.Any]


def _entry_stub(entry: 'datamodel.AnyEntry'):
    return np.empty(len(entry.dims) * (1,), dtype=entry.dtype)


@typing.overload
def shaped(
    cls: type['dataarray.OptionedClass[dataarray.PInit, dataarray.TDataArray]'],
) -> 'dataarray.TDataArray': ...


@functools.lru_cache
def dataarray_stub(cls: typing.Any, expand_dims=None) -> typing.Any:
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
    stub = stub.isel(slices)

    if expand_dims is not None:
        stub = stub.expand_dims({n: 0 for n in expand_dims}, axis=0)

    return stub


@functools.lru_cache
def get_data_model(dataclass: typing.Any):
    return datamodel.DataModel.from_dataclass(dataclass)


def freezevalues(parameters: dict) -> dict:
    return {
        k: (tuple(v) if isinstance(v, (list, np.ndarray)) else v)
        for k, v in parameters.items()
    }


_ENG_PREFIXES = {
    -30: 'q',
    -27: 'r',
    -24: 'y',
    -21: 'z',
    -18: 'a',
    -15: 'f',
    -12: 'p',
    -9: 'n',
    -6: '\N{MICRO SIGN}',
    -3: 'm',
    0: '',
    3: 'k',
    6: 'M',
    9: 'G',
    12: 'T',
    15: 'P',
    18: 'E',
    21: 'Z',
    24: 'Y',
    27: 'R',
    30: 'Q',
}


@functools.lru_cache
def describe_capture(
    this: structs.Capture | None,
    prev: structs.Capture | None = None,
    *,
    index: int,
    count: int,
) -> str:
    """generate a description of a capture"""
    if this is None:
        if prev is None:
            return 'saving last analysis'
        else:
            return 'performing last analysis'

    diffs = []

    for name in type(this).__struct_fields__:
        if name == 'start_time':
            continue
        value = getattr(this, name)
        if prev is None or value != getattr(prev, name):
            diffs.append(describe_field(this, name))

    capture_diff = ', '.join(diffs)

    if index is not None:
        progress = str(index + 1)

        if count is not None:
            progress = f'{progress}/{count}'

        progress = progress + ' '
    else:
        progress = ''

    return progress + capture_diff


@functools.lru_cache()
def format_units(value, unit='', places=None, force_prefix=None, sep=' ') -> str:
    """Format a number with SI unit prefixes"""

    sign = 1
    fmt = 'g' if places is None else f'.{places:d}f'

    if value < 0:
        sign = -1
        value = -value

    if unit.lower().startswith('db'):
        pow10 = 0
    elif force_prefix is not None:
        remap = dict(zip(_ENG_PREFIXES.values(), _ENG_PREFIXES.keys()))
        pow10 = remap[force_prefix]
    elif value != 0:
        pow10 = int(math.floor(math.log10(value) / 3) * 3)
    else:
        pow10 = 0
        # Force value to zero, to avoid inconsistencies like
        # format_eng(-0) = "0" and format_eng(0.0) = "0"
        # but format_eng(-0.0) = "-0.0"
        value = 0.0

    pow10 = np.clip(pow10, min(_ENG_PREFIXES), max(_ENG_PREFIXES))

    mant = sign * value / (10.0**pow10)
    # Taking care of the cases like 999.9..., which may be rounded to 1000
    # instead of 1 k.  Beware of the corner case of values that are beyond
    # the range of SI prefixes (i.e. > 'Y').
    if (
        force_prefix is None
        and abs(float(format(mant, fmt))) >= 1000
        and pow10 < max(_ENG_PREFIXES)
    ):
        mant /= 1000
        pow10 += 3

    unit_prefix = _ENG_PREFIXES[int(pow10)]
    if unit or unit_prefix:
        suffix = f'{sep}{unit_prefix}{unit}'
    else:
        suffix = ''

    return f'{mant:{fmt}}{suffix}'


def describe_field(capture: structs.Capture, name: str):
    meta = structs.get_capture_type_attrs(type(capture))
    attrs = meta[name]
    value = getattr(capture, name)
    value_str = describe_value(value, attrs)

    return f'{name}={value_str}'


def describe_value(value, attrs: dict, unit_prefix=None):
    if value is None:
        value_str = 'None'
    elif attrs.get('units', None) is not None and np.isfinite(value):
        unit_kws = {'force_prefix': unit_prefix, 'unit': attrs['units']}
        if isinstance(value, tuple):
            value_tup = [format_units(v, **unit_kws) for v in value]
            value_str = f'({", ".join(value_tup)})'
        else:
            value_str = format_units(value, **unit_kws)
    else:
        value_str = repr(value)

    return value_str


def channel_dataarray(
    cls,
    data: 'np.ndarray',
    capture,
    parameters: dict[str, typing.Any],
    expand_dims=None,
) -> 'xr.DataArray':
    """build an `xarray.DataArray` from an ndarray, capture information, and channel analysis keyword arguments"""
    template = dataarray_stub(cls, expand_dims)
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

        except ValueError as ex:
            exc = ex
        else:
            exc = None

        if exc is not None:
            template_shape = da[entry.name].indexes[entry.dims[0]].shape
            data_shape = np.array(arr).shape

            if template_shape == data_shape:
                raise exc
            else:
                problem = f'expected {template_shape} from template, but factory gave {data_shape}'
                name = f'{cls.__qualname__}.{entry.name}'
                raise ValueError(f'unexpected {name} coordinate shape: {problem}')

        da[entry.name].attrs.update(metadata)

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

    def to_xarray(self, expand_dims=None) -> 'xr.DataArray':
        return channel_dataarray(
            cls=self.datacls,
            data=self.data,
            capture=self.capture,
            parameters=self.parameters,
            expand_dims=expand_dims,
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
    as_xarray: typing.Literal[True]
    | typing.Literal[False]
    | typing.Literal['delayed'] = 'delayed',
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
        util.free_mempool_on_low_memory()
        with lb.stopwatch(f'analysis: {name}', logger_level='debug'):
            func = registry[type(getattr(spec, name))]

            if func_kws:
                results[name] = func(iq, capture, as_xarray=as_xarray, **func_kws)

    return results


def package_channel_analysis(
    capture: structs.Capture,
    results: dict[str, structs.ChannelAnalysis],
    expand_dims=None,
) -> 'xr.Dataset':
    # materialize as xarrays
    with lb.stopwatch('package analyses into xarray', logger_level='debug'):
        xarrays = {}
        for name, res in results.items():
            xarrays[name] = res.to_xarray(expand_dims)

        # capture.analysis_filter = dict(capture.analysis_filter)
        # capture = structs.builtins_to_struct(capture, type=type(capture))
        attrs = structs.struct_to_builtins(capture)
        if isinstance(capture, structs.FilteredCapture):
            attrs['analysis_filter'] = msgspec.to_builtins(capture.analysis_filter)
        ret = xr.Dataset(xarrays, attrs=attrs)

    return ret
