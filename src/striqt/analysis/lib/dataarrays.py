"""wrap lower-level striqt.waveform DSP calls to accept physical inputs and return xarray.DataArray"""

from __future__ import annotations as __

import collections
import dataclasses
import math
import typing

import msgspec

from .. import specs


from . import register, util

if typing.TYPE_CHECKING:
    import array_api_compat
    import numpy as np
    import xarray as xr

    from striqt.waveform._typing import ArrayType


else:
    np = util.lazy_import('numpy')
    xr = util.lazy_import('xarray')
    array_api_compat = util.lazy_import('array_api_compat')


AnalysisReturnFlag = typing.Literal[True, False, 'delayed']
_TA = typing.TypeVar('_TA', bound=AnalysisReturnFlag)


CAPTURE_DIM = 'capture'
PORT_DIM = 'port'


@dataclasses.dataclass
class AcquiredIQ:
    raw: ArrayType
    aligned: ArrayType | None
    capture: specs.Capture | None


_ENG_PREFIXES = {
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
}


@util.lru_cache()
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


def describe_field(capture: specs.Capture, name: str, *, sep='='):
    meta = specs.helpers.get_capture_type_attrs(type(capture))
    attrs = meta[name]
    value = getattr(capture, name)
    value_str = describe_value(value, attrs)

    return f'{name}{sep}{value_str}'


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


def _results_as_arrays(
    obj: tuple | list | dict | ArrayType,
) -> ArrayType | dict[str, ArrayType] | xr.Dataset | dict[str, DelayedDataArray]:
    """convert an array, or a container of arrays, into a numpy array (or container of numpy arrays)"""

    if array_api_compat.is_torch_array(obj):
        array = obj.cpu()
    elif util.is_cupy_array(obj):
        array = obj.get()
    elif array_api_compat.is_numpy_array(obj):
        return obj
    else:
        raise TypeError(f'obj type {type(obj)} is unrecognized')

    return array


def _empty_stub(dims, dtype, attrs={}):
    x = np.empty(len(dims) * (1,), dtype=dtype)
    return xr.DataArray(x, dims=dims, attrs=attrs)


@util.lru_cache()
def dataarray_stub(
    dims: tuple[str, ...] | None,
    coord_factories: tuple[register.CallableCoordinateFactory, ...],
    dtype: str,
    expand_dims: tuple[str, ...] | None = None,
) -> typing.Any:
    """return an empty array of type `cls`"""

    coord_stubs = {}
    for func in coord_factories:
        info = register.registry.coordinates[func]
        coord_stubs[info.name] = _empty_stub(info.dims, info.dtype, attrs=info.attrs)

    if dims is None or dims == ():
        dims = _infer_coord_dims(coord_factories)

    data_stub = _empty_stub(dims, dtype)
    stub = xr.DataArray(data=data_stub, dims=dims, coords=coord_stubs)

    slices = dict.fromkeys(stub.dims, slice(None, 0))
    stub = stub.isel(slices)

    if expand_dims is not None:
        stub = stub.expand_dims({n: 0 for n in expand_dims}, axis=0)

    return stub


def _reraise_coord_error(*, exc, coord, factory_info, data, name):
    template_shape = tuple([coord.indexes[d].shape for d in factory_info.dims])
    data_shape = np.array(data).shape

    if template_shape == data_shape:
        exc.args = (f'in coordinate {name!r}, {exc.args[0]}',) + exc.args[1:]
        raise exc
    else:
        problem = f'unexpected shape in coordinate {name!r}: data dimensions {factory_info.dims!r} have shape {template_shape}, but coordinate factory gave {data_shape}'
        raise ValueError(problem) from exc


def build_dataarray(
    delayed: DelayedDataArray,
    *,
    expand_dims=None,
) -> 'xr.DataArray':
    """build an `xarray.DataArray` from an ndarray, capture information, and channel analysis keyword arguments"""

    template = dataarray_stub(
        delayed.info.dims,
        delayed.info.coord_factories,
        delayed.info.dtype,
        expand_dims=expand_dims,
    )

    data = np.asarray(delayed.result)
    delayed.spec.validate()

    _validate_delayed_ndim(delayed)

    # to bypass initialization overhead, grow from the empty template
    pad = {dim: [0, data.shape[i]] for i, dim in enumerate(template.dims)}
    da = template.pad(pad)

    try:
        if da.values.ndim == 0:
            da.data = data
        else:
            da.data[:] = data
    except ValueError as ex:
        exc = ValueError(
            f'{delayed.info.name} measurement data has unexpected '
            f'shape {data.shape} -- expected {da.data.shape}'
        )
    else:
        exc = None

    if exc is not None:
        raise exc

    for coord_factory in delayed.info.coord_factories:
        factory_info = register.registry.coordinates[coord_factory]
        coord = da[factory_info.name]

        ret = coord_factory(delayed.capture, delayed.spec)
        qualname = f'{delayed.info.name}.{factory_info.name}'
        arr, metadata = register.normalize_factory_return(ret, qualname)

        try:
            coord.indexes[factory_info.dims[0]].values[:] = arr
        except ValueError as ex:
            exc = ex
        else:
            exc = None

        if exc is not None:
            _reraise_coord_error(
                exc=exc, coord=coord, factory_info=factory_info, data=arr, name=qualname
            )

        da[factory_info.name] = coord.assign_attrs(metadata)

    spec_attrs = delayed.spec.to_dict()
    return da.assign_attrs(delayed.info.attrs | spec_attrs | delayed.attrs)


@util.lru_cache()
def _infer_coord_dims(
    coord_factories: tuple[register.CallableCoordinateFactory, ...],
) -> tuple[str, ...]:
    """guess dimensions of a dataarray based on its coordinates.

    This orders the dimensions according to (1) first appearance in each
    the given list of coord factory functions and (2) first location within
    each factory.
    """

    # build an ordered list of unique coordinates
    coord_dims = {}
    for func in coord_factories:
        coord = register.registry.coordinates[func]
        if coord is None:
            continue
        empty_coords = dict.fromkeys(coord.dims or {}, None)
        coord_dims.update(empty_coords)
    return tuple(coord_dims.keys())


def _validate_delayed_ndim(delayed: DelayedDataArray) -> None:
    if delayed.info.dims is None:
        expect_dims = _infer_coord_dims(delayed.info.coord_factories)
    else:
        expect_dims = delayed.info.dims

    if isinstance(delayed.result, dict):
        raise TypeError('could not evaluate number of dimensions for dict return type')
    else:
        ndim = delayed.result.ndim

    if len(expect_dims) == ndim == 0:
        # allow scalar values
        pass
    elif len(expect_dims) + 1 != ndim:
        raise ValueError(
            f'coordinates of {delayed.info.name!r} indicate {len(expect_dims) + 1} dimensions, but the data has {ndim}'
        )


@dataclasses.dataclass
class DelayedDataArray(collections.UserDict):
    """represents the return result from a channel analysis function.

    This includes a method to convert to `xarray.DataArray`, which is
    delayed to leave the GPU time to initiate the evaluation of multiple
    analyses before we materialize them on the CPU.
    """

    capture: specs.Capture
    spec: specs.Analysis
    result: ArrayType | dict[str, ArrayType] | xr.Dataset | dict[str, DelayedDataArray]
    info: register.AnalysisInfo
    attrs: dict

    def compute(self) -> DelayedDataArray:
        return DelayedDataArray(
            # datacls=self.datacls,
            capture=self.capture,
            spec=self.spec,
            result=_results_as_arrays(self.result),
            info=self.info,
            attrs=self.attrs,
        )

    def to_xarray(self, expand_dims=None) -> 'xr.DataArray':
        return build_dataarray(self, expand_dims=expand_dims)


def select_parameter_kws(locals_: dict, omit=(PORT_DIM, 'out')) -> dict:
    """return the analysis parameters from the locals() evaluated at the beginning of analysis function"""

    items = list(locals_.items())
    return {k: v for k, v in items[1:] if k not in omit}


class EvaluationOptions(msgspec.Struct, typing.Generic[_TA]):
    as_xarray: _TA
    registry: register.AnalysisRegistry
    block_each: bool = True
    expand_dims: typing.Sequence[str] = ()

    def __post_init__(self):
        if self.as_xarray not in (True, False, 'delayed'):
            raise TypeError('as_xarray must be True, False, or "delayed"')
        assert self.registry is not None


def evaluate_by_spec(
    iq: ArrayType | AcquiredIQ,
    spec: dict[str, specs.Analysis] | specs.AnalysisGroup,
    capture: specs.Capture,
    options: EvaluationOptions,
) -> dict[str, DelayedDataArray] | dict[str, ArrayType]:
    """evaluate each analysis for the given IQ waveform"""

    if isinstance(spec, specs.AnalysisGroup):
        spec = spec.validate()
    elif isinstance(spec, dict):
        spec = options.registry.tospec().from_dict(spec)
    else:
        raise TypeError('invalid analysis spec argument')

    if not isinstance(iq, AcquiredIQ):
        iq = AcquiredIQ(raw=iq, aligned=None, capture=None)

    if isinstance(spec, dict):
        spec_dict = spec
    else:
        spec_dict = spec.to_dict()
    results: dict[str, DelayedDataArray] | dict[str, ArrayType] = {}
    as_xarray = 'delayed' if options.as_xarray else False

    if util.is_cupy_array(getattr(iq, 'raw', iq)):
        util.configure_cupy()

    for name in spec_dict.keys():
        meas = options.registry[type(getattr(spec, name))]

        with util.stopwatch(name, 'analysis'):
            func_kws = spec_dict[name]
            if not func_kws:
                continue
            if not isinstance(iq, AcquiredIQ):
                iq_sel = iq
            if iq.aligned is None or meas.prefer_unaligned_input:
                iq_sel = iq.raw
            else:
                iq_sel = iq.aligned

            ret = meas.func(iq=iq_sel, capture=capture, as_xarray=as_xarray, **func_kws)

            if options.block_each:
                results[name] = ret.compute()
            else:
                results[name] = ret

    if as_xarray == 'delayed' and options.block_each:
        return results

    for name in list(results.keys()):
        res = results[name]

        if not options.block_each:
            assert isinstance(res, DelayedDataArray)
            res = res.compute()

        if as_xarray == 'delayed':
            results[name] = res
        else:
            assert isinstance(res, DelayedDataArray)
            results[name] = res.to_xarray()

    if util.is_cupy_array(getattr(iq, 'raw', iq)):
        util.free_cupy_mempool()

    return results


def package_analysis(
    capture: specs.Capture,
    results: dict[str, DelayedDataArray] | dict[str, ArrayType],
    expand_dims=None,
) -> 'xr.Dataset':
    # materialize as xarrays
    with util.stopwatch('package dataset'):
        xarrays = {}
        for name, res in results.items():
            assert isinstance(res, DelayedDataArray)
            xarrays[name] = res.compute().to_xarray(expand_dims)

        attrs = capture.to_dict()
        if isinstance(capture, specs.FilteredCapture):
            attrs['analysis_filter'] = capture.analysis_filter.to_dict()
        ret = xr.Dataset(xarrays, attrs=attrs)

    return ret


@typing.overload
def analyze_by_spec(
    iq: ArrayType | AcquiredIQ,
    spec: dict[str, specs.Analysis] | specs.AnalysisGroup,
    capture: specs.Capture,
    options: EvaluationOptions[typing.Literal[True]],
) -> 'xr.Dataset': ...


@typing.overload
def analyze_by_spec(
    iq: ArrayType | AcquiredIQ,
    spec: dict[str, specs.Analysis] | specs.AnalysisGroup,
    capture: specs.Capture,
    options: EvaluationOptions[typing.Literal['delayed']],
) -> 'dict[str, DelayedDataArray]': ...


@typing.overload
def analyze_by_spec(
    iq: ArrayType | AcquiredIQ,
    spec: dict[str, specs.Analysis] | specs.AnalysisGroup,
    capture: specs.Capture,
    options: EvaluationOptions[typing.Literal[False]],
) -> 'dict[str, ArrayType]': ...


def analyze_by_spec(
    iq: ArrayType | AcquiredIQ,
    spec: dict[str, specs.Analysis] | specs.AnalysisGroup,
    capture: specs.Capture,
    options: EvaluationOptions,
) -> ArrayType | dict[str, ArrayType] | xr.Dataset | dict[str, DelayedDataArray]:
    """evaluate a set of different channel analyses on the iq waveform as specified by spec"""

    results = evaluate_by_spec(iq, spec, capture, options)

    if not options.as_xarray or options.as_xarray == 'delayed':
        return results
    else:
        return package_analysis(capture, results, expand_dims=options.expand_dims)
