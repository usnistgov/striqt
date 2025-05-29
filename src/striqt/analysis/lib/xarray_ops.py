"""wrap lower-level iqwaveform DSP calls to accept physical inputs and return xarray.DataArray"""

from __future__ import annotations

import collections
import functools
import dataclasses
import math
import typing

from . import register, specs, util

import array_api_compat

if typing.TYPE_CHECKING:
    import iqwaveform
    import iqwaveform.type_stubs
    import labbench as lb
    import numpy as np
    import xarray as xr
else:
    iqwaveform = util.lazy_import('iqwaveform')
    lb = util.lazy_import('labbench')
    np = util.lazy_import('numpy')
    xr = util.lazy_import('xarray')


class IQPair(typing.NamedTuple):
    aligned: 'iqwaveform.type_stubs.ArrayType'
    raw: 'iqwaveform.type_stubs.ArrayType'


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


@util.lru_cache()
def describe_capture(
    this: specs.Capture | None,
    prev: specs.Capture | None = None,
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
        if prev is not None and name == 'backend_sample_rate' and value is None:
            continue
        elif prev is None or value != getattr(prev, name):
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


def describe_field(capture: specs.Capture, name: str):
    meta = specs.get_capture_type_attrs(type(capture))
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


def _results_as_arrays(obj: tuple | list | dict | 'iqwaveform.util.Array'):
    """convert an array, or a container of arrays, into a numpy array (or container of numpy arrays)"""

    if array_api_compat.is_torch_array(obj):
        array = obj.cpu()
    elif array_api_compat.is_cupy_array(obj):
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
    dims: tuple[str, ...],
    coord_factories: tuple[callable, ...],
    dtype: str,
    expand_dims: tuple[str, ...] | None = None,
) -> typing.Any:
    """return an empty array of type `cls`"""

    coord_stubs = {}
    for func in coord_factories:
        info = register.coordinate_factory[func]
        coord_stubs[info.name] = _empty_stub(info.dims, info.dtype, attrs=info.attrs)

    if not dims:
        dims = _infer_coord_dims(coord_factories)

    data_stub = _empty_stub(dims, dtype)
    stub = xr.DataArray(data=data_stub, dims=dims, coords=coord_stubs)

    slices = dict.fromkeys(stub.dims, slice(None, 0))
    stub = stub.isel(slices)

    if expand_dims is not None:
        stub = stub.expand_dims({n: 0 for n in expand_dims}, axis=0)

    return stub


def build_dataarray(
    delayed: DelayedDataArray,
    expand_dims=None,
) -> 'xr.DataArray':
    """build an `xarray.DataArray` from an ndarray, capture information, and channel analysis keyword arguments"""
    template = dataarray_stub(
        delayed.info.dims,
        delayed.info.coord_factories,
        delayed.info.dtype,
        expand_dims=expand_dims,
    )
    data = np.asarray(delayed.data)
    delayed.spec.validate()

    _validate_delayed_ndim(delayed)

    # allow unused dimensions before those of the template
    # (for e.g. multichannel acquisition)
    target_shape = data.shape[-len(template.dims) :]

    # to bypass initialization overhead, grow from the empty template
    pad = {dim: [0, target_shape[i]] for i, dim in enumerate(template.dims)}
    da = template.pad(pad)

    try:
        da.values[:] = data
    except ValueError as ex:
        raise ValueError(
            f'{delayed.info.name} measurement data has unexpected shape {data.shape}'
        ) from ex

    for coord_factory in delayed.info.coord_factories:
        coord_info = register.coordinate_factory[coord_factory]
        ret = coord_factory(delayed.capture, delayed.spec)

        try:
            if isinstance(ret, tuple):
                arr, metadata = ret
            else:
                arr = ret
                metadata = {}

            da[coord_info.name].indexes[coord_info.dims[0]].values[:] = arr

        except ValueError as ex:
            exc = ex
        else:
            exc = None

        if exc is not None:
            template_shape = tuple(
                [da[coord_info.name].indexes[d].shape for d in coord_info.dims]
            )
            data_shape = np.array(arr).shape
            name = f'{delayed.info.name}.{coord_info.name}'

            if template_shape == data_shape:
                exc.args = (f'in coordinate {name!r}, {exc.args[0]}',) + exc.args[1:]
                raise exc
            else:
                problem = (
                    f'unexpected shape in coordinate {name!r}: '
                    f'data dimensions {coord_info.dims!r} have shape {template_shape}, '
                    f'but coordinate factory gave {data_shape}'
                )
                raise ValueError(problem) from exc

        da[coord_info.name] = da[coord_info.name].assign_attrs(metadata)

    return da.assign_attrs(delayed.info.attrs | delayed.attrs)


@util.lru_cache()
def _infer_coord_dims(coord_factories: typing.Iterable[callable]) -> list[str]:
    """guess dimensions of a dataarray based on its coordinates.

    This orders the dimensions according to (1) first appearance in each
    the given list of coord factory functions and (2) first location within
    each factory.
    """

    # build an ordered list of unique coordinates
    coord_dims = {}
    for func in coord_factories:
        coord = register.coordinate_factory[func]
        if coord is None:
            continue
        coord_dims.update(dict.fromkeys(coord.dims, None))
    return list(coord_dims.keys())


def _validate_delayed_ndim(delayed: DelayedDataArray) -> None:
    if delayed.info.dims is None:
        expect_dims = _infer_coord_dims(delayed.info.coord_factories)
    else:
        expect_dims = delayed.info.dims

    ndim = delayed.data.ndim

    if len(expect_dims) + 1 != ndim:
        raise ValueError(
            f'coordinates of {delayed.info.name!r} indicate {len(expect_dims) + 1} '
            f'dimensions, but the data has {ndim}'
        )


@dataclasses.dataclass
class DelayedDataArray(collections.UserDict):
    """represents the return result from a channel analysis function.

    This includes a method to convert to `xarray.DataArray`, which is
    delayed to leave the GPU time to initiate the evaluation of multiple
    analyses before we materialize them on the CPU.
    """

    # datacls: type
    capture: specs.RadioCapture
    spec: specs.Measurement
    data: typing.Union['iqwaveform.type_stubs.ArrayLike', dict]
    info: register.MeasurementInfo
    attrs: dict

    def compute(self) -> DelayedDataArray:
        return DelayedDataArray(
            # datacls=self.datacls,
            capture=self.capture,
            spec=self.spec,
            data=_results_as_arrays(self.data),
            info=self.info,
            attrs=self.attrs,
        )

    def to_xarray(self, expand_dims=None) -> 'xr.DataArray':
        return build_dataarray(self, expand_dims=expand_dims)


def select_parameter_kws(locals_: dict, omit=('capture', 'out')) -> dict:
    """return the analysis parameters from the locals() evaluated at the beginning of analysis function"""

    items = list(locals_.items())
    return {k: v for k, v in items[1:] if k not in omit}


def evaluate_by_spec(
    iq: 'iqwaveform.type_stubs.ArrayType' | IQPair,
    capture: specs.Capture,
    *,
    spec: str | dict | specs.Measurement,
    as_xarray: typing.Literal[True]
    | typing.Literal[False]
    | typing.Literal['delayed'] = 'delayed',
    block_each=True,
):
    """evaluate each analysis for the given IQ waveform"""

    if isinstance(spec, specs.Analysis):
        spec = spec.validate()
    else:
        spec = register.to_analysis_spec_type(register.measurement).fromdict(spec)

    spec_dict = spec.todict()
    results: dict[str, DelayedDataArray] = {}
    as_xarray = 'delayed' if as_xarray else False

    for name in spec_dict.keys():
        meas = register.measurement[type(getattr(spec, name))]

        with lb.stopwatch(f'analysis: {name}', logger_level='debug'):
            func_kws = spec_dict[name]
            if not func_kws:
                continue
            if iq.aligned is None or meas.prefer_unaligned_input:
                iq_sel = iq.raw
            else:
                iq_sel = iq.aligned

            ret = meas.func(iq=iq_sel, capture=capture, as_xarray=as_xarray, **func_kws)

            if block_each:
                results[name] = ret.compute()
            else:
                results[name] = ret

    if as_xarray:
        pass
    elif block_each:
        return results

    for name in list(results.keys()):
        if as_xarray == 'delayed':
            pass
        else:
            results[name] = results[name].compute().to_xarray()

    return results


def package_analysis(
    capture: specs.Capture,
    results: dict[str, DelayedDataArray],
    expand_dims=None,
) -> 'xr.Dataset':
    # materialize as xarrays
    with lb.stopwatch('package analyses into xarray', logger_level='debug'):
        xarrays = {}
        for name, res in results.items():
            xarrays[name] = res.compute().to_xarray(expand_dims)

        attrs = capture.todict()
        if isinstance(capture, specs.FilteredCapture):
            attrs['analysis_filter'] = capture.analysis_filter.todict()
        ret = xr.Dataset(xarrays, attrs=attrs)

    return ret


def analyze_by_spec(
    iq: 'iqwaveform.type_stubs.ArrayType' | IQPair,
    capture: specs.Capture,
    *,
    spec: str | dict | specs.Measurement,
    as_xarray: bool | typing.Literal['delayed'] = True,
    block_each: bool = True,
    expand_dims=None,
) -> 'xr.Dataset':
    """evaluate a set of different channel analyses on the iq waveform as specified by spec"""
    results = evaluate_by_spec(
        iq,
        capture,
        spec=spec,
        block_each=block_each,
        as_xarray='delayed' if as_xarray else False,
    )

    if not as_xarray or as_xarray == 'delayed':
        return results
    else:
        return package_analysis(capture, results, expand_dims=expand_dims)
