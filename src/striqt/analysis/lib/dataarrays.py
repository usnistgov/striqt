"""wrap lower-level striqt.waveform DSP calls to accept physical inputs and return xarray.DataArray"""

from __future__ import annotations as __

import collections
import dataclasses
import math
import typing

from .. import specs
from . import register, util
from .typing import TAR
from .util import array_api_compat, np, xr

import striqt.waveform as sw

import msgspec

if typing.TYPE_CHECKING:
    from .typing import Array, CoordFunc


CAPTURE_DIM = 'capture'
PORT_DIM = 'port'


@dataclasses.dataclass
class AcquiredIQ:
    """an acquired IQ waveform `x` with the capture that describes the acquisition.

    `select` returns `x` for every preference. A subclass that conditions the
    waveform (resampling, filtering, trigger alignment) overrides `select`, and a
    measurement body reaches either through `select_waveform`.
    """

    x: Array
    capture: specs.Capture | None

    def select(
        self, *, prefer_filtered: bool = True, prefer_aligned: bool = True
    ) -> Array:
        """return the waveform conditioned as `prefer_filtered` and `prefer_aligned` allow.

        Args:
            prefer_filtered: True for the waveform low-passed to the capture
                `analysis_bandwidth`, when the acquisition applies one
            prefer_aligned: True for the waveform shifted by the signal trigger, when
                the acquisition runs one

        Returns:
            shape (port, sample) array in the namespace of `x`; this class returns
            `x` unchanged
        """
        return self.x


def select_waveform(
    iq: Array | AcquiredIQ, *, prefer_filtered: bool = True, prefer_aligned: bool = True
) -> Array:
    """return the waveform to measure from an array or an `AcquiredIQ`.

    An array is returned unchanged. For an `AcquiredIQ`, the return value is
    `AcquiredIQ.select` of `iq` with the same preferences.

    Args:
        iq: shape (port, sample) array, or the acquisition that wraps one
        prefer_filtered: True for the waveform low-passed to the capture
            `analysis_bandwidth`, when the acquisition applies one
        prefer_aligned: True for the waveform shifted by the signal trigger, when the
            acquisition runs one

    Returns:
        shape (port, sample) array in the namespace of the acquisition
    """
    if isinstance(iq, AcquiredIQ):
        return iq.select(prefer_filtered=prefer_filtered, prefer_aligned=prefer_aligned)
    return iq


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
    # a mantissa like 999.9... rounds to 1000 instead of 1 k, so carry it into the
    # next prefix. Values at the largest prefix in `_ENG_PREFIXES` are not carried.
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
    obj: tuple | list | dict | Array,
) -> Array | dict[str, Array] | xr.Dataset | dict[str, DelayedDataArray]:
    """convert an array, or a container of arrays, into a numpy array (or container of numpy arrays)"""

    if array_api_compat.is_torch_array(obj):
        array = obj.cpu()
    elif sw.is_cupy_array(obj):
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
    coord_factories: tuple[CoordFunc, ...],
    dtype: str,
    expand_dims: tuple[str, ...] | None = None,
) -> 'xr.DataArray':
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
    template_shape = []
    for d in factory_info.dims:
        shape = coord.indexes[d].shape
        if isinstance(shape, tuple):
            template_shape.extend(shape)
        else:
            template_shape.append(shape)
    template_shape = tuple(template_shape)

    data_shape = np.array(data).shape

    if template_shape == data_shape:
        exc.args = (f'in coordinate {name!r}, {exc.args[0]}',) + exc.args[1:]
        raise exc
    else:
        problem = f'unexpected shape in coordinate {name!r}: data dimensions {factory_info.dims!r} have shape {template_shape}, but coordinate factory gave {data_shape}'
        raise ValueError(problem) from exc


def _restore_padded_coord_dtype(da: 'xr.DataArray', info: register.CoordInfo) -> None:
    """restore the integer or bool dtype declared for coordinate `info.name` of `da`.

    `xr.DataArray.pad` fills with nan and so promotes integer and bool coordinates
    to float. The coordinate is replaced in place with zeros of `info.dtype`, since
    `build_dataarray` overwrites every element from the coordinate factory afterward.
    Coordinates of other dtypes, or already of `info.dtype`, are left untouched.
    """

    dtype = np.dtype(info.dtype)
    coord = da[info.name]

    if dtype.kind not in 'iub' or coord.dtype == dtype:
        return

    da[info.name] = xr.zeros_like(coord, dtype=dtype)


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

    # grow from the empty template to bypass initialization overhead
    pad = {dim: (0, data.shape[i]) for i, dim in enumerate(template.dims)}
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
        _restore_padded_coord_dtype(da, factory_info)
        coord = da[factory_info.name]

        ret = coord_factory(delayed.capture, delayed.spec)
        qualname = f'{delayed.info.name}.{factory_info.name}'
        arr, metadata = register.normalize_factory_return(ret, qualname)

        try:
            idx = coord.indexes[factory_info.dims[0]]
            idx._values[:] = arr
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
def _infer_coord_dims(coord_factories: tuple[CoordFunc, ...]) -> tuple[str, ...]:
    """return the dimension names that `coord_factories` declare, in order.

    A dimension is placed at the first factory that declares the dimension, in the
    order of the dims of that factory.
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
    """wrap the result returned by a channel analysis function.

    The conversion to `xarray.DataArray` is delayed to leave the GPU time to
    initiate the evaluation of multiple analyses before they are materialized on
    the CPU.
    """

    capture: specs.Capture
    spec: specs.Analysis
    result: Array | dict[str, Array] | xr.Dataset | dict[str, DelayedDataArray]
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


class EvaluationOptions(msgspec.Struct, typing.Generic[TAR]):
    """the result type and registry for one evaluation of an analysis group.

    `as_xarray` selects the result type of `analyze_by_spec`: ``True`` for an
    `xarray.Dataset`, ``'delayed'`` for `DelayedDataArray` results, or ``False``
    for the bare arrays in the array namespace of the input. `block_each` applies
    to ``True`` and ``'delayed'`` and materializes each result to numpy as soon as
    the measurement returns. `expand_dims` names the leading dimension of each
    `xarray.DataArray` when `as_xarray` is ``True``. That dimension indexes the
    port axis of the input.
    """

    as_xarray: TAR
    registry: register.AnalysisRegistry
    block_each: bool = True
    expand_dims: typing.Sequence[str] = (PORT_DIM,)

    def __post_init__(self):
        if self.as_xarray not in (True, False, 'delayed'):
            raise TypeError('as_xarray must be True, False, or "delayed"')
        assert self.registry is not None


def _raw_array(iq: Array | AcquiredIQ) -> Array:
    """return the acquired array of `iq`, or `iq` when the argument is an array."""
    if isinstance(iq, AcquiredIQ):
        return iq.x
    return iq


def evaluate_by_spec(
    iq: Array | AcquiredIQ,
    spec: specs.AnalysisGroup,
    capture: specs.Capture,
    options: EvaluationOptions,
) -> 'dict[str, DelayedDataArray] | dict[str, Array]':
    """evaluate each enabled measurement of `spec` on the IQ waveform.

    Args:
        iq: the waveform with shape (ports, samples), or an `AcquiredIQ`. Each
            measurement body selects the conditioning through `select_waveform`
        spec: an `AnalysisGroup` such as `specs.BuiltinAnalyses`. The ``None``
            fields of `spec` are skipped
        capture: the acquisition parameters of `iq`
        options: `options.as_xarray` selects the result type. ``True`` or
            ``'delayed'`` return a `DelayedDataArray` per measurement, materialized
            to numpy when `options.block_each` is set. ``False`` returns the bare
            array of each measurement in the array namespace of `iq`, without the
            measurement attrs. `options.block_each` applies only to ``True`` and
            ``'delayed'``

    Returns:
        a dict keyed by measurement name, in the field order of `spec`

    Raises:
        TypeError: if `spec` is not an `AnalysisGroup`
    """

    spec_dict: dict[str, dict[str, typing.Any]]
    results: dict[str, DelayedDataArray] | dict[str, Array]

    if isinstance(spec, specs.AnalysisGroup):
        spec = spec.validate()
    else:
        raise TypeError('invalid analysis spec argument')

    spec_dict = spec.to_dict()
    results = {}
    as_xarray = 'delayed' if options.as_xarray else False

    if sw.is_cupy_array(_raw_array(iq)):
        sw.arrays.configure_cupy()

    for name in spec_dict.keys():
        meas = options.registry[type(getattr(spec, name))]

        with util.stopwatch(name, 'analysis'):
            func_kws = spec_dict[name]
            if not func_kws:
                continue
            ret = meas.func(iq=iq, capture=capture, as_xarray=as_xarray, **func_kws)

            if not as_xarray:
                results[name] = ret[0]
            elif options.block_each:
                results[name] = ret.compute()
            else:
                results[name] = ret

    if not as_xarray or options.block_each:
        return results

    for name in list(results.keys()):
        res = results[name]
        assert isinstance(res, DelayedDataArray)
        results[name] = res.compute()

    if sw.is_cupy_array(_raw_array(iq)):
        sw.arrays.free_cupy_mempool()

    return results


def package_analysis(
    capture: specs.Capture,
    results: dict[str, DelayedDataArray] | dict[str, Array],
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
    iq: Array | AcquiredIQ,
    spec: specs.AnalysisGroup,
    capture: specs.Capture,
    options: EvaluationOptions[typing.Literal[True]] | None = None,
) -> 'xr.Dataset': ...


@typing.overload
def analyze_by_spec(
    iq: Array | AcquiredIQ,
    spec: specs.AnalysisGroup,
    capture: specs.Capture,
    options: EvaluationOptions[typing.Literal['delayed']],
) -> 'dict[str, DelayedDataArray]': ...


@typing.overload
def analyze_by_spec(
    iq: Array | AcquiredIQ,
    spec: specs.AnalysisGroup,
    capture: specs.Capture,
    options: EvaluationOptions[typing.Literal[False]],
) -> 'dict[str, Array]': ...


def analyze_by_spec(
    iq: Array | AcquiredIQ,
    spec: specs.AnalysisGroup,
    capture: specs.Capture,
    options: EvaluationOptions | None = None,
) -> Array | dict[str, Array] | xr.Dataset | dict[str, DelayedDataArray]:
    """evaluate the enabled measurements of `spec` on the IQ waveform.

    Args:
        iq: the waveform with shape (ports, samples), or an `AcquiredIQ`. Each
            measurement body selects the conditioning through `select_waveform`
        spec: an `AnalysisGroup` such as `specs.BuiltinAnalyses`. The ``None``
            fields of `spec` are skipped
        capture: the acquisition parameters of `iq`
        options: `options.as_xarray` selects the return type. ``None`` evaluates
            with the live registry and `as_xarray` of ``True``

    Returns:
        by `options.as_xarray`:

        - ``True``: an `xarray.Dataset` of the measurements, with the `capture`
          fields as attrs and `options.expand_dims` as the leading dimension
        - ``'delayed'``: a dict of `DelayedDataArray` keyed by measurement name
        - ``False``: a dict of bare arrays keyed by measurement name, in the array
          namespace of `iq`

    Raises:
        TypeError: if `spec` is not an `AnalysisGroup`
    """

    if options is None:
        options = EvaluationOptions(as_xarray=True, registry=register.registry)

    results = evaluate_by_spec(iq, spec, capture, options)

    if not options.as_xarray or options.as_xarray == 'delayed':
        return results
    else:
        return package_analysis(capture, results, expand_dims=options.expand_dims)
