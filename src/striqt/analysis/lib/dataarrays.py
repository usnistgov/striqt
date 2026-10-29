"""wrap lower-level striqt.waveform DSP calls to accept physical inputs and return xarray.DataArray"""

from __future__ import annotations as __

import collections
import dataclasses
import math
import typing
import weakref

import msgspec

import striqt.waveform as sw

from .. import specs
from . import register, util
from .typing import TAR
from .util import array_api_compat, cached_property, np, xr

if typing.TYPE_CHECKING:
    from .typing import Array, CoordFunc


CAPTURE_DIM = 'capture'
PORT_DIM = 'port'


class Preprocessor:
    """the conditioned-waveform selector of one acquisition.

    The base class returns `x` of the acquisition for every preference. A subclass
    that conditions the waveform lazily overrides `__call__`.
    """

    def __init__(self, iq: AcquiredIQ) -> None:
        """store a weak reference to `iq`."""
        self._iq_ref: weakref.ref[AcquiredIQ] = weakref.ref(iq)

    @property
    def iq(self) -> AcquiredIQ:
        """the acquisition that owns this preprocessor.

        Raises:
            RuntimeError: when the acquisition has been garbage collected
        """
        iq = self._iq_ref()
        if iq is None:
            raise RuntimeError('the acquisition of this preprocessor was released')
        return iq

    def __call__(
        self, *, prefer_filtered: bool = True, prefer_aligned: bool = True
    ) -> Array:
        """return the waveform conditioned as the preferences request.

        Args:
            prefer_filtered: True selects the waveform low-passed to the capture
                `analysis_bandwidth` when the acquisition applies one
            prefer_aligned: True selects the waveform shifted by the signal trigger
                when the acquisition runs one

        Returns:
            the shape (port, sample) array in the namespace of `x`

        Raises:
            RuntimeError: when the acquisition has been garbage collected
        """
        return self.iq.x


@dataclasses.dataclass
class AcquiredIQ:
    """an acquired IQ waveform `x` with the capture that describes the acquisition."""

    x: Array
    """the acquired waveform with shape (port, sample)"""
    capture: specs.Capture | None
    """the acquisition parameters of `x`, or None when not known"""

    @cached_property
    def preprocess(self) -> Preprocessor:
        """return the `Preprocessor` of this acquisition, one per instance.

        A `dataclasses.replace` copy starts with a `Preprocessor` of its own.
        """
        return Preprocessor(self)


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
    # a mantissa that formats as 1000 (e.g., 999.99) is carried into the next prefix,
    # except at the largest prefix in `_ENG_PREFIXES`
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
) -> xr.DataArray:
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
        stub = stub.expand_dims(dict.fromkeys(expand_dims, 0), axis=0)

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


def _restore_padded_coord_dtype(da: xr.DataArray, info: register.CoordInfo) -> None:
    """restore the integer or bool dtype declared for coordinate `info.name` of `da`.

    The coordinate is replaced in place with zeros of `info.dtype` when `info.dtype`
    is an integer or bool dtype and the coordinate dtype differs. Other coordinates
    are left unchanged.
    """

    dtype = np.dtype(info.dtype)
    coord = da[info.name]

    if dtype.kind not in 'iub' or coord.dtype == dtype:
        return

    # `DataArray.pad` fills with nan and promotes integer and bool coordinates to float
    da[info.name] = xr.zeros_like(coord, dtype=dtype)


def build_dataarray(
    delayed: DelayedDataArray,
    *,
    expand_dims=None,
) -> xr.DataArray:
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
    except ValueError:
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
    """the result of one measurement before conversion to `xarray.DataArray`."""

    capture: specs.Capture
    spec: specs.Analysis
    result: Array | dict[str, Array] | xr.Dataset | dict[str, DelayedDataArray]
    info: register.AnalysisInfo
    attrs: dict

    def compute(self) -> DelayedDataArray:
        """return a copy with `result` materialized to numpy.

        A cupy or torch `result` is transferred from the device here; a numpy
        `result` is shared.

        Returns:
            a `DelayedDataArray` with the same `capture`, `spec`, `info`, and `attrs`

        Raises:
            TypeError: when `result` is not a numpy, cupy, or torch array
        """
        return DelayedDataArray(
            # datacls=self.datacls,
            capture=self.capture,
            spec=self.spec,
            result=_results_as_arrays(self.result),
            info=self.info,
            attrs=self.attrs,
        )

    def to_xarray(self, expand_dims=None) -> xr.DataArray:
        """return the `xarray.DataArray` of `result` with the coordinates of `info`.

        Args:
            expand_dims: names of leading dimensions added ahead of the measurement
                dimensions, or None for none

        Returns:
            a `xarray.DataArray` named `info.name`, with the attrs of `info`, the
            fields of `spec`, and `attrs`

        Raises:
            ValueError: when the shape of `result` does not match the dimensions or a
                coordinate factory returns the wrong shape
            TypeError: when `result` is a dict
        """
        return build_dataarray(self, expand_dims=expand_dims)


class EvaluationOptions(msgspec.Struct, typing.Generic[TAR]):
    """the result type and registry for one evaluation of an analysis group.

    `as_xarray` selects the result type of `analyze_by_spec`: ``True`` for an
    `xarray.Dataset`, ``'delayed'`` for `DelayedDataArray` results, or ``False``
    for the bare arrays in the array namespace of the input. `registry` supplies
    the measurement function of each spec type. `block_each` True materializes
    each result to numpy as soon as the measurement returns, when
    `as_xarray` is ``True`` or ``'delayed'``. `expand_dims` names the leading
    dimension of each `xarray.DataArray` when `as_xarray` is ``True``, and that
    dimension indexes the port axis of the input.

    Raises:
        TypeError: when `as_xarray` is not ``True``, ``False``, or ``'delayed'``
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
) -> dict[str, DelayedDataArray] | dict[str, Array]:
    """evaluate each enabled measurement of `spec` on the IQ waveform.

    Args:
        iq: the waveform with shape (port, sample), or an `AcquiredIQ` from which
            each measurement selects the conditioned waveform
        spec: an `AnalysisGroup` such as `specs.BuiltinAnalyses`; the ``None``
            fields are skipped
        capture: the acquisition parameters of `iq`
        options: the result type and the registry of the measurement functions

    Returns:
        a dict keyed by measurement name, in the field order of `spec`: a
        `DelayedDataArray` per measurement when `options.as_xarray` is ``True`` or
        ``'delayed'``, materialized to numpy when `options.block_each` is set; the
        bare array of each measurement in the array namespace of `iq`, without
        attrs, when ``False``

    Raises:
        TypeError: when `spec` is not an `AnalysisGroup`
        KeyError: when the spec type of an enabled field is not in
            `options.registry`
        specs.helpers.SpecValidationError: when a measurement validator rejects
            the ``(capture, spec)`` combination
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

    for name in spec_dict:
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
) -> xr.Dataset:
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
) -> xr.Dataset: ...


@typing.overload
def analyze_by_spec(
    iq: Array | AcquiredIQ,
    spec: specs.AnalysisGroup,
    capture: specs.Capture,
    options: EvaluationOptions[typing.Literal['delayed']],
) -> dict[str, DelayedDataArray]: ...


@typing.overload
def analyze_by_spec(
    iq: Array | AcquiredIQ,
    spec: specs.AnalysisGroup,
    capture: specs.Capture,
    options: EvaluationOptions[typing.Literal[False]],
) -> dict[str, Array]: ...


def analyze_by_spec(
    iq: Array | AcquiredIQ,
    spec: specs.AnalysisGroup,
    capture: specs.Capture,
    options: EvaluationOptions | None = None,
) -> Array | dict[str, Array] | xr.Dataset | dict[str, DelayedDataArray]:
    """evaluate the enabled measurements of `spec` on the IQ waveform.

    Args:
        iq: the waveform with shape (port, sample), or an `AcquiredIQ` from which
            each measurement selects the conditioned waveform
        spec: an `AnalysisGroup` such as `specs.BuiltinAnalyses`; the ``None``
            fields are skipped
        capture: the acquisition parameters of `iq`
        options: the result type and the registry, or ``None`` for the default
            registry with `as_xarray` ``True``

    Returns:
        by `options.as_xarray`:

        - ``True``: an `xarray.Dataset` of the measurements, with the `capture`
          fields as attrs and `options.expand_dims` as the leading dimension
        - ``'delayed'``: a dict of `DelayedDataArray` keyed by measurement name
        - ``False``: a dict of bare arrays keyed by measurement name, in the array
          namespace of `iq`

    Raises:
        TypeError: when `spec` is not an `AnalysisGroup`
        KeyError: when the spec type of an enabled field is not in
            `options.registry`
        specs.helpers.SpecValidationError: when a measurement validator rejects
            the ``(capture, spec)`` combination
    """

    if options is None:
        options = EvaluationOptions(as_xarray=True, registry=register.registry)

    results = evaluate_by_spec(iq, spec, capture, options)

    if not options.as_xarray or options.as_xarray == 'delayed':
        return results
    else:
        return package_analysis(capture, results, expand_dims=options.expand_dims)
