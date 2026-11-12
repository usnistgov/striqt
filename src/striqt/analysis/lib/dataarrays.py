"""wrap lower-level striqt.waveform DSP calls to accept physical inputs and return xarray.DataArray"""

from __future__ import annotations as __

import collections
import dataclasses
import typing
import weakref

import msgspec

import striqt.waveform as sw

from .. import specs
from . import register, util
from .typing import TAR
from .util import array_api_compat, cached_property, np, xr

if typing.TYPE_CHECKING:
    from striqt.waveform.lib.typing import DTypeLike

    from .typing import Array, CoordFunc


CAPTURE_DIM = 'capture'
PORT_DIM = 'port'


class Preprocessor:
    """the conditioned-waveform selector of one acquisition.

    A subclass that conditions the waveform overrides `__call__`.
    """

    def __init__(self, iq: AcquiredIQ) -> None:
        """store a weak reference to `iq`."""
        self._iq_ref: weakref.ref[AcquiredIQ] = weakref.ref(iq)

    @property
    def iq(self) -> AcquiredIQ:
        """the acquisition for which the preprocessor selects a waveform.

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
        """return the waveform selected by `prefer_filtered` and `prefer_aligned`.

        The base implementation returns `x` of the acquisition for every preference.

        Args:
            prefer_filtered: True prefers the waveform low-passed to the
                `analysis_bandwidth` of the capture over the unfiltered waveform
            prefer_aligned: True prefers the waveform shifted by the signal trigger
                over the unaligned waveform

        Returns:
            the shape (port, sample) array in the namespace of `x`

        Raises:
            RuntimeError: when the acquisition has been garbage collected
        """
        return self.iq.x


@dataclasses.dataclass
class AcquiredIQ:
    """an acquired IQ waveform with the parameters and power unit of the acquisition."""

    x: Array
    """the acquired waveform with shape (port, sample)"""
    capture: specs.Capture | None
    """the acquisition parameters of `x`, or None when not known"""
    power_units: sw.units.PowerUnits
    """the unit of the squared magnitude of `x`: ``'mW'`` when calibrated, ``'FS'``
    for full scale, ``'a.u.'`` for arbitrary units"""

    @cached_property
    def preprocess(self) -> Preprocessor:
        """return the `Preprocessor` of this acquisition.

        The `Preprocessor` is created on the first access and kept on the instance.
        A copy from `dataclasses.replace` is a separate instance with a
        `Preprocessor` of its own.
        """
        return Preprocessor(self)


def _results_as_arrays(
    obj: Array | dict[str, Array] | xr.Dataset | dict[str, DelayedDataArray],
) -> Array:
    """return `obj` as a numpy array.

    A cupy or torch array is copied from the device. A numpy array is returned
    unchanged.

    Raises:
        TypeError: when `obj` is not a numpy, cupy, or torch array
    """
    # the array predicates narrow the type only when the backend is importable
    array: typing.Any = obj
    if array_api_compat.is_torch_array(obj):
        return array.cpu()
    elif sw.is_cupy_array(obj):
        return array.get()
    elif array_api_compat.is_numpy_array(obj):
        return obj
    raise TypeError(f'obj type {type(obj)} is unrecognized')


def _empty_stub(dims, dtype, attrs: dict[str, typing.Any] | None = None):
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
    is an integer or bool dtype and the coordinate dtype differs.
    """

    dtype = np.dtype(info.dtype)
    coord = da[info.name]

    if dtype.kind not in 'iub' or coord.dtype == dtype:
        return

    # `DataArray.pad` fills with nan and promotes integer and bool coordinates to float
    da[info.name] = xr.zeros_like(coord, dtype=dtype)


def _coord_storage_dtype(dtype: DTypeLike) -> np.dtype:
    """return the dtype of the stored values of a coordinate registered with `dtype`.

    A bool, integer, float, or complex `dtype` is returned unchanged. Any other
    registered dtype, such as ``'str'`` or `object`, is stored as `object`.
    """
    dtype = np.dtype(dtype)
    if dtype.kind in 'biufc':
        return dtype
    return np.dtype(object)


def measurement_dims(info: register.AnalysisInfo) -> tuple[str, ...]:
    """return the dimension names of the measurement `info` after the port axis.

    Returns:
        `info.dims` when registered and not empty, otherwise the dimensions declared
        by `info.coord_factories`, each at the first factory that declares it
    """
    if info.dims is None or info.dims == ():
        return _infer_coord_dims(info.coord_factories)
    return info.dims


def measurement_attrs(
    info: register.AnalysisInfo,
    spec: specs.Analysis,
    body_attrs: typing.Mapping[str, typing.Any],
    power_units: sw.units.PowerUnits,
) -> dict[str, typing.Any]:
    """return the attrs of the output of the measurement `info` for one acquisition.

    The registered attrs of `info`, the fields of `spec`, and `body_attrs` are merged
    in that order. A key set in several sources takes the value from the last source.
    A `units` entry in a power form (`striqt.waveform.units.DB_POWER`, `POWER`, or
    `AMPLITUDE`) is resolved into `power_units`.

    Args:
        info: the registration of the measurement
        spec: the measurement parameters
        body_attrs: the attrs returned by the measurement body
        power_units: the power unit of the acquisition

    Returns:
        a new dict of the merged attrs, with the inputs unchanged
    """
    attrs = {**info.attrs, **spec.to_dict(), **body_attrs}
    return _resolve_units(attrs, power_units)


def _resolve_units(
    attrs: dict[str, typing.Any], power_units: sw.units.PowerUnits
) -> dict[str, typing.Any]:
    units = attrs.get('units', None)
    if not isinstance(units, str) or not sw.units.has_power_form(units):
        return attrs
    return {**attrs, 'units': sw.units.resolve_power_units(units, power_units)}


def measurement_coords(
    info: register.AnalysisInfo,
    capture: specs.Capture,
    spec: specs.Analysis,
    power_units: sw.units.PowerUnits | None = None,
) -> dict[str, tuple[np.ndarray, dict[str, typing.Any]]]:
    """return the coordinate values and attrs of the measurement `info` for one capture.

    Args:
        info: the registration of the measurement
        capture: the acquisition parameters that set the coordinate values
        spec: the measurement parameters that set the coordinate values
        power_units: the power unit that resolves a `units` attr in a power form, or
            None to leave the form fields in place

    Returns:
        ``(values, attrs)`` keyed by coordinate name, in the order of
        `info.coord_factories`. `values` is a numpy array with the registered
        coordinate dtype when that is a bool, integer, float, or complex dtype, and
        an `object` array otherwise. `attrs` is the registered attrs of the
        coordinate updated by the attrs that the factory returns.

    Raises:
        TypeError: when a factory returns a tuple that is not ``(values, attrs)``
        ValueError: when the values of a factory cannot be cast to the registered
            coordinate dtype
    """
    coords: dict[str, tuple[np.ndarray, dict[str, typing.Any]]] = {}

    for factory in info.coord_factories:
        coord_info = register.registry.coordinates[factory]
        qualname = f'{info.name}.{coord_info.name}'
        ret, attrs = register.normalize_factory_return(factory(capture, spec), qualname)

        try:
            values = np.asarray(ret, dtype=_coord_storage_dtype(coord_info.dtype))
        except (TypeError, ValueError) as ex:
            raise ValueError(f'in coordinate {qualname!r}, {ex}') from ex

        merged = coord_info.attrs | attrs
        if power_units is not None:
            merged = _resolve_units(merged, power_units)
        coords[coord_info.name] = (values, merged)

    return coords


def measurement_shape(
    info: register.AnalysisInfo, capture: specs.Capture, spec: specs.Analysis
) -> tuple[int, ...]:
    """return the shape of the measurement `info` after the port axis for one capture.

    Args:
        info: the registration of the measurement
        capture: the acquisition parameters that set the coordinate values
        spec: the measurement parameters that set the coordinate values

    Returns:
        one length per dimension of `measurement_dims(info)`, taken from the shape
        of the coordinate values along that dimension

    Raises:
        NotImplementedError: when a dimension of the measurement has no coordinate
            factory
        TypeError: when a factory returns a tuple that is not ``(values, attrs)``
        ValueError: when the values of a factory cannot be cast to the registered
            coordinate dtype
    """
    coords = measurement_coords(info, capture, spec)

    sizes: dict[str, int] = {}
    for factory in info.coord_factories:
        coord_info = register.registry.coordinates[factory]
        values, _ = coords[coord_info.name]
        sizes.update(zip(coord_info.dims, values.shape))

    shape = []
    for dim in measurement_dims(info):
        if dim not in sizes:
            raise NotImplementedError(
                f'dimension {dim!r} of measurement {info.name!r} has no coordinate factory'
            )
        shape.append(sizes[dim])

    return tuple(shape)


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

    coords = measurement_coords(
        delayed.info, delayed.capture, delayed.spec, delayed.power_units
    )

    for coord_factory in delayed.info.coord_factories:
        factory_info = register.registry.coordinates[coord_factory]
        _restore_padded_coord_dtype(da, factory_info)
        coord = da[factory_info.name]
        values, attrs = coords[factory_info.name]
        qualname = f'{delayed.info.name}.{factory_info.name}'

        try:
            idx = coord.indexes[factory_info.dims[0]]
            idx._values[:] = values
        except ValueError as ex:
            exc = ex
        else:
            exc = None

        if exc is not None:
            _reraise_coord_error(
                exc=exc,
                coord=coord,
                factory_info=factory_info,
                data=values,
                name=qualname,
            )

        da[factory_info.name] = coord.assign_attrs(attrs)

    return da.assign_attrs(
        measurement_attrs(
            delayed.info, delayed.spec, delayed.attrs, delayed.power_units
        )
    )


@util.lru_cache()
def _infer_coord_dims(coord_factories: tuple[CoordFunc, ...]) -> tuple[str, ...]:
    return register.coordinate_dims(register.registry, coord_factories)


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
    power_units: sw.units.PowerUnits = 'a.u.'
    """the power unit of the acquisition: ``'mW'``, ``'FS'``, or ``'a.u.'``"""

    def compute(self) -> DelayedDataArray:
        """return a copy with `result` materialized to numpy.

        A cupy or torch `result` is copied from the device. A numpy `result` is
        shared with the copy.

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
            power_units=self.power_units,
        )

    def to_xarray(self, expand_dims=None) -> xr.DataArray:
        """return the `xarray.DataArray` of `result` with the coordinates of `info`.

        Args:
            expand_dims: names of leading dimensions added ahead of the measurement
                dimensions, or None for none

        Returns:
            an `xarray.DataArray` without a name, with the attrs of `info`, the
            fields of `spec`, and `attrs`, and with a power-form `units` attr
            resolved into `power_units`

        Raises:
            ValueError: when the shape of `result` does not match the dimensions, or
                when the values of a coordinate factory differ in shape from the
                data along that dimension
            TypeError: when `result` is a dict
        """
        return build_dataarray(self, expand_dims=expand_dims)


class EvaluationOptions(msgspec.Struct, typing.Generic[TAR]):
    """the result type and registry for one evaluation of an analysis group.

    Raises:
        TypeError: when `as_xarray` is not ``True``, ``False``, or ``'delayed'``
    """

    as_xarray: TAR
    """the result type of `analyze_by_spec`: ``True`` for an `xarray.Dataset`,
    ``'delayed'`` for `DelayedDataArray` results, or ``False`` for the bare arrays
    in the array namespace of the input"""
    registry: register.AnalysisRegistry
    """the registry that supplies the measurement function of each spec type"""
    block_each: bool = True
    """True copies each result to numpy as the measurement returns, False after
    the last measurement. The field is read when `as_xarray` is ``True`` or
    ``'delayed'``."""
    expand_dims: typing.Sequence[str] = (PORT_DIM,)
    """the name of the leading dimension of each `xarray.DataArray` when
    `as_xarray` is ``True``. That dimension indexes the port axis of the input."""

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

    A measurement with a spec type that has no fields is skipped.

    Args:
        iq: the waveform with shape (port, sample), or an `AcquiredIQ` from which
            each measurement selects the conditioned waveform
        spec: an `AnalysisGroup` such as `specs.BuiltinAnalyses`, with the ``None``
            fields skipped
        capture: the acquisition parameters of `iq`
        options: the result type and the registry of the measurement functions

    Returns:
        a dict keyed by measurement name, in the field order of `spec`, by
        `options.as_xarray`:

        - ``True`` or ``'delayed'``: a `DelayedDataArray` per measurement with
          `result` copied to numpy, each as the measurement returns when
          `options.block_each` is True, otherwise after the last measurement
        - ``False``: the bare array of each measurement in the array namespace of
          `iq`, without attrs

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

    A measurement with a spec type that has no fields is skipped.

    Args:
        iq: the waveform with shape (port, sample), or an `AcquiredIQ` from which
            each measurement selects the conditioned waveform
        spec: an `AnalysisGroup` such as `specs.BuiltinAnalyses`, with the ``None``
            fields skipped
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
