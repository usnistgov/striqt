"""Implement a registry to track and wrap measurement functions that transform numpy return values
into xarray DataArray objects with labeled dimensions and coordinates.
"""

from __future__ import annotations as __

import contextlib
import functools
import typing
from fractions import Fraction
from typing import TYPE_CHECKING, Any, Callable, Literal, NamedTuple, cast

import msgspec

from .. import specs
from . import util

if TYPE_CHECKING:
    import inspect

    from striqt.waveform.lib.typing import ArrayBackend, DTypeLike

    from .dataarrays import AcquiredIQ
    from .typing import (
        RM,
        TC,
        TM,
        AnalysisFunc,
        AnalysisFuncWrapper,
        Array,
        CoordFunc,
        CoordFuncWrapper,
        Iterable,
        Measurement,
        P,
        R,
        Self,
        WrappedAnalysis,
    )
else:
    inspect = util.lazy_import('inspect')


class KwArgCache:
    """a single-element cache of a measurement result keyed on input, capture, and keywords.

    The key is the identity of the input array, the capture, and the values of the
    `fields` named at construction. The cache is disabled until entered as a context
    manager, and exit clears the entry and the callback. The cache keeps a reference
    to the input array until the entry is cleared.
    """

    _key: frozenset | None = None
    _key_iq: Array | None = None
    _value = None
    enabled = False
    _callback = None
    _callback_capture = None

    def __init__(self, fields: list[str]):
        self.name = None
        self._fields = fields
        self._callback = None

    def kw_key(self, kws: dict[str, Any]):
        if kws is None:
            return None

        kws = {k: kws[k] for k in self._fields if k in kws}

        return frozenset(kws.items())

    def clear(self):
        self._key = None
        self._key_iq = None
        self._value = None

    def lookup(self, iq: Array, kws: dict):
        if self._key is None or not self.enabled:
            return None

        if iq is self._key_iq and self.kw_key(kws) == self._key:
            return self._value
        else:
            return None

    def update(self, iq: Array, kws: dict, value):
        if not self.enabled:
            return
        self._key = self.kw_key(kws)
        self._key_iq = iq
        self._value = value

    def apply(self, func: AnalysisFunc[P, RM]) -> AnalysisFunc[P, RM]:
        """wrap `func` to return the cached result on a repeated call.

        Args:
            func: a measurement function called as ``func(x, capture, **kwargs)``

        Returns:
            a wrapper with the signature of `func`. While the cache is enabled, a
            call with the same `x` object, the same `capture`, and the same values of
            the `fields` named at construction returns the cached result. Any other
            call runs `func`, stores the result, and invokes the callback set by
            `set_callback` with the cache, the capture, the result, and the keywords.
        """

        @functools.wraps(func)
        def wrapped(
            x: Array,
            capture: specs.Capture,
            *args: P.args,
            **kwargs: P.kwargs,
        ) -> RM:
            all_kws = dict(kwargs, capture=capture)  # ty: ignore[no-matching-overload]
            match = self.lookup(x, all_kws)
            if match is not None:
                return match

            ret = func(x, capture, *args, **kwargs)
            self.update(x, all_kws, ret)
            if self._callback is None or self._callback_capture is None:
                pass
            else:
                self._callback(
                    cache=self, capture=self._callback_capture, result=ret, **kwargs
                )

            return ret

        self.name = func.__name__

        return wrapped

    def set_callback(self, func: Callable, capture=None):
        self._callback = func
        self._callback_capture = capture

    def __enter__(self):
        """enable the cache.

        Returns:
            `self`
        """
        self.enabled = True
        return self

    def __exit__(self, *exc_info):
        """disable the cache and clear the entry and the callback."""
        self.enabled = False
        self._callback = None
        self.clear()


class CoordInfo(NamedTuple):
    name: str
    func: CoordFunc
    dtype: DTypeLike
    dims: tuple[str, ...] = ()
    attrs: dict = {}  # noqa: RUF012


class CoordRegistry(dict['CoordFunc', 'CoordInfo']):
    def __call__(
        self,
        dtype: DTypeLike,
        *,
        name: str | None = None,
        dims: tuple[str, ...] | None = None,
        attrs: dict | None = None,
    ) -> CoordFuncWrapper[TC, TM, R]:
        """register the decorated function as the coordinate factory named `name`.

        The factory is called as ``func(capture, spec)`` and returns the coordinate
        values along `dims`, or None for a dimension without coordinate values.

        Args:
            dtype: numpy dtype of the coordinate values
            name: the coordinate name, or None for the `__name__` of the factory
            dims: the dimension names of the coordinate, a single name, or None
                for ``(name,)``
            attrs: attrs of the coordinate variable

        Returns:
            a decorator that registers the factory and returns it unchanged

        Raises:
            TypeError: from the returned decorator, when `name` is None and the
                factory has no `__name__`
            KeyError: from the returned decorator, when `name` is already registered
        """
        kws = locals()

        def wrapper(func: CoordFunc[TC, TM, R]) -> CoordFunc[TC, TM, R]:
            if isinstance(kws['dims'], str):
                dims = (kws['dims'],)
            else:
                dims = kws['dims']

            if kws['name'] is None:
                try:
                    name = func.__name__
                except AttributeError as ex:
                    raise TypeError(
                        'specify the coordinate name with coordinates(name, ...)'
                    ) from ex
            else:
                name = str(kws['name'])

            assert name is not None

            if name in self:
                raise KeyError(
                    f'a coordinate has already been registered for coordinate {name!r}'
                )

            if dims is None:
                dims = (name,)

            self[func] = CoordInfo(
                name=name, func=func, dims=dims, dtype=dtype, attrs=attrs or {}
            )

            return func

        return wrapper

    def __hash__(self) -> int:
        return hash(frozenset(self.items()))


class SyncInfo(NamedTuple):
    name: str
    func: WrappedAnalysis
    lag_coord_func: CoordFunc
    meas_spec_type: type[specs.Analysis]


class AlignmentSourceRegistry(dict['str | Callable', 'SyncInfo']):
    def __init__(self, registry: AnalysisRegistry) -> None:
        super().__init__()
        self._registry = registry

    def __call__(
        self,
        meas_spec_type: type[specs.Analysis],
        *,
        lag_coord_func: CoordFunc,
        name: str | None = None,
    ) -> Callable[[WrappedAnalysis[P, RM]], WrappedAnalysis[P, RM]]:
        """register the decorated measurement as a signal trigger.

        The entry is keyed by both `name` and the measurement wrapper.

        Args:
            meas_spec_type: the spec type of a measurement already registered with
                ``prefer_aligned=False``
            lag_coord_func: the coordinate factory of the lag axis of the measurement
            name: the trigger name, or None for the `__name__` of the measurement

        Returns:
            a decorator that registers the measurement wrapper and returns it
            unchanged

        Raises:
            KeyError: from the returned decorator, when `meas_spec_type` is not
                registered
            ValueError: from the returned decorator, when the measurement was
                registered with ``prefer_aligned=True``
            TypeError: from the returned decorator, when `name` is already registered
        """

        def wrapper(
            func: WrappedAnalysis[P, RM],
        ) -> WrappedAnalysis[P, RM]:
            sync_name = func.__name__ if name is None else name

            if self._registry[meas_spec_type].prefer_aligned:
                raise ValueError(
                    f'the signal_trigger measurement {func.__name__!r} must be '
                    'registered with prefer_aligned=False'
                )

            if sync_name in self:
                raise TypeError(
                    f'a signal_trigger named {sync_name} was already registered'
                )

            self[func] = self[sync_name] = SyncInfo(
                name=sync_name,
                func=func,
                lag_coord_func=lag_coord_func,
                meas_spec_type=meas_spec_type,
            )

            return func

        return wrapper


def _select_input(
    iq: Array | AcquiredIQ, *, prefer_filtered: bool, prefer_aligned: bool
) -> Array:
    """return the measurement input array selected from `iq`.

    Returns:
        `iq` when it is an array; for an `AcquiredIQ`, the waveform from `preprocess`
        with `prefer_filtered` and `prefer_aligned`
    """
    from .dataarrays import AcquiredIQ

    if isinstance(iq, AcquiredIQ):
        return iq.preprocess(
            prefer_filtered=prefer_filtered, prefer_aligned=prefer_aligned
        )
    return iq


AnalysisValidator = Callable[['specs.AnalysisCapture', Any], Any]

# func(capture: AnalysisCapture, spec, *, array_backend: ArrayBackend = 'numpy',
#      input_error: float = 0.0) -> Tolerance
# `input_error` is the relative rms amplitude error already present in the IQ, as a
# ratio
AnalysisTolerance = Callable[..., 'specs.Tolerance']


class AnalysisInfo(NamedTuple):
    name: str
    func: AnalysisFunc
    coord_factories: tuple[CoordFunc, ...]
    cache: KwArgCache | None
    dtype: str
    attrs: dict
    depends: Iterable[AnalysisFunc]
    store_compressed: bool
    prefer_filtered: bool
    prefer_aligned: bool
    dims: tuple[str, ...] | None = None
    validate: AnalysisValidator | None = None
    tolerance: AnalysisTolerance | None = None


class AnalysisRegistry(dict[type[specs.Analysis], AnalysisInfo]):
    """a registry of keyword-only arguments for decorated functions"""

    # a registry is mutated only at import time, so identity is a stable key. The
    # `__eq__` of `dict` would otherwise leave the subclass unhashable.
    __hash__ = object.__hash__

    caches: dict[AnalysisFunc, list[KwArgCache]]
    parameter_fields: dict[str, msgspec.structs.FieldInfo | None]
    parameter_defaults: dict[str, Any]

    def __init__(self):
        super().__init__()
        self.names: set[str] = set()
        self.caches = {}
        self.coordinates = CoordRegistry()
        self.signal_trigger = AlignmentSourceRegistry(self)
        self.parameter_defaults = {}
        self.parameter_fields = {}

    def measurement(
        self,
        spec_type: type[specs.Analysis],
        *,
        dtype: str,
        name: str | None = None,
        dims: tuple[str, ...] | str | None = None,
        coord_factories: Iterable[CoordFunc] | CoordFunc | None = None,
        depends: Iterable[Callable] | Callable = (),
        caches: Iterable[KwArgCache] | KwArgCache | None = None,
        store_compressed: bool = True,
        prefer_filtered: bool = True,
        prefer_aligned: bool = True,
        attrs: dict[str, Any] | None = None,
        validate: AnalysisValidator | None = None,
        tolerance: AnalysisTolerance | None = None,
    ) -> AnalysisFuncWrapper[P]:
        """register the decorated function as the measurement configured by `spec_type`.

        Registration mutates this registry when the decorator is applied. A sweep
        enables the measurement through an ``Optional[spec_type]`` field named `name`
        in an analysis group such as `specs.BuiltinAnalyses`.

        The decorated function is called as ``func(x, capture, **spec.to_dict())``
        with `x` an array of shape (port, sample), and returns an array with axes
        `dims` or an ``(array, attrs)`` tuple. The returned wrapper is called as
        ``wrapper(iq, capture, as_xarray=True, **kwargs)`` with the fields of
        `spec_type` as keywords. The `__signature__` of the wrapper and the
        ``{args}`` placeholder in the docstring are filled from `spec_type`. An
        array `iq` is passed to the function as `x`. From an `AcquiredIQ`,
        `AcquiredIQ.preprocess` selects `x` as `prefer_filtered` and
        `prefer_aligned` request. `validate` runs before the IQ is read and raises
        `specs.helpers.SpecValidationError` on a rejected combination. The wrapper
        returns, by `as_xarray`: ``True``, an `xarray.DataArray` with a leading
        ``port`` dimension; ``'delayed'``, a `DelayedDataArray`; ``False``, the
        ``(array, attrs)`` pair.

        Args:
            spec_type: the `specs.Analysis` subclass with the keyword arguments of
                the measurement as fields; one measurement per type
            dtype: numpy dtype name of the output array
            name: the output data variable name, or None for the function `__name__`
            dims: dimension names of the output axes, or None for the dimensions of
                `coord_factories`, in order
            coord_factories: functions registered with `coordinates` that produce the
                coordinate values along `dims` from ``(capture, spec)``
            depends: measurements with an intermediate this one reuses; recorded on
                the `AnalysisInfo` entry and not otherwise read
            caches: `KwArgCache` instances that `cache_context` enables for one
                capture; shared with other measurements
            store_compressed: False to store the output uncompressed in zarr
            prefer_filtered: False to measure the waveform before the
                `analysis_bandwidth` low-pass; applies only when the acquisition
                filters
            prefer_aligned: False to measure the waveform before the signal-trigger
                shift; applies only when the acquisition runs a trigger. A
                `signal_trigger` measurement must set False
            attrs: attrs of the output `xarray.DataArray`; the spec fields and any
                attrs returned by the function take precedence
            validate: a function of ``(capture, spec)`` that raises on a rejected
                combination without reading IQ; `capture` is the
                `specs.AnalysisCapture` projection, and the return value is ignored
            tolerance: a function of ``(capture, spec, *, array_backend,
                input_error)`` that returns the `specs.Tolerance` of the output
                without reading IQ

        Returns:
            a decorator that registers the decorated function and returns the wrapper

        Raises:
            ValueError: when `spec_type` is already registered
            TypeError: when a `coord_factories` entry is not callable; from the
                returned decorator, when `name` is already registered, when `caches`
                contains something other than `KwArgCache` instances, or when a field
                of `spec_type` shares a name with a field of another registered spec
                and differs in type
        """

        if isinstance(dims, str):
            dims = (dims,)

        info_kws: dict[str, Any] = {
            'name': name,
            'cache': caches,
            'dtype': dtype,
            'attrs': attrs or {},
            'store_compressed': store_compressed,
            'prefer_filtered': prefer_filtered,
            'prefer_aligned': prefer_aligned,
            'dims': dims,
            'validate': validate,
            'tolerance': tolerance,
        }

        if coord_factories is None:
            info_kws['coord_factories'] = ()
        elif callable(coord_factories):
            info_kws['coord_factories'] = (coord_factories,)
        else:
            for entry in coord_factories:
                if not callable(entry):
                    raise TypeError('coord_factories items must be callable')
            info_kws['coord_factories'] = tuple(coord_factories)

        if callable(depends):
            info_kws['depends'] = (depends,)
        else:
            info_kws['depends'] = depends

        if spec_type in self:
            name = spec_type.__qualname__
            raise ValueError(f'another measurement registered the spec {name!r}')

        def wrapper(
            func: AnalysisFunc[P, Measurement],
        ) -> WrappedAnalysis[P, Measurement]:
            @functools.wraps(func)
            def wrapped(
                iq: Array | AcquiredIQ,
                capture: specs.Capture,
                as_xarray: bool | Literal['delayed'] = True,
                *args: P.args,
                **kwargs: P.kwargs,
            ) -> Measurement:
                from .dataarrays import DelayedDataArray

                if as_xarray not in ('delayed', True, False):
                    raise ValueError(
                        'xarray argument must be one of (True, False, "delayed")'
                    )

                spec = spec_type.from_dict(kwargs)

                if validate is not None:
                    with specs.helpers.validation_path(f'.{info_kws["name"]}'):
                        validate(specs.helpers.to_analysis_capture(capture), spec)

                x = _select_input(
                    iq, prefer_filtered=prefer_filtered, prefer_aligned=prefer_aligned
                )
                ret = func(x, capture, *args, **spec.to_dict())

                data, more_attrs = normalize_factory_return(ret, name=func.__name__)

                if not as_xarray:
                    return data, more_attrs

                data = DelayedDataArray(
                    result=data,
                    capture=capture,
                    spec=spec,
                    attrs=more_attrs,
                    info=self[spec_type],
                )

                if as_xarray == 'delayed':
                    return data
                else:
                    return data.to_xarray(expand_dims=('port',))

            if info_kws['name'] is None:
                name = info_kws['name'] = func.__name__

            elif info_kws['name'] in self.names:
                raise TypeError(
                    f'a measurement named {info_kws["name"]!r} was already registered'
                )
            else:
                name = info_kws['name']
                assert isinstance(name, str)
                self.names.add(info_kws['name'])

            for field in msgspec.structs.fields(spec_type):
                if field.name is None:
                    continue
                try:
                    _, coord_default = specs.helpers.infer_coord_info(
                        field.type, allow_timestamps=False
                    )
                except TypeError:
                    self.parameter_fields[field.name] = None
                    self.parameter_defaults[field.name] = None
                    continue
                else:
                    if coord_default in self.parameter_defaults:
                        if type(coord_default) != type(self.parameter_defaults):
                            raise TypeError(
                                f'another measurement already has parameter {field.name} with a different type'
                            )
                    else:
                        self.parameter_fields[field.name] = field
                        self.parameter_defaults[field.name] = coord_default

            if caches is None:
                pass
            elif isinstance(caches, KwArgCache):
                self.caches[wrapped] = [caches]
            elif isinstance(caches, (tuple, list)) and isinstance(
                caches[0], KwArgCache
            ):
                self.caches[wrapped] = list(caches)
            else:
                raise TypeError(
                    'cache argument must be an tuple or list of KeywordArgumentCache'
                )

            self[spec_type] = AnalysisInfo(func=wrapped, **info_kws)

            # ty rejects assignment to `__signature__` on the wrapper type
            setattr(wrapped, '__signature__', _make_measurement_signature(spec_type))  # noqa: B010
            arg_doc = _make_measurement_docstring(spec_type)
            if func.__doc__ is None:
                doc = arg_doc
            elif '{args}' in func.__doc__:
                doc = func.__doc__.format(args=arg_doc)
            else:
                doc = func.__doc__
            wrapped.__doc__ = doc

            # one runtime def with `as_xarray: bool | Literal['delayed']` cannot
            # satisfy the 3-overload WrappedAnalysis Protocol, so name the type
            return cast('WrappedAnalysis[P, Measurement]', wrapped)

        return wrapper

    __call__ = measurement

    def cache_context(
        self, capture: specs.Capture, callback: Callable | None = None
    ) -> contextlib.AbstractContextManager[contextlib.ExitStack]:
        return cached_registry_context(self, capture, callback)

    def validate(self, capture: specs.Capture, analysis: specs.AnalysisGroup) -> None:
        """check each measurement in `analysis` against `capture` without IQ.

        Measurements left unset in `analysis`, and those registered without a
        `validate=` function, are skipped. Results are cached on the
        `specs.AnalysisCapture` projection of `capture`. Captures equal in that
        projection share one entry.

        Raises:
            specs.helpers.SpecValidationError: on the first invalid combination,
                with a field path rooted at ``$.analysis``
        """
        _validate_analysis_group(capture, analysis, self)

    def tolerances(
        self,
        capture: specs.Capture,
        analysis: specs.AnalysisGroup,
        *,
        array_backend: ArrayBackend = 'numpy',
        input_error: float = 0.0,
    ) -> dict[str, specs.Tolerance]:
        """return the roundoff error budget of each measurement in `analysis`.

        Results are cached on the `specs.AnalysisCapture` projection of `capture`.

        Args:
            array_backend: the array namespace of the measurement evaluation;
                selects the roundoff model
            input_error: relative rms amplitude error already present in the IQ
                handed to the measurements, as a ratio

        Returns:
            a dict of `specs.Tolerance` keyed by measurement name; measurements left
            unset in `analysis` or registered without a `tolerance=` function are
            omitted
        """
        return _analysis_group_tolerances(
            capture, analysis, self, array_backend, input_error
        )


@specs.helpers.lru_cache_on_converted(specs.AnalysisCapture, maxsize=1024)
def _validate_analysis_group(
    capture: specs.AnalysisCapture,
    analysis: specs.AnalysisGroup,
    registry: AnalysisRegistry,
) -> None:
    for name, kwargs in analysis.to_dict().items():
        if not kwargs:
            # mirror the skip in dataarrays.evaluate_by_spec: a measurement with no
            # parameters set does not run
            continue
        info = registry[type(getattr(analysis, name))]
        if info.validate is None:
            continue
        with specs.helpers.validation_path('.analysis', f'.{name}'):
            info.validate(capture, getattr(analysis, name))


@specs.helpers.lru_cache_on_converted(specs.AnalysisCapture, maxsize=1024)
def _analysis_group_tolerances(
    capture: specs.AnalysisCapture,
    analysis: specs.AnalysisGroup,
    registry: AnalysisRegistry,
    array_backend: ArrayBackend,
    input_error: float,
) -> dict[str, specs.Tolerance]:
    result: dict[str, specs.Tolerance] = {}
    for name, kwargs in analysis.to_dict().items():
        if not kwargs:
            # mirror the skip in dataarrays.evaluate_by_spec: a measurement with no
            # parameters set does not run
            continue
        info = registry[type(getattr(analysis, name))]
        if info.tolerance is None:
            continue
        result[info.name] = info.tolerance(
            capture,
            getattr(analysis, name),
            array_backend=array_backend,
            input_error=input_error,
        )
    return result


@contextlib.contextmanager
def cached_registry_context(
    registry: AnalysisRegistry,
    capture: specs.Capture,
    callback: Callable | None = None,
):
    caches: list[KwArgCache] = []
    for deps in registry.caches.values():
        caches.extend(deps)
    caches = list(set(caches))

    cm = contextlib.ExitStack()

    with cm:
        try:
            for cache in caches:
                cm.enter_context(cache)
                if callback is not None:
                    cache.set_callback(callback, capture)
            yield cm
        except:
            cm.close()
            raise


def _optional_spec_fields(group_type: type[specs.AnalysisGroup]) -> dict[str, type]:
    """return the spec type of each ``Optional[spec] = None`` field of `group_type`.

    Returns:
        the spec type of each field, keyed by field name, in field order

    Raises:
        TypeError: when a field is not an optional spec with default None, or when
            `group_type` is defined without ``omit_defaults=True``
    """
    actual: dict[str, type] = {}
    for field in msgspec.structs.fields(group_type):
        args = set(typing.get_args(field.type))
        if type(None) not in args or len(args) != 2 or field.default is not None:
            raise TypeError(
                f'{group_type.__qualname__}.{field.name} must be an Optional spec type '
                f'with default None, not {field.type!r}'
            )
        (actual[field.name],) = args - {type(None)}
    if not group_type.__struct_config__.omit_defaults:
        raise TypeError(
            f'{group_type.__qualname__} must be defined with omit_defaults=True'
        )
    return actual


def check_group_fields(
    group_type: type[specs.AnalysisGroup], expected: dict[str, type[specs.Analysis]]
) -> None:
    """raise unless `group_type` has one ``Optional[spec] = None`` field per `expected` entry.

    Field order is not checked.

    Args:
        group_type: the struct to check
        expected: the spec type of each measurement, keyed by registered name

    Raises:
        TypeError: when a field of `group_type` is not an optional spec with default
            None, when `group_type` is defined without ``omit_defaults=True``, or
            when the fields differ from `expected` in name or spec type
    """
    actual = _optional_spec_fields(group_type)
    if actual != expected:
        missing = sorted(expected.keys() - actual.keys())
        extra = sorted(actual.keys() - expected.keys())
        shared = actual.keys() & expected.keys()
        mismatched = sorted(k for k in shared if actual[k] is not expected[k])
        raise TypeError(
            f'{group_type.__qualname__} does not match the registry: missing fields '
            f'{missing!r}, unregistered fields {extra!r}, wrong spec type {mismatched!r}'
        )


def check_group_registered(
    group_type: type[specs.AnalysisGroup], registry: AnalysisRegistry
) -> None:
    """raise unless every field of `group_type` names a measurement of `registry`.

    Each field must be ``Optional[spec] = None`` with `spec` a key of `registry`, and
    the field name must equal the registered name of that measurement. Any subset of
    the registered measurements is accepted.

    Args:
        group_type: the struct to check
        registry: the registry that must contain every spec type of `group_type`

    Raises:
        TypeError: when a field of `group_type` is not an optional spec with default
            None, when `group_type` is defined without ``omit_defaults=True``, when a
            spec type is not registered, or when a field name differs from the
            registered name
    """
    for name, spec_type in _optional_spec_fields(group_type).items():
        info = registry.get(spec_type)
        if info is None:
            raise TypeError(
                f'{group_type.__qualname__}.{name}: {spec_type.__qualname__} is not a '
                'registered measurement spec type'
            )
        if info.name != name:
            raise TypeError(
                f'{group_type.__qualname__}.{name} must be named {info.name!r}, the '
                f'registered name of {spec_type.__qualname__}'
            )


class Trigger:
    def __init__(
        self,
        name_or_func: str | Callable,
        spec: specs.Analysis,
        registry: AnalysisRegistry,
    ):
        self.info: SyncInfo = registry.signal_trigger[name_or_func]
        self.meas_info = registry[self.info.meas_spec_type]

        if isinstance(spec, self.info.meas_spec_type):
            self.meas_spec = spec
        else:
            expect_type = self.info.meas_spec_type.__qualname__
            raise TypeError(f'spec must be an instance of {expect_type}')

    @classmethod
    def from_spec(
        cls, name: str, analysis: specs.AnalysisGroup, registry: AnalysisRegistry
    ) -> Self:
        info: SyncInfo = registry.signal_trigger[name]
        meas_info = registry[info.meas_spec_type]

        meas_attr = getattr(analysis, meas_info.name, None)
        meas_spec = cast(specs.Analysis, meas_attr)
        if meas_spec is None:
            raise ValueError(
                f'signal_trigger {name!r} requires an analysis specification for {meas_info.name!r}'
            )

        return cls(name, meas_spec, registry)

    def __call__(self, iq: Array | AcquiredIQ, capture: specs.Capture) -> Array:
        meas_kws = self.meas_spec.to_dict()
        ret = self.info.func(iq, capture, as_xarray=False, **meas_kws)
        return ret[0]

    def max_lag(self, capture: specs.Capture) -> int:
        lags = self.info.lag_coord_func(capture, self.meas_spec)
        step = lags[1] - lags[0]
        return step * len(lags)


def get_signal_trigger_measurement_name(name: str, registry: AnalysisRegistry) -> str:
    info: SyncInfo = registry.signal_trigger[name]
    meas_info = registry[info.meas_spec_type]
    return meas_info.name


def normalize_factory_return(ret, name: str) -> tuple[Array, dict[str, Any]]:
    """normalize the coordinate and data factory returns into (data, metadata)"""

    if not isinstance(ret, tuple):
        arr = ret
        attrs = {}
    elif len(ret) == 2:
        arr, attrs = ret
    else:
        raise TypeError(
            f'tuple returned by {name!r} coordinate factory must have length 2, not {len(ret)}. return a list or array if this was meant as data.'
        )

    if not isinstance(attrs, dict):
        raise TypeError(
            f'second item of {name!r} coordinate factory return tuple must be dict.return an array or list if this was meant as data.'
        )
    else:
        attrs = {
            k: (str(v) if isinstance(v, Fraction) else v) for k, v in attrs.items()
        }

    return arr, attrs


registry: AnalysisRegistry = AnalysisRegistry()


@util.lru_cache()
def _make_measurement_signature(spec_cls):
    """
    Generates an inspect.Signature object from a specs.Analysis subclass.
    """
    kw_parameters = inspect.signature(spec_cls).parameters

    parameters = [
        inspect.Parameter(
            'iq',
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation='Array | AcquiredIQ',
        ),
        inspect.Parameter(
            'capture',
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=specs.Capture,
        ),
    ]

    return inspect.Signature(parameters + list(kw_parameters.values()))


def _make_measurement_docstring(spec_cls):
    extra_prepend = {
        'iq': 'IQ waveform of shape (port, sample), or an `AcquiredIQ` from which the conditioned waveform is selected',
        'capture': 'the acquisition parameters of `iq`',
        'as_xarray': "True for an `xarray.DataArray`, False for an ``(array, attrs)`` tuple, or 'delayed' for a `DelayedDataArray`",
    }
    extra_types = {
        'iq': 'Array or AcquiredIQ',
        'capture': 'striqt.analysis.specs.Capture',
        'as_xarray': "(bool or 'delayed')",
    }
    return specs.doc.struct_args_docstring(
        spec_cls, extra_prepend=extra_prepend, extra_types=extra_types
    )
