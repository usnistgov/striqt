"""Implement a registry to track and wrap measurement functions that transform numpy return values
into xarray DataArray objects with labeled dimensions and coordinates.
"""

from __future__ import annotations as __

import collections
import contextlib
import functools
import textwrap
from typing import Any, Callable, cast, Literal, NamedTuple, TYPE_CHECKING, Union
from fractions import Fraction

import msgspec

from .. import specs
from . import util

if TYPE_CHECKING:
    import inspect
    from .typing import (
        AnalysisFunc,
        AnalysisFuncWrapper,
        Array,
        CoordFunc,
        CoordFuncWrapper,
        Iterable,
        Measurement,
        P,
        RM,
        Self,
        WrappedAnalysis,
        WrappedCoord,
    )
else:
    inspect = util.lazy_import('inspect')


class KwArgCache:
    """A single-element cache of measurement results.

    It is keyed on captures and keyword arguments.

    """

    _key: frozenset | None = None
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
        self._value = None

    def lookup(self, kws: dict):
        if self._key is None or not self.enabled:
            return None

        if self.kw_key(kws) == self._key:
            return self._value
        else:
            return None

    def update(self, kws: dict, value):
        if not self.enabled:
            return
        self._key = self.kw_key(kws)
        self._value = value

    def apply(self, func: 'AnalysisFunc[P, RM]') -> 'AnalysisFunc[P, RM]':
        @functools.wraps(func)
        def wrapped(
            iq: Array,
            capture: specs.Capture,
            *args: P.args,
            **kwargs: P.kwargs,
        ) -> RM:
            all_kws = dict(kwargs, capture=capture)  # ty: ignore
            match = self.lookup(all_kws)
            if match is not None:
                return match

            ret = func(iq, capture, *args, **kwargs)
            self.update(all_kws, ret)
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
        self.enabled = True
        return self

    def __exit__(self, *exc_info):
        self.enabled = False
        self._callback = None
        self.clear()


class CoordInfo(NamedTuple):
    name: str
    func: CoordFunc
    dtype: str
    dims: tuple[str, ...] = ()
    attrs: dict = {}


class CoordRegistry(dict['CoordFunc', 'CoordInfo']):
    def __call__(
        self,
        dtype,
        *,
        name: str | None = None,
        dims: tuple[str, ...] | None = None,
        attrs={},
    ) -> CoordFuncWrapper:
        """register a coordinate factory function.

        The factory function should return an iterable containing coordinate
        values, or None to indicate a named placeholder for a dimension.
        """
        kws = locals()

        def wrapper(func):
            if isinstance(kws['dims'], str):
                dims = tuple((kws['dims'],))
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
                name=name, func=func, dims=dims, dtype=dtype, attrs=attrs
            )

            return func

        return wrapper

    def __hash__(self):  # pyright: ignore
        return hash(frozenset(self.items()))


class SyncInfo(NamedTuple):
    name: str
    func: Callable
    lag_coord_func: CoordFunc
    meas_spec_type: type[specs.Analysis]


class AlignmentSourceRegistry(dict['str | Callable', 'SyncInfo']):
    def __call__(
        self,
        meas_spec_type: type[specs.Analysis],
        *,
        lag_coord_func: CoordFunc,
        name=None,
    ) -> AnalysisFuncWrapper:
        """register a coordinate factory function.

        The proper dimension to evaluate in the data is determined from
        the supplied coordinate factory function.

        Arguments:
            coord_factory: the coordinate factory used to define the measurement.
        """
        info_kws = {
            'name': name,
            'lag_coord_func': lag_coord_func,
            'meas_spec_type': meas_spec_type,
        }

        def wrapper(
            func: AnalysisFunc[P, RM],
        ) -> WrappedAnalysis[P, RM]:
            if info_kws['name'] is None:
                info_kws['name'] = func.__name__

            if info_kws['name'] in self:
                raise TypeError(
                    f'a signal_trigger named {info_kws["name"]} was already registered'
                )

            self[func] = self[info_kws['name']] = SyncInfo(func=func, **info_kws)

            return func  # type: ignore

        return wrapper

    def to_spec(
        self, base: type[specs.AnalysisGroup] = specs.AnalysisGroup
    ) -> type[specs.AnalysisGroup]:
        return to_analysis_spec_type(self, base)


PreferIQSource = Literal['aligned', 'pre_filter', 'pre_align']


class AnalysisInfo(NamedTuple):
    name: str
    func: AnalysisFunc
    coord_factories: tuple[CoordFunc, ...]
    prefer_iq_source: PreferIQSource
    cache: KwArgCache | None
    dtype: str
    attrs: dict
    depends: Iterable[AnalysisFunc]
    store_compressed: bool
    dims: tuple[str, ...] | None = None


class AnalysisRegistry(dict[type[specs.Analysis], AnalysisInfo]):
    """a registry of keyword-only arguments for decorated functions"""

    caches: dict[AnalysisFunc, list[KwArgCache]]
    parameter_fields: dict[str, 'msgspec.structs.FieldInfo|None']
    parameter_defaults: dict[str, Any]

    def __init__(self):
        super().__init__()
        self.depends_on: dict[Callable, set[Callable]] = {}
        self.names: set[str] = set()
        self.caches = {}
        self.use_unaligned_input: set[Callable] = set()
        self.coordinates = CoordRegistry()
        self.signal_trigger = AlignmentSourceRegistry()
        self.parameter_defaults = {}
        self.parameter_fields = {}

    def __or__(self, other):
        result = super().__or__(other)
        assert isinstance(result, AnalysisRegistry)
        result.depends_on = self.depends_on | other.depends_on
        result.names = self.names | other.names
        result.caches = self.caches | other.caches
        result.use_unaligned_input = (
            self.use_unaligned_input | other.use_unaligned_input
        )
        result.coordinates = self.coordinates | other.coordinates
        result.signal_trigger = self.signal_trigger | other.signal_trigger
        return result

    def measurement(
        self,
        spec_type: type[specs.Analysis],
        *,
        dtype: str,
        name: str | None = None,
        dims: tuple[str, ...] | str | None = None,
        coord_factories: Iterable[WrappedCoord] | WrappedCoord | None = None,
        depends: Iterable[Callable] | Callable = [],
        caches: Iterable[KwArgCache] | KwArgCache | None = None,
        prefer_iq_source: PreferIQSource = 'aligned',
        store_compressed=True,
        attrs={},
    ) -> AnalysisFuncWrapper:
        """add decorated `func` and its keyword arguments in the self.tostruct() schema"""

        if isinstance(dims, str):
            dims = (dims,)

        info_kws: dict[str, Any] = dict(
            name=name,
            prefer_iq_source=prefer_iq_source,
            cache=caches,
            dtype=dtype,
            attrs=attrs,
            store_compressed=store_compressed,
            dims=dims,
        )

        if coord_factories is None:
            info_kws['coord_factories'] = tuple()
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
                iq: Array,
                capture: specs.Capture,
                as_xarray: bool = True,
                *args: P.args,
                **kwargs: P.kwargs,
            ) -> Measurement:
                from .dataarrays import DelayedDataArray

                # handle the additional argument 'as_xarray' that allows
                # the return of a _DelayedDataArray result for fast serialization and
                # xarray object instantiation
                if as_xarray not in ('delayed', True, False):
                    raise ValueError(
                        'xarray argument must be one of (True, False, "delayed")'
                    )

                spec = spec_type.from_dict(kwargs)
                ret = func(iq, capture, *args, **spec.to_dict())

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
                    # TODO: this should
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

            self.depends_on[wrapped] = set()
            for dep in info_kws['depends']:
                self.depends_on[dep].add(wrapped)

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

            setattr(wrapped, '__signature__', _make_measurement_signature(spec_type))
            doc = f'{wrapped.__doc__}\n{_make_measurement_docstring(spec_type)}'
            setattr(wrapped, '__doc__', doc)

            return wrapped  # type: ignore

        return wrapper

    __call__ = measurement

    def tospec(
        self, base: type[specs.AnalysisGroup] = specs.AnalysisGroup
    ) -> type[specs.AnalysisGroup]:
        return to_analysis_spec_type(self, base)

    def cache_context(self, capture: specs.Capture, callback: Callable | None = None):
        return cached_registry_context(self, capture, callback)


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


def to_analysis_spec_type(
    registry: AlignmentSourceRegistry | AnalysisRegistry,
    base: type[specs.AnalysisGroup] = specs.AnalysisGroup,
) -> type[specs.AnalysisGroup]:
    """return a Struct subclass type representing a specification for calls to all registered functions"""

    fields = [
        (info.name, Union[struct_type, None], None)
        for struct_type, info in registry.items()
    ]

    cls = msgspec.defstruct(
        'Analysis',
        cast(list[tuple[str, type, Any]], fields),
        bases=(base,),
        kw_only=True,
        omit_defaults=True,
        frozen=True,
    )

    return cast(type[specs.AnalysisGroup], cls)


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

    def __call__(self, iq: Array, capture: specs.Capture) -> Array:
        meas_kws = self.meas_spec.to_dict()
        ret = self.info.func(iq, capture, as_xarray=False, **meas_kws)
        return ret[0]

    def max_lag(self, capture: specs.Capture) -> int:
        lags = self.info.lag_coord_func(capture, self.meas_spec)
        step = lags[1] - lags[0]
        return step * len(lags)


def get_trigger(
    name: str, analysis: specs.AnalysisGroup, registry: AnalysisRegistry
) -> Trigger:
    return Trigger.from_spec(name, analysis, registry)


def get_signal_trigger_measurement_name(name: str, registry: AnalysisRegistry) -> str:
    info: SyncInfo = registry.signal_trigger[name]
    meas_info = registry[info.meas_spec_type]
    return meas_info.name


def normalize_factory_return(ret, name: str) -> tuple['Array', dict[str, Any]]:
    """normalize the coordinate and data factory returns into (data, metadata)"""

    if not isinstance(ret, tuple):
        arr = ret
        attrs = {}
    elif len(ret) == 2:
        arr, attrs = ret
    else:
        raise TypeError(
            f'tuple returned by {repr(name)} coordinate factory must have length 2, not {len(ret)}. return a list or array if this was meant as data.'
        )

    if not isinstance(attrs, dict):
        raise TypeError(
            f'second item of {repr(name)} coordinate factory return tuple must be dict.return an array or list if this was meant as data.'
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
            'iq', inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation='Array'
        ),
        inspect.Parameter(
            'capture',
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=specs.Capture,
        ),
    ]

    return inspect.Signature(parameters + list(kw_parameters.values()))


def _make_measurement_docstring(spec_cls):
    assert specs.Analysis.__doc__ is not None

    skip = len(specs.Analysis.__mro__) + 1
    docs = [
        cls.__doc__.strip()
        for cls in spec_cls.__mro__[-skip::-1]
        if cls.__doc__ is not None
    ]
    args = textwrap.indent('\n'.join(docs), 4 * ' ')
    return f'{specs.Analysis.__doc__.rstrip()}\n{args}'
