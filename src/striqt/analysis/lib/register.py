"""Implement a registry to track and wrap measurement functions that transform numpy return values
into xarray DataArray objects with labeled dimensions and coordinates.
"""

from __future__ import annotations

import collections
import functools
import msgspec
import typing
import typing_extensions
from . import specs, util


if typing.TYPE_CHECKING:
    _P = typing_extensions.ParamSpec('_P')
    _R = typing_extensions.TypeVar('_R')
    import labbench as lb
    import iqwaveform
else:
    lb = util.lazy_import('labbench')


class _MeasurementCallable(typing.Protocol):
    def __call__(
        iq: 'iqwaveform.type_stubs.ArrayType', capture: specs.Capture, **kwargs
    ) -> typing.Union[
        'iqwaveform.type_stubs.ArrayType',
        tuple['iqwaveform.type_stubs.ArrayType', dict[str]],
    ]: ...


_TMeasCallable = typing.TypeVar('_TMeasCallable', bound=_MeasurementCallable)


class _CoordinateFactoryCallable(typing.Protocol):
    def __call__(
        capture: specs.Capture, spec: specs.Measurement
    ) -> typing.Union[
        'iqwaveform.type_stubs.ArrayType',
        tuple['iqwaveform.type_stubs.ArrayType', dict[str]],
    ]: ...


_TCoordCallable = typing.TypeVar('_TCoordCallable', bound=_CoordinateFactoryCallable)


class KeywordArgumentCache:
    """A single-element cache of measurement results.

    It is keyed on captures and keyword arguments.

    """

    _key: frozenset = None
    _value = None
    enabled = False

    def __init__(self, fields: list[str]):
        self._fields = fields

    def kw_key(self, kws: dict[str, typing.Any]):
        if kws is None:
            return None

        kws = {k: kws[k] for k in self._fields}

        return frozenset(kws.items())

    def clear(self):
        self.update(None, None)

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

    def apply(self, func: typing.Callable[_P, _R]) -> typing.Callable[_P, _R]:
        @functools.wraps(func)
        def wrapped(*args, **kws):
            match = self.lookup(kws)
            if match is not None:
                return match

            ret = func(*args, **kws)
            self.update(kws, ret)
            return ret

        return wrapped

    def __enter__(self):
        self.enabled = True
        return self

    def __exit__(self, *args):
        self.enabled = False
        self.clear()


class CoordinateInfo(typing.NamedTuple):
    name: str
    func: _CoordinateFactoryCallable
    dtype: str
    dims: tuple[str, ...] = None
    attrs: dict = {}


class _CoordinateRegistry(
    collections.UserDict[_CoordinateFactoryCallable, CoordinateInfo]
):
    def __call__(
        self,
        dtype,
        *,
        name: str | None = None,
        dims: tuple[str, ...] | None = None,
        attrs={},
    ) -> typing.Callable[[_TCoordCallable], _TCoordCallable]:
        """register a coordinate factory function.

        The factory function should return an iterable containing coordinate
        values, or None to indicate a named placeholder for a dimension.
        """
        kws = locals()

        def wrapper(func: _TCoordCallable) -> _TCoordCallable:
            if isinstance(kws['dims'], str):
                dims = tuple((kws['dims'],))
            else:
                dims = kws['dims']

            if kws['name'] is not None:
                name = str(kws['name'])
            else:
                try:
                    name = func.__name__
                except AttributeError as ex:
                    raise TypeError(
                        'specify the coordinate name with coordinates(name, ...)'
                    ) from ex

            if name in self:
                raise KeyError(
                    f'a coordinate has already been registered for coordinate {name!r}'
                )

            if dims is None:
                dims = (name,)

            self[func] = CoordinateInfo(
                name=name, func=func, dims=dims, dtype=dtype, attrs=attrs
            )

            return func

        return wrapper


coordinate_factory = _CoordinateRegistry()


class AlignmentInfo(typing.NamedTuple):
    name: str
    func: callable
    lag_coord_func: _CoordinateFactoryCallable
    meas_spec_type: type[specs.Measurement]


class _AlignmentSourceRegistry(collections.UserDict[str, AlignmentInfo]):
    def __call__(
        self,
        meas_spec_type: type[specs.Measurement],
        *,
        lag_coord_func: _CoordinateFactoryCallable,
        name=None,
    ) -> typing.Callable[[_TMeasCallable], _TMeasCallable]:
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

        def wrapper(func):
            if info_kws['name'] is None:
                info_kws['name'] = func.__name__

            if info_kws['name'] in self:
                raise TypeError(
                    f'an alignment source named {info_kws["name"]} was already registered'
                )

            self[info_kws['name']] = AlignmentInfo(func=func, **info_kws)

            return func

        return wrapper


alignment_source = _AlignmentSourceRegistry()


class MeasurementInfo(typing.NamedTuple):
    name: str
    func: _MeasurementCallable
    coord_factories: list[callable]
    prefer_unaligned_input: bool
    cache: KeywordArgumentCache | None
    dtype: str
    attrs: dict
    depends: typing.Iterable[_MeasurementCallable]
    dims: typing.Iterable[str] | str | None = (None,)


class _MeasurementRegistry(
    collections.UserDict[type[specs.Measurement], MeasurementInfo]
):
    """a registry of keyword-only arguments for decorated functions"""

    def __init__(self):
        super().__init__()
        self.depends_on: dict[callable, set[callable]] = {}
        self.names: set[str] = set()
        self.caches: dict[callable, callable] = {}
        self.use_unaligned_input: set[callable] = set()

    def __call__(
        self,
        spec_type: type[specs.Measurement],
        *,
        dtype: str,
        name: str | None = None,
        dims: typing.Iterable[str] | str | None = None,
        coord_factories: typing.Iterable[callable] | callable | None = None,
        depends: typing.Iterable[callable] = [],
        caches: typing.Iterable[KeywordArgumentCache] | None = None,
        prefer_unaligned_input=False,
        attrs={},
    ) -> typing.Callable[[_TMeasCallable], _TMeasCallable]:
        """add decorated `func` and its keyword arguments in the self.tostruct() schema"""

        if isinstance(dims, str):
            dims = (dims,)

        if coord_factories is None:
            coord_factories = ()
        elif callable(coord_factories):
            coord_factories = (coord_factories,)
        else:
            for entry in coord_factories:
                if not callable(entry):
                    raise TypeError('each coord_factories item must be callable')
            coord_factories = tuple(coord_factories)

        if callable(depends):
            depends = (depends,)

        info_kws = dict(
            name=name,
            coord_factories=coord_factories,
            prefer_unaligned_input=prefer_unaligned_input,
            cache=caches,
            dtype=dtype,
            attrs=attrs,
            depends=depends,
            dims=dims,
        )

        def wrapper(func: _TMeasCallable) -> _TMeasCallable:
            @functools.wraps(func)
            def wrapped(iq, capture, *, as_xarray=True, **kws):
                from .xarray_ops import DelayedDataArray

                # handle the additional argument 'as_xarray' that allows
                # the return of a _DelayedDataArray result for fast serialization and
                # xarray object instantiation
                if as_xarray not in ('delayed', True, False):
                    raise ValueError(
                        'xarray argument must be one of (True, False, "delayed")'
                    )

                spec = spec_type.fromdict(kws)
                ret = func(iq, capture, **spec.todict())

                if not as_xarray:
                    return ret

                if isinstance(ret, (list, tuple)) and len(ret) == 2:
                    data, more_attrs = ret
                else:
                    data = ret
                    more_attrs = {}

                data = DelayedDataArray(
                    data=data,
                    capture=capture,
                    spec=spec,
                    attrs=more_attrs,
                    info=self[spec_type],
                )

                if as_xarray == 'delayed':
                    return data
                else:
                    return data.to_xarray()

            if info_kws['name'] is None:
                info_kws['name'] = func.__name__

            if info_kws['name'] in self.names:
                raise TypeError(
                    f'a measurement named {info_kws["name"]!r} was already registered'
                )
            else:
                self.names.add(info_kws['name'])

            self.depends_on[wrapped] = []
            for dep in depends:
                self.depends_on[dep].append(wrapped)

            if caches is None:
                pass
            elif isinstance(caches, KeywordArgumentCache):
                self.caches[wrapped] = [caches]
            elif isinstance(caches, (tuple, list)) and isinstance(
                caches[0], KeywordArgumentCache
            ):
                self.caches[wrapped] = list(caches)
            else:
                raise TypeError(
                    'cache argument must be an tuple or list of KeywordArgumentCache'
                )

            self[spec_type] = MeasurementInfo(func=wrapped, **info_kws)

            return wrapped

        return wrapper

    def cache_context(self) -> typing.ContextManager:
        all_caches = []
        for caches in self.caches.values():
            all_caches.extend(caches)
        return lb.sequentially(*all_caches)


measurement = _MeasurementRegistry()


def to_analysis_spec(
    registry: _AlignmentSourceRegistry | _MeasurementRegistry,
    base: type[specs.Analysis] = specs.Analysis,
) -> type[specs.Analysis]:
    """return a Struct subclass type representing a specification for calls to all registered functions"""

    fields = [
        (info.name, typing.Union[struct_type, None], None)
        for struct_type, info in registry.items()
    ]

    return msgspec.defstruct(
        'Analysis',
        fields,
        bases=(base,),
        kw_only=True,
        forbid_unknown_fields=True,
        omit_defaults=True,
        frozen=True,
        cache_hash=True,
    )


class AlignmentCaller(typing.Protocol):
    def __init__(self, name: str, analysis: specs.Analysis):
        self.info: AlignmentInfo = alignment_source[name]
        self.meas_info = measurement[self.info.meas_spec_type]

        self.meas_spec = getattr(analysis, self.meas_info.name, None)
        if self.meas_spec is None:
            raise ValueError(
                f'alignment source {name!r} requires an analysis '
                f'specification for {self.meas_info.name!r}'
            )

        self.meas_kws = self.meas_spec.todict()

    def __call__(
        self, iq: 'iqwaveform.type_stubs.ArrayType', capture: specs.Capture
    ) -> float:
        return self.info.func(iq, capture, **self.meas_kws)

    def max_lag(self, capture):
        lags = self.info.lag_coord_func(capture, self.meas_spec)
        step = lags[1] - lags[0]
        return step * len(lags)


@util.lru_cache()
def get_aligner(name: str, analysis: specs.Analysis) -> AlignmentCaller:
    return AlignmentCaller(name, analysis)


def get_alignment_measurement_name(name: str):
    info: AlignmentInfo = alignment_source[name]
    meas_info = measurement[info.meas_spec_type]
    return meas_info.name
