"""Implement a registry to track and wrap measurement functions that transform numpy return values
into xarray DataArray objects with labeled dimensions and coordinates.
"""

from __future__ import annotations

import collections
import functools
import msgspec
import typing

from . import specs


_P = typing.ParamSpec('_P')
_R = typing.TypeVar('_R')


class Cache:
    """A single-element cache of measurement results.

    It is keyed on captures and keyword arguments.

    """

    _key: frozenset = None
    _value = None
    enabled = False

    @staticmethod
    def kw_key(kws):
        if kws is None:
            return None

        kws = dict(kws)
        del kws['iq'], kws['as_xarray']
        return frozenset(kws.items())

    @classmethod
    def clear(cls):
        cls.update(None, None)

    @classmethod
    def lookup(cls, kws: dict):
        if cls._key is None or not cls.enabled:
            return None

        if cls.kw_key(kws) == cls._key:
            return cls._value
        else:
            return None

    @classmethod
    def update(cls, kws: dict, value):
        if not cls.enabled:
            return
        cls._key = cls.kw_key(kws)
        cls._value = value

    @classmethod
    def cached_calls(cls, func):
        @functools.wraps(func)
        def wrapped(**kws):
            match = cls.lookup(kws)
            if match is not None:
                return match

            ret = func(**kws)
            cls.update(kws, ret)
            return ret

        return wrapped

    def __enter__(self):
        self.enabled = True
        return self

    def __exit__(self, *args):
        self.enabled = False
        self.clear()


# TODO: future approach to registering coordinates, rather than dataclasses?
class DelayedCoordinate(typing.NamedTuple):
    name: str
    func: callable
    dtype: str
    dims: tuple[str, ...] = None
    attrs: dict = {}

class _CoordinateRegistry(collections.UserDict[str, DelayedCoordinate]):
    def __call__[**_P, _R](
        self,
        dtype,
        *,
        name: str | None = None,
        dims: tuple[str, ...] | None = None,
        attrs={},
    ) -> typing.Callable[_P, _R]:
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

            self[name] = DelayedCoordinate(
                name=name, func=func, dims=dims, dtype=dtype, attrs=attrs
            )

            return func

        return wrapper

    def get(self, key):
        if not callable(key):
            raise TypeError('coordinate key must be a registered callable')
        try:
            return super().get(key)
        except KeyError:
            raise TypeError(
                f'callable {repr(key)} has not been registerd as a coordinate'
            )


coordinate_factory = _CoordinateRegistry()


class _MeasurementRegistry(collections.UserDict):
    """a registry of keyword-only arguments for decorated functions"""

    def __init__(self):
        super().__init__()
        self.depends_on: dict[callable, set[callable]] = {}
        self.names: set[str] = set()
        self.caches: dict[callable, callable] = {}

    def __call__[**_P, _R](
        self,
        name: str | None = None,
        *,
        dims: typing.Iterable[str] | str | None = None,
        coord_funcs: typing.Iterable[callable] | callable | None = None,
        depends: typing.Iterable[callable] = [],
        spec_type: type[specs.Measurement],
        dtype: str,
        cache: Cache | None = None,
        attrs={},
    ) -> typing.Callable[_P, _R]:
        """add decorated `func` and its keyword arguments in the self.tostruct() schema"""
        if isinstance(dims, str):
            dims = (dims,)

        if coord_funcs is None:
            coord_funcs = ()
        elif callable(coord_funcs):
            coord_funcs = (coord_funcs,)
        else:
            for entry in coord_funcs:
                if not callable(entry):
                    raise TypeError('each coord_funcs item must be callable')
            coord_funcs = tuple(coord_funcs)

        if callable(depends):
            depends = (depends,)

        kws = locals()

        def wrapper(func: typing.Callable[P, R]) -> typing.Callable[P, R]:
            if kws['name'] is None:
                name = func.__name__
            else:
                name = kws['name ']

            if name in self.names:
                raise TypeError(
                    f'a measurement named {repr(name)} was already registered'
                )
            else:
                self.names.add(name)

            @functools.wraps(func)
            def wrapped(iq, capture, **kws):
                from .xarray_ops import _DelayedDataArray

                # injects and handles an additional argument, 'as_xarray', which allows
                # the return of a ChannelAnalysis result for fast serialization and
                # xarray object instantiation
                as_xarray = kws.pop('as_xarray', True)
                if as_xarray not in ('delayed', True, False):
                    raise ValueError(
                        'xarray argument must be one of (True, False, "delayed")'
                    )

                spec = spec_type.fromdict(kws)
                ret = func(iq, capture, **spec.todict())

                if not as_xarray:
                    return ret

                if isinstance(ret, (list, tuple)) and len(ret) == 2:
                    data, metadata = ret
                    metadata = attrs | metadata
                else:
                    data = ret
                    metadata = attrs

                data = _DelayedDataArray(
                    data=data,
                    capture=capture,
                    name=name,
                    dims=dims,
                    coord_factories=coord_funcs,
                    spec=spec,
                    dtype=dtype,
                    attrs=metadata,
                )

                if as_xarray == 'delayed':
                    return data
                else:
                    return data.to_xarray()

            name = wrapped.__name__
            if name in self:
                raise TypeError(
                    f'another function named {repr(name)} has already been registered'
                )

            self.depends_on[wrapped] = []
            for dep in depends:
                self.depends_on[dep].append(wrapped)

            if cache is None:
                pass
            elif not isinstance(cache, Cache):
                raise TypeError('cache argument must be an instance of Cache')
            else:
                self.caches[wrapped] = cache

            self[spec_type] = wrapped

            return wrapped

        return wrapper

    def container_spec(self) -> type[specs.MeasurementSet]:
        """return a Struct subclass type representing a specification for calls to all registered functions"""
        fields = [
            (func.__name__, typing.Union[struct_type, None], None)
            for struct_type, func in self.items()
        ]

        return msgspec.defstruct(
            'Analysis',
            fields,
            bases=(specs.MeasurementSet,),
            kw_only=True,
            forbid_unknown_fields=True,
            omit_defaults=True,
            frozen=True,
        )


measurement = _MeasurementRegistry()
