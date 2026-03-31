"""helper functions for specification data structures and their aliases"""

from __future__ import annotations as __

import functools
import fractions
from typing import (
    Any,
    Callable,
    Iterator,
    ItemsView,
    Iterable,
    KeysView,
    Mapping,
    TYPE_CHECKING,
    TypeVar,
    ValuesView,
    cast,
    get_origin,
    overload,
)
import warnings
import msgspec
from striqt.waveform.lib import util

_T = TypeVar('_T')
_K = TypeVar('_K')
_V = TypeVar('_V')


if TYPE_CHECKING:
    import typing_extensions

    _P = typing_extensions.ParamSpec('_P')
    _R = TypeVar('_R', covariant=True)

    class _MetaP(typing_extensions.Protocol[_P, _R]):
        def __call__(
            self,
            standard_name: str,
            units: str | None = None,
            *args: _P.args,
            **kws: _P.kwargs,
        ) -> _R: ...

    def _like_meta(
        _: Callable[_P, _R], /
    ) -> Callable[[_MetaP[_P, _R]], _MetaP[_P, _R]]:
        def impl(x: _MetaP[_P, _R]) -> _MetaP[_P, _R]:
            return x

        return impl
else:

    def _like_meta(_):
        def impl(x):
            return x

        return impl


class frozendict(Mapping[_K, _V]):
    """
    An immutable dictionary that supports hashing
    """

    __slots__ = ['_hash', '_dict']
    _dict: dict[_K, _V]
    _hash: int | None

    @classmethod
    def fromkeys(cls, seq: Iterable[_K], value: _V | None = None) -> frozendict[_K, _V]:
        return cls(dict.fromkeys(seq, value))

    def __new__(cls, *args: Any, **kwargs: Any) -> frozendict[_K, _V]:
        inst = super().__new__(cls)
        inst._dict = dict(*args, **kwargs)  # pyright: ignore
        inst._hash = None
        return inst

    def __reduce__(self) -> tuple[Any, ...]:
        return (self.__class__, (self._dict,))

    def __getitem__(self, key: _K) -> _V:
        return self._dict[key]

    def __contains__(self, key: object) -> bool:
        return key in self._dict

    def copy(self) -> frozendict[_K, _V]:
        return self.__class__(self)

    def __iter__(self) -> Iterator[_K]:
        return iter(self._dict)

    def __len__(self) -> int:
        return len(self._dict)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self._dict!r})'

    def __hash__(self) -> int:
        if self._hash is not None:
            return self._hash

        h = 0
        for key, value in self.items():
            try:
                h ^= hash((key, value))
            except:
                raise TypeError(f'unhashable frozendict entry {(key, value)!r}')
        self._hash = h

        return self._hash

    def __or__(self, other: Any) -> frozendict[_K, _V]:
        if not isinstance(other, (dict, self.__class__)):
            return NotImplemented
        new = dict(self)
        new.update(other)
        return self.__class__(new)

    def __ror__(self, other: Any) -> dict[Any, Any]:
        if isinstance(other, dict):
            return other | self._dict
        elif isinstance(other, frozendict):
            return other._dict | self._dict
        elif hasattr(other, '__len__') or hasattr(other, '__iter__'):
            return dict(other) | self._dict
        else:
            raise TypeError('unsupported mapping')

    def __ior__(self, other: Any) -> frozendict[_K, _V]:
        raise TypeError(f"'{self.__class__.__name__}' object is frozen")

    def items(self) -> ItemsView[_K, _V]:  # noqa: D102
        return self._dict.items()

    def keys(self) -> KeysView[_K]:  # noqa: D102
        return self._dict.keys()

    def values(self) -> ValuesView[_V]:  # noqa: D102
        return self._dict.values()

    def update(self, other: dict[_K, _V], /):
        raise TypeError(f"'{self.__class__.__name__}' object is frozen")


@_like_meta(msgspec.Meta)
def Meta(standard_name: str, units: str | None = None, **kws) -> msgspec.Meta:
    """annotation that is used to generate 'standard_name' and 'units' fields of xarray attrs objects"""
    extra = {'standard_name': standard_name}
    if units is not None:
        # xarray objects with units == None cannot be saved to netcdf;
        # in this case, omit
        extra['units'] = units
    return msgspec.Meta(description=standard_name, extra=extra, **kws)


@util.lru_cache()
def get_capture_type_attrs(capture_cls: type[msgspec.Struct]) -> dict[str, Any]:
    """return attrs metadata for each field in `capture`"""
    attrs = {}

    for field in msgspec.structs.fields(capture_cls):
        if isinstance(field.type, msgspec.inspect.UnionType):
            types = field.type.types
        else:
            types = [field.type]

        for type_ in types:
            type_extra = getattr(type_, 'extra', {})
            if len(type_extra) > 0:
                attrs[field.name] = type_extra
                break
        else:
            attrs[field.name] = {}

    return attrs


@functools.cache
def _warn_on_capture_lookup_miss(capture_value, capture_attr, error_label, default):
    warnings.warn(
        f'{error_label} is missing key {capture_attr}=={capture_value!r}; using default {default}'
    )


@util.lru_cache()
def _enc_hook(obj):
    if isinstance(obj, frozendict):
        return obj._dict
    elif isinstance(obj, fractions.Fraction):
        return str(obj)
    elif hasattr(obj, '__float__'):
        return float(obj)
    else:
        return obj


def _dec_hook(type_, obj):
    schema_cls = get_origin(type_) or type_

    if issubclass(schema_cls, (int, float)) and hasattr(obj, '__float__'):
        return float(obj)
    elif issubclass(schema_cls, fractions.Fraction):
        return fractions.Fraction(obj)
    else:
        return obj


@overload
def freeze(obj: dict[_K, _V], max_depth: int | None = None) -> 'frozendict[_K, _V]':
    pass


@overload
def freeze(
    obj: tuple[_V, ...] | list[_V], max_depth: int | None = None
) -> tuple[_V, ...]:
    pass


@overload
def freeze(obj: _T, max_depth: int | None = None) -> _T:
    pass


def freeze(
    obj: Mapping[_K, _V] | tuple[_V, ...] | list[_V] | _T,
    max_depth: int | None = None,
) -> 'frozendict[_K, _V]|tuple[_V, ...]|_T':
    """recursively transform list and dict into tuple and frozendict"""
    if isinstance(obj, (list, tuple)):
        nd = None if max_depth is None else max_depth - 1
        if nd is None or nd > 0:
            ret = tuple([freeze(v, nd) for v in obj])
            return ret  # type: ignore
        else:
            return tuple(obj) # type: ignore
    elif isinstance(obj, dict):
        nd = None if max_depth is None else max_depth - 1
        if nd is None or nd > 0:
            mapping = {k: freeze(v, nd) for k, v in obj.items()}
            return frozendict(mapping)
        else:
            return frozendict(obj)
    else:
        return obj  # type: ignore


@overload
def unfreeze(obj: Mapping[_K, _V], max_depth: int | None = None) -> 'dict[_K, _V]':
    pass


@overload
def unfreeze(obj: tuple[_V, ...] | list[_V], max_depth: int | None = None) -> list[_V]:
    pass


@overload
def unfreeze(obj: _V, max_depth: int | None = None) -> _V:
    pass


def unfreeze(
    obj: Mapping[_K, _V] | tuple[_V, ...] | list[_V] | _V,
    max_depth: int | None = None,
) -> 'dict[_K, _V]|list[_V]|_V':
    """Recursively transform dict into frozendict"""
    if isinstance(obj, (list, tuple)):
        nd = None if max_depth is None else max_depth - 1
        if nd is None or nd > 0:
            ret = [unfreeze(v, nd) for v in obj]
            return ret # type: ignore
        else:
            return list(obj)

    if isinstance(obj, (dict, frozendict)):
        nd = None if max_depth is None else max_depth - 1
        if nd is None or nd > 0:
            ret = {k: unfreeze(v, nd) for k, v in obj.items()}
            return ret # type: ignore
        else:
            return dict(obj)
    else:
        return obj  # type: ignore


def convert_dict(obj: Any, type: type[_T]) -> _T:
    return msgspec.convert(obj, type=type, strict=False, dec_hook=_dec_hook)


def convert_spec(other: Any, type: type[_T]) -> _T:
    return msgspec.convert(
        other, type=type, strict=False, from_attributes=True, dec_hook=_dec_hook
    )


def _inspect_container_depth(type_: msgspec.inspect.Type) -> int:
    """returns the maximum depth needed to freeze the given msgspec type"""
    from msgspec import inspect as mi

    if not isinstance(type_, mi.Type):
        type_ = mi.type_info(type_)

    if isinstance(type_, mi.DictType):
        return 1 + _inspect_container_depth(type_.value_type)
    elif isinstance(type_, mi.TupleType):
        return 1 + max(_inspect_container_depth(t) for t in type_.item_types)
    elif isinstance(type_, mi.ListType):
        return 1 + _inspect_container_depth(type_.item_type)
    elif isinstance(type_, mi.Metadata):
        return _inspect_container_depth(type_.type)
    elif isinstance(type_, mi.UnionType):
        return max(_inspect_container_depth(t) for t in type_.types)
    else:
        return 0


@functools.cache
def inspect_freeze_depths(spec_cls: type[msgspec.Struct]) -> dict[str, int]:
    """returns the maximum depth to freeze the given msgspec Struct"""

    fields = msgspec.structs.fields(spec_cls)
    depths = {}
    for field in fields:
        n = _inspect_container_depth(field.type)
        if n > 0:
            depths[field.name] = n
    return depths


# %% msgspec field type introspection
def infer_coord_info(
    type_: msgspec.inspect.Type, allow_timestamps=True
) -> tuple[dict, Any]:
    """returns an (attrs, default_value) pair for the given msgspec field type"""
    from msgspec import inspect as mi

    if allow_timestamps or TYPE_CHECKING:
        # don't force pandas imports, for lazy import support
        from pandas import Timestamp
    else:
        Timestamp = None

    BUILTINS: dict[type[mi.Type], Any] = {
        mi.FloatType: 0.0,
        mi.BoolType: False,
        mi.IntType: 0,
        mi.StrType: '',
        mi.DictType: {},
    }

    if isinstance(type_, mi.Type):
        type_key = type_
    else:
        type_key = mi.type_info(type_)

    if isinstance(type_key, tuple(BUILTINS.keys())):
        # dicey if subclasses show up
        return {}, BUILTINS[type(type_key)]
    elif isinstance(type_key, mi.CustomType):
        if Timestamp is not None:
            return {}, Timestamp(0)
        else:
            try:
                return {}, type_key.cls()
            except Exception as ex:
                name = type_key.cls.__qualname__
                raise TypeError(f'failed to make default for type {name!r}') from ex
    elif isinstance(type_key, mi.Metadata):
        return type_key.extra or {}, infer_coord_info(type_key.type, allow_timestamps)[
            1
        ]
    elif isinstance(type_key, mi.LiteralType):
        return {}, type(type_key.values[0])
    elif isinstance(type_key, mi.UnionType):
        UNION_SKIP = (mi.NoneType, mi.VarTupleType)
        types = [t for t in type_key.types if not isinstance(t, UNION_SKIP)]
        if len(types) == 1:
            return infer_coord_info(types[0], allow_timestamps)
        else:
            names = tuple(type(t).__qualname__ for t in types)
            raise TypeError(
                f'cannot determine xarray type for union of msgspec types {names!r}'
            )
    else:
        raise TypeError(f'unsupported msgspec field type {type_key!r}')
