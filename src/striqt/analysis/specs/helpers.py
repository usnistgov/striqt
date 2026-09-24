"""helper functions for specification data structures and their aliases"""

from __future__ import annotations as __

import contextlib
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
import msgspec
from striqt.waveform.lib import util

_T = TypeVar('_T')
_K = TypeVar('_K')
_V = TypeVar('_V')


if TYPE_CHECKING:
    import typing_extensions
    from typing_extensions import Concatenate

    from striqt.waveform.lib.typing import LRUWrapped

    from . import structs

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
        inst._dict = cast('dict[_K, _V]', dict(*args, **kwargs))
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


# %% validation of (capture, analysis spec) combinations
class SpecValidationError(msgspec.ValidationError):
    """a validation failure that carries a msgspec-style field path.

    Subclassing `msgspec.ValidationError` is what lets the path survive: msgspec
    re-raises a plain `ValueError` from `__post_init__` as its own
    `ValidationError` with its own path, but propagates a `ValidationError`
    subclass untouched.

    `path` is one nested chain of field accesses within a single document root,
    which renders as msgspec does it (`$.analysis.spectrogram`) and leads the
    message. `locations` are additional sibling roots, rendered as an `at ... on ...`
    trailer in the order given. The split exists because a sweep failure is not
    located by one chain: the same incompatibility is pinned down jointly by a
    `captures:` entry, the `loops:` point that generated it, and the measurement key
    that rejected it. Those are siblings in the document, so collapsing them into
    `path` would render a field chain that does not exist.
    """

    def __init__(
        self,
        message: str,
        path: tuple[str, ...] = (),
        locations: tuple[str, ...] = (),
    ) -> None:
        self.message = message
        self.path = tuple(path)
        self.locations = tuple(locations)
        super().__init__(str(self))

    def __str__(self) -> str:
        text = self.message
        if self.path:
            text = f'${"".join(self.path)}: {text}'
        if self.locations:
            text += ' - at ' + ' on '.join(f'${loc}' for loc in self.locations)
        return text

    def prepend(self, *parts: str) -> SpecValidationError:
        return type(self)(self.message, tuple(parts) + self.path, self.locations)

    def at(self, *locations: str) -> SpecValidationError:
        """prepend sibling document locations, which render in an `at ... on ...` trailer"""
        return type(self)(self.message, self.path, tuple(locations) + self.locations)


@contextlib.contextmanager
def validation_path(*parts: str) -> Iterator[None]:
    """re-raise a validation failure with `parts` prepended to its field path"""

    try:
        yield
    except SpecValidationError as ex:
        raise ex.prepend(*parts) from ex.__cause__
    except (ValueError, TypeError, msgspec.ValidationError) as ex:
        raise SpecValidationError(str(ex), parts) from ex


@util.lru_cache(4096)
def convert_spec_cached(spec_cls: type[_T], spec: Any) -> _T:
    """project `spec` onto the fields that `spec_cls` declares.

    msgspec hands back `spec` itself when it is already exactly `spec_cls`, so an
    argument that was projected already adds no second cache entry.
    """
    return convert_spec(spec, type=spec_cls)


def to_analysis_capture(capture: structs.Capture) -> structs.AnalysisCapture:
    """project a capture down to the fields the analysis layer can read"""
    from . import structs

    return convert_spec_cached(structs.AnalysisCapture, capture)


@overload
def lru_cache_on_converted(
    spec_type: type[msgspec.Struct], /, *, maxsize: int | None = 128
) -> Callable[
    [Callable[Concatenate[Any, _P], _R]], LRUWrapped[Concatenate[Any, _P], _R]
]:
    pass


@overload
def lru_cache_on_converted(
    capture_type: type[msgspec.Struct],
    spec_type: type[msgspec.Struct],
    /,
    *,
    maxsize: int | None = 128,
) -> Callable[
    [Callable[Concatenate[Any, Any, _P], _R]],
    LRUWrapped[Concatenate[Any, Any, _P], _R],
]:
    pass


def lru_cache_on_converted(
    *spec_types: type[msgspec.Struct], maxsize: int | None = 128
) -> Callable[[Callable[..., _R]], LRUWrapped[..., _R]]:
    """cache the decorated function, keyed on projections of its leading arguments.

    Each of the first `len(spec_types)` positional arguments is converted before the
    cache lookup, so callers that differ only in fields the target types do not declare
    share one entry. Doing this by hand takes a conversion wrapper stacked over
    `lru_cache`; in the other order the cache silently keys on the unprojected argument
    instead.

    The decorated function annotates the projected arguments with the type its *body*
    receives, while callers may pass anything `convert_spec` accepts, so the overloads
    above type those positions as `Any`.
    """

    def wrapper(func: Callable[..., _R]) -> LRUWrapped[..., _R]:
        cached = util.lru_cache(maxsize)(func)

        @functools.wraps(func)
        def wrapped(*args: Any, **kwargs: Any) -> _R:
            if len(args) < len(spec_types):
                raise TypeError(
                    f'{func.__name__} needs its first {len(spec_types)} argument(s) '  # ty: ignore[unresolved-attribute]
                    'passed by position, since they are converted before the cache '
                    'lookup'
                )

            converted = tuple(
                convert_spec_cached(spec_cls, arg)
                for spec_cls, arg in zip(spec_types, args)
            )

            return cached(*converted, *args[len(spec_types) :], **kwargs)

        # functools.wraps does not carry these over from the lru_cache wrapper
        cast(Any, wrapped).cache_clear = cached.cache_clear
        cast(Any, wrapped).cache_info = cached.cache_info

        return cast('LRUWrapped[..., _R]', wrapped)

    return wrapper


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


@util.lru_cache()
def get_capture_field_types(capture_cls: type[msgspec.Struct]) -> dict[str, Any]:
    """return the annotated type of each field in `capture_cls`"""
    return {field.name: field.type for field in msgspec.structs.fields(capture_cls)}


@util.lru_cache()
def _enc_hook(obj) -> Any:
    if isinstance(obj, frozendict):
        return obj._dict
    elif isinstance(obj, fractions.Fraction):
        return str(obj)
    elif hasattr(obj, '__float__'):
        return float(obj)
    else:
        return obj


@util.lru_cache()
def _enc_hook_no_tuple_keys(obj: Any) -> Any:
    """like `_enc_hook`, but with dictionary tuple keys encoded as JSON array text"""

    out = _enc_hook(obj)
    if isinstance(out, dict) and any(isinstance(k, tuple) for k in out):
        out = {
            msgspec.json.encode(list(k)).decode() if isinstance(k, tuple) else k: v
            for k, v in out.items()
        }
    return out


def _dec_hook(type_, obj):
    schema_cls = get_origin(type_) or type_

    if issubclass(schema_cls, (int, float)) and hasattr(obj, '__float__'):
        return float(obj)
    elif issubclass(schema_cls, fractions.Fraction):
        if isinstance(obj, float):
            # a YAML/JSON float is the author's decimal (.001), not its binary expansion
            return fractions.Fraction(obj).limit_denominator(10**10)
        return fractions.Fraction(obj)
    else:
        return obj


def _schema_hook(cls):
    if issubclass(cls, fractions.Fraction):
        return {'type': ['string', 'number']}
    elif issubclass(cls, frozendict):
        return {'type': 'object'}
    else:
        raise TypeError(f'no schema handler available for type {cls!r}')


def json_schema(cls: type[msgspec.Struct]) -> dict:
    return msgspec.json.schema(cls, schema_hook=_schema_hook)


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
            return ret
        else:
            return tuple(obj)
    elif isinstance(obj, dict):
        nd = None if max_depth is None else max_depth - 1
        if nd is None or nd > 0:
            mapping = {k: freeze(v, nd) for k, v in obj.items()}
            return frozendict(mapping)
        else:
            return frozendict(obj)
    else:
        return cast('_T', obj)


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
            return ret
        else:
            return list(obj)  # ty: ignore[invalid-return-type]

    if isinstance(obj, (dict, frozendict)):
        nd = None if max_depth is None else max_depth - 1
        if nd is None or nd > 0:
            ret = {k: unfreeze(v, nd) for k, v in obj.items()}
            return ret
        else:
            return dict(obj)
    else:
        return cast('_V', obj)


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
        info = infer_coord_info(type_key.type, allow_timestamps)[1]
        return type_key.extra or {}, info
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
