"""helper functions for specification data structures and their aliases"""

from __future__ import annotations as __

import functools
import fractions
import typing
import warnings

if typing.TYPE_CHECKING:
    from immutabledict import immutabledict
import msgspec


_T = typing.TypeVar('_T')
_K = typing.TypeVar('_K')
_V = typing.TypeVar('_V')


def Meta(standard_name: str, units: str | None = None, **kws) -> msgspec.Meta:
    """annotation that is used to generate 'standard_name' and 'units' fields of xarray attrs objects"""
    extra = {'standard_name': standard_name}
    if units is not None:
        # xarray objects with units == None cannot be saved to netcdf;
        # in this case, omit
        extra['units'] = units
    return msgspec.Meta(description=standard_name, extra=extra, **kws)


@functools.lru_cache()
def get_capture_type_attrs(capture_cls: type[msgspec.Struct]) -> dict[str, typing.Any]:
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


def maybe_lookup_with_capture_key(
    capture: msgspec.Struct,
    value: _T | typing.Mapping[typing.Any, _T],
    capture_attr: str,
    error_label: str,
    default: _T | None = None,
) -> _T | None:
    """evaluate a lookup table based on an attribute in a capture object.

    Returns:
    - If value is dict-like, returns value[getattr(capture, capture_attr)]
    - Otherwise, returns value
    """
    if hasattr(value, 'keys') and hasattr(value, '__getitem__'):
        value = typing.cast(typing.Mapping, value)
        try:
            capture_value = getattr(capture, capture_attr)
        except AttributeError:
            raise AttributeError(
                f'can only look up {error_label!r} if the capture has {capture_attr!r} field'
            )
        try:
            return value[capture_value]
        except KeyError:
            _warn_on_capture_lookup_miss(
                capture_value=capture_value,
                capture_attr=capture_attr,
                error_label=error_label,
                default=default,
            )
            return default

    elif typing.TYPE_CHECKING:
        return typing.cast(_T, value)
    else:
        return value


def _enc_hook(obj):
    import numpy as np
    from immutabledict import immutabledict

    if isinstance(obj, immutabledict):
        return dict(obj)
    if isinstance(obj, fractions.Fraction):
        return str(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    else:
        return obj


def _dec_hook(type_, obj):
    import numpy as np

    schema_cls = typing.get_origin(type_) or type_

    if issubclass(schema_cls, (int, float)) and isinstance(obj, np.floating):
        return float(obj)
    elif issubclass(schema_cls, fractions.Fraction):
        return fractions.Fraction(obj)
    else:
        return obj


@typing.overload
def _deep_freeze(obj: dict[_K, _V]) -> 'immutabledict[_K, _V]':
    pass


@typing.overload
def _deep_freeze(obj: tuple[_V, ...] | list[_V]) -> tuple[_V, ...]:
    pass


@typing.overload
def _deep_freeze(obj: _T) -> _T:
    pass


def _deep_freeze(
    obj: typing.Mapping[_K, _V] | tuple[_V, ...] | list[_V] | _T,
) -> 'immutabledict[_K, _V]|tuple[_V, ...]|_T':
    """Recursively transform dict into immutabledict"""
    if isinstance(obj, (list, tuple)):
        return tuple(_deep_freeze(v) for v in obj)
    if isinstance(obj, dict):
        from immutabledict import immutabledict

        mapping = {k: _deep_freeze(v) for k, v in obj.items()}
        return immutabledict(mapping)
    else:
        return obj  # type: ignore

@typing.overload
def _unfreeze(obj: typing.Mapping[_K, _V]) -> 'dict[_K, _V]':
    pass


@typing.overload
def _unfreeze(obj: tuple[_V, ...] | list[_V]) -> list[_V]:
    pass


@typing.overload
def _unfreeze(obj: _T) -> _T:
    pass

def _unfreeze(
    obj: typing.Mapping[_K, _V] | tuple[_V, ...] | list[_V] | _T,
) -> 'dict[_K, _V]|list[_V]|_T':
    """Recursively transform dict into immutabledict"""
    from immutabledict import immutabledict
    
    if isinstance(obj, (list, tuple)):
        return [_unfreeze(v) for v in obj]
    if isinstance(obj, (dict, immutabledict)):
        mapping = {k: _unfreeze(v) for k, v in obj.items()}
        return dict(mapping)
    else:
        return obj  # type: ignore


@functools.lru_cache()
def _private_fields(cls: type[msgspec.Struct]) -> tuple[str, ...]:
    return tuple([n for n in cls.__struct_fields__ if n.startswith('_')])


def convert_dict(obj: typing.Any, type: type[_T]) -> _T:
    return msgspec.convert(obj, type=type, strict=False, dec_hook=_dec_hook)


def convert_spec(other: typing.Any, type: type[_T]) -> _T:
    return msgspec.convert(
        other, type=type, strict=False, from_attributes=True, dec_hook=_dec_hook
    )
