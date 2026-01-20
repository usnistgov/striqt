"""helper functions for specification data structures and their aliases"""

from __future__ import annotations as __

import functools
import fractions
import numbers
import typing
import warnings

import msgspec


_T = typing.TypeVar('_T')


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

    if isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, fractions.Fraction):
        return str(obj)
    else:
        return obj


def _dec_hook(type_, obj):
    import numpy as np

    origin_cls = typing.get_origin(type_) or type_

    if issubclass(origin_cls, fractions.Fraction):
        return fractions.Fraction(obj)
    elif issubclass(origin_cls, numbers.Number) and isinstance(obj, np.floating):
        return float(obj)
    else:
        return obj


def _deep_hash(obj: typing.Mapping | typing.Sequence) -> int:
    """compute the hash of a dict or other mapping based on its key, value pairs.

    The hash is evaluated recursively for nested structures.
    """
    if isinstance(obj, (tuple, list)):
        keys = ()
        values = obj
    elif isinstance(obj, dict):
        keys = frozenset(obj.keys())
        values = obj.values()
    else:
        return hash(obj)

    deep_values = tuple(
        _deep_hash(v) if isinstance(v, (tuple, list, dict)) else v for v in values
    )

    return hash(keys) ^ hash(deep_values)


@functools.lru_cache()
def _private_fields(cls: type[msgspec.Struct]) -> tuple[str, ...]:
    return tuple([n for n in cls.__struct_fields__ if n.startswith('_')])


def convert_dict(obj: typing.Any, type: type[_T]) -> _T:
    return msgspec.convert(obj, type=type, strict=False, dec_hook=_dec_hook)


def convert_spec(other: typing.Any, type: type[_T]) -> _T:
    return msgspec.convert(
        other, type=type, strict=False, from_attributes=True, dec_hook=_dec_hook
    )
