"""data structures that specify waveform characteristics"""

from __future__ import annotations
import fractions
import functools
import numbers
import typing
from typing import Annotated
import warnings

import msgspec

from . import util

if typing.TYPE_CHECKING:
    import pandas as pd
    import numpy as np

    _T = typing.TypeVar('_T')

else:
    pd = util.lazy_import('pandas')
    np = util.lazy_import('numpy')


WindowType = typing.Union[str, tuple[str, float]]


def _enc_hook(obj):
    if isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, fractions.Fraction):
        return str(obj)
    else:
        return obj


def _dec_hook(type_, obj):
    origin_cls = typing.get_origin(type_) or type_
    np_float_types = (np.float16, np.float32, np.float64)

    if issubclass(origin_cls, pd.Timestamp):
        return pd.to_datetime(obj)
    elif issubclass(origin_cls, fractions.Fraction):
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


def meta(standard_name: str, units: str | None = None, **kws) -> msgspec.Meta:
    """annotation that is used to generate 'standard_name' and 'units' fields of xarray attrs objects"""
    extra = {'standard_name': standard_name}
    if units is not None:
        # xarray objects with units == None cannot be saved to netcdf;
        # in this case, omit
        extra['units'] = units
    return msgspec.Meta(description=standard_name, extra=extra, **kws)


@util.lru_cache()
def _private_fields(capture_cls: type[SpecBase]) -> tuple[str, ...]:
    return tuple([n for n in capture_cls.__struct_fields__ if not n.startswith('_')])


def convert_dict(obj: typing.Any, type: type[_T]) -> _T:
    return msgspec.convert(
        obj, type=type, strict=False, dec_hook=_dec_hook, builtin_types=(pd.Timestamp,)
    )


class SpecBase(msgspec.Struct, kw_only=True, frozen=True, cache_hash=True):
    """Base type for structures that support validated
    (de)serialization.

    It is a `msgspec.Struct` class with some often-used utility
    methods that fix encoding and decoding hooks for extra types.
    """

    def replace(self, **attrs) -> typing.Self:
        """returns a copy of self with changed attributes.

        See also:
            Python standard library `copy.replace`
        """
        return msgspec.structs.replace(self, **attrs).validate()

    def todict(self, skip_private=False) -> dict:
        """return a dictinary representation of `self`"""
        map = msgspec.to_builtins(
            self, builtin_types=(pd.Timestamp,), enc_hook=_enc_hook
        )

        if skip_private:
            for name in _private_fields(type(self)):
                del map[name]

        return map

    def tojson(self) -> bytes:
        return msgspec.json.encode(self, enc_hook=_enc_hook)

    @classmethod
    def fromdict(cls: type[_T], d: dict) -> _T:
        return convert_dict(d, type=cls)

    @classmethod
    def fromspec(cls: type[_T], other: SpecBase) -> _T:
        return msgspec.convert(
            other,
            type=cls,
            strict=False,
            from_attributes=True,
            dec_hook=_dec_hook,
            builtin_types=(pd.Timestamp,),
        )

    @classmethod
    def fromjson(cls: type[_T], d: str | bytes) -> _T:
        return msgspec.json.decode(d, type=cls, strict=False, dec_hook=_dec_hook)

    def validate(self) -> typing.Self:
        return self.fromdict(self.todict())


class _SlowHashSpecBase(SpecBase, kw_only=True, frozen=True, cache_hash=True):
    def __hash__(self):
        try:
            return msgspec.Struct.__hash__(self)
        except TypeError:
            # presume a dict or tuple from here on
            pass

        # attr names come with the type, so get them for free here
        h = hash(type(self))

        # work through the values
        for name in self.__struct_fields__:
            value = getattr(self, name)
            if isinstance(value, (tuple, dict)):
                h ^= _deep_hash(value)
            else:
                h ^= hash(value)

        return h


class CaptureBase(SpecBase, kw_only=True, frozen=True):
    """bare minimum information about an IQ acquisition"""

    # acquisition
    duration: Annotated[float, meta('duration of the capture', 's')] = 0.1
    sample_rate: Annotated[float, meta('IQ sample rate', 'S/s')] = 15.36e6
    analysis_bandwidth: Annotated[float, meta('Analysis bandwidth', 'Hz')] = float(
        'inf'
    )

    def __post_init__(self):
        if not util.isroundmod(self.duration * self.sample_rate, 1):
            raise ValueError(
                f'duration {self.duration!r} is not an integer multiple of sample period'
            )


class AnalysisFilter(SpecBase, kw_only=True, frozen=True, cache_hash=True):
    nfft: int = 8192
    window: typing.Union[tuple[str, ...], str] = 'hamming'
    nfft_out: int | None = None


class FilteredCapture(CaptureBase, kw_only=True, frozen=True, cache_hash=True):
    # filtering and resampling
    analysis_filter: AnalysisFilter = msgspec.field(default_factory=AnalysisFilter)
    # analysis_filter: dict = msgspec.field(
    #     default_factory=lambda: {'nfft': 8192, 'window': 'hamming'}
    # )


class AnalysisKeywords(typing.TypedDict):
    as_xarray: typing.NotRequired[bool | typing.Literal['delayed']]


class Measurement(
    _SlowHashSpecBase,
    forbid_unknown_fields=True,
    cache_hash=True,
    kw_only=True,
    frozen=True,
):
    """
    Returns:
        Analysis result of type `(xarray.DataArray if as_xarray else type(iq))`

    Args:
        iq (numpy.ndarray or cupy.ndarray): the M-channel input waveform of shape (M,N)
        capture:
        as_xarray (bool): True to return xarray.DataArray or False to match type(iq)
    """

    def __hash__(self):
        try:
            return msgspec.Struct.__hash__(self)
        except TypeError:
            # presume a dict or tuple from here on
            pass

        # attr names come with the type, so get them for free here
        h = hash(type(self))

        # work through the values
        for name in self.__struct_fields__:
            value = getattr(self, name)
            if isinstance(value, (tuple, dict)):
                h ^= _deep_hash(value)
            else:
                h ^= hash(value)

        return h


class Analysis(
    _SlowHashSpecBase,
    forbid_unknown_fields=True,
    cache_hash=True,
    kw_only=True,
    frozen=True,
):
    """base class for a set of Measurement specifications"""

    pass


@util.lru_cache()
def get_capture_type_attrs(capture_cls: type[CaptureBase]) -> dict[str, typing.Any]:
    """return attrs metadata for each field in `capture`"""
    info = msgspec.inspect.type_info(capture_cls)

    attrs = {}

    for field in info.fields:
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
    capture: CaptureBase,
    value: _T | typing.Mapping[str, _T],
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
                f'can only look up {error_label} when an attribute {capture_attr!r} exists in the capture type'
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

    else:
        return typing.cast(_T, value)


class AcquisitionInfo(SpecBase, kw_only=True, frozen=True):
    pass
