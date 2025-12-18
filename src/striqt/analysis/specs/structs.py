"""data structures that specify waveform characteristics"""

from __future__ import annotations as __

import fractions
import functools
from math import inf
import numbers
import typing
import warnings

import msgspec

from . import types

if typing.TYPE_CHECKING:
    _T = typing.TypeVar('_T')


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
def _private_fields(capture_cls: type[SpecBase]) -> tuple[str, ...]:
    return tuple([n for n in capture_cls.__struct_fields__ if n.startswith('_')])


def convert_dict(obj: typing.Any, type: type[_T]) -> _T:
    return msgspec.convert(obj, type=type, strict=False, dec_hook=_dec_hook)


def convert_spec(other: typing.Any, type: type[_T]) -> _T:
    return msgspec.convert(
        other, type=type, strict=False, from_attributes=True, dec_hook=_dec_hook
    )


class SpecBase(
    msgspec.Struct,
    kw_only=True,
    frozen=True,
    forbid_unknown_fields=True,
    cache_hash=True,
):
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
        if len(attrs) == 0:
            return self
        return msgspec.structs.replace(self, **attrs).validate()

    def to_dict(self, skip_private=False) -> dict:
        """return a dictinary representation of `self`"""
        map = msgspec.to_builtins(self, enc_hook=_enc_hook)

        if skip_private:
            for name in _private_fields(type(self)):
                del map[name]

        return map

    def to_json(self) -> bytes:
        return msgspec.json.encode(self, enc_hook=_enc_hook)

    @classmethod
    def from_dict(cls: type[_T], d: dict) -> _T:
        return convert_dict(d, type=cls)

    @classmethod
    def from_spec(cls: type[_T], other: SpecBase) -> _T:
        return convert_spec(other, type=cls)

    @classmethod
    def from_json(cls: type[_T], d: str | bytes) -> _T:
        return msgspec.json.decode(d, type=cls, strict=False, dec_hook=_dec_hook)

    def validate(self) -> typing.Self:
        return self.from_dict(self.to_dict())


class _SlowHashSpecBase(SpecBase, kw_only=True, frozen=True):
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


class Capture(SpecBase, kw_only=True, frozen=True):
    """bare minimum information about an IQ acquisition"""

    # acquisition
    duration: types.DurationType = 0.1
    sample_rate: types.SampleRateType = 15.36e6
    analysis_bandwidth: types.AnalysisBandwidthType = inf

    def __post_init__(self):
        from ..lib import util

        if not util.isroundmod(self.duration * self.sample_rate, 1):
            raise ValueError(
                f'duration {self.duration!r} is not an integer multiple of sample period'
            )


class _CaptureKeywords(typing.TypedDict, total=False):
    duration: types.DurationType
    sample_rate: types.SampleRateType
    analysis_bandwidth: types.AnalysisBandwidthType


class AnalysisFilter(SpecBase, kw_only=True, frozen=True):
    nfft: int = 8192
    window: typing.Union[tuple[str, ...], str] = 'hamming'
    nfft_out: int | None = None


class FilteredCapture(Capture, kw_only=True, frozen=True):
    # filtering and resampling
    analysis_filter: AnalysisFilter = msgspec.field(default_factory=AnalysisFilter)
    # analysis_filter: dict = msgspec.field(
    #     default_factory=lambda: {'nfft': 8192, 'window': 'hamming'}
    # )


class AnalysisKeywords(typing.TypedDict):
    as_xarray: typing.NotRequired[bool | typing.Literal['delayed']]


class Measurement(_SlowHashSpecBase, kw_only=True, frozen=True):
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


class Analysis(_SlowHashSpecBase, kw_only=True, frozen=True):
    """base class for a set of Measurement specifications"""

    pass


@functools.lru_cache()
def get_capture_type_attrs(capture_cls: type[Capture]) -> dict[str, typing.Any]:
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
    capture: Capture,
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
