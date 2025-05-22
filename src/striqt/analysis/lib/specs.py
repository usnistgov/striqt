"""data structures that specify waveform characteristics"""

from __future__ import annotations
import typing
from typing import Annotated

import msgspec

from . import util

if typing.TYPE_CHECKING:
    import pandas as pd
    import numpy as np
else:
    pd = util.lazy_import('pandas')
    np = util.lazy_import('numpy')


_T = typing.TypeVar('_T')
_BUILTIN_TYPES = (pd.Timestamp,)


def _enc_hook(obj):
    if isinstance(obj, (np.float16, np.float32, np.float64)):
        return float(obj)
    else:
        return obj


def _dec_hook(type_, obj):
    if typing.get_origin(type_) is pd.Timestamp:
        return pd.to_datetime(obj)
    elif isinstance(obj, (np.float16, np.float32, np.float64)):
        return float(obj)
    else:
        return obj


def meta(standard_name: str, units: str | None = None, **kws) -> msgspec.Meta:
    """annotation that is used to generate 'standard_name' and 'units' fields of xarray attrs objects"""
    extra = {'standard_name': standard_name}
    if units is not None:
        # xarray objects with units == None cannot be saved to netcdf;
        # in this case, omit
        extra['units'] = units
    return msgspec.Meta(description=standard_name, extra=extra, **kws)


class StructBase(msgspec.Struct, kw_only=True, frozen=True, cache_hash=True):
    """Base type for structures that support validated
    (de)serialization.

    It is a `msgspec.Struct` class with some often-used utility
    methods that fix encoding and decoding hooks for extra types.
    """

    def replace(self, **attrs) -> type[typing.Self]:
        """returns a copy of self with changed attributes.

        See also:
            Python standard library `copy.replace`
        """
        return msgspec.structs.replace(self, **attrs).validate()

    def todict(self) -> dict:
        """return a dictinary representation of `self`"""
        return msgspec.to_builtins(
            self, builtin_types=_BUILTIN_TYPES, enc_hook=_enc_hook
        )

    def tojson(self) -> bytes:
        return msgspec.json.encode(self, enc_hook=_enc_hook)

    @classmethod
    def fromdict(cls: type[_T], d: dict) -> _T:
        return msgspec.convert(
            d, type=cls, strict=False, dec_hook=_dec_hook, builtin_types=_BUILTIN_TYPES
        )

    @classmethod
    def fromspec(cls: type[_T], other: StructBase) -> _T:
        return msgspec.convert(
            other,
            type=cls,
            strict=False,
            from_attributes=True,
            dec_hook=_dec_hook,
            builtin_types=_BUILTIN_TYPES,
        )

    @classmethod
    def fromjson(cls: type[_T], d: str | bytes) -> _T:
        return msgspec.json.decode(d, type=cls, strict=False, dec_hook=_dec_hook)

    def validate(self) -> type[typing.Self]:
        return self.fromdict(self.todict())


class Capture(StructBase, kw_only=True, frozen=True):
    """bare minimum information about an IQ acquisition"""

    # acquisition
    duration: Annotated[float, meta('duration of the capture', 's')] = 0.1
    sample_rate: Annotated[float, meta('IQ sample rate', 'S/s')] = 15.36e6
    analysis_bandwidth: Annotated[float, meta('Analysis bandwidth', 'Hz')] = float(
        'inf'
    )


class AnalysisFilter(StructBase, kw_only=True, frozen=True, cache_hash=True):
    nfft: int = 8192
    window: typing.Union[tuple[str, ...], str] = 'hamming'
    nfft_out: int = None


class FilteredCapture(Capture, kw_only=True, frozen=True, cache_hash=True):
    # filtering and resampling
    analysis_filter: AnalysisFilter = msgspec.field(default_factory=AnalysisFilter)
    # analysis_filter: dict = msgspec.field(
    #     default_factory=lambda: {'nfft': 8192, 'window': 'hamming'}
    # )


class AnalysisKeywords(typing.TypedDict):
    """base class for groups of keyword arguments that define calls to a set of analysis functions"""

    as_xarray: typing.NotRequired[bool | typing.Literal['delayed']]


class Measurement(
    StructBase, forbid_unknown_fields=True, cache_hash=True, kw_only=True, frozen=True
):
    """base class for groups of keyword arguments that define calls to a set of analysis functions"""

    pass


class Analysis(
    StructBase, forbid_unknown_fields=True, cache_hash=True, kw_only=True, frozen=True
):
    """base class for a set of Measurement specifications"""

    pass


@util.lru_cache()
def get_capture_type_attrs(capture_cls: type[Capture]) -> dict[str]:
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
