from __future__ import annotations
import functools
import typing
from typing import Annotated
from typing import Optional
from . import util
from frozendict import frozendict
import msgspec


if typing.TYPE_CHECKING:
    import pandas as pd
else:
    pd = util.lazy_import('pandas')


def meta(standard_name: str, unit: str | None = None, **kws) -> msgspec.Meta:
    """annotation that is used to generate 'standard_name' and 'units' fields of xarray attrs objects"""
    return msgspec.Meta(
        description=standard_name,
        extra={'standard_name': standard_name, 'units': unit},
        **kws,
    )


@functools.wraps(functools.partial(msgspec.to_builtins, builtin_types=[]))
def struct_to_builtins(*args, **kws):
    kws = dict(kws, builtin_types=(frozendict, pd.Timestamp))
    return msgspec.to_builtins(*args, **kws)


builtins_to_struct = functools.partial(msgspec.convert)
builtins_to_struct.__name__ = 'builtins_to_struct'


def copy_struct(struct: msgspec.Struct, **update_fields):
    """return a copy of struct, optionally with changes to its fields"""

    mapping = struct_to_builtins(struct)
    mapping = dict(mapping, **update_fields)
    return builtins_to_struct(mapping, type(struct))


class Capture(msgspec.Struct, kw_only=True, frozen=True):
    """bare minimum information about an IQ acquisition"""

    # acquisition
    duration: Annotated[float, meta('duration of the capture', 's')] = 0.1
    sample_rate: Annotated[float, meta('IQ sample rate', 'S/s')] = 15.36e6
    analysis_bandwidth: Optional[Annotated[float, meta('Analysis bandwidth', 'Hz')]] = (
        None
    )


class FilteredCapture(Capture):
    # filtering and resampling
    analysis_filter: dict = msgspec.field(
        default_factory=lambda: frozendict({'nfft': 8192, 'window': 'hamming'})
    )


class ChannelAnalysis(msgspec.Struct):
    """base class for groups of keyword arguments that define calls to multiple analysis functions"""


@functools.lru_cache
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
