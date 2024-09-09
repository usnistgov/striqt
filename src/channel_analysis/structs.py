from __future__ import annotations
import functools
import typing
from typing import Annotated as A
from typing import Optional

from frozendict import frozendict
import msgspec
from msgspec import to_builtins


def meta(standard_name: str, unit: str | None = None) -> msgspec.Meta:
    """annotation that is used to generate 'standard_name' and 'units' fields of xarray attrs objects"""
    return msgspec.Meta(
        description=standard_name, extra={'standard_name': standard_name, 'units': unit}
    )


@functools.lru_cache
def get_attrs(struct: typing.Type[msgspec.Struct], field: str) -> dict[str, str]:
    """get an attrs dict for xarray based on Annotated type hints with `meta`"""
    hints = typing.get_type_hints(struct, include_extras=True)

    try:
        metas = hints[field].__metadata__
    except AttributeError:
        return {}

    if len(metas) == 0:
        return {}
    elif len(metas) == 1 and isinstance(metas[0], msgspec.Meta):
        return metas[0].extra
    else:
        raise TypeError(
            'Annotated[] type hints must contain exactly one msgspec.Meta object'
        )


class Capture(msgspec.Struct, kw_only=True, frozen=True):
    """bare minimum information about an IQ acquisition"""

    # acquisition
    duration: A[float, meta('duration of the capture', 's')] = 0.1
    sample_rate: A[float, meta('IQ sample rate', 'S/s')] = 15.36e6
    analysis_bandwidth: A[Optional[float], meta('Analysis bandwidth', 'Hz')] = None


class FilteredCapture(Capture):
    # filtering and resampling
    analysis_filter: dict = msgspec.field(
        default_factory=lambda: frozendict({'nfft': 8192, 'window': 'hamming'})
    )


class ChannelAnalysis(msgspec.Struct):
    """base class for groups of keyword arguments that define calls to multiple analysis functions"""
