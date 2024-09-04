from __future__ import annotations
from collections import UserDict
import functools
import inspect
import typing
from typing import Annotated as A
from typing import Optional

from frozendict import frozendict
import msgspec
from msgspec import to_builtins


TFunc = typing.Callable[..., typing.Any]


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


class FilteredCapture(Capture):
    # filtering and resampling
    analysis_bandwidth: A[Optional[float], meta('DSP filter bandwidth', 'Hz')] = None
    analysis_filter: dict = msgspec.field(
        default_factory=lambda: frozendict({'nfft': 8192, 'window': 'hamming'})
    )


class KeywordArguments(msgspec.Struct):
    """base class for the keyword argument parameters of an analysis function"""


class ChannelAnalysis(msgspec.Struct):
    """base class for groups of keyword arguments that define calls to multiple analysis functions"""


class KeywordConfigRegistry(UserDict):
    """a registry of keyword-only arguments for decorated functions"""

    def __init__(self, base_struct=None):
        super().__init__()
        self.base_struct = base_struct

    @staticmethod
    def _param_to_field(name, p: inspect.Parameter):
        """convert an inspect.Parameter to a msgspec.Struct field"""
        if p.annotation is inspect._empty:
            raise TypeError(
                f'to register this function, keyword-only argument "{name}" needs a type annotation'
            )

        if p.default is inspect._empty:
            return (name, p.annotation)
        else:
            return (name, p.annotation, p.default)

    def __call__(self, func: TFunc) -> TFunc:
        """add decorated `func` and its keyword arguments in the self.tostruct() schema"""
        name = func.__name__

        params = inspect.signature(func).parameters

        kws = [
            self._param_to_field(k, p)
            for k, p in params.items()
            if p.kind is inspect.Parameter.KEYWORD_ONLY
        ]

        struct_type = msgspec.defstruct(name, kws, bases=(KeywordArguments,))

        # validate the struct
        msgspec.json.schema(struct_type)

        self[struct_type] = func

        return func

    def spec_type(self) -> ChannelAnalysis:
        """return a Struct subclass type representing a specification for calls to all registered functions"""
        fields = [
            (func.__name__, typing.Union[struct_type, None], None)
            for struct_type, func in self.items()
        ]

        return msgspec.defstruct(
            'channel_analysis',
            fields,
            bases=(self.base_struct,) if self.base_struct else None,
            kw_only=True,
            forbid_unknown_fields=True,
            omit_defaults=True,
        )
