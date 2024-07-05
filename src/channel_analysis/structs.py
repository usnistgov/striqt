from __future__ import annotations
import inspect
import msgspec
from collections import UserDict
import typing
from frozendict import frozendict
# from iqwaveform.power_analysis import isroundmod


TDecoratedFunc = typing.Callable[..., typing.Any]


class Capture(msgspec.Struct, kw_only=True, frozen=True):
    """bare minimum information about an IQ acquisition"""

    # acquisition
    duration: float
    sample_rate: float

    # def __post_init__(self):
    #     if not isroundmod(self.duration, 1/self.sample_rate):
    #         raise ValueError("duration must consist of a counting number of sample periods")


class FilteredCapture(Capture):
    # filtering and resampling
    analysis_bandwidth: typing.Optional[float] = None
    analysis_filter: dict = msgspec.field(
        default_factory=lambda: frozendict({'fft_size': 1024, 'window': 'hamming'})
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

    def __call__(self, func: TDecoratedFunc) -> TDecoratedFunc:
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
