"""wrap lower-level iqwaveform DSP calls to accept physical inputs and return xarray.DataArray"""

from __future__ import annotations

import collections
import functools
import inspect
import typing

import msgspec
from xarray_dataclasses.dataarray import DataClass

from ..._api import structs

from ..._api import util
from .capture import ChannelAnalysisResult

if typing.TYPE_CHECKING:
    import numpy as np
else:
    np = util.lazy_import('numpy')


TFunc = typing.Callable[..., typing.Any]


class KeywordArguments(msgspec.Struct):
    """base class for the keyword argument parameters of an analysis function"""


class ChannelAnalysisRegistryDecorator(collections.UserDict):
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

    def __call__(self, xarray_datacls: DataClass, metadata={}) -> TFunc:
        """add decorated `func` and its keyword arguments in the self.tostruct() schema"""

        def wrapper(func: TFunc):
            name = func.__name__
            sig = inspect.signature(func)
            params = sig.parameters

            @functools.wraps(func)
            def wrapped(iq, capture, **kws):
                bound = sig.bind(iq=iq, capture=capture, **kws)
                call_params = bound.kwargs
                ret = func(*bound.args, **bound.kwargs)

                if isinstance(ret, (list, tuple)) and len(ret) == 2:
                    result, ret_metadata = ret
                    ret_metadata = dict(metadata, **ret_metadata)
                else:
                    result = ret
                    ret_metadata = metadata

                return ChannelAnalysisResult(
                    xarray_datacls,
                    result,
                    capture,
                    parameters=call_params,
                    attrs=ret_metadata,
                )

            sig_kws = [
                self._param_to_field(k, p)
                for k, p in params.items()
                if p.kind is inspect.Parameter.KEYWORD_ONLY
            ]

            struct_type = msgspec.defstruct(name, sig_kws, bases=(KeywordArguments,))

            # validate the struct
            msgspec.json.schema(struct_type)

            self[struct_type] = wrapped

            return wrapped

        return wrapper

    def spec_type(self) -> structs.ChannelAnalysis:
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
