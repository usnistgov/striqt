"""Implement a registry to track and wrap measurement functions that transform numpy return values
into xarray DataArray objects with labeled dimensions and coordinates.
"""

from __future__ import annotations

import collections
import functools
import inspect
import msgspec
import typing

from . import structs
from . import util
from .xarray_ops import (
    ChannelAnalysisResult,
    evaluate_channel_analysis,
    package_channel_analysis,
)


if typing.TYPE_CHECKING:
    from xarray_dataclasses import dataarray
    import array_api_compat
    import iqwaveform
    import xarray as xr
else:
    dataarray = util.lazy_import('xarray_dataclasses.dataarray')
    array_api_compat = util.lazy_import('array_api_compat')
    iqwaveform = util.lazy_import('iqwaveform')
    xr = util.lazy_import('xarray')


TFunc = typing.Callable[..., typing.Any]


def _results_as_arrays(obj: tuple | list | dict | 'iqwaveform.util.Array'):
    """convert an array, or a container of arrays, into a numpy array (or container of numpy arrays)"""

    if array_api_compat.is_torch_array(obj):
        array = obj.cpu()
    elif array_api_compat.is_cupy_array(obj):
        array = obj.get()
    else:
        raise TypeError(f'obj type {type(obj)} is unrecognized')

    return array


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


class KeywordArguments(structs.StructBase):
    """base class for the keyword argument parameters of an analysis function"""


class ChannelAnalysisRegistry(collections.UserDict):
    """a registry of keyword-only arguments for decorated functions"""

    def __init__(self, base_struct=None):
        super().__init__()
        self.base_struct = base_struct

    def __call__(self, xarray_datacls: 'dataarray.DataClass', metadata={}) -> TFunc:
        """add decorated `func` and its keyword arguments in the self.tostruct() schema"""

        def wrapper(func: TFunc):
            name = func.__name__
            if name in self:
                raise TypeError(
                    f'another function named {repr(name)} has already been registered'
                )
            sig = inspect.signature(func)
            params = sig.parameters

            @functools.wraps(func)
            def wrapped(iq, capture, **kws):
                # injects and handles an additional argument, 'as_xarray', which allows
                # the return of a ChannelAnalysis result for fast serialization and
                # xarray object instantiation
                as_xarray = kws.pop('as_xarray', True)
                if as_xarray == 'delayed':
                    delay_xarray = True
                elif as_xarray in (True, False):
                    delay_xarray = False
                else:
                    raise ValueError(
                        'xarray argument must be one of (True, False, "delayed")'
                    )

                bound = sig.bind(iq=iq, capture=capture, **kws)
                call_params = bound.kwargs

                ret = func(*bound.args, **bound.kwargs)

                if not as_xarray:
                    return ret

                if isinstance(ret, (list, tuple)) and len(ret) == 2:
                    result, ret_metadata = ret
                    ret_metadata = dict(metadata, **ret_metadata)
                else:
                    result = ret
                    ret_metadata = metadata

                try:
                    result = _results_as_arrays(result)
                except TypeError as ex:
                    msg = f'improper return type from {func.__name__}'
                    raise TypeError(msg) from ex

                result_obj = ChannelAnalysisResult(
                    xarray_datacls,
                    result,
                    capture,
                    parameters=call_params,
                    attrs=ret_metadata,
                )

                if delay_xarray:
                    return result_obj
                else:
                    return result_obj.to_xarray()

            sig_kws = [
                _param_to_field(k, p)
                for k, p in params.items()
                if p.kind is inspect.Parameter.KEYWORD_ONLY and not k.startswith('_')
            ]

            struct_type = msgspec.defstruct(
                name,
                sig_kws,
                bases=(KeywordArguments,),
                forbid_unknown_fields=True,
                frozen=True,
            )

            def hook(type_):
                return type_

            # validate the struct
            msgspec.json.schema(struct_type, schema_hook=hook)

            self[struct_type] = wrapped

            return wrapped

        return wrapper

    def spec_type(self) -> type[structs.ChannelAnalysis]:
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


register_xarray_measurement = ChannelAnalysisRegistry(structs.ChannelAnalysis)


def analyze_by_spec(
    iq: 'iqwaveform.util.Array',
    capture: structs.Capture,
    *,
    spec: str | dict | structs.ChannelAnalysis,
    as_xarray: typing.Literal[True]
    | typing.Literal[False]
    | typing.Literal['delayed'] = True,
    expand_dims=None,
) -> 'xr.Dataset':
    """evaluate a set of different channel analyses on the iq waveform as specified by spec"""

    results = evaluate_channel_analysis(
        iq,
        capture,
        spec=spec,
        registry=register_xarray_measurement,
        as_xarray='delayed' if as_xarray else False,
    )

    if as_xarray and as_xarray != 'delayed':
        return package_channel_analysis(capture, results, expand_dims=expand_dims)
    else:
        return results
