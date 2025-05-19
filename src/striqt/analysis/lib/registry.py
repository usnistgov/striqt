"""Implement a registry to track and wrap measurement functions that transform numpy return values
into xarray DataArray objects with labeled dimensions and coordinates.
"""

from __future__ import annotations

import collections
import functools
import inspect
import msgspec
import typing

from . import specs


if typing.TYPE_CHECKING:
    import xarray_dataclasses


TFunc = typing.Callable[..., typing.Any]


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


class KeywordArguments(specs.StructBase):
    """base class for the keyword argument parameters of an analysis function"""


class _AnalysisRegistry(collections.UserDict):
    """a registry of keyword-only arguments for decorated functions"""

    spec_base_type = specs.Analysis

    def __init__(self):
        super().__init__()
        self.bases: dict[str, set[callable]] = {}
        self.names = set()

    def __call__[**P, R](
        self,
        xarray_datacls: 'xarray_dataclasses.datamodel.DataClass',
        basis: str,
        spec_type: type[specs.Analysis],
        metadata={},
    ) -> typing.Callable[P, R]:
        """add decorated `func` and its keyword arguments in the self.tostruct() schema"""

        def wrapper(func: TFunc):
            name = func.__name__
            if name in self.names:
                raise TypeError(f'a function named {repr(name)} was already registered')
            else:
                self.names.add(name)

            # sig = inspect.signature(func)

            @functools.wraps(func)
            def wrapped(iq, capture, **kws):
                from .xarray_ops import _DelayedDataArray

                # injects and handles an additional argument, 'as_xarray', which allows
                # the return of a ChannelAnalysis result for fast serialization and
                # xarray object instantiation
                as_xarray = kws.pop('as_xarray', True)
                if as_xarray not in ('delayed', True, False):
                    raise ValueError(
                        'xarray argument must be one of (True, False, "delayed")'
                    )

                spec = spec_type.fromdict(kws)
                ret = func(iq, capture, **spec.todict())

                if not as_xarray:
                    return ret

                if isinstance(ret, (list, tuple)) and len(ret) == 2:
                    result, ret_metadata = ret
                    ret_metadata = dict(metadata, **ret_metadata)
                else:
                    result = ret
                    ret_metadata = metadata

                result = _DelayedDataArray(
                    xarray_datacls,
                    result,
                    capture,
                    spec=spec,
                    attrs=ret_metadata,
                )

                if as_xarray == 'delayed':
                    return result
                else:
                    return result.to_xarray()

            name = wrapped.__name__
            if name in self:
                raise TypeError(
                    f'another function named {repr(name)} has already been registered'
                )
            # sig = inspect.signature(wrapped)
            # params = sig.parameters

            # sig_kws = [
            #     _param_to_field(k, p)
            #     for k, p in params.items()
            #     if p.kind is inspect.Parameter.KEYWORD_ONLY and not k.startswith('_')
            # ]

            # self.bases.setdefault(basis, []).append(name)
            # self.spec_types[name] = spec_type

            # def hook(type_):
            #     return type_

            # # validate the struct
            # msgspec.json.schema(struct_type, schema_hook=hook)

            self[spec_type] = wrapped

            return wrapped

        return wrapper

    def spec_types(self) -> type[specs.Analysis]:
        """return a Struct subclass type representing a specification for calls to all registered functions"""
        fields = [
            (func.__name__, typing.Union[struct_type, None], None)
            for struct_type, func in self.items()
        ]

        return msgspec.defstruct(
            'Analysis',
            fields,
            bases=(self.spec_base_type,),
            kw_only=True,
            forbid_unknown_fields=True,
            omit_defaults=True,
            frozen=True,
        )


measurement = _AnalysisRegistry()


# TODO: future approach to registering coordinates, rather than dataclasses?
# class _RegisteredCoordinate(typing.NamedTuple):
#     dims: tuple[str, ...]
#     dtype: str
#     standard_name: str
#     units: typing.Optional[str]
#     func: callable


# class _CoordinateRegistry(collections.UserDict[str, _RegisteredCoordinate]):
#     def __call__(self, name: str|None = None, *, dims: tuple[str, ...], dtype, standard_name, units=None):
#         if isinstance(dims, str):
#             dims = tuple((dims,))
#         else:
#             dims = tuple(dims)
#             if len(dims) == 0:
#                 raise ValueError('dims must be a string or an iterable of strings')

#         if name is not None:
#             name = str(name)

#         def wrapper(func):
#             if name is None: # noqa: F823
#                 try:
#                     name = (func.__name__,)
#                 except AttributeError as ex:
#                     raise TypeError('specify the coordinate name with register_coordinate(name, ...)')from ex

#             if name in self:
#                 raise KeyError(f'a coordinate has already been registered for dimension {dims!r}')

#             self[name] = _RegisteredCoordinate(
#                 dims=dims, dtype=dtype, standard_name=standard_name, units=units
#             )

#             return func

#         return wrapper

# coordinate = _CoordinateRegistry()
