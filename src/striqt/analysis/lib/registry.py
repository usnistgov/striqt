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


# TODO: future approach to registering coordinates, rather than dataclasses?
class DelayedCoordinate(typing.NamedTuple):
    name: str
    func: callable
    dtype: str
    dims: tuple[str, ...] = None
    attrs: dict = {}


class _CoordinateRegistry(collections.UserDict[str, DelayedCoordinate]):
    def __call__[**P, R](
        self,
        dtype,
        *,
        name: str | None = None,
        dims: tuple[str, ...] | None = None,
        attrs={},
    ) -> typing.Callable[P, R]:
        kws = locals()

        def wrapper(func):
            if isinstance(kws['dims'], str):
                dims = tuple((kws['dims'],))
            else:
                dims = kws['dims']

            if kws['name'] is not None:
                name = str(kws['name'])
            else:
                try:
                    name = func.__name__
                except AttributeError as ex:
                    raise TypeError(
                        'specify the coordinate name with coordinates(name, ...)'
                    ) from ex

            if name in self:
                raise KeyError(
                    f'a coordinate has already been registered for coordinate {name!r}'
                )

            if dims is None:
                dims = (name,)

            self[name] = DelayedCoordinate(
                name=name, func=func, dims=dims, dtype=dtype, attrs=attrs
            )

            return func

        return wrapper

    def get(self, key):
        if not callable(key):
            print(repr(key))
            raise TypeError('coordinate key must be a registered callable')
        try:
            return super().get(key)
        except KeyError:
            raise TypeError(
                f'callable {repr(key)} has not been registerd as a coordinate'
            )

    def guess_dims(self, funcs: typing.Iterable[callable]) -> list[str]:
        """guess dimensions of a dataarray based on its coordinates"""
        dims = {}
        for func in funcs:
            coord = self.get(func)
            if coord is None:
                continue
            dims.update(dict.fromkeys(coord.dims, None))
        return list(dims.keys())


coordinate_factory = _CoordinateRegistry()


class _MeasurementRegistry(collections.UserDict):
    """a registry of keyword-only arguments for decorated functions"""

    def __init__(self):
        super().__init__()
        self.depends_on: dict[callable, set[callable]] = {}
        self.names: set[str] = set()

    def __call__[**P, R](
        self,
        # xarray_datacls: 'xarray_dataclasses.datamodel.DataClass',
        name: str | None = None,
        *,
        # coords: typing.Iterable[],
        dims: typing.Iterable[str] | str | None = None,
        coord_funcs: typing.Iterable[callable] | callable | None = None,
        depends: typing.Iterable[callable] = [],
        spec_type: type[specs.Measurement],
        dtype: str,
        attrs={},
    ) -> typing.Callable[P, R]:
        """add decorated `func` and its keyword arguments in the self.tostruct() schema"""
        if isinstance(dims, str):
            dims = (dims,)

        if coord_funcs is None:
            coord_funcs = ()
        elif callable(coord_funcs):
            coord_funcs = (coord_funcs,)
        else:
            for entry in coord_funcs:
                if not callable(entry):
                    raise TypeError('each coord_funcs item must be callable')

        if callable(depends):
            depends = (depends,)

        kws = locals()

        def wrapper(func: TFunc):
            if kws['name'] is None:
                name = func.__name__
            else:
                name = kws['name ']

            if name in self.names:
                raise TypeError(
                    f'a measurement named {repr(name)} was already registered'
                )
            else:
                self.names.add(name)

            if kws['dims'] is None:
                dims = coordinate_factory.guess_dims(coord_funcs)
            else:
                dims = kws['dims']

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
                    ret_metadata = dict(attrs, **ret_metadata)
                else:
                    result = ret
                    ret_metadata = attrs

                result = _DelayedDataArray(
                    result,
                    capture,
                    name=name,
                    dims=dims,
                    coord_factories=coord_funcs,
                    spec=spec,
                    dtype=dtype,
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

            self.depends_on[wrapped] = []
            for dep in depends:
                self.depends_on[dep].append(wrapped)

            # self.spec_types[name] = spec_type

            # def hook(type_):
            #     return type_

            # # validate the struct
            # msgspec.json.schema(struct_type, schema_hook=hook)

            self[spec_type] = wrapped

            return wrapped

        return wrapper

    def spec_types(self) -> type[specs.Measurement]:
        """return a Struct subclass type representing a specification for calls to all registered functions"""
        fields = [
            (func.__name__, typing.Union[struct_type, None], None)
            for struct_type, func in self.items()
        ]

        return msgspec.defstruct(
            'Analysis',
            fields,
            bases=(specs.Measurement,),
            kw_only=True,
            forbid_unknown_fields=True,
            omit_defaults=True,
            frozen=True,
        )


measurement = _MeasurementRegistry()
