from __future__ import annotations
import inspect
import msgspec
from collections import UserDict
import typing


TDecoratedFunc = typing.Callable[..., typing.Any]


class KeywordArgumentStruct(msgspec.Struct):
    pass


class AnalysisStruct(msgspec.Struct):
    pass


class KeywordConfigRegistry(UserDict):
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

    def include(self, func: TDecoratedFunc) -> TDecoratedFunc:
        """add decorated `func` and its keyword arguments in the self.tostruct() schema"""
        name = func.__name__

        params = inspect.signature(func).parameters

        kws = [
            self._param_to_field(k, p)
            for k, p in params.items()
            if p.kind is inspect.Parameter.KEYWORD_ONLY
        ]

        struct = msgspec.defstruct(name, kws, bases=(KeywordArgumentStruct,))

        # validate the struct
        msgspec.json.schema(struct)

        self[func] = struct

        return func

    def tostruct(self) -> msgspec.Struct:
        """return a Struct representing a specification for calls to all registered functions"""
        fields = [
            (func.__name__, typing.Union[struct, None], None)
            for func, struct in self.items()
        ]

        return msgspec.defstruct(
            'channel_analysis',
            fields,
            bases=(self.base_struct,) if self.base_struct else None,
        )


registry = KeywordConfigRegistry(AnalysisStruct)


def from_any(obj: str | dict | registry.base_struct) -> registry.base_struct:
    """return a channel analysis specification from a yaml string,
    dictionary of dictionaries, or channel analysis specification
    """
    struct = registry.tostruct()

    if isinstance(obj, (dict,registry.base_struct)):
        return msgspec.convert(obj, struct)
    elif isinstance(obj, str):
        return msgspec.yaml.decode(obj, type=struct)
    else:
        return TypeError('unrecognized type')
