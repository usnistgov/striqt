from __future__ import annotations
import inspect
import msgspec
from collections import UserDict
import typing


TDecoratedFunc = typing.Callable[..., typing.Any]


class _CallConfig(msgspec.Struct):
    pass


class _AnalysisConfig(msgspec.Struct):
    pass


class KeywordConfigRegistry(UserDict):
    def __init__(self, field_struct=None, parent_struct=None):
        super().__init__()
        self.field_cls = field_struct
        self.parent_cls = parent_struct

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
        name = func.__name__

        params = inspect.signature(func).parameters

        kws = [
            self._param_to_field(k, p)
            for k, p in params.items()
            if p.kind is inspect.Parameter.KEYWORD_ONLY
        ]

        struct = msgspec.defstruct(
            name, kws, bases=(self.field_cls,) if self.field_cls else None
        )

        # validate the struct
        msgspec.json.schema(struct)

        self[func] = struct

        return func

    def tospec(self) -> msgspec.Struct:
        """return a Struct representing a specification for calls to all registered functions"""
        fields = [
            (func.__name__, typing.Union[struct, None], None)
            for func, struct in self.items()
        ]

        return msgspec.defstruct(
            'channel_analysis',
            fields,
            bases=(self.parent_cls,) if self.parent_cls else None,
        )


registry = KeywordConfigRegistry(
    field_struct=_CallConfig, parent_struct=_AnalysisConfig
)


def from_any(obj: str | dict | registry.parent_cls) -> registry.parent_cls:
    """return a channel analysis specification from a yaml string,
    dictionary of dictionaries, or channel analysis specification
    """
    struct = registry.tospec()

    if isinstance(obj, registry.parent_cls):
        return obj
    elif isinstance(obj, dict):
        return struct(**obj)
    elif isinstance(obj, str):
        return msgspec.yaml.decode(obj, type=struct)
    else:
        return TypeError('unrecognized type')
