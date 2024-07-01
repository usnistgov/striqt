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
            raise TypeError(f'to register this function, keyword-only argument "{name}" of needs a type annotation')

        if p.default is inspect._empty:
            return (name, p.annotation)
        else:
            return (name, p.annotation, p.default)

    def addfunc(self, func: callable):
        """introspect keyword-only arguments in callable and add a corresponding msgspec.Struct to self"""
        name = func.__name__

        params = inspect.signature(func).parameters

        kws = [
            self._param_to_field(k, p)
            for k, p in params.items()
            if p.kind is inspect.Parameter.KEYWORD_ONLY
        ]

        struct = msgspec.defstruct(
            name,
            kws,
            bases=(self.field_cls,)
        )

        # validate the struct
        msgspec.json.schema(struct)

        self[func] = struct

    def decorator_factory(self) -> callable[[TDecoratedFunc],TDecoratedFunc]:
        """return a callable """
        def registry_decorator(func: TDecoratedFunc) -> TDecoratedFunc:
            self.addfunc(func)
            return func
        
        return registry_decorator

    def tospec(self) -> msgspec.Struct:
        """return a Struct representing a specification for calls to all registered functions"""
        fields = [
            (func.__name__, typing.Union[struct,None], None)
            for func, struct in self.items()
        ]

        return msgspec.defstruct(
            'channel_analysis',
            fields,
            bases=(_AnalysisConfig,)
        )
    

registry = KeywordConfigRegistry(field_struct=_CallConfig, parent_struct=_AnalysisConfig)
