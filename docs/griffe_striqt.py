"""griffe extension that substitutes striqt's runtime docstrings into the static tree.

Each hook imports the striqt object it documents, and griffe loads the whole `striqt`
package, so a documentation build imports every striqt module (SoapySDR through
`striqt.sensor`, matplotlib and plotly through `striqt.figures`).
"""

from __future__ import annotations

import ast
import inspect
from typing import Any

import griffe


def _runtime_object(path: str) -> Any:
    """return the object at the dotted `path` under ``striqt``, or None.

    None is returned for a path outside ``striqt`` and for a name absent at runtime
    (a class under ``if TYPE_CHECKING``, an optional dependency).
    """
    if not path.startswith('striqt.'):
        return None
    try:
        return griffe.dynamic_import(path)
    except (ImportError, AttributeError):
        return None


def _measurement_spec_type(func: Any) -> type | None:
    """return the spec type registered for the measurement wrapper `func`, or None."""
    from striqt.analysis import registry

    for spec_type, info in registry.items():
        if info.func is func:
            return spec_type
    return None


def _struct_field_docs(cls: Any) -> tuple[dict[str, str], dict[str, str]]:
    """return the ``(descriptions, types)`` of the fields of a msgspec Struct `cls`.

    Both dicts are empty when `cls` is not a Struct subclass.
    """
    import msgspec

    from striqt.analysis.specs.doc import describe_msgspec_fields

    if not (isinstance(cls, type) and issubclass(cls, msgspec.Struct)):
        return {}, {}
    return describe_msgspec_fields(cls)


def _annotation_string(annotation: Any) -> str | None:
    """return the display string of a runtime parameter `annotation`; None when empty."""
    if annotation is inspect.Parameter.empty:
        return None
    if isinstance(annotation, str):
        return annotation
    if isinstance(annotation, type):
        return f'{annotation.__module__}.{annotation.__qualname__}'
    return inspect.formatannotation(annotation)


def _measurement_parameters(
    runtime: Any, spec_type: type, static: griffe.Parameters
) -> griffe.Parameters:
    """return griffe parameters from the `__signature__` of the wrapper `runtime`.

    A parameter named as a field of `spec_type` is annotated with the display string
    of the field type. A parameter also present in `static` keeps the static
    annotation expression, which mkdocstrings renders as a cross-reference.
    """
    _, types = _struct_field_docs(spec_type)
    static_annotations = {p.name: p.annotation for p in static}

    parameters = []
    for p in inspect.signature(runtime).parameters.values():
        if p.name in types:
            annotation: str | griffe.Expr | None = types[p.name]
        elif p.name in static_annotations:
            annotation = static_annotations[p.name]
        else:
            annotation = _annotation_string(p.annotation)
        default = None if p.default is inspect.Parameter.empty else repr(p.default)
        parameters.append(
            griffe.Parameter(
                p.name,
                annotation=annotation,
                kind=griffe.ParameterKind[p.kind.name.lower()],
                default=default,
            )
        )
    return griffe.Parameters(*parameters)


def _binding_docstring(obj: Any) -> str | None:
    """return the docstring of a bound sensor controller class `obj`, or None.

    The docstring consists of the source backend name with its summary line, the
    ``sensor_binding`` key of the binding, and the spec types to which the ``source``
    block, the captures, and the action decode.
    """
    from striqt.sensor.lib import bindings, controller

    if not (isinstance(obj, type) and issubclass(obj, controller.Controller)):
        return None

    source_cls = obj.sensor.source_cls
    schema = obj.schema
    action_type = obj.sensor.sweep_spec_cls.__mro__[1]
    key = next((k for k, v in bindings.registry.items() if v is obj), None)

    backend = f'`{source_cls.__name__}` backend'
    summary = (inspect.getdoc(source_cls) or '').partition('\n')[0].rstrip('.')
    if summary:
        backend += f' ({summary})'
    selector = f', selected by ``sensor_binding: {key}``' if key else ''

    return (
        f'control the {backend}{selector}.\n\n'
        f'The ``source`` block of the action decodes to `{schema.source.__qualname__}`, '
        f'each entry of ``captures`` to `{schema.capture.__qualname__}`, and the '
        f'action to `{action_type.__qualname__}`.'
    )


def _docstring(
    text: str, parent: griffe.Object, agent: griffe.Visitor | griffe.Inspector
) -> griffe.Docstring:
    """return a docstring of `text` on `parent` with the parser of `agent`.

    A docstring built without a parser is rendered as one block of text.
    """
    return griffe.Docstring(
        text,
        parent=parent,
        parser=agent.docstring_parser,
        parser_options=agent.docstring_options,
    )


class StriqtRuntimeDocs(griffe.Extension):
    """substitute the runtime docstrings and signatures of striqt into the static tree.

    The hooks set the filled docstring and the `__signature__` of the wrapper on a
    registered measurement, the `Meta` description and the field type on each field
    of a msgspec Struct, and a docstring naming the backend and spec types on a
    module attribute bound by `bind_sensor`.
    """

    def on_function_instance(
        self,
        *,
        node: ast.AST | griffe.ObjectNode,
        func: griffe.Function,
        agent: griffe.Visitor | griffe.Inspector,
        **kwargs: Any,
    ) -> None:
        """replace the docstring and parameters of a registered measurement."""
        if isinstance(node, griffe.ObjectNode) or not func.decorators:
            return
        runtime = _runtime_object(func.path)
        if runtime is None:
            return
        spec_type = _measurement_spec_type(runtime)
        if spec_type is None:
            return

        doc = inspect.getdoc(runtime)
        if doc and func.docstring is not None:
            func.docstring.value = doc
        elif doc:
            func.docstring = _docstring(doc, func, agent)
        func.parameters = _measurement_parameters(runtime, spec_type, func.parameters)

    def on_class_members(
        self,
        *,
        node: ast.AST | griffe.ObjectNode,
        cls: griffe.Class,
        agent: griffe.Visitor | griffe.Inspector,
        **kwargs: Any,
    ) -> None:
        """set the docstring and annotation of each field of a msgspec Struct."""
        if isinstance(node, griffe.ObjectNode) or not cls.bases:
            return
        descriptions, types = _struct_field_docs(_runtime_object(cls.path))

        for name, member in cls.members.items():
            if isinstance(member, griffe.Alias) or not isinstance(
                member, griffe.Attribute
            ):
                continue
            if name in types:
                member.annotation = types[name]
            if member.docstring is None and descriptions.get(name):
                member.docstring = _docstring(descriptions[name], member, agent)

    def on_attribute_instance(
        self,
        *,
        node: ast.AST | griffe.ObjectNode,
        attr: griffe.Attribute,
        agent: griffe.Visitor | griffe.Inspector,
        **kwargs: Any,
    ) -> None:
        """set the docstring of a module attribute bound by `bind_sensor`."""
        # `_binding_docstring` imports `striqt.sensor`, and only that package binds
        # sensors
        if (
            isinstance(node, griffe.ObjectNode)
            or attr.docstring is not None
            or not attr.path.startswith('striqt.sensor.')
            or not isinstance(attr.value, griffe.ExprCall)
        ):
            return
        doc = _binding_docstring(_runtime_object(attr.path))
        if doc:
            attr.docstring = _docstring(doc, attr, agent)
