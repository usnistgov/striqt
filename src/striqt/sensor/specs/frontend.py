"""editor schema of an action spec class: each field with its owning class, kind,
choices, per-port flag, default, units, constraints, and nesting.

Field hints are read from `msgspec.structs.fields`, which holds the resolved
`Annotated` hints with the `PER_PORT` marker and the `Meta` extras (`standard_name`,
`units`).
"""

from __future__ import annotations as __

import dataclasses
import fractions
import math
import typing
from typing import Any, Literal, Optional, Union, get_args, get_origin

import msgspec

import striqt.analysis as sa
from striqt.analysis.specs import helpers
from striqt.analysis.specs.helpers import field_owners, qualname, resolve_hint

from . import captures, structs

Kind = Literal[
    'number',
    'integer',
    'string',
    'boolean',
    'literal',
    'fraction',
    'struct',
    'list',
    'dict',
    'opaque',
]
SectionKind = Literal[
    'struct', 'struct_list', 'tagged_list', 'optional_struct_dict', 'dict', 'opaque'
]

# the generic parameters of `ActionSpec` and `CapturePlan`, which an unbound action
# class leaves in its field hints; `striqt.sensor.lib.typing` declares each bound as
# a string, which `resolve_hint` does not evaluate
_TYPEVAR_BOUNDS: dict[str, type] = {
    'SS': structs.Source,
    'SC': structs.SensorCapture,
    'SP': structs.Peripherals,
    'SA': sa.specs.AnalysisGroup,
}

# top-level fields that an editor shows but does not edit, whatever the hint
_OPAQUE_SECTIONS = frozenset({'adjust_captures', 'plot_hint', 'mock_source'})


@dataclasses.dataclass(frozen=True)
class FieldSchema:
    """the editor schema of one struct field, or of one element of a list or dict field."""

    name: str
    """field name; empty for the element schema of a list or dict"""
    owner: str
    """qualified name of the class that first declares the field in the MRO"""
    kind: Kind
    """value kind, one of `Kind`"""
    choices: tuple[Any, ...] = ()
    """`Literal` values in declaration order, with the True and False of a bool member first"""
    per_port: bool = False
    """True when the hint carries the `PER_PORT` marker"""
    optional: bool = False
    """True when None is accepted"""
    required: bool = False
    """True when the field has no default"""
    default: Any = None
    """JSON-able default (a `Fraction` as a string, a struct as a dict, a non-finite float as a string); None when the field is required or defaults to None"""
    units: Optional[str] = None
    """the `units` of the outermost `Meta`"""
    standard_name: Optional[str] = None
    """the `standard_name` of the outermost `Meta`"""
    description: Optional[str] = None
    """the `Meta.description` when it differs from `standard_name`"""
    ge: Optional[float] = None
    """lower bound (inclusive) from the `Meta` constraints"""
    gt: Optional[float] = None
    """lower bound (exclusive) from the `Meta` constraints"""
    le: Optional[float] = None
    """upper bound (inclusive) from the `Meta` constraints"""
    lt: Optional[float] = None
    """upper bound (exclusive) from the `Meta` constraints"""
    length: Optional[int] = None
    """element count of a fixed-length tuple of kind ``'list'``; None for a variadic tuple"""
    item: Optional[FieldSchema] = None
    """element schema, for kind ``'list'`` or ``'dict'``"""
    fields: tuple[FieldSchema, ...] = ()
    """fields of the struct, for kind ``'struct'`` without `variants`"""
    tag: Optional[str] = None
    """tag field name of a tagged union of structs"""
    variants: tuple[StructSchema, ...] = ()
    """the structs of a tagged union, in declaration order"""
    read_only: bool = False
    """True for kind ``'opaque'``"""


@dataclasses.dataclass(frozen=True)
class StructSchema:
    """the editor schema of one struct class: the fields in declaration order with the tag of the class."""

    cls: str
    """qualified class name"""
    tag_value: Optional[str]
    """the tag of a tagged struct, else None"""
    fields: tuple[FieldSchema, ...]
    """the field schemas in `__struct_fields__` order"""


@dataclasses.dataclass(frozen=True)
class SectionSchema:
    """one top-level field of an action spec as an editor section.

    `schema` describes the element for ``'struct_list'`` (a `StructSchema`),
    ``'tagged_list'`` (a `FieldSchema` with `tag` and `variants`), and ``'dict'`` (a
    `FieldSchema`); the struct for ``'struct'`` and ``'optional_struct_dict'`` (a
    `StructSchema`); and the field itself for ``'opaque'`` (a `FieldSchema`).
    """

    name: str
    """field name on the action spec"""
    kind: SectionKind
    """section kind, one of `SectionKind`"""
    field: FieldSchema
    """the schema of the top-level field, with the default and description of the field"""
    schema: Union[StructSchema, FieldSchema]
    """the element or struct schema selected by `kind`"""
    read_only: bool
    """True for kind ``'opaque'``"""


@dataclasses.dataclass(frozen=True)
class SpecSchema:
    """the editor schema of an action spec class: one section per top-level field."""

    action_cls: str
    """qualified name of the action spec class"""
    sensor_binding: Optional[str]
    """the ``sensor_binding`` tag of a bound class; None for an unbound class"""
    sections: tuple[SectionSchema, ...]
    """the sections in `__struct_fields__` order"""


@dataclasses.dataclass(frozen=True)
class _Shape:
    """the kind of a resolved hint and the nested schemas of that kind"""

    kind: Kind
    choices: tuple[Any, ...] = ()
    length: Optional[int] = None
    item: Optional[FieldSchema] = None
    fields: tuple[FieldSchema, ...] = ()
    tag: Optional[str] = None
    variants: tuple[StructSchema, ...] = ()


def _is_class(hint: Any) -> bool:
    # on python 3.9, isinstance(tuple[int, ...], type) is True
    return isinstance(hint, type) and get_origin(hint) is None


def _is_struct(hint: Any) -> bool:
    return _is_class(hint) and issubclass(hint, msgspec.Struct)


def _literal_choices(members: tuple[Any, ...]) -> Optional[tuple[Any, ...]]:
    """return the choices of a union of `Literal` and bool members, else None."""
    if not members or not all(m is bool or get_origin(m) is Literal for m in members):
        return None
    choices: list[Any] = [True, False] if bool in members else []
    for m in members:
        if m is not bool:
            choices.extend(get_args(m))
    return tuple(choices)


def _tagged_union(
    members: tuple[Any, ...],
) -> Optional[tuple[str, tuple[type[msgspec.Struct], ...]]]:
    """return the tag field and the structs of a tagged union, else None."""
    if len(members) < 2 or not all(_is_struct(m) for m in members):
        return None
    tag_fields = {m.__struct_config__.tag_field for m in members}
    tags = [m.__struct_config__.tag for m in members]
    if len(tag_fields) != 1 or None in tag_fields or None in tags:
        return None
    structs_ = typing.cast('tuple[type[msgspec.Struct], ...]', tuple(members))
    return tag_fields.pop(), structs_


def _element(hint: Any, owner: type) -> FieldSchema:
    return _field_schema('', hint, owner=owner, default=None, required=False)


def _shape_one(hint: Any, owner: type) -> _Shape:
    """classify a single resolved hint."""
    if hint is bool:
        return _Shape('boolean')
    if hint is int:
        return _Shape('integer')
    if hint is float:
        return _Shape('number')
    if hint is str:
        return _Shape('string')
    if _is_class(hint) and issubclass(hint, fractions.Fraction):
        return _Shape('fraction')
    if _is_struct(hint):
        return _Shape('struct', fields=struct_schema(hint).fields)

    origin = get_origin(hint)
    args = get_args(hint)
    if origin is tuple and args:
        if args[-1] is Ellipsis:
            return _Shape('list', item=_element(args[0], owner))
        if all(a == args[0] for a in args):
            return _Shape('list', item=_element(args[0], owner), length=len(args))
        return _Shape('opaque')
    if origin is list and args:
        return _Shape('list', item=_element(args[0], owner))
    if origin is dict and len(args) == 2:
        key, value = args
        if resolve_hint(key, bounds=_TYPEVAR_BOUNDS).members != (str,) or value is Any:
            return _Shape('opaque')
        return _Shape('dict', item=_element(value, owner))
    return _Shape('opaque')


def _shape(members: tuple[Any, ...], owner: type) -> _Shape:
    """classify the resolved members of a hint.

    The first matching rule sets the kind:

    - no member: ``'opaque'``
    - `Literal` and bool members only: ``'literal'``
    - one member: the kind of that type (`_shape_one`)
    - int and float: ``'number'``
    - structs tagged on one field: ``'struct'`` with `variants`
    - any other mix: ``'opaque'``
    """
    if not members:
        return _Shape('opaque')
    if members != (bool,):
        choices = _literal_choices(members)
        if choices is not None:
            return _Shape('literal', choices=choices)
    if len(members) == 1:
        return _shape_one(members[0], owner)
    if set(members) <= {int, float}:
        return _Shape('number')
    tagged = _tagged_union(members)
    if tagged is not None:
        tag, variants = tagged
        return _Shape(
            'struct', tag=tag, variants=tuple(struct_schema(v) for v in variants)
        )
    return _Shape('opaque')


def _first(metas: tuple[msgspec.Meta, ...], attr: str) -> Any:
    for meta in metas:
        value = getattr(meta, attr, None)
        if value is not None:
            return value
    return None


def _first_extra(metas: tuple[msgspec.Meta, ...], key: str) -> Optional[str]:
    for meta in metas:
        if meta.extra and meta.extra.get(key) is not None:
            return meta.extra[key]
    return None


def _jsonable(value: Any) -> Any:
    """return `value` with each tuple as a list and each non-finite float as a string."""
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, float) and not math.isfinite(value):
        # msgspec.json writes inf and nan as null
        return str(value)
    return value


def _json_default(value: Any) -> Any:
    if value is None or value is msgspec.NODEFAULT:
        return None
    builtins = msgspec.to_builtins(value, enc_hook=helpers._enc_hook_no_tuple_keys)
    return _jsonable(builtins)


def _raw_default(info: msgspec.structs.FieldInfo) -> Any:
    """return the default of `info`, from the default factory when one is set.

    A factory that raises `TypeError` or `ValueError` gives None, and a required
    field gives `msgspec.NODEFAULT`.
    """
    if info.default_factory is not msgspec.NODEFAULT:
        factory = typing.cast('typing.Callable[[], Any]', info.default_factory)
        try:
            return factory()
        except (TypeError, ValueError):
            # a struct factory with required fields of its own
            return None
    return info.default


def _field_schema(
    name: str, hint: Any, *, owner: type, default: Any, required: bool
) -> FieldSchema:
    """return the editor schema of one field from its resolved type hint.

    Args:
        name: the field name; empty for the element of a list or dict
        hint: the field annotation from `msgspec.structs.fields`, with its
            `Annotated` wrappers
        owner: the class that first declares the field
        default: the default value, or None
        required: True when the field has no default

    Returns:
        a `FieldSchema` with `per_port` from the `PER_PORT` marker, `optional` from
        a None member, `kind` and the nested schemas from the remaining members, and
        the units, names, and bounds from the outermost `Meta` that sets each
    """
    resolved = resolve_hint(hint, bounds=_TYPEVAR_BOUNDS)
    shape = _shape(resolved.members, owner)
    standard_name = _first_extra(resolved.metas, 'standard_name')
    description = _first(resolved.metas, 'description')
    if description == standard_name:
        description = None
    return FieldSchema(
        name=name,
        owner=qualname(owner),
        kind=shape.kind,
        choices=shape.choices,
        per_port=resolved.per_port,
        optional=resolved.optional,
        required=required,
        default=None if required else _json_default(default),
        units=_first_extra(resolved.metas, 'units'),
        standard_name=standard_name,
        description=description,
        ge=_first(resolved.metas, 'ge'),
        gt=_first(resolved.metas, 'gt'),
        le=_first(resolved.metas, 'le'),
        lt=_first(resolved.metas, 'lt'),
        length=shape.length,
        item=shape.item,
        fields=shape.fields,
        tag=shape.tag,
        variants=shape.variants,
        read_only=shape.kind == 'opaque',
    )


def struct_schema(cls: type[msgspec.Struct]) -> StructSchema:
    """return the editor schema of the fields of the struct class `cls`.

    Returns:
        a `StructSchema` with one `FieldSchema` per entry of `__struct_fields__`,
        each with the owner from `field_owners`
    """
    owners = field_owners(cls)
    fields = tuple(
        _field_schema(
            info.name,
            info.type,
            owner=owners[info.name],
            default=_raw_default(info),
            required=info.required,
        )
        for info in msgspec.structs.fields(cls)
    )
    tag = cls.__struct_config__.tag
    return StructSchema(
        cls=qualname(cls),
        tag_value=None if tag is None else str(tag),
        fields=fields,
    )


def _is_optional_struct_dict(struct: StructSchema) -> bool:
    """return True when every field of `struct` is an optional struct without variants."""
    return bool(struct.fields) and all(
        f.kind == 'struct' and f.optional and not f.variants for f in struct.fields
    )


def _struct_of(hint: Any) -> Optional[type[msgspec.Struct]]:
    members = resolve_hint(hint, bounds=_TYPEVAR_BOUNDS).members
    if len(members) == 1 and _is_struct(members[0]):
        return typing.cast('type[msgspec.Struct]', members[0])
    return None


def _section(spec_cls: type, name: str, hint: Any, field: FieldSchema) -> SectionSchema:
    """classify the top-level field `name` of `spec_cls` with the hint `hint` as a section.

    The first matching rule sets the section kind:

    - a name in `_OPAQUE_SECTIONS`, or a field of kind ``'opaque'``: ``'opaque'``
    - a struct with one or more fields, all optional structs (an analysis group):
      ``'optional_struct_dict'``
    - any other struct: ``'struct'``
    - a list of a tagged union: ``'tagged_list'``
    - a list of one struct: ``'struct_list'``, with the capture type of `spec_cls`
      for ``captures``
    - a dict: ``'dict'``
    - any other field: ``'opaque'``
    """
    opaque = SectionSchema(name, 'opaque', field, field, True)
    if name in _OPAQUE_SECTIONS or field.kind == 'opaque':
        return opaque

    if field.kind == 'struct' and not field.variants:
        cls = _struct_of(hint)
        if cls is None:
            return opaque
        struct = struct_schema(cls)
        if _is_optional_struct_dict(struct):
            return SectionSchema(name, 'optional_struct_dict', field, struct, False)
        return SectionSchema(name, 'struct', field, struct, False)

    if field.kind == 'list' and field.item is not None and field.item.kind == 'struct':
        if field.item.variants:
            return SectionSchema(name, 'tagged_list', field, field.item, False)
        if name == 'captures':
            element: Any = captures.get_capture_type(typing.cast('Any', spec_cls))
        else:
            element = get_args(resolve_hint(hint, bounds=_TYPEVAR_BOUNDS).members[0])[0]
        cls = _struct_of(element)
        if cls is None:
            return opaque
        return SectionSchema(name, 'struct_list', field, struct_schema(cls), False)

    if field.kind == 'dict' and field.item is not None:
        return SectionSchema(name, 'dict', field, field.item, False)

    return opaque


def spec_schema(spec_cls: type[structs.ActionSpec]) -> SpecSchema:
    """return the editor schema of the action spec class `spec_cls`.

    Args:
        spec_cls: a bound action class from a sensor binding, or an unbound class
            such as `Sweep`; a generic parameter of an unbound class resolves to
            the bound of the parameter (`Source`, `SensorCapture`, `Peripherals`,
            `AnalysisGroup`), and a field with a struct default resolves to the
            class of the default

    Returns:
        a `SpecSchema` with one section per top-level field in declaration order
        and `sensor_binding` from the tag of `spec_cls`
    """
    owners = field_owners(spec_cls)
    sections = []
    for info in msgspec.structs.fields(spec_cls):
        default = _raw_default(info)
        hint = info.type
        if isinstance(hint, typing.TypeVar) and isinstance(default, msgspec.Struct):
            hint = type(default)
        field = _field_schema(
            info.name,
            hint,
            owner=owners[info.name],
            default=default,
            required=info.required,
        )
        sections.append(_section(spec_cls, info.name, hint, field))
    tag = spec_cls.__struct_config__.tag
    return SpecSchema(
        action_cls=qualname(spec_cls),
        sensor_binding=None if tag is None else str(tag),
        sections=tuple(sections),
    )


def schema_json(spec_cls: type[structs.ActionSpec]) -> dict[str, Any]:
    """return `spec_schema(spec_cls)` as a dict of JSON-able values.

    Returns:
        the `dataclasses.asdict` form of the schema after a round trip through
        `msgspec.json`, so tuples are lists and every leaf is a JSON value
    """
    encoded = msgspec.json.encode(dataclasses.asdict(spec_schema(spec_cls)))
    return msgspec.json.decode(encoded)
