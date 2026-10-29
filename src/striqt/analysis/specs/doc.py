from __future__ import annotations as __

import functools
import types
import typing

import msgspec


def format_type_hint(hint) -> str:
    """return the display string of a type hint, without module paths."""
    # 1. Base cases: None, Any, and Ellipsis
    if hint is None or hint is type(None):
        return 'None'
    if hint is typing.Any:
        return 'Any'
    if hint is Ellipsis:
        return '...'

    origin = typing.get_origin(hint)
    args = typing.get_args(hint)

    if origin:
        # `types.UnionType` is absent on Python 3.9
        if origin is typing.Union or (
            hasattr(types, 'UnionType') and origin is types.UnionType
        ):
            return ' | '.join(format_type_hint(a) for a in args)

        # literals
        if origin is typing.Literal:
            return f'Literal[{", ".join(repr(a) for a in args)}]'

        origin_name = getattr(origin, '__name__', str(origin)).replace('typing.', '')

        # Recurse through the arguments
        if args:
            formatted_args = ', '.join(format_type_hint(a) for a in args)
            return f'{origin_name}[{formatted_args}]'

        return origin_name

    # standard classes and built-ins (e.g., int, str, MyStruct)
    if hasattr(hint, '__name__'):
        return hint.__name__

    return str(hint).replace('typing.', '')


def clean_type_string(type_str):
    """rewrite ``Literal[`` as ``one of [`` in a type string."""
    import re

    type_str = re.sub(r'\bLiteral\[', 'one of [', type_str)

    # # 2. Remap nested module paths to top-level paths
    # for old_path, new_path in MODULE_REMAPS.items():
    #     type_str = type_str.replace(old_path, new_path)

    return type_str


def parse_type_and_meta(hint):
    """split a type hint into the un-annotated hint and its `msgspec.Meta` objects.

    Returns:
        a ``(hint, metas)`` tuple: `hint` with every `Annotated` wrapper removed,
        recursively through the type arguments, and the list of the `msgspec.Meta`
        objects found; a `Literal` hint is returned unchanged with an empty list
    """

    origin = typing.get_origin(hint)
    args = typing.get_args(hint)

    if origin is typing.Annotated:
        clean_base, metas = parse_type_and_meta(args[0])
        msgspec_metas = [m for m in args[1:] if isinstance(m, msgspec.Meta)]
        return clean_base, msgspec_metas + metas

    if origin is typing.Literal:
        return hint, []

    if args:
        cleaned_args = []
        all_metas = []

        for arg in args:
            if arg is Ellipsis:
                cleaned_args.append(Ellipsis)
            else:
                clean_arg, metas = parse_type_and_meta(arg)
                cleaned_args.append(clean_arg)
                all_metas.extend(metas)

        try:
            if hasattr(types, 'UnionType') and origin is types.UnionType:
                import operator
                from functools import reduce

                clean_type = reduce(operator.or_, cleaned_args)
            else:
                clean_type = origin[tuple(cleaned_args)]
        except Exception:
            clean_type = hint

        return clean_type, all_metas

    return hint, []


@functools.cache
def describe_msgspec_fields(obj: type[msgspec.Struct]):
    """return the ``(descriptions, types)`` dicts of the fields of `obj`, keyed by name.

    A description is the first `Meta.description` of the field, followed by the
    constraints and units in parentheses. A type is the display string of the field
    annotation. Both dicts are empty when the type hints of `obj` do not resolve.
    Results are cached per struct type.
    """
    descriptions = {}
    types = {}

    try:
        hints = typing.get_type_hints(obj, include_extras=True)
        struct_fields = {f.name for f in msgspec.structs.fields(obj)}
    except Exception:
        return {}, {}

    for field_name, field_type in hints.items():
        if field_name in struct_fields:
            clean_type, metas = parse_type_and_meta(field_type)

            description = ''
            constraints = []

            for meta in metas:
                if meta.description and not description:
                    description = meta.description

                if getattr(meta, 'gt', None) is not None:
                    constraints.append(f'> {meta.gt}')
                if getattr(meta, 'ge', None) is not None:
                    constraints.append(f'>= {meta.ge}')
                if getattr(meta, 'lt', None) is not None:
                    constraints.append(f'< {meta.lt}')
                if getattr(meta, 'le', None) is not None:
                    constraints.append(f'<= {meta.le}')
                if getattr(meta, 'multiple_of', None) is not None:
                    constraints.append(f'multiple of {meta.multiple_of}')
                if getattr(meta, 'min_length', None) is not None:
                    constraints.append(f'min_length={meta.min_length}')
                if getattr(meta, 'max_length', None) is not None:
                    constraints.append(f'max_length={meta.max_length}')
                if getattr(meta, 'pattern', None) is not None:
                    constraints.append(f"pattern='{meta.pattern}'")

                if meta.extra and 'units' in meta.extra:
                    constraints.append(f'units: {meta.extra["units"]}')

            if constraints:
                constraint_str = f' *(Constraints: {", ".join(constraints)})*'
                if description:
                    description += constraint_str
                else:
                    description = constraint_str.strip()

            raw_type_str = format_type_hint(clean_type)
            final_type_str = clean_type_string(raw_type_str)

            descriptions[field_name] = description
            types[field_name] = final_type_str

    return descriptions, types


def struct_args_docstring(
    obj,
    indent_spaces: int = 4,
    extra_prepend: dict[str, str] | None = None,
    extra_append: dict[str, str] | None = None,
    extra_types: dict[str, str] | None = None,
) -> str:
    """return the ``Args:`` entries of a Google-style docstring for the fields of `obj`.

    Args:
        obj: a `msgspec.Struct` type or instance
        indent_spaces: indentation of each entry, in spaces
        extra_prepend: descriptions of entries placed before the fields, keyed by name
        extra_append: descriptions of entries placed after the fields, keyed by name
        extra_types: type strings that replace the formatted field types, keyed by name

    Returns:
        the entries, one per name, separated by blank lines; an empty string when
        there are no entries
    """
    if not isinstance(obj, type):
        obj = type(obj)
        return struct_args_docstring(**locals())
    descriptions, types = describe_msgspec_fields(obj)
    descriptions = (extra_prepend or {}) | descriptions | (extra_append or {})
    types = types | (extra_types or {})

    if not descriptions and not types:
        return ''

    lines = []
    base_indent = ' ' * indent_spaces

    # both dicts have the field names as keys; only the `extra_*` names can differ
    for field_name in descriptions:
        type_str = types.get(field_name, 'Any')
        desc = descriptions.get(field_name, '')

        if desc:
            # Google style: param_name (type): description
            lines.append(f'{base_indent}{field_name} ({type_str}): {desc}')
        else:
            # Fallback if there is no description provided
            lines.append(f'{base_indent}{field_name} ({type_str}):')

    return '\n\n'.join(lines)
