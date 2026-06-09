from __future__ import annotations as __
from IPython.utils.text import indent
import functools
import msgspec
import typing
import types


def format_type_hint(hint) -> str:
    """
    Recursively formats a Python type hint into a clean, readable string,
    replacing the need for Sphinx's stringify_annotation.
    """
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
        # modern pipe syntax for unions
        if origin is typing.Union or (
            hasattr(types, 'UnionType') and origin is types.UnionType
        ):
            return ' | '.join(format_type_hint(a) for a in args)

        # literals
        if origin is typing.Literal:
            return f'Literal[{", ".join(repr(a) for a in args)}]'

        # Get the clean name of the container (e.g., 'list', 'dict', 'tuple')
        origin_name = getattr(origin, '__name__', str(origin)).replace('typing.', '')

        # Recurse through the arguments
        if args:
            formatted_args = ', '.join(format_type_hint(a) for a in args)
            return f'{origin_name}[{formatted_args}]'

        return origin_name

    # standard classes and built-ins (e.g., int, str, MyStruct)
    if hasattr(hint, '__name__'):
        return hint.__name__

    # fallback for edge cases
    return str(hint).replace('typing.', '')


def clean_type_string(type_str):
    """Applies string replacements to clean up Literal and module paths."""
    # 1. Clean up Literal
    import re

    type_str = re.sub(r'\bLiteral\[', 'one of [', type_str)

    # # 2. Remap nested module paths to top-level paths
    # for old_path, new_path in MODULE_REMAPS.items():
    #     type_str = type_str.replace(old_path, new_path)

    return type_str


def parse_type_and_meta(hint):
    """
    Recursively walks a type hint to extract msgspec.Meta objects.
    Safely handles Literal values, Ellipsis, Unions, and generic containers.
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
    """
    Extracts descriptions, constraints, and clean type strings from a msgspec.Struct.
    Returns a tuple of two dictionaries: (descriptions, types).
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
    obj, indent_spaces: int = 4, extra_prepend={}, extra_append={}, extra_types={}
) -> str:
    """
    Generates a 'Args:' docstring block from parsed field metadata.

    Args:
        obj: the Struct to document
        indent_spaces (int): The number of spaces to indent the arguments.

    Returns:
        str: A fully formatted Google-style docstring segment.
    """
    if not isinstance(obj, type):
        obj = type(obj)
        return struct_args_docstring(**locals())
    descriptions, types = describe_msgspec_fields(obj)
    descriptions = extra_prepend | descriptions | extra_append
    types = types | extra_types

    if not descriptions and not types:
        return ''

    lines = []
    base_indent = ' ' * indent_spaces

    # We iterate over descriptions.keys() assuming describe_msgspec_fields
    # populates both dictionaries with the exact same keys in the same order.
    for field_name in descriptions.keys():
        type_str = types.get(field_name, 'Any')
        desc = descriptions.get(field_name, '')

        if desc:
            # Google style: param_name (type): description
            lines.append(f'{base_indent}{field_name} ({type_str}): {desc}')
        else:
            # Fallback if there is no description provided
            lines.append(f'{base_indent}{field_name} ({type_str}):')

    return '\n\n'.join(lines)
