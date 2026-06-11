import click
import json
import typing

if typing.TYPE_CHECKING:
    from pathlib import Path


@click.command('convert a yaml sweep spec file to json format')
@click.argument('path', type=click.Path(exists=True, dir_okay=False))
@click.option(
    '--yaml/',
    '-y',
    is_flag=True,
    show_default=True,
    default=False,
    help='output a flattened yaml file instead of json',
)
def cli(path, yaml: bool):
    out_path = run(path, yaml)
    print(f'wrote "{out_path!s}"')


def run(path, yaml: bool) -> 'Path':
    import striqt.sensor as ss
    import striqt.analysis as sa
    import msgspec
    import sys
    from pathlib import Path

    if path.endswith('.yaml') or path.endswith('.yml'):
        spec = ss.read_yaml_spec(path)
    elif path.endswith('.json'):
        spec = ss.read_json_spec(path)
    else:
        raise click.ClickException(f'unrecognized file extension for {path}')

    if yaml:
        out_path = Path(path).with_suffix('.yaml')
        if out_path == Path(path):
            out_path = out_path.with_stem(out_path.stem + '-flat')
        s = msgspec.yaml.encode(spec, enc_hook=sa.specs.helpers._enc_hook)
        with open(out_path, 'wb') as stream:
            stream.write(s)
    else:
        out_path = Path(path).with_suffix('.json')
        s = msgspec.json.encode(spec, enc_hook=sa.specs.helpers._enc_hook_no_tuple_keys)
        d = json.loads(s)
        with open(out_path, 'w') as stream:
            stream.write(_custom_indent(d))

    return out_path


def _custom_indent(obj, indent_level=2, current_indent=0):
    space = ' ' * indent_level

    if isinstance(obj, dict):
        if not obj:
            return '{}'
        items = []
        for k, v in obj.items():
            # Indent mappings (dictionaries)
            formatted_val = _custom_indent(
                v, indent_level, current_indent + indent_level
            )
            items.append(
                f'{space * (current_indent + indent_level)}{json.dumps(k)}: {formatted_val.lstrip()}'
            )
        return '{\n' + ',\n'.join(items) + f'\n{space * current_indent}}}'

    elif isinstance(obj, list):
        if not obj:
            return '[]'
        items = [_custom_indent(v, indent_level, current_indent) for v in obj]
        # Keep lists completely compact on a single line
        return '[' + ', '.join(items) + ']'

    else:
        # Base types (strings, numbers, booleans, null)
        return json.dumps(obj)


if __name__ == '__main__':
    cli()  # pyright: ignore
