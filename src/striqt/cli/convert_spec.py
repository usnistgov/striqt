import click
import json


def custom_indent(obj, indent_level=2, current_indent=0):
    space = ' ' * indent_level

    if isinstance(obj, dict):
        if not obj:
            return '{}'
        items = []
        for k, v in obj.items():
            # Indent mappings (dictionaries)
            formatted_val = custom_indent(
                v, indent_level, current_indent + indent_level
            )
            items.append(
                f'{space * (current_indent + indent_level)}{json.dumps(k)}: {formatted_val.lstrip()}'
            )
        return '{\n' + ',\n'.join(items) + f'\n{space * current_indent}}}'

    elif isinstance(obj, list):
        if not obj:
            return '[]'
        items = [custom_indent(v, indent_level, current_indent) for v in obj]
        # Keep lists completely compact on a single line
        return '[' + ', '.join(items) + ']'

    else:
        # Base types (strings, numbers, booleans, null)
        return json.dumps(obj)


@click.command('convert a yaml sweep spec file to json format')
@click.argument('path', type=click.Path(exists=True, dir_okay=False))
def run(path):
    import striqt.sensor as ss
    import striqt.analysis as sa
    import msgspec
    import sys
    from pathlib import Path

    out_path = Path(path).with_suffix('.json')
    spec = ss.read_yaml_spec(path)
    s = msgspec.json.encode(spec, enc_hook=sa.specs.helpers._enc_hook)
    d = json.loads(s)

    with open(out_path, 'w') as stream:
        stream.write(custom_indent(d))

    print(f'wrote "{out_path!s}"')


if __name__ == '__main__':
    run()  # pyright: ignore
