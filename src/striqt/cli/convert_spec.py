import click


@click.command('convert a yaml sweep spec file to json format')
@click.argument('path', type=click.Path(exists=True, dir_okay=False))
def run(path):
    import striqt.sensor as ss
    import striqt.analysis as sa
    import msgspec
    import json
    import sys
    from pathlib import Path

    out_path = Path(path).with_suffix('.json')
    spec = ss.read_yaml_spec(path)
    s = msgspec.json.encode(spec, enc_hook=sa.specs.helpers._enc_hook)
    d = json.loads(s)

    with open(out_path, 'w') as stream:
        stream.write(json.dumps(d, indent=True))


if __name__ == '__main__':
    run()  # pyright: ignore
