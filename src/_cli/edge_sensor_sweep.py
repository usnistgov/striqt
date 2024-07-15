"""this is installed into the shell PATH environment as configured by pyproject.toml"""

import click
from pathlib import Path

import sys

print('*', sys.argv)


@click.command(
    'Run a radio spectrum sensor acquisition sweep according to a configuration file.'
)
@click.argument('yaml_path', type=click.Path(exists=True, dir_okay=False))
@click.option(
    '--output-path',
    default=None,
    type=click.Path(),
    help='output file path; if unspecified, follows yaml file name',
)
@click.option(
    '--force/',
    '-f',
    is_flag=True,
    show_default=True,
    default=False,
    help='overwrite an existing output; otherwise, attempt append on existing data',
)
def run(yaml_path: Path, output_path, force):
    if output_path is None:
        output_path = Path(yaml_path).with_suffix('.zarr.zip')

    # defer imports to here to make the command line --help snappier
    from edge_sensor.actions import sweep
    from edge_sensor.structs import read_yaml_sweep

    run_spec, sweep_fields = read_yaml_sweep(yaml_path)

    from edge_sensor.radio import airt
    import labbench as lb
    from channel_analysis import dump

    lb.show_messages('info')

    with airt.AirT7201B() as sdr:
        data = sweep(sdr, run_spec, sweep_fields)

    if force:
        mode = 'w'
    else:
        mode = 'a'

    dump(output_path, data, mode)

    click.echo(f'wrote to {output_path}')
