"""this is installed into the shell PATH environment as configured by pyproject.toml"""

import click
from pathlib import Path
from typing import Optional


@click.command(
    'Run a radio spectrum sensor acquisition sweep according to a configuration file.'
)
@click.argument('yaml_path', type=click.Path(exists=True, dir_okay=False))
@click.option(
    '--output-path/',
    '-o',
    default=None,
    type=click.Path(),
    help='output file path (default: YAML_PATH with .zarr.zip suffix)',
)
@click.option(
    '--remote/',
    '-r',
    show_default=True,
    type=str,
    default=None,
    help='run remotely on the specified host (host or host:port) instead of this machine',
)
@click.option(
    '--force/',
    '-f',
    is_flag=True,
    show_default=True,
    default=False,
    help='overwrite an existing output; otherwise, attempt append on existing data',
)
@click.option(
    '--verbose/',
    '-v',
    is_flag=True,
    show_default=True,
    default=False,
    help='print debug',
)
def run(
    yaml_path: Path, output_path: Optional[str], remote: Optional[str], force: bool, verbose: bool
):
    if output_path is None:
        output_path = Path(yaml_path).with_suffix('.zarr.zip')

    # defer imports to here to make the command line --help snappier
    from edge_sensor.actions import CAPTURE_DIM
    from edge_sensor import read_yaml_sweep

    sweep_spec, sweep_fields = read_yaml_sweep(yaml_path)

    from edge_sensor.controller import SweepController, connect
    from edge_sensor.iq_corrections import read_calibration_corrections

    import labbench as lb
    import xarray as xr
    from channel_analysis import dump

    if verbose:
        lb.show_messages('debug')
    else:
        lb.show_messages('info')

    if remote is None:
        controller = SweepController()
    else:
        controller = connect(remote).root

    if sweep_spec.radio_setup.calibration is None:
        calibration = None
    else:
        calibration = read_calibration_corrections(sweep_spec.radio_setup.calibration)

    generator = list(controller.iter_sweep(sweep_spec, sweep_fields, calibration))
    data = xr.concat(generator, CAPTURE_DIM)

    if force:
        mode = 'w'
    else:
        mode = 'a'

    dump(output_path, data, mode)

    click.echo(f'wrote to {output_path}')
