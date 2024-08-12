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
    yaml_path: Path, output_path: str, remote: Optional[str], force: bool, verbose: bool
):
    if output_path is None:
        output_path = Path(yaml_path).with_suffix('.zarr.zip')

    # defer imports to here to make the command line --help snappier
    from edge_sensor.actions import iter_sweep, CAPTURE_DIM
    from edge_sensor.structs import read_yaml_sweep

    sweep_spec, sweep_fields = read_yaml_sweep(yaml_path)

    from edge_sensor.radio import find_radio_cls_by_name
    from edge_sensor.radio.util import prepare_gpu
    from edge_sensor.util import set_cuda_mem_limit
    from edge_sensor.controller import SweepController, connect

    import labbench as lb
    import xarray as xr
    from channel_analysis import dump

    if verbose:
        lb.show_messages('debug')
    else:
        lb.show_messages('info')

    # try:
    #     set_cuda_mem_limit()
    # except ModuleNotFoundError:
    #     pass

    # radio_type = find_radio_cls_by_name(sweep_spec.radio_setup.driver)

    # radio = radio_type()
    # if sweep_spec.radio_setup.resource is not None:
    #     radio.resource = sweep_spec.radio_setup.resource

    # prep = prepare_gpu(radio, sweep_spec.captures, sweep_spec.channel_analysis, sweep_fields)

    # with lb.concurrently(radio, prep):
    #     radio.setup(sweep_spec.radio_setup)
    #     sweep_it = iter_sweep(radio, sweep_spec, sweep_fields)
    #     data = xr.concat(sweep_it, CAPTURE_DIM)

    if remote is None:
        controller = SweepController()
    else:
        host, *extra = remote.split(',', 1)
        if len(extra) == 0:
            port = 4567
        else:
            port = int(extra[0])

        conn = connect(host, port=port)
        controller = conn.root

    generator = list(controller.iter_sweep(sweep_spec, sweep_fields))
    print(generator[0])
    data = xr.concat(generator, CAPTURE_DIM)
    print(data)

    if force:
        mode = 'w'
    else:
        mode = 'a'

    dump(output_path, data, mode)

    click.echo(f'wrote to {output_path}')
