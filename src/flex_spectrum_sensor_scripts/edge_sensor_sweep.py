"""this is installed into the shell PATH environment as configured by pyproject.toml"""

import click
from pathlib import Path


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
@click.option(
    '--verbose/',
    '-v',
    is_flag=True,
    show_default=True,
    default=False,
    help='print debug',
)
def run(yaml_path: Path, output_path, force, verbose):
    if output_path is None:
        output_path = Path(yaml_path).with_suffix('.zarr.zip')

    # defer imports to here to make the command line --help snappier
    from edge_sensor.actions import concat_sweeps, sweep_iterator
    from edge_sensor.structs import read_yaml_sweep

    sweep_spec, sweep_fields = read_yaml_sweep(yaml_path)

    from edge_sensor.radio import find_radio_cls_by_name
    from edge_sensor.radio.util import prepare_gpu
    from edge_sensor.util import set_cuda_mem_limit

    import labbench as lb
    from channel_analysis import dump

    if verbose:
        lb.show_messages('debug')
    else:
        lb.show_messages('info')

    try:
        set_cuda_mem_limit()
    except ModuleNotFoundError:
        pass

    radio_type = find_radio_cls_by_name(sweep_spec.radio_setup.driver)
    radio = radio_type()
    prep = prepare_gpu(radio, sweep_spec.captures, sweep_spec.channel_analysis, sweep_fields)

    with lb.concurrently(radio, prep):
        radio.setup(sweep_spec.radio_setup)
        sweep_it = sweep_iterator(radio, sweep_spec, sweep_fields)
        data = concat_sweeps(sweep_it, radio, sweep_spec, sweep_fields)

    if force:
        mode = 'w'
    else:
        mode = 'a'

    dump(output_path, data, mode)

    click.echo(f'wrote to {output_path}')
