"""this is installed into the shell PATH environment as configured by pyproject.toml"""

from __future__ import annotations
import click
from pathlib import Path
import typing
import sys
import importlib.util


def lazy_import(module_name: str):
    """postponed imports of the module with the specified name.

    The import is not performed until the module is accessed in the code. This
    reduces the total time to import labbench by waiting to import the module
    until it is used.
    """

    # see https://docs.python.org/3/library/importlib.html#implementing-lazy-imports
    try:
        ret = sys.modules[module_name]
        return ret
    except KeyError:
        pass

    spec = importlib.util.find_spec(module_name)
    if spec is None:
        raise ImportError(f'no module found named "{module_name}"')
    spec.loader = importlib.util.LazyLoader(spec.loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


if typing.TYPE_CHECKING:
    import edge_sensor
    import labbench as lb
    import channel_analysis
    import xarray as xr
    import pickle
else:
    edge_sensor = lazy_import('edge_sensor')
    lb = lazy_import('labbench')
    channel_analysis = lazy_import('channel_analysis')
    xr = lazy_import('xarray')
    pickle = lazy_import('pickle')


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
    yaml_path: Path,
    output_path: str | None,
    remote: str | None,
    force: bool,
    verbose: bool,
):
    if output_path is None:
        output_path = Path(yaml_path).with_suffix('.zarr.zip')

    sweep_spec = edge_sensor.read_yaml_sweep(yaml_path)

    if verbose:
        lb.show_messages('debug')
    else:
        lb.show_messages('info')

    if remote is None:
        controller = edge_sensor.SweepController()
    else:
        controller = edge_sensor.connect(remote).root

    if sweep_spec.radio_setup.calibration is None:
        calibration = None
    else:
        calibration = edge_sensor.read_calibration_corrections(
            sweep_spec.radio_setup.calibration
        )

    results = list(controller.iter_sweep(sweep_spec, calibration))
    dataset = xr.concat(results, edge_sensor.CAPTURE_DIM)

    edge_sensor.dump(output_path, dataset, mode='w' if force else 'a')

    lb.logger.info(f'wrote to {output_path}')
