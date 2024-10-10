from __future__ import annotations

import click
import functools
from pathlib import Path
import typing
import sys
from datetime import datetime
import importlib.util
from socket import gethostname


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
    from IPython.core import ultratb
    import labbench as lb
    import xarray as xr
    import edge_sensor
    import channel_analysis
    import zarr
else:
    edge_sensor = lazy_import('edge_sensor')
    lb = lazy_import('labbench')
    channel_analysis = lazy_import('channel_analysis')
    xr = lazy_import('xarray')
    ultratb = lazy_import('IPython.core.ultratb')
    zarr = lazy_import('zarr')


HOSTNAME = gethostname()


def _chain_decorators(decorators: list[callable], func: callable) -> callable:
    for option in decorators:
        func = option(func)
    return func


# %% Sweep script
def click_sensor_sweep(description: typing.Optional[str] = None):
    """decorates a function to serve as the main function in a sweep CLI with click"""

    if description is None:
        description = 'Run a radio spectrum sensor acquisition sweep according to a configuration file.'

    click_decorators = (
        click.command(description),
        click.argument('yaml_path', type=click.Path(exists=True, dir_okay=False)),
        click.option(
            '--output-path/',
            '-o',
            default=None,
            type=click.Path(),
            help='output file path (default: YAML_PATH with .zarr.db suffix)',
        ),
        click.option(
            '--remote/',
            '-r',
            show_default=True,
            type=str,
            default=None,
            help='run on the specified remote host (at host or host:port)',
        ),
        click.option(
            '--store/',
            '-s',
            type=click.Choice(['zip', 'directory', 'db'], case_sensitive=True),
            default=None,
            help='override config file data store setting: "zip" for single acquisition, "directory" to support appending acquisitions',
        ),
        click.option(
            '--force/',
            '-f',
            is_flag=True,
            show_default=True,
            default=False,
            help='overwrite an existing output; otherwise, attempt append on existing data',
        ),
        click.option(
            '--debug/',
            '-d',
            is_flag=True,
            show_default=True,
            default=False,
            help='if set, drop to an IPython debug on exception',
        ),
        click.option(
            '--verbose/',
            '-v',
            is_flag=True,
            show_default=True,
            default=False,
            help='print debug',
        ),
    )

    def decorate(func):
        return _chain_decorators(click_decorators, func)

    return decorate


Store = typing.TypeVar('Store', bound='zarr.storage.Store')
Controller = typing.TypeVar('Controller', bound='zarr.storage.Store')
Sweep = typing.TypeVar('Sweep', bound='edge_sensor.Sweep')
Dataset = typing.TypeVar('Dataset', bound='xr.Dataset')


def get_file_format_fields(sweep_spec, controller, yaml_path):
    radio_id = controller.radio_id(sweep_spec.radio_setup.driver)
    fields = edge_sensor._api.captures.capture_fields_with_aliases(
        sweep_spec.captures[0], radio_id, sweep_spec.output.coord_aliases
    )
    fields['start_time'] = datetime.now().strftime('%Y%m%d-%Hh%Mm%S')
    fields['yaml_name'] = Path(yaml_path).stem
    fields['radio_id'] = radio_id

    return fields


def init_sensor_sweep(
    *,
    yaml_path: Path,
    output_path: str | None,
    store: str | None,
    remote: str | None,
    force: bool,
    verbose: bool,
    debug: bool,
    sweep_cls: type = None,
    adjust_captures: dict = {},
    open_store: bool = True,
) -> tuple[Store, Controller, Sweep, Dataset]:
    if sweep_cls is None:
        sweep_cls = edge_sensor.Sweep

    sweep_spec = edge_sensor.read_yaml_sweep(
        yaml_path, sweep_cls=sweep_cls, adjust_captures=adjust_captures
    )

    if verbose:
        lb.util.force_full_traceback(True)
        lb.show_messages('debug')
    else:
        lb.show_messages('info')

    if remote is None:
        controller = edge_sensor.SweepController(sweep_spec.radio_setup)
    else:
        controller = edge_sensor.connect(remote).root

    if debug:
        lb.util.force_full_traceback(True)
        sys.excepthook = ultratb.FormattedTB(
            mode='Verbose', color_scheme='Linux', call_pdb=1
        )

    path_fields = get_file_format_fields(sweep_spec, controller, yaml_path)

    if sweep_spec.radio_setup.calibration is None:
        calibration = None
    else:
        path = Path(sweep_spec.radio_setup.calibration.format(**path_fields))
        if not path.is_absolute():
            path = Path(yaml_path).parent.absolute()/path
        path = str(path.absolute())
        calibration = edge_sensor.read_calibration_corrections(path)

    if not open_store:
        return None, controller, sweep_spec, calibration

    if store is None:
        if sweep_spec.output.store is None:
            click.echo(
                'specify output.store in the yaml file or use -s NAME on the command line'
            )
            sys.exit(1)

        store = sweep_spec.output.store.lower()
    else:
        store = store.lower()

    yaml_path = Path(yaml_path)
    if output_path is None:
        spec_path = Path(sweep_spec.output.path)
        spec_path = spec_path.expanduser() # e.g., "~/" -> "/home/user/"
        if spec_path is None:
            click.echo(
                'specify output.path in the yaml file or use -o PATH on the command line'
            )
            sys.exit(1)
        spec_path = Path(str(spec_path).format(**path_fields))

        if store == 'directory':
            fixed_path = spec_path.with_suffix('.zarr')
        else:
            fixed_path = spec_path.with_suffix('.zarr.zip')

    else:
        fixed_path = str(output_path).format(**path_fields)

    Path(fixed_path).parent.mkdir(parents=True, exist_ok=True)

    store = channel_analysis.open_store(fixed_path, mode='w' if force else 'a')

    return store, controller, sweep_spec, calibration


# %% Server scripts
_CLICK_SERVER = (
    click.command('Host a server for remote control over the spectrum sensor'),
    click.argument('host', type=str, default=HOSTNAME),
    click.option(
        '--port/',
        '-p',
        show_default=True,
        default=4567,
        type=int,
        help='TCP port to serve',
    ),
    click.option(
        '--driver/',
        '-d',
        show_default=True,
        default='NullRadio',
        type=str,
        help='name of the default driver to load',
    ),
    click.option(
        '--verbose/',
        '-v',
        is_flag=True,
        show_default=True,
        default=False,
        help='print debug',
    ),
)


def click_server(func):
    return _chain_decorators(_CLICK_SERVER, func)


def run_server(host: str, port: int, driver: str, verbose: bool):
    # defer imports to here to make the command line --help snappier
    from edge_sensor._api.controller import start_server
    import labbench as lb

    if verbose:
        lb.util.force_full_traceback(True)
        lb.show_messages('debug')
    else:
        lb.show_messages('info')

    edge_sensor.start_server(host=host, port=port, default_driver=driver)
