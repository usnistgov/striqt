from __future__ import annotations

import click
from pathlib import Path
import typing
import sys
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


def _apply_exception_hooks(controller, sweep, debug: bool, remote: bool|None):
    lb.util.force_full_traceback(True)

    def hook(*args):
        if debug:
            print('entering debugger')
            debugger = ultratb.FormattedTB(
                mode='Verbose', color_scheme='Linux', call_pdb=1
            )
            debugger(*args)

        if not remote:
            controller.close_radio(sweep.radio_setup)

    sys.excepthook = hook


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
            help='override the output file path in the yaml specification',
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
            '--store-backend/',
            '-s',
            type=click.Choice(['zip', 'directory', 'db'], case_sensitive=True),
            default=None,
            help='override yaml file data store setting: "zip" for single acquisition, "directory" to support appending acquisitions',
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
Controller = typing.TypeVar('Controller', bound='edge_sensor.SweepController')
Sweep = typing.TypeVar('Sweep', bound='edge_sensor.Sweep')
Dataset = typing.TypeVar('Dataset', bound='xr.Dataset')


def init_sensor_sweep(
    *,
    yaml_path: Path,
    output_path: str | None,
    store_backend: str | None,
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

    sweep = edge_sensor.read_yaml_sweep(
        yaml_path, sweep_cls=sweep_cls, adjust_captures=adjust_captures
    )

    if store_backend is None and sweep.output.store is None:
        click.echo(
            'specify output.store in the yaml file or use -s <NAME> on the command line'
        )
        sys.exit(1)

    if output_path is None and sweep.output.path is None:
        click.echo(
            'specify output.path in the yaml file or use -o PATH on the command line'
        )
        sys.exit(1)

    if verbose:
        lb.util.force_full_traceback(True)
        lb.show_messages('debug')
    else:
        lb.show_messages('info')

    if remote is None:
        controller = edge_sensor.SweepController(sweep.radio_setup)
    else:
        controller = edge_sensor.connect(remote).root

    _apply_exception_hooks(controller, sweep, debug=debug, remote=remote)

    # reload the yaml now that radio_id can be known to fully format any filenames
    radio_id = controller.radio_id(sweep.radio_setup.driver)
    sweep = edge_sensor.read_yaml_sweep(
        yaml_path,
        sweep_cls=sweep_cls,
        adjust_captures=adjust_captures,
        radio_id=radio_id,
    )

    calls = {}

    calls['calibration'] = lb.Call(
        edge_sensor.read_calibration_corrections, sweep.radio_setup.calibration
    )

    if open_store:
        calls['store'] = lb.Call(
            edge_sensor.open_store,
            sweep,
            radio_id=radio_id,
            yaml_path=yaml_path,
            output_path=output_path,
            store_backend=store_backend,
            force=force,
        )

    opened = lb.concurrently(**calls)
    opened.setdefault('store', None)
    opened.setdefault('calibration', None)

    return opened['store'], controller, sweep, opened['calibration']


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
    from edge_sensor.api.controller import start_server
    import labbench as lb

    if verbose:
        lb.util.force_full_traceback(True)
        lb.show_messages('debug')
    else:
        lb.show_messages('info')

    edge_sensor.start_server(host=host, port=port, default_driver=driver)
