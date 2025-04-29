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
    import zarr
else:
    edge_sensor = lazy_import('edge_sensor')
    lb = lazy_import('labbench')
    xr = lazy_import('xarray')
    ultratb = lazy_import('IPython.core.ultratb')
    zarr = lazy_import('zarr')


HOSTNAME = gethostname()


def _chain_decorators(decorators: list[callable], func: callable) -> callable:
    for option in decorators:
        func = option(func)
    return func


def _apply_exception_hooks(controller, sweep, debug: bool, remote: bool | None):
    def hook(*args):
        if debug:
            print('entering debugger')
            lb.util.force_full_traceback(True)
            debugger = ultratb.FormattedTB(
                mode='Verbose', color_scheme='Linux', call_pdb=1
            )
            debugger(*args)

        if remote:
            print('closing in exception hook')
            controller.close_radio(sweep.radio_setup)

    sys.excepthook = hook


def _do_expensive_imports():
    import iqwaveform
    import xarray
    import pandas
    import numpy


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


def _run_click_plotter(
    plot_func: callable,
    zarr_path: str,
    center_frequency=None,
    interactive=False,
    no_save=False,
    data_variable=[],
    sweep_index=-1,
    **plot_func_kws,
):
    """handle keyword arguments passed in from click, and call plot_func()"""

    from matplotlib import pyplot as plt
    import channel_analysis
    from pathlib import Path
    import numpy as np

    if interactive:
        plt.ion()
    else:
        plt.ioff()

    plt.style.use('iqwaveform.ieee_double_column')

    # index on the following fields in order, matching the input options
    dataset = channel_analysis.load(zarr_path).set_xindex(
        ['channel', 'center_frequency', 'start_time', 'sweep_start_time']
    )

    valid_freqs = tuple(dataset.indexes['center_frequency'].levels[1])
    if center_frequency is None:
        fcs = valid_freqs
    elif center_frequency in valid_freqs:
        fcs = [center_frequency]
        dataset = dataset.sel(center_frequency=fcs)
    else:
        raise ValueError(
            f'no frequency {center_frequency} in data set - must be one of {valid_freqs}'
        )

    valid_vars = tuple(dataset.data_vars.keys())
    if len(data_variable) == 0:
        variables = valid_vars
    elif len(set(data_variable) - set(valid_vars)) == 0:
        variables = list(data_variable)
        drop_set = set(dataset.data_vars.keys()) - set(variables)
        dataset = dataset.drop_vars(list(drop_set))
    else:
        invalid = tuple(set(data_variable) - set(valid_vars))
        raise ValueError(
            f'data variables {invalid} are not in data set - must be one of {valid_vars}'
        )

    sweep_start_time = np.atleast_1d(dataset.sweep_start_time)[sweep_index]
    dataset = dataset.sel(sweep_start_time=sweep_start_time).load()

    if no_save:
        output_path = None
    else:
        output_path = Path(zarr_path).parent / Path(zarr_path).name.split('.', 1)[0]
        output_path.mkdir(exist_ok=True)

    plot_func(dataset, output_path, interactive, **plot_func_kws)

    if interactive:
        input('press enter to quit')


def click_capture_plotter(description: typing.Optional[str] = None):
    """decorate a function to handle single-capture plots of zarr or zarr.zip files"""

    if description is None:
        description = 'plot signal analysis from zarr or zarr.zip files'

    click_decorators = (
        click.command(description),
        click.argument('zarr_path', type=click.Path(exists=True, dir_okay=True)),
        click.option(
            '--interactive/',
            '-i',
            is_flag=True,
            show_default=True,
            default=False,
            help='',
        ),
        click.option(
            '--center-frequency/',
            '-f',
            type=float,
            default=None,
            help='if specified, plot for only this frequency',
        ),
        click.option(
            '--sweep-index/',
            '-s',
            type=int,
            show_default=True,
            default=-1,
            help='sweep index to plot (-1 for last)',
        ),
        click.option(
            '--data-variable',
            '-d',
            type=str,
            multiple=True,
            default=[],
            help='plot only the specified variable if specified',
        ),
        click.option(
            '--no-save/',
            '-n',
            is_flag=True,
            show_default=True,
            default=False,
            help="don't save the resulting plots",
        ),
    )

    def decorate(func):
        def wrapped(*args, **kws):
            return _run_click_plotter(func, *args, **kws)

        return _chain_decorators(click_decorators, wrapped)

    return decorate


def _connect_controller(remote, sweep):
    if remote is None:
        return edge_sensor.SweepController(sweep)
    else:
        return edge_sensor.connect(remote).root


DataStoreManager = typing.TypeVar(
    'DataStoreManager', bound='edge_sensor.io.DataStoreManager'
)
Controller = typing.TypeVar('Controller', bound='edge_sensor.SweepController')
Sweep = typing.TypeVar('Sweep', bound='edge_sensor.Sweep')
Dataset = typing.TypeVar('Dataset', bound='xr.Dataset')


def init_sweep_cli(
    *,
    yaml_path: Path,
    output_path: str | None = None,
    store_backend: str | None,
    remote: str | None,
    force: bool = False,
    verbose: bool = False,
    debug: bool = False,
    sweep_cls: type[Sweep] | None = None,
    store_manager_cls: type[DataStoreManager] | None = None,
) -> tuple[DataStoreManager, Controller, Sweep, Dataset]:
    if sweep_cls is None:
        sweep_cls = edge_sensor.Sweep

    if store_manager_cls is None:
        store_manager_cls = edge_sensor.io.AppendingDataManager

    # first read, without knowing radio_id
    sweep = edge_sensor.read_yaml_sweep(yaml_path, sweep_cls=sweep_cls)

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

    # start by connecting to the controller, so that the radio id can be used
    # as a file naming field
    controller = _connect_controller(remote, sweep)
    _apply_exception_hooks(controller, sweep, debug=debug, remote=remote)
    radio_id = controller.radio_id(sweep.radio_setup.driver)
    sweep = edge_sensor.read_yaml_sweep(
        yaml_path,
        sweep_cls=sweep_cls,
        radio_id=radio_id,
    )

    # now, open the store
    store = store_manager_cls(
        sweep, output_path=output_path, store_backend=store_backend, force=force
    )

    calls = {}
    calls['calibration'] = lb.Call(
        edge_sensor.read_calibration_corrections,
        sweep.radio_setup.calibration,
    )
    calls['store'] = lb.Call(store.open)

    with lb.stopwatch('load store and prepare calibrations', logger_level='debug'):
        opened = lb.concurrently(**calls)

    return store, controller, sweep, opened.get('calibration', None)


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
