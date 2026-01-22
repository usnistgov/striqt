"""CLI tools are broken out into this separate module in order to avoid"""

import click
import typing


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
            show_default=True,
            default=False,
            count=True,
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
    from striqt import analysis
    import striqt.figures
    from striqt.analysis.lib.dataarrays import PORT_DIM
    from pathlib import Path
    import numpy as np

    if interactive:
        plt.ion()
    else:
        plt.ioff()

    plt.style.use('striqt.figures.ieee_double_column')

    # index on the following fields in order, matching the input options
    dataset = analysis.load(zarr_path)
    if 'channel' in dataset.coords:
        dataset = dataset.rename_vars({'channel': 'port'})

    if 'start_time' in dataset.dims:
        index_dims = [PORT_DIM, 'center_frequency', 'start_time', 'sweep_start_time']
    else:
        index_dims = [PORT_DIM, 'center_frequency']
    dataset = dataset.set_xindex(index_dims)

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

    if 'sweep_start_time' in index_dims:
        sweep_start_time = np.atleast_1d(dataset.sweep_start_time)[sweep_index]
        dataset = dataset.sel(sweep_start_time=sweep_start_time).load()
    else:
        dataset = dataset.load()

    if no_save:
        output_path = None
    else:
        output_path = Path(zarr_path).parent / Path(zarr_path).name.split('.', 1)[0]
        output_path.mkdir(exist_ok=True)

    plot_func(dataset, output_path, interactive, **plot_func_kws)

    if interactive:
        analysis.util.blocking_input('press enter to quit')


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
            '--index/',
            '-n',
            type=int,
            show_default=True,
            default=-1,
            help='sweep index to plot (-1 for last)',
        ),
        click.option(
            '--data-variable/',
            '-d',
            type=str,
            multiple=True,
            default=[],
            help='plot only the specified variable if specified',
        ),
        click.option(
            '--style',
            '-s',
            type=click.Choice(['presentation_half_width', 'presentation_full_width', 'ieee', 'ieee_double_column']),
            default='presentation_half_width',
            help="matplotlib style sheet to use"
        ),
        click.option(
            '--no-save',
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


# %% Server scripts


def click_server(func):
    import socket

    HOSTNAME = socket.gethostname()

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
            default='NoSource',
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

    return _chain_decorators(_CLICK_SERVER, func)
