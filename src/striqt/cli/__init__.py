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
