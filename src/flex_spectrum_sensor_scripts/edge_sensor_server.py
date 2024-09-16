"""this is installed into the shell PATH environment as configured by pyproject.toml"""

import click
from socket import gethostname

HOSTNAME = gethostname()


@click.command('Host a server for remote control over the spectrum sensor')
@click.argument('host', type=str, default=HOSTNAME)
@click.option(
    '--port/',
    '-p',
    show_default=True,
    default=4567,
    type=int,
    help='TCP port to serve',
)
@click.option(
    '--driver/',
    '-d',
    show_default=True,
    default='NullRadio',
    type=str,
    help='name of the default driver to load',
)
@click.option(
    '--verbose/',
    '-v',
    is_flag=True,
    show_default=True,
    default=False,
    help='print debug',
)
def run(host: str, port: int, driver: str, verbose: bool):
    # defer imports to here to make the command line --help snappier
    from edge_sensor.controller import start_server
    import labbench as lb

    if verbose:
        lb.show_messages('debug')
    else:
        lb.show_messages('info')

    start_server(host=host, port=port, default_driver=driver)


if __name__ == '__main__':
    run()
