"""the ``striqt-studio`` console entry point"""

import click


@click.command()
@click.argument('path', type=click.Path(exists=True, dir_okay=False), required=True)
@click.option(
    '--host',
    '-h',
    default='0.0.0.0',
    show_default=True,
    help='Host to bind the server to',
)
@click.option(
    '--port',
    '-p',
    default=8000,
    show_default=True,
    help='Port to bind the server to',
)
def cli(path: str, host: str, port: int) -> None:
    """serve a browser page that previews the sweep at PATH.

    PATH is a YAML or JSON sweep specification. The page is served at
    http://HOST:PORT.
    """
    from .server import run_server

    run_server(path, host=host, port=port)
