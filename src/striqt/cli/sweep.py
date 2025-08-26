"""this is installed into the shell PATH environment as configured by pyproject.toml"""

from . import click_sensor_sweep
import sys
import logging
import io


class StderrStreamToLog(io.StringIO):
    def write(self, data: str):
        if data.startswith(('DEBUG', 'INFO', 'WARNING', 'ERROR')):
            sys.stdout.write(data)
        else:
            try:
                from striqt.analysis.lib import util

                util.get_logger('controller').warning(str(data))
            except BaseException as ex:
                print('uncaught build: ', ex)


@click_sensor_sweep(
    'Run an acquisition and analysis sweep with a software-defined radio'
)
def run(**kws):
    # instantiate sweep objects
    from striqt.sensor.lib import frontend

    cli_objs = None

    do_tui = kws.pop('tui')

    if do_tui and sys.stdout.isatty():
        # full TUI
        app = None
        from striqt.sensor.lib import tui

        app = tui.SweepHUDApp(kws)
        app.run()

        cli_objs = app.cli_objs

        if app._exception is None:
            pass
        elif not hasattr(app, '_exc_info'):
            sys.exit(1)
        else:
            frontend.maybe_start_debugger(cli_objs, app._exc_info)
            sys.exit(1)

    else:
        # simple CLI
        try:
            cli_objs = frontend.init_sweep_cli(**kws)

            from striqt.analysis.lib import util

            util.show_messages(logging.INFO)
            generator = frontend.iter_sweep_cli(
                cli_objs,
                remote=kws.get('remote', None),
            )

            for _ in generator:
                pass
        except BaseException:
            from rich.console import Console

            console = Console()
            console.print_exception(
                show_locals=False,
                width=None,
                suppress=['rich', 'zarr', 'xarray', 'pandas'],
            )
            frontend.maybe_start_debugger(cli_objs, sys.exc_info())
            sys.exit(1)


if __name__ == '__main__':
    run()
