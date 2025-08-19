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
    import wurlitzer

    app = None

    do_tui = kws.pop('tui')

    # stderr = StderrStreamToLog()
    # pipes = wurlitzer.pipes(stderr=stderr, stdout=None)
    # stdout_r, stderr_r = pipes.__enter__()

    if do_tui and sys.stdout.isatty():
        from striqt.sensor.lib import tui

        app = tui.SweepHUDApp(kws)
        app.run()

        cli_objs = app.cli_objs

        if app._exception is not None:
            exc_info = app._exc_info
        else:
            exc_info = (None, None, None)

    else:
        cli_objs = None
        try:
            cli_objs = frontend.init_sweep_cli(**kws)

            if kws['verbose']:
                log_level = logging.DEBUG
            else:
                log_level = logging.INFO

            logging.basicConfig(level=log_level)

            generator = frontend.iter_sweep_cli(
                cli_objs,
                remote=kws.get('remote', None),
            )

            for _ in generator:
                pass
        except BaseException:
            exc_info = sys.exc_info()
        else:
            exc_info = (None, None, None)

    frontend.maybe_start_debugger(cli_objs, exc_info)

if __name__ == '__main__':
    run()
