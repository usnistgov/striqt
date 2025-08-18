"""this is installed into the shell PATH environment as configured by pyproject.toml"""

from . import click_sensor_sweep
import sys
import logging


class Handler(logging.Handler):
    def emit(self, record):
        if record.name == 'striqt.analysis':
            print(record.name, record.extra, record.getMessage(), record.args)


@click_sensor_sweep(
    'Run an acquisition and analysis sweep with a software-defined radio'
)
def run(**kws):
    # instantiate sweep objects
    from striqt.sensor.lib import frontend

    app = None

    try:
        if sys.stdout.isatty():
            from striqt.sensor.lib import tui

            app = tui.SweepHUDApp(kws)
            app.run()

        else:
            cli_objs = frontend.init_sweep_cli(**kws)

            if kws['verbose']:
                log_level = logging.DEBUG
            else:
                log_level = logging.INFO

            logging.basicConfig(level=log_level)

            handler = Handler()

            for name in ('source', 'analysis', 'sink'):
                logger = frontend.util.get_logger(name)
                logger.setLevel(log_level)
                logger.logger.addHandler(handler)

            generator = frontend.iter_sweep_cli(
                cli_objs,
                remote=kws.get('remote', None),
            )

            for _ in generator:
                pass
    except:
        if app is not None and hasattr(app, 'cli_objs'):
            cli_objs = app.cli_objs
        frontend.maybe_start_debugger(cli_objs)


if __name__ == '__main__':
    run()
