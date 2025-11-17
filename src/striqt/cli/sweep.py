"""this is installed into the shell PATH environment as configured by pyproject.toml"""

from striqt.sensor.lib import connection
from . import click_sensor_sweep
import sys
import logging


@click_sensor_sweep(
    'Run an acquisition and analysis sweep with a software-defined radio'
)
def run(**kws):
    # instantiate sweep objects
    from striqt.sensor.lib import frontend, sweeps

    cli_objs = None

    do_tui = kws.pop('tui')
    debugger = frontend.DebugOnException(enable=kws['debug'])

    yaml_path = kws['yaml_path']

    open_kws = {
        'output_path': kws['output_path'],
        'store_backend': kws['store_backend'],
        'except_context': debugger,
    }

    if do_tui and sys.stdout.isatty():
        # full TUI
        app = None
        from striqt.sensor.lib import tui

        app = tui.SweepHUDApp(kws)
        app.run()

        cli_objs = app.resources

        if app._exception is None:
            pass
        elif not hasattr(app, '_exc_info'):
            sys.exit(1)
        else:
            # exception printing is handled by SweepHUDApp
            frontend.maybe_start_debugger(cli_objs, app._exc_info)
            sys.exit(1)

    else:
        frontend.log_verbosity(kws['verbose'])

        with connection.open_sensor_from_yaml(yaml_path, **open_kws) as ctx:
            sweep_iter = sweeps.SweepIterator(ctx.resources, always_yield=True)
            for _ in sweep_iter:
                pass


if __name__ == '__main__':
    run()
