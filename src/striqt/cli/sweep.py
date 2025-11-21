"""this is installed into the shell PATH environment as configured by pyproject.toml"""

from . import click_sensor_sweep
import sys


@click_sensor_sweep('Run a swept acquisition and analysis from a yaml file')
def run(**kws):
    # instantiate sweep objects
    from striqt import sensor

    do_tui = kws.pop('tui')
    except_handler = sensor.util.DebugOnException(enable=kws['debug'], verbose=kws['verbose'])
    sys.excepthook = except_handler.run

    yaml_path = kws['yaml_path']

    if do_tui and sys.stdout.isatty():
        # full TUI
        from striqt.sensor.lib import tui

        app = tui.SweepHUDApp(kws)
        app.run()

        if app._exception is None:
            pass
        elif not hasattr(app, '_exc_info'):
            sys.exit(1)
        else:
            # exception printing is handled by SweepHUDApp; force the context exit
            sensor.util.exit_context(except_handler, app._exc_info)
            sys.exit(1)

    else:
        sensor.util.log_verbosity(kws['verbose'])
        spec = sensor.read_yaml_spec(
            yaml_path,
            output_path=kws['output_path'],
            store_backend=kws['store_backend'],
        )

        with sensor.open_sensor(spec, yaml_path, except_handler) as ctx:
            sweep_iter = sensor.SweepIterator(ctx.resources, always_yield=True)
            for _ in sweep_iter:
                pass

if __name__ == '__main__':
    run()
