#!/usr/bin/env python3

"""this is installed into the shell PATH environment as configured by pyproject.toml"""

import click
from . import click_sensor_sweep
import sys


def adjust_port(spec, port):
    from striqt import sensor

    loops = []
    for i, loop in enumerate(spec.loops):
        if loop.field == 'port':
            new_loop = sensor.specs.List(field='port', values=(port,))
            loops.append(new_loop)
            loops.extend(spec.loops[i + 1 :])
            return spec.replace(loops=loops)
        else:
            loops.append(loop)

    # no loop - replace the captures instead
    captures = [c.replace(port=port) for c in spec.captures]
    return spec.replace(captures=captures)


@click_sensor_sweep('Run a swept acquisition and analysis from a yaml file')
@click.option(
    '--port/',
    default=None,
    prompt=True,
    required=False,
    prompt_required=False,
    type=click.IntRange(0, 1),
    help='limit the calibration to the hardware input port at the specified index',
)
def run(**kws):
    # instantiate sweep objects
    from striqt import sensor

    do_tui = kws.pop('tui')
    except_handler = sensor.util.DebugOnException(
        enable=kws['debug'], verbose=kws['verbose']
    )
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

        if kws['port'] is not None:
            spec = adjust_port(spec, kws['port'])

        with sensor.open_resources(spec, yaml_path, except_handler) as resources:
            sweep = sensor.iterate_sweep(
                resources, yield_values=False, always_yield=True
            )
            for _ in sweep:
                pass


if __name__ == '__main__':
    run()
