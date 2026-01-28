#!/usr/bin/env python3

"""this is installed into the shell PATH environment as configured by pyproject.toml"""

import click
from . import click_sensor_sweep
import sys

label_check_failed = False

def adjust_port(spec, port):
    from striqt import sensor as ss

    loops = []
    for i, loop in enumerate(spec.loops):
        if loop.field == 'port':
            new_loop = ss.specs.List(field='port', values=(port,))
            loops.append(new_loop)
            loops.extend(spec.loops[i + 1 :])
            return spec.replace(loops=loops)
        else:
            loops.append(loop)

    # no loop - replace the captures instead
    captures = [c.replace(port=port) for c in spec.captures]
    return spec.replace(captures=captures)


def check_labels(spec, source_id):
    from striqt import sensor as ss
    from pprint import pformat

    global label_check_failed

    labels = ss.specs.helpers.list_all_labels(spec, source_id)

    if len(labels) == 0:
        return
    
    while True:
        response = ss.util.blocking_input(f'''
            the sweep will produce these different label coordinates. 
            
            {pformat(labels)}

            are they correct? (y/n)
            '''
        )

        if response.lower() == 'y':
            break
        elif response.lower() == 'n':
            label_check_failed = True
            break


@click_sensor_sweep('Run a swept acquisition and analysis from a yaml file')
@click.option(
    '--port/',
    default=None,
    prompt=True,
    required=False,
    prompt_required=False,
    type=click.IntRange(min=0),
    help='limit the acquisition the specified input port index',
)
def run(*, yaml_path, debug, verbose, port, **kws):
    # instantiate sweep objects
    from striqt import sensor

    except_handler = sensor.util.DebugOnException(enable=debug, verbose=verbose)
    sys.excepthook = except_handler.run

    sensor.util.log_verbosity(verbose)
    spec = sensor.read_yaml_spec(
        yaml_path,
        output_path=kws['output_path'],
    )

    if port is not None:
        spec = adjust_port(spec, port)

    with sensor.open_resources(spec, yaml_path, except_handler) as resources:
        if label_check_failed:
            raise KeyboardInterrupt

        sweep = sensor.iterate_sweep(resources, yield_values=False, always_yield=True)
        for _ in sweep:
            pass


if __name__ == '__main__':
    run()  # type: ignore
