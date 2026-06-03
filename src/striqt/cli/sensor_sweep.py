#!/usr/bin/env python3

"""this is installed into the shell PATH environment as configured by pyproject.toml"""

import click
from . import click_sensor_sweep
import sys


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
def run(*, path, debug, skip_confirm, verbose, port, **kws):
    import striqt.sensor as ss
    import striqt.analysis as sa

    def confirm_labels(sweep: ss.specs.Sweep, source_id: str):
        from striqt import sensor as ss
        from pprint import pformat

        labels = ss.specs.helpers.list_capture_adjustments(spec, source_id)

        if len(labels) == 0:
            return

        dict_repr = pformat(labels, indent=2, sort_dicts=False)
        info = (
            '▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀\n'
            'the following unique field values will be set based on adjust_captures:\n'
            f' {dict_repr[1:-1]} \n'
            '▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀\n'
        )

        while not skip_confirm:
            response = sa.util.blocking_input(
                f'{info}\nconfirm: are these correct? (y/n) '
            )

            if response.lower() == 'y':
                break
            elif response.lower() == 'n':
                raise KeyboardInterrupt

    except_handler = ss.util.DebugOnException(enable=debug, verbose=verbose)
    sys.excepthook = except_handler.run

    ss.util.log_verbosity(verbose)
    if path.endswith('yaml'):
        spec = ss.read_yaml_spec(path, output_path=kws['output_path'])
    elif path.endswith('json'):
        spec = ss.read_json_spec(path, output_path=kws['output_path'])
    else:
        raise click.ClickException('expected file to have .json or .yaml suffix')

    if port is not None:
        spec = adjust_port(spec, port)

    ctx = ss.open_resources(spec, path, on_source_opened=confirm_labels)
    with ctx as resources:
        sweep = ss.iterate_sweep(resources, yield_values=False, always_yield=True)
        for _ in sweep:
            pass


if __name__ == '__main__':
    run()  # pyright: ignore
