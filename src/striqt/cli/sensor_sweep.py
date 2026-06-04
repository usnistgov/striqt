#!/usr/bin/env python3

"""this is installed into the shell PATH environment as configured by pyproject.toml"""

import click
from typing import Callable, Union, TYPE_CHECKING

if TYPE_CHECKING:
    import striqt.sensor as ss


def confirm_labels(spec: 'ss.specs.Sweep', source_id: str):
    import striqt.sensor as ss
    import striqt.analysis as sa
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

    while True:
        answer = sa.util.blocking_input(f'{info}\nconfirm: are these correct? (y/n) ')
        if answer.lower() == 'y':
            break
        elif answer.lower() == 'n':
            raise KeyboardInterrupt


def run(path: str, *, output_path=None, check_func: Union[Callable, None] = None):
    import striqt.sensor as ss
    from pathlib import Path

    if isinstance(path, Path):
        path = str(path)

    if path.endswith('.yaml') or path.endswith('.yml'):
        spec = ss.read_yaml_spec(path, output_path=output_path)
    elif path.endswith('.json'):
        spec = ss.read_json_spec(path, output_path=output_path)
    else:
        raise click.ClickException('expected file to have .json or .yaml suffix')

    ctx = ss.open_resources(spec, path, on_source_opened=check_func)
    with ctx as resources:
        sweep = ss.iterate_sweep(resources, yield_values=False, always_yield=True)
        for _ in sweep:
            pass


@click.command('run a sweep from a json/yaml specification')
@click.argument('path', type=click.Path(exists=True, dir_okay=False))
@click.option(
    '--debug/',
    '-d',
    is_flag=True,
    show_default=True,
    default=False,
    help='invoke an IPython debug prompt on exception',
)
@click.option(
    '--skip-confirm/',
    '-s',
    is_flag=True,
    show_default=True,
    default=False,
    help='skip y/n confirmation before start',
)
@click.option(
    '--verbose/',
    '-v',
    show_default=True,
    default=False,
    count=True,
    help='add detail to messages',
)
@click.option(
    '--output-path/',
    '-o',
    default=None,
    type=click.Path(),
    help='override the output file path in the yaml specification',
)
def cli(path: str, output_path, debug: bool, verbose: bool, skip_confirm: bool):
    import striqt.sensor as ss
    import sys

    except_handler = ss.util.DebugOnException(enable=debug, verbose=verbose)
    sys.excepthook = except_handler.run
    ss.util.log_verbosity(verbose)

    run(
        path,
        output_path=output_path,
        check_func=None if skip_confirm else confirm_labels,
    )


if __name__ == '__main__':
    cli()  # pyright: ignore
