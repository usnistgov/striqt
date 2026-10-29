#!/usr/bin/env python3

"""console entry point ``sensor-sweep``, installed on the shell PATH by pyproject.toml"""

from typing import TYPE_CHECKING, Any, Callable, Union

import click

if TYPE_CHECKING:
    import striqt.sensor as ss
    from striqt.sensor.lib.resources import Resources


def confirm_labels(spec: 'ss.specs.Sweep', source_id: str):
    from pprint import pformat

    import striqt.analysis as sa
    import striqt.sensor as ss

    labels = ss.specs.sequencing.list_capture_adjustments(spec, source_id)
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


def log_overloads(dd: 'ss.lib.compute.DelayedDataset') -> None:
    """log a WARNING on the ``analysis`` logger when a port of `dd.capture` overloaded.

    Args:
        dd: the analyzed capture
    """
    import striqt.analysis as sa
    from striqt.sensor.lib.compute import summary

    message = summary.describe_overloads(summary.overloads(dd))
    if message is not None:
        sa.util.get_logger('analysis').warning(message)


def log_spectrogram_snr(
    resources: 'Resources',
    cache: Any,
    capture: 'ss.specs.SensorCapture',
    result: Any,
    **kwargs: Any,
) -> None:
    """log the peak SNR of each port at INFO on the ``analysis`` logger.

    One line per port is logged when `cache` is a spectrogram cache and
    `sweep.source.calibration` is set.

    Args:
        resources: the open resources of the run
        cache: the filled measurement cache
        capture: the capture under analysis
        result: the cached ``(spectrogram, attrs)`` pair
        **kwargs: the measurement keywords, unused
    """
    import striqt.analysis as sa
    from striqt.sensor.lib.compute import summary

    if 'spectrogram' not in cache.name:
        return

    sweep = resources['sweep_spec']
    entries = summary.spectrogram_snr(
        sweep, capture, result, format_path=resources['format_path']
    )
    logger = sa.util.get_logger('analysis')
    for entry in entries or ():
        logger.info(
            summary.describe_spectrogram_snr(
                entry,
                source_id=resources['source'].source_id,
                adjust_spec=sweep.adjust_captures,
            )
        )


def run(path: str, *, output_path=None, check_func: Union[Callable, None] = None):
    from pathlib import Path

    import striqt.sensor as ss

    if isinstance(path, Path):
        path = str(path)

    if path.endswith('.yaml') or path.endswith('.yml'):
        spec = ss.read_yaml_spec(path, output_path=output_path)
    elif path.endswith('.json'):
        spec = ss.read_json_spec(path, output_path=output_path)
    else:
        raise click.ClickException('expected file to have .json or .yaml suffix')

    ctx = ss.open_resources(
        spec,
        path,
        on_source_opened=check_func,
        on_analyzed=log_overloads,
        on_sink_cached=log_spectrogram_snr,
    )
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
def cli(*, path: str, debug: bool, verbose: bool, skip_confirm: bool, output_path):
    import sys

    import striqt.sensor as ss

    except_handler = ss.util.DebugOnException(enable=debug, verbose=verbose)
    sys.excepthook = except_handler.run
    ss.util.log_verbosity(verbose)

    func = None if skip_confirm else confirm_labels

    run(path, output_path=output_path, check_func=func)


if __name__ == '__main__':
    cli()
