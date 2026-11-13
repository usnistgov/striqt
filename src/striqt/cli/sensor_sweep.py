#!/usr/bin/env python3

"""the ``sensor-sweep`` console entry point"""

import functools
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

import click

if TYPE_CHECKING:
    from pathlib import Path

    import striqt.sensor as ss
    from striqt.sensor.lib.resources import Resources
    from striqt.sensor.lib.typing import PromptCallback


def confirm_labels(
    spec: 'ss.specs.Sweep',
    source_id: str,
    prompt: 'Optional[PromptCallback]' = None,
) -> None:
    """block on a y/n prompt that lists the capture adjustments of `spec` at `source_id`.

    Nothing is prompted when `spec.adjust_captures` sets no field. The prompt repeats
    until the answer is ``y`` or ``n``.

    Args:
        spec: the sweep with the `adjust_captures` block
        source_id: the id of the open source, selecting the adjustments
        prompt: hook called with the prompt text, returning the operator's answer;
            None reads one line from stdin through `blocking_input`

    Raises:
        KeyboardInterrupt: if the operator answers ``n``
    """
    from pprint import pformat

    import striqt.analysis as sa
    import striqt.sensor as ss

    labels = ss.specs.sequencing.list_capture_adjustments(spec, source_id)
    if len(labels) == 0:
        return
    if prompt is None:
        prompt = sa.util.blocking_input

    dict_repr = pformat(labels, indent=2, sort_dicts=False)
    info = (
        '▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀\n'
        'the following unique field values will be set based on adjust_captures:\n'
        f' {dict_repr[1:-1]} \n'
        '▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀\n'
    )

    while True:
        answer = prompt(f'{info}\nconfirm: are these correct? (y/n) ')
        if answer.lower() == 'y':
            break
        elif answer.lower() == 'n':
            raise KeyboardInterrupt


def log_overloads(dd: 'ss.lib.compute.DelayedDataset') -> None:
    """log a WARNING on the ``analysis`` logger when a port of `dd.capture` overloaded.

    Args:
        dd: the delayed dataset of the analyzed acquisition
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

    One record per port with a finite SNR is logged when `cache` is a spectrogram
    cache and `sweep.source.calibration` is set.

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


def run(
    path: 'Union[str, Path]',
    *,
    output_path: Optional[str] = None,
    check_func: Union[Callable[..., None], None] = None,
    prompt: 'Optional[PromptCallback]' = None,
) -> None:
    """run the action spec in the yaml or json file at `path`.

    Args:
        path: the yaml or json spec file
        output_path: replaces `sink.path` of the spec when given
        check_func: hook run once with ``(sweep, source_id, prompt=prompt)`` after
            the source opens and before the first capture, or None for no hook
        prompt: hook called with the text of each operator prompt, returning the
            operator's answer; None reads one line from stdin through
            `blocking_input`

    Raises:
        click.ClickException: if the suffix of `path` is none of ``.yaml``, ``.yml``,
            or ``.json``
    """
    import striqt.sensor as ss

    path = str(path)

    try:
        spec = ss.read_spec(path, output_path=output_path)
    except ValueError as ex:
        raise click.ClickException(str(ex)) from ex

    if check_func is None:
        on_source_opened = None
    else:
        on_source_opened = functools.partial(check_func, prompt=prompt)

    ctx = ss.open_resources(
        spec,
        path,
        on_source_opened=on_source_opened,
        on_analyzed=log_overloads,
        on_sink_cached=log_spectrogram_snr,
        prompt=prompt,
    )
    with ctx as resources:
        sweep = ss.iterate_sweep(resources, yield_values=False, always_yield=True)
        for _ in sweep:
            pass


@click.command()
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
def cli(
    *,
    path: str,
    debug: bool,
    verbose: bool,
    skip_confirm: bool,
    output_path: Optional[str],
) -> None:
    """run the sweep specified at PATH.

    PATH is a YAML or JSON sweep specification. The results are written to the sink
    path in the specification, or to `--output-path`. A y/n prompt on the capture
    adjustments precedes a sweep that has any, unless `--skip-confirm` is given.
    """
    import sys

    import striqt.sensor as ss

    except_handler = ss.util.DebugOnException(enable=debug, verbose=verbose)
    sys.excepthook = except_handler.run
    ss.util.log_verbosity(verbose)

    func = None if skip_confirm else confirm_labels

    run(path, output_path=output_path, check_func=func)


if __name__ == '__main__':
    cli()
