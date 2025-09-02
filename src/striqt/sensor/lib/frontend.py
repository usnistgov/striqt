from __future__ import annotations
from pathlib import Path
import logging
import sys
import typing

import click

from . import calibration, controller, io, peripherals, sinks, specs, util

if typing.TYPE_CHECKING:
    import xarray as xr
    import labbench as lb
else:
    xr = util.lazy_import('xarray')
    lb = util.lazy_import('labbench')


def get_controller(remote, sweep):
    if remote is None:
        return controller.SweepController(sweep)
    else:
        return controller.connect(remote).root


class DebugOnException:
    def __init__(
        self,
        enable: bool = False,
    ):
        self.enable = enable

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.run(*args)

    def run(self, etype, exc, tb):
        if self.enable and (etype, exc, tb) != (None, None, None):
            print('entering debugger')
            from IPython.core import ultratb

            lb.util.force_full_traceback(True)
            if not hasattr(sys, 'last_value'):
                sys.last_value = exc
            if not hasattr(sys, 'last_traceback'):
                sys.last_traceback = tb
            debugger = ultratb.FormattedTB(mode='Plain', call_pdb=1)
            debugger(etype, exc, tb)


class CLIObjects(typing.NamedTuple):
    sink: sinks.SinkBase
    controller: controller.SweepController
    peripherals: peripherals.PeripheralsBase
    debugger: DebugOnException
    sweep_spec: specs.Sweep
    calibration: 'xr.Dataset'


class SweepSpecClasses(typing.NamedTuple):
    sink_cls: typing.Type[sinks.SinkBase]
    peripherals_cls: typing.Type[peripherals.PeripheralsBase]


def _get_extension_classes(sweep_spec: specs.Sweep) -> SweepSpecClasses:
    ext = sweep_spec.extensions.todict()

    import_cls = io._import_extension
    return SweepSpecClasses(
        peripherals_cls=import_cls(ext, 'peripherals'),
        sink_cls=import_cls(ext, 'sink'),
    )


class RotatingJSONFileHandler(logging.handlers.RotatingFileHandler):
    def __init__(self, path, *args, **kws):
        path = Path(path)
        if path.exists() and path.stat().st_size > 2:
            self.empty = False
        else:
            self.empty = True

        self.terminator = ''

        super().__init__(path, *args, **kws)

        self.stream.write('[\n')

    def emit(self, rec):
        super().emit(rec)
        self.stream.write(',\n')

    def close(self):
        self.stream.write('\n]')
        super().close()


def _log_to_file(output: specs.Output):
    Path(output.log_path).parent.mkdir(exist_ok=True, parents=True)

    logger = logging.getLogger('labbench')
    formatter = lb._host.JSONFormatter()
    handler = RotatingJSONFileHandler(
        output.log_path, maxBytes=50_000_000, backupCount=5, encoding='utf8'
    )

    handler.setFormatter(formatter)
    handler.setLevel(logging.DEBUG)

    if hasattr(handler, '_striqt_handler'):
        logger.removeHandler(handler._striqt_handler)

    logger.setLevel(lb.util._LOG_LEVEL_NAMES[output.log_level])
    logger.addHandler(handler)
    logger._striqt_handler = handler


def init_sweep_cli(
    *,
    yaml_path: Path,
    output_path: typing.Optional[str] = None,
    store_backend: typing.Optional[str] = None,
    remote: typing.Optional[str] = None,
    force: bool = False,
    verbose: bool = False,
    debug: bool = False,
) -> CLIObjects:
    # now re-read the yaml, using sweep_cls as the schema, but without knowledge of
    sweep_spec = io.read_yaml_sweep(yaml_path)
    debug_handler = DebugOnException(debug)

    if '{' in sweep_spec.output.path:
        # in this case, we're still waiting to fill in radio_id
        open_sink_early = False
    else:
        open_sink_early = True

    if store_backend is None and sweep_spec.output.store is None:
        click.echo(
            'specify output.store in the yaml file or use -s <NAME> on the command line'
        )
        sys.exit(1)

    if output_path is None and sweep_spec.output.path is None:
        click.echo(
            'specify output.path in the yaml file or use -o PATH on the command line'
        )
        sys.exit(1)

    lb.show_messages('warning')

    # start by connecting to the controller, so that the radio id can be used
    # as a file naming field
    peripherals = None
    sink = None
    controller = None

    try:
        calls = {}
        calls['controller'] = lb.Call(get_controller, remote, sweep_spec)
        if open_sink_early:
            if output_path is None:
                output_path = sweep_spec.output.path

            yaml_classes = _get_extension_classes(sweep_spec)
            # now, open the store
            sink = yaml_classes.sink_cls(
                sweep_spec, output_path=output_path, store_backend=store_backend
            )
            calls['open sink'] = lb.Call(sink.open)

        with util.stopwatch(
            f'open {", ".join(calls)}',
            'controller',
            logger_level=logging.INFO,
            threshold=1,
        ):
            controller = util.concurrently_with_fg(calls, False)['controller']

        yaml_classes = _get_extension_classes(sweep_spec)
        radio_id = controller.radio_id(sweep_spec.radio_setup.driver)
        sweep_spec = io.read_yaml_sweep(
            yaml_path,
            radio_id=radio_id,
        )

        if sweep_spec.output.log_path is not None:
            _log_to_file(sweep_spec.output)

        peripherals = yaml_classes.peripherals_cls(sweep_spec)

        # open the rest
        calls = {}
        calls['calibration'] = lb.Call(
            calibration.read_calibration,
            sweep_spec.radio_setup.calibration,
        )
        calls['peripherals'] = lb.Call(peripherals.open)
        if not open_sink_early:
            if output_path is None:
                output_path = sweep_spec.output.path

            sink = yaml_classes.sink_cls(
                sweep_spec, output_path=output_path, store_backend=store_backend
            )
            calls['open sink'] = lb.Call(sink.open)

        with util.stopwatch(
            f'load {", ".join(calls)}',
            'controller',
            logger_level=logging.INFO,
            threshold=0.25,
        ):
            opened = lb.concurrently(**calls)

        peripherals.set_source(controller.radios[sweep_spec.radio_setup.driver])

    except BaseException as ex:
        if debug_handler.enable:
            print(ex)
            debug_handler.run(*sys.exc_info())
            sys.exit(1)
        else:
            raise

    return CLIObjects(
        sink=sink,
        controller=controller,
        sweep_spec=sweep_spec,
        peripherals=peripherals,
        debugger=debug_handler,
        calibration=opened.get('calibration', None),
    )


def _extract_traceback(
    exc_type: type[BaseException],
    exc_value: BaseException,
    traceback,
    *,
    show_locals: bool = False,
    locals_hide_dunder: bool = True,
    locals_hide_sunder: bool = False,
    _visited_exceptions: typing.Optional[set[BaseException]] = None,
):
    """labbench implemented ConcurrentException as a container for
    exceptions that occur in multiple threads.

    A similar feature was added to python 3.11, ExceptionGroup.
    rich.traceback supports displaying this, so we can
    extract the exceptions in the same way here.

    """

    from rich.traceback import (
        Frame,
        Stack,
        Trace,
        Traceback,
        _SyntaxError,
        walk_tb,
        LOCALS_MAX_LENGTH,
        LOCALS_MAX_STRING,
    )
    from rich import pretty
    from itertools import islice
    import inspect
    import os

    stacks: list[Stack] = []
    is_cause = False

    from rich import _IMPORT_CWD

    notes: list[str] = getattr(exc_value, '__notes__', None) or []

    grouped_exceptions: set[BaseException] = (
        set() if _visited_exceptions is None else _visited_exceptions
    )

    def safe_str(_object: typing.Any) -> str:
        """Don't allow exceptions from __str__ to propagate."""
        try:
            return str(_object)
        except Exception:
            return '<exception str() failed>'

    while True:
        stack = Stack(
            exc_type=safe_str(exc_type.__name__),
            exc_value=safe_str(exc_value),
            is_cause=is_cause,
            notes=notes,
        )

        if sys.version_info >= (3, 11):
            if isinstance(exc_value, (BaseExceptionGroup, ExceptionGroup)):
                stack.is_group = True
                for exception in exc_value.exceptions:
                    if exception in grouped_exceptions:
                        continue
                    grouped_exceptions.add(exception)
                    stack.exceptions.append(
                        Traceback.extract(
                            type(exception),
                            exception,
                            exception.__traceback__,
                            show_locals=show_locals,
                            locals_hide_dunder=locals_hide_dunder,
                            locals_hide_sunder=locals_hide_sunder,
                            _visited_exceptions=grouped_exceptions,
                        )
                    )

        if isinstance(exc_value, lb.util.ConcurrentException):
            stack.is_group = True
            for exception in exc_value.thread_exceptions:
                if exception in grouped_exceptions:
                    continue
                grouped_exceptions.add(exception)
                stack.exceptions.append(
                    Traceback.extract(
                        type(exception),
                        exception,
                        exception.__traceback__,
                        show_locals=show_locals,
                        locals_hide_dunder=locals_hide_dunder,
                        locals_hide_sunder=locals_hide_sunder,
                        _visited_exceptions=grouped_exceptions,
                    )
                )

        if isinstance(exc_value, SyntaxError):
            stack.syntax_error = _SyntaxError(
                offset=exc_value.offset or 0,
                filename=exc_value.filename or '?',
                lineno=exc_value.lineno or 0,
                line=exc_value.text or '',
                msg=exc_value.msg,
                notes=notes,
            )

        stacks.append(stack)
        append = stack.frames.append

        def get_locals(
            iter_locals: typing.Iterable[tuple[str, object]],
        ) -> typing.Iterable[tuple[str, object]]:
            """Extract locals from an iterator of key pairs."""
            if not (locals_hide_dunder or locals_hide_sunder):
                yield from iter_locals
                return
            for key, value in iter_locals:
                if locals_hide_dunder and key.startswith('__'):
                    continue
                if locals_hide_sunder and key.startswith('_'):
                    continue
                yield key, value

        for frame_summary, line_no in walk_tb(traceback):
            filename = frame_summary.f_code.co_filename

            last_instruction: typing.Optional[tuple[tuple[int, int], tuple[int, int]]]
            last_instruction = None
            if sys.version_info >= (3, 11):
                instruction_index = frame_summary.f_lasti // 2
                instruction_position = next(
                    islice(
                        frame_summary.f_code.co_positions(),
                        instruction_index,
                        instruction_index + 1,
                    )
                )
                (
                    start_line,
                    end_line,
                    start_column,
                    end_column,
                ) = instruction_position
                if (
                    start_line is not None
                    and end_line is not None
                    and start_column is not None
                    and end_column is not None
                ):
                    last_instruction = (
                        (start_line, start_column),
                        (end_line, end_column),
                    )

            if filename and not filename.startswith('<'):
                if not os.path.isabs(filename):
                    filename = os.path.join(_IMPORT_CWD, filename)
            if frame_summary.f_locals.get('_rich_traceback_omit', False):
                continue

            frame = Frame(
                filename=filename or '?',
                lineno=line_no,
                name=frame_summary.f_code.co_name,
                locals=(
                    {
                        key: pretty.traverse(
                            value,
                            max_length=LOCALS_MAX_LENGTH,
                            max_string=LOCALS_MAX_STRING,
                        )
                        for key, value in get_locals(frame_summary.f_locals.items())
                        if not (inspect.isfunction(value) or inspect.isclass(value))
                    }
                    if show_locals
                    else None
                ),
                last_instruction=last_instruction,
            )
            append(frame)
            if frame_summary.f_locals.get('_rich_traceback_guard', False):
                del stack.frames[:]

        if not grouped_exceptions:
            cause = getattr(exc_value, '__cause__', None)
            if cause is not None and cause is not exc_value:
                exc_type = cause.__class__
                exc_value = cause
                # __traceback__ can be None, e.g. for exceptions raised by the
                # 'multiprocessing' module
                traceback = cause.__traceback__
                is_cause = True
                continue

            cause = exc_value.__context__
            if cause is not None and not getattr(
                exc_value, '__suppress_context__', False
            ):
                exc_type = cause.__class__
                exc_value = cause
                traceback = cause.__traceback__
                is_cause = False
                continue
        # No cover, code is reached but coverage doesn't recognize it.
        break  # pragma: no cover

    trace = Trace(stacks=stacks)

    return trace


def print_exception():
    from rich.console import Console
    from rich.traceback import Traceback

    console = Console()

    trace = _extract_traceback(*sys.exc_info(), show_locals=False)

    traceback = Traceback(
        trace,
        width=None,
        show_locals=False,
        suppress=['rich', 'labbench', 'zarr', 'xarray', 'pandas'],
    )

    console.print(traceback)


def maybe_start_debugger(cli_objects: CLIObjects | None, exc_info):
    if cli_objects is not None and cli_objects.debugger.enable:
        cli_objects.debugger.run(*exc_info)


def iter_sweep_cli(
    cli: CLIObjects,
    *,
    remote=None,
):
    # pull out the cli elements that have context
    cli_objects = cli
    *cli_context, sweep, cal = cli_objects

    with lb.sequentially(*cli_context):
        try:
            reuse_iq = cli.sweep_spec.radio_setup.reuse_iq
            # iterate through the sweep specification, yielding a dataset for each capture
            sweep_iter = cli.controller.iter_sweep(
                sweep,
                calibration=cal,
                prepare=False,
                always_yield=True,
                reuse_compatible_iq=reuse_iq,  # calibration-specific optimization
            )

            sweep_iter.set_peripherals(cli.peripherals)
            sweep_iter.set_writer(cli.sink)

            # step through captures
            for _ in sweep_iter:
                yield
        except:
            raise
        else:
            cli.sink.flush()


def iterate_sweep_cli(
    cli: CLIObjects,
    *,
    remote=None,
):
    # pull out the cli elements that have context
    cli_objects = cli
    *cli_context, sweep, cal = cli_objects

    with lb.sequentially(*cli_context):
        try:
            reuse_iq = cli.sweep_spec.radio_setup.reuse_iq
            # iterate through the sweep specification, yielding a dataset for each capture
            sweep_iter = cli.controller.iter_sweep(
                sweep,
                calibration=cal,
                prepare=False,
                always_yield=True,
                reuse_compatible_iq=reuse_iq,  # calibration-specific optimization
            )

            sweep_iter.set_peripherals(cli.peripherals)
            sweep_iter.set_writer(cli.sink)

            # step through captures
            for _ in sweep_iter:
                yield None

            cli.sink.flush()
        except BaseException as ex:
            if cli_objects.debugger.enable:
                print(ex)
                cli_objects.debugger.run(*sys.exc_info())
                sys.exit(1)
            else:
                raise
