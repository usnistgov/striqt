"""open the hardware and software resources of an action as one context.

The source, peripherals, sink, calibration file, log file, and compute warmup open
concurrently in `util.threadpool`. A failure in one closes the others.
"""

from __future__ import annotations as __

import contextlib
import dataclasses
import os
import sys
from contextlib import AbstractContextManager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Generic, cast

import typing_extensions

import striqt.analysis as sa

from .. import specs
from . import bindings, compute, controller, io, util
from .peripherals import NoPeripherals
from .sinks import SinkBase
from .sources import buffers
from .typing import PC, PS, SA, SC, SP, SS, Peripherals

if TYPE_CHECKING:
    import xarray as xr

    from .actions.core import ActionRunner
    from .typing import (
        AnalysisSelector,
        AnalyzedCallback,
        CachedCallback,
        P,
        PassThroughWrapper,
        PromptCallback,
        R,
        SourceOpenCallback,
        SweepCallbacksKws,
    )


@dataclasses.dataclass(frozen=True)
class SweepCallbacks:
    """the hooks of one run of an action.

    An exception from any hook ends the run. `select_analysis`, `on_analyzed`, and
    `on_sink_cached` run in the analysis thread. That thread blocks until the hook
    returns.
    """

    on_source_opened: SourceOpenCallback | None = None
    """hook run once with ``(sweep, source_id)`` in a worker thread after the source
    opens and before the first capture"""
    on_analyzed: AnalyzedCallback | None = None
    """hook run with the `DelayedDataset` of each acquisition before the sink
    receives the dataset; the `extra_data` of the dataset lacks the peripheral
    entries"""
    on_sink_cached: CachedCallback | None = None
    """hook run with ``(resources, cache, capture, result, **kwargs)`` each time a
    measurement fills a `KwArgCache`; `kwargs` are the measurement keywords"""
    select_analysis: AnalysisSelector | None = None
    """hook run with ``(capture, analysis)`` before each analysis, returning the
    group to evaluate in place of the adjusted `analysis`; None selects the whole
    group"""
    prompt: PromptCallback | None = None
    """hook run with the text of each operator prompt, returning the operator's
    answer; None reads one line from stdin through `blocking_input`"""


if TYPE_CHECKING:
    # the Generic parameters are for type checking only on python < 3.10

    class Resources(typing_extensions.TypedDict, Generic[SS, SP, SC, SA, PS, PC]):
        """the open resources over which an action runs.

        `action_spec` is the spec the action runs. `runner` is the action built from
        `action_spec`. `sweep_spec` is the `Sweep` for which the hardware is open.
        `callbacks` are the `SweepCallbacks` of the run; a missing key means none.
        """

        source: controller.Controller[SS, SP, SC, SA, PS, PC]
        sink: SinkBase
        peripherals: Peripherals[SP, SC]
        except_context: typing_extensions.NotRequired[AbstractContextManager]
        action_spec: specs.ActionSpec[Any, SP, SA]
        sweep_spec: specs.Sweep[SS, SP, SC, SA]
        runner: ActionRunner[Any, SS, SP, SC, SA, PS, PC]
        calibration: xr.Dataset | None
        format_path: specs.paths.PathFormatter | None
        callbacks: typing_extensions.NotRequired[SweepCallbacks]

    class AnyResources(
        typing_extensions.TypedDict,
        Generic[SS, SP, SC, SA, PS, PC],
        total=False,
    ):
        """`Resources` with every key optional"""

        source: controller.Controller[SS, SP, SC, SA, PS, PC]
        sink: SinkBase
        peripherals: Peripherals[SP, SC]
        except_context: typing_extensions.NotRequired[AbstractContextManager]
        action_spec: specs.ActionSpec[Any, SP, SA]
        sweep_spec: specs.Sweep[SS, SP, SC, SA]
        runner: ActionRunner[Any, SS, SP, SC, SA, PS, PC]
        calibration: xr.Dataset | None
        format_path: specs.paths.PathFormatter | None
        callbacks: typing_extensions.NotRequired[SweepCallbacks]

else:
    # the Generic parameters are for type checking only on python < 3.10
    class Resources(typing_extensions.TypedDict):
        """the open resources over which an action runs.

        `action_spec` is the spec the action runs. `runner` is the action built from
        `action_spec`. `sweep_spec` is the `Sweep` for which the hardware is open.
        `callbacks` are the `SweepCallbacks` of the run; a missing key means none.
        """

        source: controller.Controller
        sink: SinkBase
        peripherals: typing_extensions.NotRequired[Peripherals]
        except_context: typing_extensions.NotRequired[AbstractContextManager]
        action_spec: specs.ActionSpec
        sweep_spec: specs.Sweep
        runner: ActionRunner
        calibration: xr.Dataset | None
        format_path: specs.paths.PathFormatter | None
        callbacks: typing_extensions.NotRequired[SweepCallbacks]

    class AnyResources(typing_extensions.TypedDict, total=False):
        """`Resources` with every key optional"""

        source: controller.Controller
        sink: SinkBase
        peripherals: typing_extensions.NotRequired[Peripherals]
        except_context: typing_extensions.NotRequired[AbstractContextManager]
        action_spec: specs.ActionSpec
        sweep_spec: specs.Sweep
        runner: ActionRunner
        calibration: xr.Dataset | None
        format_path: specs.paths.PathFormatter | None
        callbacks: typing_extensions.NotRequired[SweepCallbacks]


def prompt_hook(resources: Resources[SS, SP, SC, SA, PS, PC]) -> PromptCallback:
    """return the operator prompt of ``resources['callbacks']``, or
    `sa.util.blocking_input` when the key is missing or the hook is None."""
    callbacks = resources.get('callbacks')
    if callbacks is None or callbacks.prompt is None:
        return sa.util.blocking_input
    return callbacks.prompt


def _timeit(desc: str = '') -> PassThroughWrapper[P, R]:
    return sa.util.stopwatch(
        desc, 'sweep', threshold=0.5, logger_level=util.logging.INFO
    )


def _open_sink(
    runner: ActionRunner[Any, SS, SP, SC, SA, PS, PC],
    format_path: specs.paths.PathFormatter | None = None,
) -> SinkBase[SC]:
    with sa.util.stopwatch('open sink', 'sweep', 0.5, util.logging.INFO):
        return runner.open_sink(format_path)


class ConnectionManager(
    contextlib.ExitStack,
    Generic[SS, SP, SC, SA, PS, PC],
):
    """the open contexts of one action spec, opened on entry and closed on exit.

    Construction stores the spec and its options; nothing opens until `__enter__`,
    which yields the `Resources` dict. `set_callbacks` sets the hooks of the runs
    before entry.
    """

    _resources: AnyResources[SS, SP, SC, SA, PS, PC]

    def __init__(
        self,
        spec: specs.ActionSpec[SS, SP, SA],
        spec_path: str | Path | None = None,
        *,
        test_only: bool = False,
    ):
        super().__init__()
        self._spec = spec
        self._spec_path = spec_path
        self._test_only = test_only
        self._callbacks = SweepCallbacks()
        self._opened = False
        self._resources = AnyResources()

    def set_callbacks(
        self,
        callbacks: SweepCallbacks | None = None,
        /,
        **hooks: typing_extensions.Unpack[SweepCallbacksKws],
    ) -> ConnectionManager[SS, SP, SC, SA, PS, PC]:
        """set the hooks of the runs over the resources and return the manager.

        `hooks` replace the named fields of `callbacks`, or of the hooks set so far
        when `callbacks` is None; the fields of `SweepCallbacks` are the accepted
        keywords.

        Raises:
            RuntimeError: if the resources are already open, since
                `on_source_opened` runs during the open
        """
        if self._opened:
            raise RuntimeError('set the callbacks before entering the resources')
        base = self._callbacks if callbacks is None else callbacks
        self._callbacks = dataclasses.replace(base, **hooks)
        return self

    # ExitStack is inherited for the callback registry. `with` yields the collected
    # resources rather than the stack
    def __enter__(self) -> Resources[SS, SP, SC, SA, PS, PC]:  # ty: ignore[invalid-method-override]
        if not self._opened:
            self._opened = True
            _open(self)
        return self.resources

    @util.cached_property
    def resources(self) -> Resources[SS, SP, SC, SA, PS, PC]:
        missing = Resources.__required_keys__ - set(self._resources.keys())

        # at runtime, __required_keys__ includes the NotRequired fields
        missing = missing - {'peripherals', 'except_context', 'callbacks'}
        if len(missing) == 0:
            return cast(Resources[SS, SP, SC, SA, PS, PC], self._resources)
        else:
            raise TypeError(f'connections {missing!r} are incomplete')


def _setup_logging(sink: specs.Sink, formatter):
    log_path = formatter(sink.log_path)
    util.log_to_file(log_path, sink.log_level)


def _open_devices(
    conn: ConnectionManager,
    ctrl_cls: type[controller.Controller],
    spec: specs.Sweep,
    skip_peripherals: bool = False,
    format_path: specs.paths.PathFormatter | None = None,
):
    """open the source and peripherals, with `NoPeripherals` when `skip_peripherals`"""

    source = util.threadpool.submit(ctrl_cls.from_sweep_spec, spec, format_path)

    peripherals_cls: type[Peripherals] = (
        NoPeripherals if skip_peripherals else ctrl_cls.sensor.peripherals_cls
    )
    peripherals = util.threadpool.submit(
        _timeit('open peripherals')(peripherals_cls), spec
    )

    with util.ExceptionStack() as exc:
        with exc.defer():
            source = conn._resources['source'] = source.result()
            conn.enter_context(source)

        with exc.defer():
            peripherals = conn._resources['peripherals'] = peripherals.result()
            # ty 0.0.81 does not accept a Protocol with __enter__/__exit__ as an
            # AbstractContextManager
            conn.enter_context(peripherals)  # ty: ignore[invalid-argument-type]

    # setup runs only after both the source and the peripherals are open
    peripherals.setup(spec.captures, spec.loops)


def resident_bytes_bound(
    sweep: specs.Sweep[Any, Any, Any, Any], source_id: str | None = None
) -> int:
    """return an upper bound (bytes) on the raw IQ resident during a run of `sweep`.

    The bound counts ``sweep.source.bursts.buffers + 2`` acquisitions: the queued
    segments, the one under analysis, and the one being written. Each acquisition is
    sized as the largest read over the captures of
    `specs.sequencing.repeat_captures`: the port count of the capture, times 8 bytes
    per complex64 sample, times the source samples of the record, the correction
    overlaps, and the transient and strobe holdoff of a first burst segment. The
    holdoff of a first segment is read into the receive buffer and dropped from the
    acquisition. The reads of the later segments of a burst include no holdoff. The
    bound exceeds the resident bytes by at least the holdoff of each acquisition. The
    bound covers the `x` arrays alone, not the resampled, filtered, or analyzed
    copies.

    Args:
        sweep: the sweep to size, with the `bursts` policy of its source
        source_id: the source id for `specs.sequencing.repeat_captures`, or None
    """
    captures = specs.sequencing.repeat_captures(sweep, source_id)
    largest = 0
    for capture in captures:
        analysis = specs.sequencing.adjust_analysis(
            sweep.analysis, capture.adjust_analysis
        )
        overlaps = compute.get_correction_overlaps(capture, sweep.source, analysis)
        count = buffers.get_read_count(
            capture, sweep.source, include_holdoff=True, overlap=sum(overlaps)
        )
        ports = len(specs.captures.split_capture_ports(capture))
        largest = max(largest, ports * 8 * count)
    return (sweep.source.bursts.buffers + 2) * largest


def _prepare_sweep(spec: specs.Sweep, callbacks: SweepCallbacks) -> None:
    """run `callbacks.on_source_opened` after listing the capture adjustments of `spec`.

    The `resident_bytes_bound` of `spec` is logged at INFO on the ``sweep`` logger.
    """
    source_id = controller.lookup.id(spec.source)
    specs.sequencing.list_capture_adjustments(spec, source_id=source_id)

    bound = resident_bytes_bound(spec, source_id)
    sa.util.get_logger('sweep').info(
        f'resident IQ bound: {bound} bytes ({bound / 1e6:.1f} MB) over '
        f'{spec.source.bursts.buffers} queued acquisitions'
    )

    if callbacks.on_source_opened is not None:
        callbacks.on_source_opened(spec, source_id)


def open_resources(
    spec: specs.ActionSpec[SS, SP, SA],
    spec_path: str | Path | None = None,
    *,
    test_only: bool = False,
) -> ConnectionManager[SS, SP, SC, SA, PS, PC]:
    """return the manager of the sensor hardware and software contexts that run `spec`.

    On entry, the source, peripherals, sink, calibration file, and log file open
    concurrently for the `sweep` of the action runner built from `spec`, and the
    compute warmup runs in the calling thread; the manager yields the open
    `Resources` and closes every resource on exit. The hooks of the runs are set
    with `ConnectionManager.set_callbacks` before entry; an exception from a hook
    ends the run. Progress is logged at INFO on the ``sweep`` logger.

    Args:
        spec: the action spec
        spec_path: the file from which `spec` was read, or None; on entry the
            process working directory becomes the directory of the file, and the
            ``{spec_path}`` path field is set
        test_only: if True, open `NoPeripherals` in place of the peripherals and
            skip the compute warmup

    Raises:
        TypeError: on entry, if no action runner is registered for the type of `spec`
        KeyError: on entry, if no sensor binding is registered for the type of `spec`
        ExceptionGroup: on entry, if more than one resource fails to open; a single
            failure raises that exception
    """
    return ConnectionManager(spec, spec_path, test_only=test_only)


@sa.util.stopwatch('open resources', 'sweep', 1.0, sa.util.INFO)
def _open(conn: ConnectionManager[SS, SP, SC, SA, PS, PC]) -> None:
    """open every resource of `conn` into `conn._resources`, and close every opened one on a failure."""

    import numpy as np  # noqa: F401  avoids import concurrency bugs on python < 3.15

    from .compute import prepare_compute

    logger = sa.util.get_logger('sweep')
    logger.log(sa.util.INFO, 'opening sensor resources')

    from . import actions

    spec = conn._spec
    spec_path = conn._spec_path
    test_only = conn._test_only
    callbacks = conn._callbacks

    fmt = specs.paths.PathFormatter(spec, spec_path=spec_path)

    if spec_path is not None:
        os.chdir(str(Path(spec_path).parent))

    runner = actions.registry.lookup(spec)(spec)
    sweep = runner.sweep
    ctrl_cls = bindings.get_controller(spec)
    conn._resources['sweep_spec'] = sweep

    exc = util.ExceptionStack('failed to open resources', cancel_on_except=True)
    with util.share_thread_interrupts():
        devices = util.threadpool.submit(
            _open_devices, conn, ctrl_cls, sweep, test_only, fmt
        )
        prep_sweep = util.threadpool.submit(_prepare_sweep, sweep, callbacks)
        sink = util.threadpool.submit(_open_sink, runner, fmt)

        with exc.defer():
            # foreground thread 1: initialize warmup sweeps
            try:
                compute_iter = prepare_compute(sweep, skip_warmup=test_only)
                next(compute_iter)
            except Exception:
                util.cancel_threads()
                raise

        if sweep.source.calibration is not None:
            read_calibration = cast(
                'Callable[..., xr.Dataset | None]', io.read_calibration
            )
            cal = util.threadpool.submit(
                _timeit('read calibration')(read_calibration),
                sweep.source.calibration,
                fmt,
            )
        else:
            cal = None
        if sweep.sink.log_path is not None:
            log_setup = util.threadpool.submit(_setup_logging, sweep.sink, fmt)
        else:
            log_setup = None

        with exc.defer():
            # finish any warmups
            try:
                for _ in compute_iter:
                    pass
            except Exception:
                util.cancel_threads()
                raise

        with exc.defer():
            sink = conn._resources['sink'] = sink.result()
            conn.enter_context(sink)
        with exc.defer():
            devices.result()
        with exc.defer():
            if cal is not None:
                cal = cal.result()
            conn._resources['calibration'] = cal
        with exc.defer():
            if log_setup is not None:
                log_setup.result()
        with exc.defer():
            prep_sweep.result()

    try:
        exc.handle()
    except:
        conn.__exit__(*sys.exc_info())
        raise

    conn._resources['action_spec'] = spec
    conn._resources['runner'] = runner
    conn._resources['format_path'] = fmt
    conn._resources['callbacks'] = callbacks
