"""open the hardware and software resources of an action as one context.

The source, peripherals, sink, calibration file, log file, and compute warmup open
concurrently in `util.threadpool`. A failure in one closes the others.
"""

from __future__ import annotations as __

import contextlib
import dataclasses
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, ContextManager, Generic, cast

import typing_extensions

import striqt.analysis as sa

from .. import specs
from . import bindings, controller, io, util
from .peripherals import NoPeripherals
from .sinks import SinkBase
from .typing import PC, PS, SA, SC, SP, SS, Peripherals

if TYPE_CHECKING:
    import xarray as xr

    from .actions.core import ActionRunner
    from .typing import (
        AnalyzedCallback,
        CachedCallback,
        P,
        PassThroughWrapper,
        R,
        SourceOpenCallback,
    )


@dataclasses.dataclass(frozen=True)
class SweepCallbacks:
    """the hooks of one run of an action.

    An exception from any hook ends the run. `on_analyzed` and `on_sink_cached` run
    in the analysis thread. That thread blocks until the hook returns. The `kwargs`
    of `on_sink_cached` are the measurement keywords.
    """

    on_source_opened: SourceOpenCallback | None = None
    """hook run once with ``(sweep, source_id)`` in a worker thread after the source
    opens and before the first capture"""
    on_analyzed: AnalyzedCallback | None = None
    """hook run with the `DelayedDataset` of each capture before the sink receives
    the dataset"""
    on_sink_cached: CachedCallback | None = None
    """hook run with ``(resources, cache, capture, result, **kwargs)`` each time a
    measurement fills a `KwArgCache`"""


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
        except_context: typing_extensions.NotRequired[ContextManager]
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
        except_context: typing_extensions.NotRequired[ContextManager]
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
        except_context: typing_extensions.NotRequired[ContextManager]
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
        except_context: typing_extensions.NotRequired[ContextManager]
        action_spec: specs.ActionSpec
        sweep_spec: specs.Sweep
        runner: ActionRunner
        calibration: xr.Dataset | None
        format_path: specs.paths.PathFormatter | None
        callbacks: typing_extensions.NotRequired[SweepCallbacks]


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
    _resources: AnyResources[SS, SP, SC, SA, PS, PC]

    def __init__(self, sweep_spec: specs.Sweep[SS, SP, SC, SA]):
        super().__init__()
        self._resources = AnyResources(sweep_spec=sweep_spec)

    # ExitStack is inherited for the callback registry. `with` yields the collected
    # resources rather than the stack
    def __enter__(self) -> Resources[SS, SP, SC, SA, PS, PC]:  # ty: ignore[invalid-method-override]
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


def _prepare_sweep(spec: specs.Sweep, callbacks: SweepCallbacks) -> None:
    """list the capture adjustments of `spec`, then run `callbacks.on_source_opened`"""
    source_id = controller.lookup.id(spec.source)
    specs.sequencing.list_capture_adjustments(spec, source_id=source_id)

    if callbacks.on_source_opened is not None:
        callbacks.on_source_opened(spec, source_id)


@sa.util.stopwatch('open resources', 'sweep', 1.0, sa.util.INFO)
def open_resources(
    spec: specs.ActionSpec[SS, SP, SA],
    spec_path: str | Path | None = None,
    *,
    test_only: bool = False,
    on_source_opened: SourceOpenCallback | None = None,
    on_analyzed: AnalyzedCallback | None = None,
    on_sink_cached: CachedCallback | None = None,
) -> ConnectionManager[SS, SP, SC, SA, PS, PC]:
    """open the sensor hardware and software contexts needed to run `spec`.

    The source, peripherals, sink, calibration file, and log file open concurrently
    for the `sweep` of the action runner built from `spec`. The compute warmup runs
    in the calling thread. The returned `ConnectionManager` yields the open
    `Resources` on entry and closes every resource on exit. An exception from a hook
    ends the run. Progress is logged at INFO on the ``sweep`` logger.

    Args:
        spec: the action spec
        spec_path: the file from which `spec` was read, or None; sets the process
            working directory and the ``{spec_path}`` path field
        test_only: if True, open `NoPeripherals` in place of the peripherals and
            skip the compute warmup
        on_source_opened: hook run once with ``(sweep, source_id)`` in a worker
            thread after the source opens and before the first capture
        on_analyzed: hook run in the analysis thread with the `DelayedDataset` of
            each capture, before the sink receives the dataset
        on_sink_cached: hook run in the analysis thread with ``(resources, cache,
            capture, result, **kwargs)`` each time a measurement fills a `KwArgCache`

    Raises:
        TypeError: if no action runner is registered for the type of `spec`
        KeyError: if no sensor binding is registered for the type of `spec`
        ExceptionGroup: if more than one resource fails to open; a single failure
            raises that exception
    """

    import numpy as np  # noqa: F401  avoids import concurrency bugs on python < 3.15

    from .compute import prepare_compute

    logger = sa.util.get_logger('sweep')
    logger.log(sa.util.INFO, 'opening sensor resources')

    from . import actions

    fmt = specs.paths.PathFormatter(spec, spec_path=spec_path)

    if spec_path is not None:
        os.chdir(str(Path(spec_path).parent))

    callbacks = SweepCallbacks(
        on_source_opened=on_source_opened,
        on_analyzed=on_analyzed,
        on_sink_cached=on_sink_cached,
    )

    runner = actions.registry.lookup(spec)(spec)
    sweep = runner.sweep
    ctrl_cls = bindings.get_controller(spec)
    conn = ConnectionManager(sweep_spec=sweep)

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
    conn._resources['sweep_spec'] = sweep
    conn._resources['runner'] = runner
    conn._resources['format_path'] = fmt
    conn._resources['callbacks'] = callbacks

    return conn
