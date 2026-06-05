"""dynamic imports and context management of multiple resources"""

from __future__ import annotations as __

import contextlib
import functools
import importlib
import os
import sys
from typing import Any, cast, ContextManager, Generic, TYPE_CHECKING
from pathlib import Path

import typing_extensions

from . import bindings, controller, io, sources, util
from .sinks import SinkBase
from .typing import Peripherals, SS, SP, SC, PS, PC
from .. import specs

import striqt.analysis as sa

if TYPE_CHECKING:
    from .typing import PassThroughWrapper, SourceOpenCallback
    import xarray as xr

    # python < 3.10 workaround

    class Resources(typing_extensions.TypedDict, Generic[SS, SP, SC, PS, PC]):
        """Sensor resources needed to run a sweep"""

        source: controller.Controller[SS, SC, PS, PC]
        sink: SinkBase
        peripherals: Peripherals[SP, SC]
        except_context: typing_extensions.NotRequired[ContextManager]
        sweep_spec: specs.Sweep[SS, SP, SC]
        calibration: 'xr.Dataset|None'
        format_path: specs.helpers.PathFormatter | None

    class AnyResources(
        typing_extensions.TypedDict,
        Generic[SS, SP, SC, PS, PC],
        total=False,
    ):
        """Sensor resources needed to run a sweep"""

        source: controller.Controller[SS, SC, PS, PC]
        sink: SinkBase
        peripherals: Peripherals[SP, SC]
        except_context: typing_extensions.NotRequired[ContextManager]
        sweep_spec: specs.Sweep[SS, SP, SC]
        calibration: 'xr.Dataset|None'
        format_path: specs.helpers.PathFormatter | None

else:
    # python < 3.10 workaround
    class Resources(typing_extensions.TypedDict):
        """Sensor resources needed to run a sweep"""

        source: sources.Controller
        sink: SinkBase
        peripherals: typing_extensions.NotRequired[Peripherals]
        except_context: typing_extensions.NotRequired[ContextManager]
        sweep_spec: specs.Sweep
        calibration: 'xr.Dataset|None'
        format_path: specs.helpers.PathFormatter | None

    class AnyResources(typing_extensions.TypedDict, total=False):
        """Sensor resources needed to run a sweep"""

        source: sources.Controller
        sink: SinkBase
        peripherals: typing_extensions.NotRequired[Peripherals]
        except_context: typing_extensions.NotRequired[ContextManager]
        sweep_spec: specs.Sweep
        calibration: 'xr.Dataset|None'
        format_path: specs.helpers.PathFormatter | None


def _timeit(desc: str = '') -> PassThroughWrapper:
    return sa.util.stopwatch(
        desc, 'sweep', threshold=0.5, logger_level=util.logging.INFO
    )


def _open_sink(
    spec: specs.Sweep[Any, Any, SC],
    default_cls: type[SinkBase] | None,
    format_path: specs.helpers.PathFormatter | None = None,
) -> SinkBase[SC]:
    with sa.util.stopwatch('open sink', 'sweep', 0.5, util.logging.INFO):
        if spec.extensions.sink is not None:
            mod_name, *sub_names, obj_name = spec.extensions.sink.rsplit('.')
            mod = importlib.import_module(mod_name)
            for name in sub_names:
                mod = getattr(mod, name)
            sink_cls: type[SinkBase] = getattr(mod, obj_name)
        elif default_cls is not None:
            sink_cls = default_cls
        else:
            raise TypeError('no sink class in sensor binding or spec .extensions.sink')

        return sink_cls(spec, format_path)


class ConnectionManager(
    contextlib.ExitStack,
    Generic[SS, SP, SC, PS, PC],
):
    _resources: AnyResources[SS, SP, SC, PS, PC]

    def __init__(self, sweep_spec: specs.Sweep[SS, SP, SC]):
        super().__init__()
        self._resources = AnyResources(sweep_spec=sweep_spec)

    def __enter__(self):  # pyright: ignore
        return self.resources

    @util.cached_property
    def resources(self) -> Resources[SS, SP, SC, PS, PC]:
        missing = Resources.__required_keys__ - set(self._resources.keys())

        # TODO: troubleshoot why runtime __required_keys__ includes NotRequired fields
        missing = missing - {'peripherals', 'except_context'}
        if len(missing) == 0:
            return cast(Resources[SS, SP, SC, PS, PC], self._resources)
        else:
            raise TypeError(f'connections {missing!r} are incomplete')


def _setup_logging(sink: specs.Sink, formatter):
    log_path = formatter(sink.log_path)
    util.log_to_file(log_path, sink.log_level)


def _open_devices(
    conn: ConnectionManager,
    bind: bindings.SensorBinding,
    spec: specs.Sweep,
    skip_peripherals: bool = False,
    format_path: specs.helpers.PathFormatter | None = None,
):
    """open source and optionally peripherals"""

    source = util.threadpool.submit(bind.controller.from_sweep_spec, spec, format_path)

    if not skip_peripherals:
        peripherals = util.threadpool.submit(
            _timeit('open peripherals')(bind.peripherals), spec
        )
    else:
        peripherals = None

    with util.ExceptionStack() as exc:
        with exc.defer():
            source = conn._resources['source'] = source.result()
            conn.enter_context(source)

        with exc.defer():
            if peripherals is not None:
                peripherals = conn._resources['peripherals'] = peripherals.result()
                conn.enter_context(peripherals)

    # the peripherals wait until both the source and the
    if peripherals is not None:
        peripherals.setup(spec.captures, spec.loops)


def _prepare_sweep(spec: specs.Sweep, callback: SourceOpenCallback | None = None):
    """after the source opens, enumerate the sweep and invoke callback"""
    source_id = controller.lookup.id(spec.source)
    specs.helpers.list_capture_adjustments(spec, source_id=source_id)

    if callback is not None:
        callback(spec, source_id)


@sa.util.stopwatch('open resources', 'sweep', 1.0, sa.util.INFO)
def open_resources(
    spec: specs.Sweep[SS, SP, SC],
    spec_path: str | Path | None = None,
    *,
    test_only: bool = False,
    on_source_opened: SourceOpenCallback | None = None,
) -> ConnectionManager[SS, SP, SC, PS, PC]:
    """open the sensor hardware and software contexts needed to run the given sweep.

    The returned Connections object contains the resulting context. All of its resources
    are then open and ready to run the sweep.
    """

    from .compute import prepare_compute
    import numpy as np  # python < 3.15 workaround import concurrency bugs

    logger = sa.util.get_logger('sweep')
    logger.log(sa.util.INFO, 'opening sensor resources')

    fmt = specs.helpers.PathFormatter(spec, spec_path=spec_path)

    if spec_path is not None:
        os.chdir(str(Path(spec_path).parent))

    bind = bindings.get_binding(spec)
    conn = ConnectionManager(sweep_spec=spec)

    exc = util.ExceptionStack('failed to open resources', cancel_on_except=True)
    with util.share_thread_interrupts():
        devices = util.threadpool.submit(
            _open_devices, conn, bind, spec, test_only, fmt
        )
        prep_sweep = util.threadpool.submit(_prepare_sweep, spec, on_source_opened)
        sink = util.threadpool.submit(_open_sink, spec, bind.sink, fmt)

        with exc.defer():
            # foreground thread 1: initialize warmup sweeps
            try:
                # prioritize compute as we get started; load up buffers
                compute_iter = prepare_compute(spec, skip_warmup=test_only)
                next(compute_iter)
            except Exception:
                util.cancel_threads()
                raise

        if spec.source.calibration is not None:
            cal = util.threadpool.submit(
                _timeit('read calibration')(io.read_calibration),
                spec.source.calibration,
                fmt,
            )
        else:
            cal = None
        if spec.sink.log_path is not None:
            log_setup = util.threadpool.submit(_setup_logging, spec.sink, fmt)
        else:
            log_setup = None

        with exc.defer():
            # finish any warmups
            try:
                # prioritize compute as we get started; load up buffers
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

    conn._resources['sweep_spec'] = spec
    conn._resources['format_path'] = fmt

    return conn
