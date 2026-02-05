"""dynamic imports and context management of multiple resources"""

from __future__ import annotations as __

import contextlib
import functools
import importlib
import os
import sys
import threading
import typing
from pathlib import Path

from . import bindings, io, util
from .peripherals import PeripheralsBase
from .sinks import SinkBase
from .sources import get_source_id, SourceBase, _PS, _PC
from .. import specs
from ..specs import _TC, _TP, _TS

import typing_extensions
import striqt.analysis as sa


if typing.TYPE_CHECKING:
    import xarray as xr

    # typing workarounds for python < 3.10
    _P = typing_extensions.ParamSpec('_P')
    _R = typing.TypeVar('_R')

    class Resources(
        typing_extensions.TypedDict, typing.Generic[_TS, _TP, _TC, _PS, _PC]
    ):
        """Sensor resources needed to run a sweep"""

        source: SourceBase[_TS, _TC, _PS, _PC]
        sink: SinkBase
        peripherals: PeripheralsBase[_TP, _TC]
        except_context: typing_extensions.NotRequired[typing.ContextManager]
        sweep_spec: specs.Sweep[_TS, _TP, _TC]
        calibration: 'xr.Dataset|None'
        alias_func: specs.helpers.PathAliasFormatter | None

    class AnyResources(
        typing_extensions.TypedDict,
        typing.Generic[_TS, _TP, _TC, _PS, _PC],
        total=False,
    ):
        """Sensor resources needed to run a sweep"""

        source: SourceBase[_TS, _TC, _PS, _PC]
        sink: SinkBase
        peripherals: PeripheralsBase[_TP, _TC]
        except_context: typing_extensions.NotRequired[typing.ContextManager]
        sweep_spec: specs.Sweep[_TS, _TP, _TC]
        calibration: 'xr.Dataset|None'
        alias_func: specs.helpers.PathAliasFormatter | None

else:
    # workaround for python < 3.10
    class Resources(typing_extensions.TypedDict):
        """Sensor resources needed to run a sweep"""

        source: SourceBase
        sink: SinkBase
        peripherals: typing_extensions.NotRequired[PeripheralsBase]
        except_context: typing_extensions.NotRequired[typing.ContextManager]
        sweep_spec: specs.Sweep
        calibration: 'xr.Dataset|None'
        alias_func: specs.helpers.PathAliasFormatter | None

    class AnyResources(typing_extensions.TypedDict, total=False):
        """Sensor resources needed to run a sweep"""

        source: SourceBase
        sink: SinkBase
        peripherals: typing_extensions.NotRequired[PeripheralsBase]
        except_context: typing_extensions.NotRequired[typing.ContextManager]
        sweep_spec: specs.Sweep
        calibration: 'xr.Dataset|None'
        alias_func: specs.helpers.PathAliasFormatter | None


def _timeit(desc: str = '') -> typing.Callable[[util._Tfunc], util._Tfunc]:
    return sa.util.stopwatch(
        desc, 'sweep', threshold=0.5, logger_level=util.logging.INFO
    )


def import_sink_cls(spec: specs.Extension, lazy: bool = False) -> type[SinkBase]:
    if spec.sink is None:
        raise TypeError('extension sink was not specified')
    mod_name, *sub_names, obj_name = spec.sink.rsplit('.')
    if lazy:
        mod = util.lazy_import(mod_name)
    else:
        mod = importlib.import_module(mod_name)
    for name in sub_names:
        mod = getattr(mod, name)

    return getattr(mod, obj_name)


class ConnectionManager(
    contextlib.ExitStack,
    typing.Generic[_TS, _TP, _TC, _PS, _PC],
):
    _resources: AnyResources[_TS, _TP, _TC, _PS, _PC]

    def __init__(self, sweep_spec: specs.Sweep[_TS, _TP, _TC]):
        super().__init__()
        self._resources = AnyResources(sweep_spec=sweep_spec)

    def __enter__(self):  # type: ignore
        return self.resources

    @functools.cached_property
    def resources(self) -> Resources[_TS, _TP, _TC, _PS, _PC]:
        missing = Resources.__required_keys__ - set(self._resources.keys())

        # TODO: troubleshoot why runtime __required_keys__ includes NotRequired fields
        missing = missing - {'peripherals', 'except_context'}
        if len(missing) == 0:
            return typing.cast(Resources[_TS, _TP, _TC, _PS, _PC], self._resources)
        else:
            raise TypeError(f'connections {missing!r} are incomplete')


def _setup_logging(sink: specs.Sink, formatter):
    util.safe_import('xarray')
    log_path = formatter(sink.log_path)
    util.log_to_file(log_path, sink.log_level)


class _SourceOpenCallback(typing.Protocol):
    def __call__(self, sweep: specs.Sweep, source_id: str) -> None: ...


def _open_devices(
    conn: ConnectionManager,
    binding: bindings.SensorBinding,
    spec: specs.Sweep,
    skip_peripherals: bool = False,
    on_source_opened: _SourceOpenCallback | None = None,
):
    """open source and any peripherals"""

    def _post_source_open():
        source_id = get_source_id(spec.source)
        specs.helpers.list_capture_adjustments(spec, source_id=source_id)

        if on_source_opened is not None:
            on_source_opened(spec, source_id)

    1 // 0

    source = util.threadpool.submit(
        _timeit('open sensor source')(binding.source.from_spec),
        spec.source,
        captures=spec.captures,
        loops=spec.loops,
        reuse_iq=spec.options.reuse_iq,
    )

    source_callback = util.threadpool.submit(_post_source_open)

    if not skip_peripherals:
        peripherals = util.threadpool.submit(
            _timeit('open peripherals')(binding.peripherals), spec
        )
    else:
        peripherals = None

    with util.ExceptionGrouper() as exc:
        with exc.defer():
            source = conn._resources['source'] = source.result()
            conn.enter_context(source)

        with exc.defer():
            source_callback.result()

        with exc.defer():
            if peripherals is not None:
                peripherals = conn._resources['peripherals'] = peripherals.result()
                conn.enter_context(peripherals)

    if peripherals is not None:
        peripherals.setup(spec.captures, spec.loops)


@sa.util.stopwatch('open resources', 'sweep', 1.0, sa.util.PERFORMANCE_INFO)
def open_resources(
    spec: specs.Sweep[_TS, _TP, _TC],
    spec_path: str | Path | None = None,
    except_context: typing.ContextManager | None = None,
    *,
    test_only: bool = False,
    on_source_opened: _SourceOpenCallback | None = None,
) -> ConnectionManager[_TS, _TP, _TC, _PS, _PC]:
    """open the sensor hardware and software contexts needed to run the given sweep.

    The returned Connections object contains the resulting context. All of its resources
    have been opened and set up as needed to run the specified sweep.
    """

    from .compute import prepare_compute

    formatter = specs.helpers.PathAliasFormatter(spec, spec_path=spec_path)

    if spec_path is not None:
        os.chdir(str(Path(spec_path).parent))

    bind = bindings.get_binding(spec)
    conn = ConnectionManager(sweep_spec=spec)

    if spec.extensions.sink is not None:
        sink_cls = import_sink_cls(spec.extensions, lazy=True)
    elif bind.sink is not None:
        sink_cls = bind.sink
    else:
        raise TypeError('no sink class in sensor binding or extensions.sink spec')

    exc = util.ExceptionGrouper('failed to open resources')
    with util.share_thread_interrupts():
        # background threads
        sink = util.threadpool.submit(
            _timeit('open sink')(sink_cls), spec, alias_func=formatter
        )

        devices = util.threadpool.submit(
            _open_devices,
            conn,
            bind,
            spec,
            skip_peripherals=test_only,
            on_source_opened=on_source_opened,
        )

        if spec.source.calibration is not None:
            cal = util.threadpool.submit(
                _timeit('read calibration')(io.read_calibration),
                spec.source.calibration,
                formatter,
            )
        else:
            cal = None

        if spec.sink.log_path is not None:
            log_setup = util.threadpool.submit(_setup_logging, spec.sink, formatter)
        else:
            log_setup = None

        # foreground thread
        prepare_compute(spec, skip_warmup=test_only)

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

        # if except_context is not None:
        #     conn.enter_context(except_context)
        #     conn._resources['except_context'] = except_context

    try:
        exc.handle()
    except:
        conn.__exit__(*sys.exc_info())
        raise

    conn._resources['sweep_spec'] = spec
    conn._resources['alias_func'] = formatter

    return conn


def open_sensor_from_yaml(
    yaml_path: Path,
    *,
    except_context: typing.ContextManager | None = None,
    output_path: str | None = None,
    store_backend: str | None = None,
) -> ConnectionManager[typing.Any, typing.Any, typing.Any, typing.Any, typing.Any]:
    spec = io.read_yaml_spec(yaml_path)

    sink = spec.sink
    if output_path is not None:
        sink = sink.replace(path=output_path)
    if store_backend is not None:
        sink = sink.replace(store=store_backend)
    spec = spec.replace(output=sink)

    return open_resources(spec, yaml_path, except_context)
