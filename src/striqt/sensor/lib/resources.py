"""mangle dynamic imports and context management of multiple resources"""

from __future__ import annotations

import collections
import contextlib
import functools
import importlib
import os
import sys
import threading
import typing
from pathlib import Path

from . import bindings, calibration, captures, io, specs, util

from .peripherals import PeripheralsBase
from .sinks import SinkBase
from .sources import SourceBase
from .specs import _TC, _TP, _TS


if typing.TYPE_CHECKING:
    import xarray as xr
    import typing_extensions

    # typing workarounds for python < 3.10
    _P = typing_extensions.ParamSpec('_P')
    _R = typing.TypeVar('_R')

    class Resources(typing_extensions.TypedDict, typing.Generic[_TS, _TP, _TC]):
        """Sensor resources needed to run a sweep"""

        source: SourceBase[_TS, _TC]
        sink: SinkBase
        peripherals: PeripheralsBase[_TP, _TC]
        except_context: typing_extensions.NotRequired[typing.ContextManager]
        sweep_spec: specs.Sweep[_TS, _TP, _TC]
        calibration: 'xr.Dataset|None'
        alias_func: captures.PathAliasFormatter | None

    class AnyResources(
        typing_extensions.TypedDict, typing.Generic[_TS, _TP, _TC], total=False
    ):
        """Sensor resources needed to run a sweep"""

        source: SourceBase[_TS, _TC]
        sink: SinkBase
        peripherals: PeripheralsBase[_TP, _TC]
        except_context: typing_extensions.NotRequired[typing.ContextManager]
        sweep_spec: specs.Sweep[_TS, _TP, _TC]
        calibration: 'xr.Dataset|None'
        alias_func: captures.PathAliasFormatter | None

else:
    # workaround for python < 3.10
    class Resources(typing.TypedDict):
        """Sensor resources needed to run a sweep"""

        source: SourceBase
        sink: SinkBase
        peripherals: PeripheralsBase
        except_context: typing_extensions.NotRequired[typing.ContextManager]
        sweep_spec: specs.Sweep
        calibration: 'xr.Dataset|None'
        alias_func: captures.PathAliasFormatter | None

    class AnyResources(typing.TypedDict, total=False):
        """Sensor resources needed to run a sweep"""

        source: SourceBase
        sink: SinkBase
        peripherals: PeripheralsBase
        except_context: typing_extensions.NotRequired[typing.ContextManager]
        sweep_spec: specs.Sweep
        calibration: 'xr.Dataset|None'
        alias_func: captures.PathAliasFormatter | None


def import_sink_cls(
    spec: specs.Extension,
) -> type[SinkBase]:
    if spec.sink is None:
        raise TypeError('extension sink was not specified')
    mod_name, *sub_names, obj_name = spec.sink.rsplit('.')
    mod = importlib.import_module(mod_name)
    for name in sub_names:
        mod = getattr(mod, name)
    return getattr(mod, obj_name)


class Call(util.Call[util._P, util._R]):
    _dest = None

    def returns(self, d) -> typing_extensions.Self:
        self._dest = d
        return self

    def __call__(self) -> _R | None:
        name = threading.current_thread().name
        with util.stopwatch(name, 'sweep', 0.5, util.logging.INFO):
            result = super().__call__()
            if self._dest is not None:
                self._dest[name] = result
            return result


class ConnectionManager(
    contextlib.ExitStack,
    typing.Generic[_TS, _TP, _TC],
):
    _resources: AnyResources[_TS, _TP, _TC]

    def __init__(self, sweep_spec: specs.Sweep[_TS, _TP, _TC]):
        super().__init__()
        self._resources = AnyResources(sweep_spec=sweep_spec)

    def open(
        self, func: typing.Callable[_P, _R], *args: _P.args, **kws: _P.kwargs
    ) -> Call:
        def wrapper():
            result = func(*args, **kws)
            self.enter_context(obj)  # type: ignore
            return result

        return Call(wrapper).returns(self._resources)

    def get(
        self, func: typing.Callable[_P, _R], *args: _P.args, **kws: _P.kwargs
    ) -> Call:
        return Call(func, *args, **kws).returns(self._resources)

    def enter(self, ctx, name):
        self._resources[name] = self.enter_context(ctx)

    def log_call(
        self, func: typing.Callable[_P, _R], *args: _P.args, **kws: _P.kwargs
    ) -> Call:
        return Call(func, *args, **kws)

    @functools.cached_property
    def resources(self) -> Resources[_TS, _TP, _TC]:
        missing = Resources.__required_keys__ - set(self._resources.keys())
        if len(missing) == 0:
            return typing.cast(Resources[_TS, _TP, _TC], self._resources)
        else:
            raise TypeError(f'connections {missing!r} are incomplete')


def _setup_logging(sink: specs.Sink, formatter):
    log_path = formatter(sink.log_path)
    util.log_to_file(log_path, sink.log_level)


def _open_devices(conn: ConnectionManager, binding: bindings.SensorBinding, spec: specs.Sweep):
    """the source and any peripherals"""

    calls = {
        'source': Call(conn.open, binding.source, spec.source, analysis=spec.analysis),
        'peripherals': Call(conn.open, binding.peripherals, spec),
    }

    util.concurrently(calls)

    # run peripherals setup after the source is fully initialized, in case
    # it could produce spurious inputs during source initialization
    conn._resources['peripherals'].setup(spec.captures, spec.loops)  # type: ignore


@util.stopwatch("open resources", "sweep", 1.0, util.PERFORMANCE_INFO)
def open_sensor(
    spec: specs.Sweep[_TS, _TP, _TC],
    spec_path: str | Path | None = None,
    except_context: typing.ContextManager | None = None,
) -> ConnectionManager[_TS, _TP, _TC]:
    """open the sensor hardware and software contexts needed to run the given sweep.

    The returned Connections object contains the resulting context. All of its resources
    have been opened and set up as needed to run the specified sweep.
    """

    from .sweeps import prepare_compute

    formatter = captures.PathAliasFormatter(spec, spec_path=spec_path)

    if spec_path is not None:
        os.chdir(str(Path(spec_path).parent))

    bind = bindings.get_binding(spec)
    conn = ConnectionManager(sweep_spec=spec)

    if spec.extensions.sink is not None:
        sink_cls = import_sink_cls(spec.extensions)
    elif bind.sink is not None:
        sink_cls = bind.sink
    else:
        raise TypeError('no sink class in sensor binding or extensions.sink spec')

    is_cupy = spec.source.array_backend == 'cupy'

    try:
        calls = {
            'compute': conn.log_call(prepare_compute, spec),
            'sink': conn.open(sink_cls, spec, alias_func=formatter),
            'calibration': conn.get(calibration.read_calibration, spec.source.calibration, formatter),
            'devices': conn.log_call(_open_devices, conn, bind, spec)
        }

        if spec.sink.log_path is not None:
            calls['log_to_file'] = Call(_setup_logging, spec.sink, formatter)

        util.concurrently_with_fg(calls)

        if except_context is not None:
            conn.enter(except_context, 'except_context')

        conn._resources['sweep_spec'] = spec
        conn._resources['alias_func'] = formatter

    except BaseException as ex:
        if except_context is not None:
            print(ex)
            except_context.__exit__(*sys.exc_info())
        else:
            raise

        conn.__exit__(*sys.exc_info())
        raise

    return conn


def open_sensor_from_yaml(
    yaml_path: Path,
    *,
    except_context: typing.ContextManager | None = None,
    output_path: str | None = None,
    store_backend: str | None = None,
) -> ConnectionManager[typing.Any, typing.Any, typing.Any]:
    spec = io.read_yaml_spec(yaml_path)

    sink = spec.sink
    if output_path is not None:
        sink = sink.replace(path=output_path)
    if store_backend is not None:
        sink = sink.replace(store=store_backend)
    spec = spec.replace(output=sink)

    return open_sensor(spec, yaml_path, except_context)
