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


if typing.TYPE_CHECKING:
    import xarray as xr
    import typing_extensions

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
    class Resources(typing.TypedDict):
        """Sensor resources needed to run a sweep"""

        source: SourceBase
        sink: SinkBase
        peripherals: PeripheralsBase
        except_context: typing_extensions.NotRequired[typing.ContextManager]
        sweep_spec: specs.Sweep
        calibration: 'xr.Dataset|None'
        alias_func: specs.helpers.PathAliasFormatter | None

    class AnyResources(typing.TypedDict, total=False):
        """Sensor resources needed to run a sweep"""

        source: SourceBase
        sink: SinkBase
        peripherals: PeripheralsBase
        except_context: typing_extensions.NotRequired[typing.ContextManager]
        sweep_spec: specs.Sweep
        calibration: 'xr.Dataset|None'
        alias_func: specs.helpers.PathAliasFormatter | None


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


class Call(util.Call[util._P, util._R]):
    _dest = None

    def __init__(self, func, *args, **kws):
        def wrapper(*a, **k):
            if threading.current_thread() == threading.main_thread():
                name = 'compute'
            else:
                name = threading.current_thread().name
            with util.stopwatch(name, 'sweep', 0.5, util.logging.INFO):
                result = func(*a, **k)
                if self._dest is not None:
                    self._dest[name] = result
                return result

        super().__init__(wrapper, *args, **kws)

    def return_into(self, d) -> typing_extensions.Self:
        self._dest = d
        return self


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

    def open(
        self, func: typing.Callable[_P, _R], *args: _P.args, **kws: _P.kwargs
    ) -> Call[[], _R]:
        def wrapper():
            obj = func(*args, **kws)
            self.enter_context(obj)  # type: ignore
            return obj

        return Call(wrapper).return_into(self._resources)

    def get(
        self, func: typing.Callable[_P, _R], *args: _P.args, **kws: _P.kwargs
    ) -> Call[_P, _R]:
        return Call(func, *args, **kws).return_into(self._resources)

    def enter(self, ctx, name):
        self._resources[name] = self.enter_context(ctx)

    def log_call(
        self, func: typing.Callable[_P, _R], *args: _P.args, **kws: _P.kwargs
    ) -> Call[_P, _R]:
        return Call(func, *args, **kws)

    @functools.cached_property
    def resources(self) -> Resources[_TS, _TP, _TC, _PS, _PC]:
        missing = Resources.__required_keys__ - set(self._resources.keys())
        if len(missing) == 0:
            return typing.cast(Resources[_TS, _TP, _TC, _PS, _PC], self._resources)
        else:
            raise TypeError(f'connections {missing!r} are incomplete')


def _setup_logging(sink: specs.Sink, formatter):
    util.safe_import('xarray')
    log_path = formatter(sink.log_path)
    util.log_to_file(log_path, sink.log_level)


class _SourceOpenCallback(typing.Protocol):
    def __call__(self, sweep: specs.Sweep, source_id: str) -> None:
        ...


def _open_devices(
    conn: ConnectionManager,
    binding: bindings.SensorBinding,
    spec: specs.Sweep,
    skip_peripherals: bool = False,
    on_source_opened: _SourceOpenCallback|None = None
):
    """the source and any peripherals"""

    def _post_source_open():
        source_id = get_source_id(spec.source)
        specs.helpers.list_all_labels(spec, source_id=source_id)

        if on_source_opened is not None:
            on_source_opened(spec, source_id)


    calls: dict[str, typing.Any] = {
        'source': conn.open(
            binding.source.from_spec,
            spec.source,
            captures=spec.captures,
            loops=spec.loops,
            reuse_iq=spec.options.reuse_iq,
        ),
        'source_opened': util.Call(_post_source_open)
    }

    if not skip_peripherals:
        calls['peripherals'] = conn.open(binding.peripherals, spec)

    util.concurrently(calls)

    if conn._resources.get('peripherals', None) is None:
        # an exception happened, and we're in teardown
        return

    assert 'source' in conn._resources
    if conn._resources['source'] is not None:
        specs.helpers.list_all_labels(spec, source_id=conn._resources['source'].id)

    calls = {}
    if not skip_peripherals:
        assert 'peripherals' in conn._resources
        conn._resources['peripherals'].setup(
            spec.captures,
            spec.loops
        )


@util.stopwatch('open resources', 'sweep', 1.0, util.PERFORMANCE_INFO)
def open_resources(
    spec: specs.Sweep[_TS, _TP, _TC],
    spec_path: str | Path | None = None,
    except_context: typing.ContextManager | None = None,
    *,
    test_only: bool = False,
    source_callback: typing.Callable|None = None
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

    try:
        calls = {
            # 'compute' MUST be first to run in the foreground.
            # otherwise, any cuda-dependent imports will hang.
            'compute': conn.log_call(prepare_compute, spec, skip_warmup=test_only),
            'sink': conn.open(sink_cls, spec, alias_func=formatter),
            'devices': util.Call(
                _open_devices, conn, bind, spec, skip_peripherals=test_only, source_callback=source_callback
            ),
        }

        if spec.source.calibration is not None:
            calls['calibration'] = conn.get(
                io.read_calibration, spec.source.calibration, formatter
            )
        else:
            conn._resources['calibration'] = None

        if spec.sink.log_path is not None:
            calls['log_to_file'] = Call(_setup_logging, spec.sink, formatter)

        util.concurrently_with_fg(calls)

        if except_context is None:
            conn._resources['except_context'] = None
        else:
            conn.enter(except_context, 'except_context')

        if test_only:
            conn._resources['peripherals'] = None

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
) -> ConnectionManager[typing.Any, typing.Any, typing.Any, typing.Any, typing.Any]:
    spec = io.read_yaml_spec(yaml_path)

    sink = spec.sink
    if output_path is not None:
        sink = sink.replace(path=output_path)
    if store_backend is not None:
        sink = sink.replace(store=store_backend)
    spec = spec.replace(output=sink)

    return open_resources(spec, yaml_path, except_context)
