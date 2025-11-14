"""mangle dynamic imports and context management of multiple resources"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
import typing
from pathlib import Path

from ..bindings import get_binding, registry
from . import (
    calibration,
    captures,
    io,
    peripherals,
    sinks,
    specs,
    sweeps,
    util,
)

if typing.TYPE_CHECKING:
    import striqt.waveform as iqwaveform

    _P = typing.ParamSpec('_P')
    _R = typing.TypeVar('_R')

else:
    iqwaveform = util.lazy_import('striqt.waveform')


def _import_sink_cls(
    spec: specs.SweepSpec[specs._TS, specs._TC],
) -> type[sinks.SinkBase]:
    import importlib

    mod_name, *sub_names, obj_name = spec.extensions.sink.rsplit('.')
    mod = importlib.import_module(mod_name)
    for name in sub_names:
        mod = getattr(mod, name)
    return getattr(mod, obj_name)


def _import_extensions(
    spec: specs.ExtensionSpec, alias_func: captures.PathAliasFormatter | None = None
):
    """import an extension class from a dict representation of structs.Extensions

    Arguments:
        spec: specification structure for the extension imports
        alias_func: formatter to fill aliases in the import path
    """
    import importlib

    if spec.import_path is None:
        pass
    else:
        if alias_func is None:
            p = spec.import_path
        else:
            p = alias_func(spec.import_path)

        if p != sys.path[0]:
            assert isinstance(p, (str, Path))
            sys.path.insert(0, str(p))

    from ..bindings import get_registry

    if spec.import_name is None:
        return

    start_count = len(get_registry())
    importlib.import_module(spec.import_name)
    if len(get_registry()) - start_count == 0:
        logger = util.get_logger('controller')
        import_name = spec.import_name
        logger.warning(
            f'imported extension module {import_name!r}, but it did not bind a sensor'
        )


def run_warmup(input_spec: specs.SweepSpec[specs._TS, specs._TC]):
    if not input_spec.source.warmup_sweep:
        return

    if input_spec.source.array_backend == 'cupy':
        iqwaveform.set_max_cupy_fft_chunk(input_spec.source.cupy_max_fft_chunk_size)

    warmup_spec = sweeps.design_warmup_sweep(input_spec)

    if len(warmup_spec.captures) == 0:
        return

    source = registry.warmup.source(warmup_spec.source, analysis=warmup_spec.analysis)

    with source:
        resources = sweeps.Resources(source=source, sweep_spec=warmup_spec)

        warmup_iter = sweeps.iter_sweep(resources, always_yield=True, calibration=None)

        for _ in warmup_iter:
            pass


def _load_calibration(
    spec: specs.SweepSpec, alias_func: captures.PathAliasFormatter | None
):
    p = spec.source.calibration
    if p is None:
        return

    if alias_func is not None:
        p = alias_func(spec.source.calibration)

    calibration.read_calibration(p)


class ConnectionManager(contextlib.ExitStack):
    resources: sweeps.Resources

    def __init__(self):
        self.resources = sweeps.Resources()

    def open(
        self, name, func: typing.Callable[_P, _R], *args: _P.args, **kws: _P.kwargs
    ):
        self.resources[name] = obj = func(*args, **kws)
        self.enter_context(obj)  # type: ignore


def open_sensor_from_spec(
    spec: specs.SweepSpec[specs._TS, specs._TC],
    spec_path: str | Path | None = None,
    except_context: typing.ContextManager | None = None,
) -> ConnectionManager:
    """open the sensor hardware and software contexts needed to run the given sweep.

    The returned Connections object contains the resulting context. All of its resources
    have been opened and set up as needed to run the specified sweep.
    """

    timer_kws = dict(threshold=1, logger_suffix='controller', logger_level=logging.INFO)
    format_aliases = captures.PathAliasFormatter(spec, spec_path=spec_path)

    if spec_path is not None:
        os.chdir(str(Path(spec_path).parent))

    _import_extensions(spec.extensions, format_aliases)

    if spec.sink.log_path is not None:
        util.log_to_file(spec.sink.log_path, spec.sink.log_level)

    binding = get_binding(spec)
    sink_cls = _import_sink_cls(spec)

    conns = ConnectionManager()

    try:
        calls = {
            'source': util.Call(
                conns.open,
                'source',
                binding.source,
                spec.source,
                analysis=spec.analysis,
            ),
            'sink': util.Call(
                conns.open, 'sink', sink_cls, spec, alias_func=format_aliases
            ),
            'warmup': util.Call(run_warmup, spec),
            'load_calibration': util.Call(_load_calibration, spec, format_aliases),
            'peripherals': util.Call(
                conns.open, 'peripherals', binding.peripherals, spec
            ),
        }

        with util.stopwatch(f'open {", ".join(calls)}', threshold=1, **timer_kws):  # type: ignore
            util.concurrently_with_fg(calls, False)

        with util.stopwatch(f'setup {", ".join(calls)}', **timer_kws):  # type: ignore
            if 'peripherals' in conns.resources:
                assert 'source' in conns.resources
                conns.resources['peripherals'].setup(conns.resources['source'])

        if except_context is not None:
            conns.enter_context(except_context)

        conns.resources['sweep_spec'] = spec

    except BaseException as ex:
        if except_context is not None:
            print(ex)
            except_context.__exit__(*sys.exc_info())
        else:
            raise

        conns.__exit__(*sys.exc_info())
        raise

    return conns


def open_sensor_from_yaml(
    yaml_path: Path,
    except_context: typing.ContextManager | None = None,
    output_path: typing.Optional[str] = None,
    store_backend: typing.Optional[str] = None,
) -> ConnectionManager:
    spec = io.read_yaml_sweep(yaml_path)

    sink = spec.sink
    if output_path is not None:
        sink = sink.replace(path=output_path)
    if store_backend is not None:
        sink = sink.replace(store=store_backend)
    spec = spec.replace(output=sink)

    return open_sensor_from_spec(spec, yaml_path, except_context)
