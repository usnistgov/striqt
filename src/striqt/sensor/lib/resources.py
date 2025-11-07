"""mangle dynamic imports and context management of multiple resources"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
import sys
import typing

from . import calibration, io, peripherals, sinks, specs, sweeps, util
from ..bindings import get_binding, Warmup


if typing.TYPE_CHECKING:
    import striqt.waveform as iqwaveform
else:
    iqwaveform = util.lazy_import('striqt.waveform')


class SweepSpecClasses(typing.NamedTuple):
    sink_cls: typing.Type[sinks.SinkBase]
    peripherals_cls: typing.Type[peripherals.PeripheralsBase]


def _get_sink(spec: specs.SweepSpec[specs._TS, specs._TC]) -> sinks.SinkBase:
    import importlib

    mod_name, *sub_names, obj_name = spec.extensions.sink.rsplit('.')
    mod = importlib.import_module(mod_name)
    for name in sub_names:
        mod = getattr(mod, name)
    cls = getattr(mod, obj_name)
    return cls(spec)


def run_warmup(input_spec: specs.SweepSpec[specs._TS, specs._TC]):
    if not input_spec.source.warmup_sweep:
        return

    if input_spec.source.array_backend == 'cupy':
        iqwaveform.set_max_cupy_fft_chunk(input_spec.source.cupy_max_fft_chunk_size)

    warmup_spec = sweeps.design_warmup_sweep(input_spec)

    if len(warmup_spec.captures) == 0:
        return

    source = Warmup.source(warmup_spec.source, analysis=warmup_spec.analysis)

    with source:
        resources = sweeps.Resources(source=source, sweep_spec=warmup_spec)

        warmup_iter = sweeps.iter_sweep(
            resources, always_yield=True, calibration=None, quiet=True
        )

        for _ in warmup_iter:
            pass


class Connections(contextlib.ExitStack):
    resources: sweeps.Resources

    def __init__(self):
        self.resources = sweeps.Resources()

    def open_by_name(self, **contexts: typing.Unpack[sweeps.ConnectionResources]):
        for name, obj in contexts.items():
            self.resources[name] = obj
            self.enter_context(obj)  # type: ignore


def open_sensor_from_spec(
    input_spec: specs.SweepSpec[specs._TS, specs._TC],
    origin_path: str | Path,
    except_context: typing.ContextManager | None = None,
) -> Connections:
    """open the sensor hardware and software contexts needed to run the given sweep.

    The returned Connections object contains the resulting context. All of its resources
    have been opened and set up as needed to run the specified sweep.
    """

    timer_kws = dict(threshold=1, logger_suffix='controller', logger_level=logging.INFO)
    binding = get_binding(input_spec)
    source = binding.source(input_spec.source, analysis=input_spec.analysis)

    try:
        aliased_spec = io.fill_aliases(origin_path, input_spec, source.id)
    except (RuntimeError, ConnectionError):
        # source.id requires a connection
        aliased_spec = None

        if input_spec.sink.path is None or '{' in input_spec.sink.path:
            # still waiting to fill in radio_id
            open_sink_early = False
        else:
            # there are no aliases to fill in
            open_sink_early = True
    else:
        open_sink_early = True

    conns = Connections()

    try:
        calls = {}
        calls['radio'] = util.Call(conns.open_by_name, source=source)

        if open_sink_early:
            calls['sink'] = util.Call(conns.open_by_name, sink=_get_sink(input_spec))

        with util.stopwatch(f'open {", ".join(calls)}', threshold=1, **timer_kws):  # type: ignore
            util.concurrently_with_fg(calls, False)

        assert 'radio' in conns.resources
        if aliased_spec is None:
            aliased_spec = io.fill_aliases(
                origin_path,
                input_spec,
                source_id=source.id,
            )

        if aliased_spec.sink.log_path is not None:
            util.log_to_file(
                aliased_spec.sink.log_path,
                aliased_spec.sink.log_level,
            )

        # open the rest
        calls = {}
        if aliased_spec.source.calibration is not None:
            calls['calibration'] = util.Call(
                calibration.read_calibration,
                aliased_spec.source.calibration,
            )

        if not open_sink_early:
            calls['sink'] = util.Call(conns.open_by_name, sink=_get_sink(aliased_spec))

        peripherals = binding.peripherals(aliased_spec, source=source)
        calls['peripherals'] = util.Call(conns.open_by_name, peripherals=peripherals)

        with util.stopwatch(f'setup {", ".join(calls)}', **timer_kws):  # type: ignore
            util.concurrently(**calls)

        # run this here after any radio stream has already been established to
        # avoid side-effects during the radio setup
        if 'peripherals' in conns.resources:
            conns.resources['peripherals'].setup()

        if except_context is not None:
            conns.open_by_name(except_context=except_context)

        conns.resources['sweep_spec'] = aliased_spec

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
) -> Connections:
    unaliased_spec = io.read_yaml_sweep(yaml_path)

    output = unaliased_spec.sink
    if output_path is not None:
        output = output.replace(path=output_path)
    if store_backend is not None:
        output = output.replace(store=store_backend)
    unaliased_spec = unaliased_spec.replace(output=output)

    calls = {}
    calls['context'] = util.Call(
        open_sensor_from_spec, unaliased_spec, yaml_path, except_context
    )
    calls['warmup'] = util.Call(run_warmup, unaliased_spec)
    return util.concurrently_with_fg(calls)['context']
