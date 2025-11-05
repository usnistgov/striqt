from __future__ import annotations

import contextlib
import logging
from pathlib import Path
import sys
import typing

from . import calibration, io, peripherals, sinks, specs, sweeps, util
from .sources import find_radio_cls_by_name, SourceBase


if typing.TYPE_CHECKING:
    import striqt.waveform as iqwaveform
else:
    iqwaveform = util.lazy_import('striqt.waveform')


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


def _get_source(radio_setup: specs.RadioSetup) -> SourceBase:
    radio_cls = find_radio_cls_by_name(radio_setup.driver)
    return radio_cls(resource=radio_setup.resource)


def _do_warmup(sweep_spec):
    if sweep_spec.radio_setup.array_backend == 'cupy':
        striqt.waveform.set_max_cupy_fft_chunk(
            sweep_spec.radio_setup.cupy_max_fft_chunk_size
        )

    if not sweep_spec.radio_setup.warmup_sweep:
        return

    warmup_sweep = sweeps.design_warmup_sweep(sweep_spec)

    if len(warmup_sweep.captures) == 0:
        return

    source = _get_source(warmup_sweep.radio_setup)

    with source:
        resources = sweeps.Resources(radio=source, sweep_spec=warmup_sweep)

        warmup_iter = sweeps.iter_sweep(
            resources, always_yield=True, calibration=None, quiet=True
        )

        for _ in warmup_iter:
            pass


class Connections(contextlib.ExitStack):
    resources: sweeps.Resources

    def __init__(self):
        self.resources = sweeps.Resources()

    def open_by_name(self, name, obj):
        self.resources[name] = obj
        self.enter_context(obj)


def open_sensor_from_spec(
    unaliased_spec: specs.Sweep,
    origin_path: str | Path,
    except_context: typing.ContextManager | None = None,
) -> Connections:
    """open the sensor hardware and software contexts needed to run the given sweep.

    The returned Connections object contains the resulting context. All of its resources
    have been opened and set up as needed to run the specified sweep.
    """

    timer_kws = dict(logger_suffix='controller', logger_level=logging.INFO)
    ext_types = None

    if '{' in unaliased_spec.output.path:
        # in this case, we're still waiting to fill in radio_id
        open_sink_early = False
    else:
        open_sink_early = True

    conns = Connections()

    try:
        calls = {}
        calls['radio'] = util.Call(
            conns.open_by_name, 'radio', _get_source(unaliased_spec.radio_setup)
        )

        if open_sink_early:
            ext_types = _get_extension_classes(unaliased_spec)
            calls['sink'] = util.Call(
                conns.open_by_name, 'sink', ext_types.sink_cls(unaliased_spec)
            )

        with util.stopwatch('open ' + ', '.join(calls), threshold=1, **timer_kws):
            util.concurrently_with_fg(calls, False)

        aliased_spec = io.fill_aliases(
            origin_path,
            unaliased_spec,
            radio_id=conns.resources['radio'].id,
        )

        if ext_types is None:
            ext_types = _get_extension_classes(aliased_spec)

        if aliased_spec.output.log_path is not None:
            util.log_to_file(
                aliased_spec.output.log_path,
                aliased_spec.output.log_level,
            )

        # open the rest
        calls = {}
        calls['calibration'] = util.Call(
            calibration.read_calibration,
            aliased_spec.radio_setup.calibration,
        )

        if not open_sink_early:
            calls['sink'] = util.Call(
                conns.open_by_name, 'sink', ext_types.sink_cls(aliased_spec)
            )

        calls['peripherals'] = util.Call(
            conns.open_by_name,
            'peripherals',
            ext_types.peripherals_cls(aliased_spec, source=conns.resources['radio']),
        )

        calls['radio'] = util.Call(
            conns.resources['radio'].setup,
            aliased_spec.radio_setup,
            analysis=aliased_spec.analysis,
        )

        with util.stopwatch('setup ' + ', '.join(calls), threshold=1, **timer_kws):
            util.concurrently(**calls)

        # in case peripherals enable e.g. an amplifier, run this here after
        # any radio stream has already been established to avoid undesired
        # input signals during the readio setup
        conns.resources['peripherals'].setup()

        if except_context is not None:
            conns.open_by_name('exception_handler', except_context)

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
    except_context: typing.Callable | None = None,
    output_path: typing.Optional[str] = None,
    store_backend: typing.Optional[str] = None,
) -> Connections:
    unaliased_spec = io.read_yaml_sweep(yaml_path)

    output = unaliased_spec.output.replace
    if output_path is not None:
        output = output.replace(path=output_path)
    if store_backend is not None:
        output = output.replace(store=store_backend)
    unaliased_spec = unaliased_spec.replace(output=output)

    calls = {}
    calls['context'] = util.Call(
        open_sensor_from_spec, unaliased_spec, yaml_path, except_context
    )
    calls['warmup'] = _do_warmup(unaliased_spec)
    return util.concurrently_with_fg(calls)['context']
