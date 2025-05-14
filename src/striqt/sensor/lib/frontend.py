from pathlib import Path
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
        if self.enable:
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

    if verbose:
        lb.util.force_full_traceback(True)
        lb.show_messages('debug')
    else:
        lb.show_messages('info')

    # start by connecting to the controller, so that the radio id can be used
    # as a file naming field
    peripherals = None
    sink = None
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
        with lb.stopwatch(f'open {", ".join(calls)}', logger_level='info', threshold=1):
            controller = util.concurrently_with_fg(calls, False)['controller']

        yaml_classes = _get_extension_classes(sweep_spec)
        radio_id = controller.radio_id(sweep_spec.radio_setup.driver)
        sweep_spec = io.read_yaml_sweep(
            yaml_path,
            radio_id=radio_id,
        )

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

        with lb.stopwatch(
            f'load {", ".join(calls)}', logger_level='info', threshold=0.25
        ):
            opened = lb.concurrently(**calls)

    except BaseException:
        debug_handler.run(*sys.exc_info())
        if not debug_handler.enable:
            raise

    return CLIObjects(
        sink=sink,
        controller=controller,
        sweep_spec=sweep_spec,
        peripherals=peripherals,
        debugger=debug_handler,
        calibration=opened.get('calibration', None),
    )


def execute_sweep_cli(
    cli: CLIObjects,
    *,
    remote=None,
):
    # pull out the cli elements that have context
    *cli_context, sweep, cal = cli

    with lb.sequentially(*cli_context):
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
            pass

        cli.sink.flush()
