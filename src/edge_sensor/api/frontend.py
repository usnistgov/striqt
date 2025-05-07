from pathlib import Path
import sys
import typing

import msgspec
import click

from . import calibration, controller, io, peripherals, sinks, structs, util

if typing.TYPE_CHECKING:
    import xarray as xr
    import labbench as lb
else:
    xr = util.lazy_import('xarray')
    lb = util.lazy_import('labbench')


def _connect_controller(remote, sweep):
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
    sweep_spec: structs.Sweep
    calibration: 'xr.Dataset'


class SweepSpecClasses(typing.NamedTuple):
    sink_cls: typing.Type[sinks.SinkBase]
    peripherals_cls: typing.Type[peripherals.PeripheralsBase]


def _get_extension_classes(sweep_spec: structs.Sweep) -> SweepSpecClasses:
    ext = msgspec.to_builtins(sweep_spec.extensions)

    import_cls = io._import_extension
    return SweepSpecClasses(
        peripherals_cls=import_cls(ext, 'peripherals'),
        sink_cls=import_cls(ext, 'sink'),
    )


def init_sweep_cli(
    *,
    yaml_path: Path,
    output_path: typing.Optional[str] = None,
    store_backend: typing.Optional[str],
    remote: typing.Optional[str],
    force: bool = False,
    verbose: bool = False,
    debug: bool = False,
) -> CLIObjects:
    # now re-read the yaml, using sweep_cls as the schema, but without knowledge of
    sweep_spec = io.read_yaml_sweep(yaml_path)

    debug_handler = DebugOnException(debug)

    if 'None' in sweep_spec.output.path:
        # in this case, we're still waiting to fill in radio_id
        open_writer_early = False
    else:
        open_writer_early = True

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
    try:
        calls = {}
        calls['controller'] = lb.Call(_connect_controller, remote, sweep_spec)
        if open_writer_early:
            yaml_classes = _get_extension_classes(sweep_spec)
            # now, open the store
            sink = yaml_classes.sink_cls(
                sweep_spec, output_path=output_path, store_backend=store_backend
            )
            calls['file store'] = lb.Call(sink.open)
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
            calibration.read_calibration_corrections,
            sweep_spec.radio_setup.calibration,
        )
        calls['peripherals'] = lb.Call(peripherals.open)
        if not open_writer_early:
            # now, open the store
            sink = yaml_classes.sink_cls(
                sweep_spec, output_path=output_path, store_backend=store_backend
            )
            calls['writer'] = lb.Call(sink.open)

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
    reuse_compatible_iq: bool = False,
    remote=None,
):
    # pull out the cli elements that have context
    *cli_context, sweep, cal = cli

    import time
    t0 = time.perf_counter()
    with lb.sequentially(*cli_context):
        print(time.perf_counter() - t0)
        # iterate through the sweep specification, yielding a dataset for each capture
        sweep_iter = cli.controller.iter_sweep(
            sweep,
            calibration=cal,
            prepare=False,
            always_yield=True,
            reuse_compatible_iq=reuse_compatible_iq,  # calibration-specific optimization
        )

        sweep_iter.set_peripherals(cli.peripherals)
        sweep_iter.set_writer(cli.sink)

        # step through captures
        for _ in sweep_iter:
            pass
