from pathlib import Path
import sys
import typing

import click

from . import calibration, controller, io, peripherals, structs, util, writers

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


class CLIObjects(typing.NamedTuple):
    sweep_spec: structs.Sweep
    writer: writers.WriterBase
    controller: controller.SweepController
    calibration: 'xr.Dataset'
    peripherals: peripherals.PeripheralsBase


class SweepSpecClasses(typing.NamedTuple):
    writer_cls: typing.Type[writers.WriterBase]
    peripherals_cls: typing.Type[peripherals.PeripheralsBase]


def _get_extension_classes(sweep_spec: structs.Sweep) -> SweepSpecClasses:
    exts = sweep_spec.extensions
    import_cls = io._import_extension
    return SweepSpecClasses(
        peripherals_cls=import_cls(exts.peripherals),
        writer_cls=import_cls(exts.writer),
    )


def _apply_exception_hooks(
    controller=None,
    *,
    sweep=None,
    debug: bool = False,
    remote: typing.Optional[bool] = False,
):
    def hook(*args):
        from IPython.core import ultratb

        if debug:
            print('entering debugger')
            lb.util.force_full_traceback(True)
            debugger = ultratb.FormattedTB(
                mode='Verbose', color_scheme='Linux', call_pdb=1
            )
            debugger(*args)

        if remote:
            print('closing in exception hook')
            controller.close_radio(sweep.radio_setup)

    sys.excepthook = hook


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
    if not remote:
        _apply_exception_hooks(debug=debug)
    try:
        controller = _connect_controller(remote, sweep_spec)
        if remote:
            _apply_exception_hooks(controller, sweep_spec, debug=debug, remote=remote)

        yaml_classes = _get_extension_classes(sweep_spec)
        radio_id = controller.radio_id(sweep_spec.radio_setup.driver)
        sweep_spec = io.read_yaml_sweep(
            yaml_path,
            sweep_cls=yaml_classes.sweep_cls,
            radio_id=radio_id,
        )

        peripherals = yaml_classes.peripherals_cls(sweep_spec)

        # now, open the store
        writer = yaml_classes.writer_cls(
            sweep_spec,
            output_path=output_path,
            store_backend=store_backend,
            force=force,
        )

        calls = {}
        calls['calibration'] = lb.Call(
            calibration.read_calibration_corrections,
            sweep_spec.radio_setup.calibration,
        )
        calls['writer'] = lb.Call(writer.open)
        calls['peripherals'] = lb.Call(peripherals.open)

        with lb.stopwatch(
            'load data writer, peripherals, and calibrations', logger_level='debug'
        ):
            opened = lb.concurrently(**calls)

    except BaseException:
        import traceback

        traceback.print_exc()
        raise

    return CLIObjects(
        writer=writer,
        controller=controller,
        sweep_spec=sweep_spec,
        calibration=opened.get('calibration', None),
        peripherals=peripherals,
    )


def execute_sweep(
    cli: CLIObjects,
    *,
    reuse_compatible_iq: bool = False,
    remote=None,
):
    try:
        # iterate through the sweep specification, yielding a dataset for each capture
        sweep_iter = cli.controller.iter_sweep(
            cli.sweep_spec,
            calibration=cli.calibration,
            prepare=False,
            always_yield=True,
            reuse_compatible_iq=reuse_compatible_iq,  # calibration-specific optimization
        )

        sweep_iter.set_callbacks(
            arm_func=cli.peripherals.arm,
            acquire_func=cli.peripherals.acquire,
            intake_func=cli.writer.append,
        )

        # step through captures
        for _ in sweep_iter:
            pass

        cli.writer.flush()

    except BaseException:
        import traceback

        traceback.print_exc()
        # this is handled by hooks in sys.excepthook, which may
        # trigger the IPython debugger (if configured) and then close the radio
        raise

    else:
        if remote is not None:
            cli.controller.close_radio(cli.sweep_spec.radio_setup)


def server_cli(host: str, port: int, driver: str, verbose: bool):
    # defer imports to here to make the command line --help snappier
    import labbench as lb

    if verbose:
        lb.util.force_full_traceback(True)
        lb.show_messages('debug')
    else:
        lb.show_messages('info')

    controller.start_server(host=host, port=port, default_driver=driver)
