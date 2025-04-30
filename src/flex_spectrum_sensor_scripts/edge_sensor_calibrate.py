#!/usr/bin/env python3

import click
from flex_spectrum_sensor_scripts import click_sensor_sweep


@click_sensor_sweep('Run a Y-factor calibration sweep with speed optimizations')
@click.option(
    '--channel/',
    default=None,
    prompt=True,
    required=False,
    prompt_required=False,
    type=click.IntRange(0, 1),
    help='limit the calibration to the hardware input port at the specified index',
)
def run(*, channel, **kws):
    from edge_sensor.api import cli

    cli_objs = cli.init_sweep_cli(**kws)

    # apply the channel setting
    if channel is not None:
        import msgspec

        variables = msgspec.structs.replace(
            cli_objs.sweep_spec.calibration_variables, channel=(channel,)
        )
        sweep_spec = msgspec.structs.replace(
            cli_objs.sweep_spec, calibration_variables=variables
        )
        cli_objs = cli.CLIObjects(sweep_spec, *cli_objs[1:])
        cli_objs.peripherals.set_sweep(sweep_spec)

    cli.execute_sweep(
        cli_objs,
        reuse_compatible_iq=True,
        remote=kws.get('remote', None),
    )


if __name__ == '__main__':
    run(standalone_mode=False)
