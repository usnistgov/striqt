#!/usr/bin/env python3

import click
from . import click_sensor_sweep


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
    from edge_sensor.api import frontend

    cli_objs = frontend.init_sweep_cli(**kws)

    # apply the channel setting
    if channel is not None:
        variables = cli_objs.sweep_spec.calibration_variables.replace(
            channel=(channel,)
        )
        sweep_spec = cli_objs.sweep_spec.replace(calibration_variables=variables)
        cli_objs = frontend.CLIObjects(sweep_spec, *cli_objs[1:])
        cli_objs.peripherals.set_sweep(sweep_spec)

    frontend.execute_sweep_cli(
        cli_objs,
        remote=kws.get('remote', None),
    )


if __name__ == '__main__':
    run(standalone_mode=False)
