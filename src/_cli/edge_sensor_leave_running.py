"""this is installed into the shell PATH environment as configured by pyproject.toml"""

import click
from pathlib import Path


@click.command(
    'Continuously run radio spectrum sensor acquisition sweeps without saving'
)
@click.argument('yaml_path', type=click.Path(exists=True, dir_okay=False))
def run(yaml_path: Path):
    print('Connecting...')

    # defer imports to here to make the command line --help snappier
    from edge_sensor.actions import sweep
    from edge_sensor.structs import read_yaml_sweep

    run_spec, sweep_fields = read_yaml_sweep(yaml_path)

    from edge_sensor.radio import airt
    import labbench as lb

    lb.show_messages('warning')

    sdr = airt.AirT7201B()

    with sdr:
        print('Running; press Ctrl+C to stop')
        while True:
            sweep(sdr, run_spec, sweep_fields)
