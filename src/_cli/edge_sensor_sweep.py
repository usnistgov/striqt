"""this is installed into the shell PATH environment as configured by pyproject.toml"""

import click
from pathlib import Path

@click.command()
@click.argument('yaml_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('output_path', type=click.Path(file_okay=False))
@click.option('--force/-f', is_flag=True, help="overwrite an existing output; otherwise, attempt to append")
def run(yaml_path: Path, output_path: Path, force:bool):
    from edge_sensor.actions import sweep
    from edge_sensor.structs import read_yaml_sweep

    run_spec, sweep_fields = read_yaml_sweep(yaml_path)

    from edge_sensor.radio import airt
    import labbench as lb
    from channel_analysis import dump
    
    lb.show_messages('info')

    with airt.AirT7201B() as sdr:
        data = sweep(sdr, run_spec, sweep_fields)

    if force:
        mode='w'
    else:
        mode='a'

    dump(output_path, data, mode)