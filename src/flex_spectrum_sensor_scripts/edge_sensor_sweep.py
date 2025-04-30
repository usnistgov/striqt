"""this is installed into the shell PATH environment as configured by pyproject.toml"""

from flex_spectrum_sensor_scripts import click_sensor_sweep


@click_sensor_sweep(
    'Run an acquisition and analysis sweep with a software-defined radio'
)
def run(**kws):
    # instantiate sweep objects
    from edge_sensor.api import cli
    cli_objs = cli.init_sweep_cli(**kws)

    cli.execute_sweep(
        cli_objs,
        remote=kws.get('remote', None),
    )


if __name__ == '__main__':
    run()
