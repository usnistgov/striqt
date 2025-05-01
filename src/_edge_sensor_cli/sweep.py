"""this is installed into the shell PATH environment as configured by pyproject.toml"""

from . import click_sensor_sweep


@click_sensor_sweep(
    'Run an acquisition and analysis sweep with a software-defined radio'
)
def run(**kws):
    # instantiate sweep objects
    from edge_sensor.api import frontend

    cli_objs = frontend.init_sweep_cli(**kws)

    frontend.execute_sweep_cli(
        cli_objs,
        remote=kws.get('remote', None),
    )


if __name__ == '__main__':
    run()
