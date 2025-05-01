"""this is installed into the shell PATH environment as configured by pyproject.toml"""

from flex_spectrum_sensor_scripts import click_server


@click_server
def run(**kws):
    from edge_sensor.api import cli

    cli.server_cli(**kws)


if __name__ == '__main__':
    run()
