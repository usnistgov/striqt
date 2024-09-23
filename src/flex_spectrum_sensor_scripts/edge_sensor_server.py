"""this is installed into the shell PATH environment as configured by pyproject.toml"""

from flex_spectrum_sensor_scripts import click_server, run_server

@click_server
def run(**kws):
    run_server(**kws)


if __name__ == '__main__':
    run()
