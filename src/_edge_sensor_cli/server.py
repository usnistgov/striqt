"""this is installed into the shell PATH environment as configured by pyproject.toml"""

from . import click_server


@click_server
def run(*, host, port, driver, verbose, **kws):
    import labbench as lb
    from edge_sensor.api import controller

    if verbose:
        lb.util.force_full_traceback(True)
        lb.show_messages('debug')
    else:
        lb.show_messages('info')

    controller.start_server(host=host, port=port, default_driver=driver)


if __name__ == '__main__':
    run()
