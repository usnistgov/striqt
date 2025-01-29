"""this is installed into the shell PATH environment as configured by pyproject.toml"""

from __future__ import annotations

import sys

from flex_spectrum_sensor_scripts import (
    click_sensor_sweep,
    init_sensor_sweep,
    edge_sensor,
    lb,
    xr,
)


def get_file_format_fields(dataset: 'xr.Dataset'):
    return {
        name: coord.values[0]
        for name, coord in dataset.coords.items()
        if edge_sensor.CAPTURE_DIM in coord.dims
    }


@click_sensor_sweep(
    'Run an acquisition and analysis sweep with a software-defined radio'
)
def run(**kws):
    # instantiate sweep objects
    store, controller, sweep_spec, calibration = init_sensor_sweep(**kws)

    # acquire and analyze each capture in the sweep
    results = [
        result
        for result in controller.iter_sweep(sweep_spec, calibration)
        if result is not None
    ]

    with lb.stopwatch('merging results', logger_level='debug'):
        dataset = xr.concat(results, edge_sensor.CAPTURE_DIM)

    with lb.stopwatch(f'write to {sweep_spec.output.path}'):
        edge_sensor.dump(store, dataset)


if __name__ == '__main__':
    run()
