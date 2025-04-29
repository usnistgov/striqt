"""this is installed into the shell PATH environment as configured by pyproject.toml"""

from __future__ import annotations

from flex_spectrum_sensor_scripts import (
    click_sensor_sweep,
    init_sweep_cli,
    edge_sensor,
)


@click_sensor_sweep(
    'Run an acquisition and analysis sweep with a software-defined radio'
)
def run(**kws):
    # instantiate sweep objects
    store: 'edge_sensor.io.AppendingDataManager'
    sweep_spec: 'edge_sensor.Sweep'

    store, controller, sweep_spec, calibration = init_sweep_cli(
        **kws,
        sweep_cls=edge_sensor.Sweep,
        store_manager_cls=edge_sensor.io.AppendingDataManager,
    )

    try:
        # may need to revisit this to restore remote operation
        # prepare = kws.get('remote', None)
        # results = [
        #     result
        #     for result in sweep_iter
        #     if result is not None
        # ]

        sweep_iter = controller.iter_sweep(
            sweep_spec,
            calibration=calibration,
            prepare=False,
            always_yield=True,
        )

        sweep_iter.set_callbacks(intake_func=store.append)

        for _ in sweep_iter:
            pass

        store.flush()

    except BaseException:
        import traceback

        traceback.print_exc()
        # this is handled by hooks in sys.excepthook, which may
        # trigger the IPython debugger (if configured) and then close the radio
        raise
    else:
        if not kws['remote']:
            controller.close_radio(sweep_spec.radio_setup)

if __name__ == '__main__':
    run()
