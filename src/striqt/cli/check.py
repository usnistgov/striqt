#!/usr/bin/env python

import click


@click.command('runtime information about running a sweep')
@click.argument('yaml_path', type=click.Path(exists=True, dir_okay=False))
def run(yaml_path):
    print('Initializing...')
    # instantiate sweep objects
    from striqt.sensor.lib import frontend, calibration
    from striqt import sensor
    from pprint import pprint
    from striqt.sensor.lib.io import _get_capture_format_fields
    import labbench as lb
    from pathlib import Path
    import pandas as pd

    lb.show_messages('warning')

    init_sweep = sensor.read_yaml_sweep(yaml_path)
    print(f'Testing connect with driver {init_sweep.radio_setup.driver!r}...')
    controller = frontend.get_controller(None, init_sweep)
    radio_id = controller.radio_id(init_sweep.radio_setup.driver)
    print(f'Connected, radio_id is {radio_id!r}')
    sweep = sensor.read_yaml_sweep(yaml_path, radio_id=radio_id)

    print('\nCalibration info')
    print(60 * '=')
    if sweep.radio_setup.calibration is None:
        print('Configured for uncalibrated operation')
    elif not Path(sweep.radio_setup.calibration).exists():
        print('No file at configured path!')
    else:
        cal = calibration.read_calibration(sweep.radio_setup.calibration)
        summary = calibration.summarize_calibration(cal)
        with pd.option_context('display.max_rows', None):
            print(summary.sort_index(axis=1).sort_index(axis=0))

    print('\nPaths')
    print(60 * '=')
    expanded_paths = {
        'output.path': (sweep.output.path, init_sweep.output.path),
        'extensions.import_path': (
            sweep.extensions.import_path,
            init_sweep.extensions.import_path,
        ),
        'radio_setup.calibration': (
            sweep.radio_setup.calibration,
            init_sweep.radio_setup.calibration,
        ),
    }

    for name, (pe, pu) in expanded_paths.items():
        print(f'{name}:')
        print(f'\tRaw input: {pu!r}')
        print(f'\tEvaluated: {pe!r}')
        print('\tExists: ', 'yes' if Path(pe).exists() else 'no')


    kws = {'sweep': sweep, 'radio_id': radio_id, 'yaml_path': yaml_path}
    field_sets = {}
    for c in sweep.captures:
        fields = _get_capture_format_fields(c, **kws)
        fields = [dict(fields, channel=[i]) for i in fields['channel']]
        for f in fields:
            for k, v in f.items():
                if k in sweep.defaults.__struct_fields__ and k != 'start_time':
                    continue
                field_sets.setdefault(k, set()).add(v)

    print('\n\nAlias {field} names and expanded values:')
    print(60 * '=')
    pprint(field_sets, width=40)


if __name__ == '__main__':
    run()
