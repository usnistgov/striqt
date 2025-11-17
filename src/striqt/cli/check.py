#!/usr/bin/env python

import click


@click.command('runtime information about running a sweep')
@click.argument('yaml_path', type=click.Path(exists=True, dir_okay=False))
def run(yaml_path):
    print('Initializing...')
    # instantiate sweep objects
    from striqt.sensor.lib import captures, frontend, calibration, resources, util
    from striqt import sensor
    from pprint import pprint
    from striqt.sensor.lib.io import _get_capture_format_fields
    import labbench as lb
    from pathlib import Path
    import pandas as pd
    import itertools

    util.show_messages(util.logging.WARNING)

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
        if pe is None:
            continue
        print('\tExists: ', 'yes' if Path(pe).exists() else 'no')

    kws = {'sweep': sweep, 'radio_id': radio_id, 'yaml_path': yaml_path}
    field_sets = {}
    splits = (captures.split_capture_ports(c) for c in sweep.captures)
    for c in itertools.chain(*splits):
        fields = _get_capture_format_fields(c, **kws)
        for k, v in fields.items():
            if k in sweep.defaults.__struct_fields__ and k != 'start_time':
                continue
            field_sets.setdefault(k, set()).add(v)

    print('\n\nUnique alias field coordinates in output:')
    print(60 * '=')
    cfields = set(sweep.defaults.__struct_fields__)
    afields = set(field_sets.keys()) - cfields
    pprint({k: field_sets[k] for k in afields}, width=40)

    print('\n\nUnique capture field coordinates in output:')
    print(60 * '=')
    omit = {'start_time', 'delay'}
    pprint({k: field_sets[k] for k in (cfields - omit) if k in field_sets}, width=40)


if __name__ == '__main__':
    run()
