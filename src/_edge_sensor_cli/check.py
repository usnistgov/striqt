#!/usr/bin/env python

import click


@click.command('runtime information about running a sweep')
@click.argument('yaml_path', type=click.Path(exists=True, dir_okay=False))
def run(yaml_path):
    print('Initializing...')
    # instantiate sweep objects
    from edge_sensor.api import frontend
    import edge_sensor
    from pprint import pprint
    from edge_sensor.api.io import _get_default_format_fields
    import labbench as lb

    lb.show_messages('warning')

    sweep = edge_sensor.read_yaml_sweep(yaml_path)
    print(f'Testing connect with driver {sweep.radio_setup.driver!r}...')
    controller = frontend.get_controller(None, sweep)
    radio_id = controller.radio_id(sweep.radio_setup.driver)
    sweep = edge_sensor.read_yaml_sweep(yaml_path, radio_id=radio_id)
    print('Success!')

    print('\n\nExpanded paths')
    print(60 * '=')
    expanded_paths = {
        'output.path': sweep.output.path,
        'extensions.import_path': sweep.extensions.import_path,
        'radio_setup.calibration': sweep.radio_setup.calibration,
    }
    pprint(expanded_paths, width=60)

    print('\n\nAlias {field} names and expanded values:')
    print(60 * '=')
    all_fields = _get_default_format_fields(
        sweep, radio_id=radio_id, yaml_path=yaml_path
    )
    fields = {
        k: v
        for k, v in all_fields.items()
        if k == 'start_time' or k not in sweep.defaults.__struct_fields__
    }
    pprint(fields, width=40)


if __name__ == '__main__':
    run()
