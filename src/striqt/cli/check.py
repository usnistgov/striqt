#!/usr/bin/env python

import click

from striqt.sensor.lib import calibration, resources
from striqt.sensor.lib.specs import helpers


@click.command('runtime information about running a sweep')
@click.argument('yaml_path', type=click.Path(exists=True, dir_okay=False))
def run(yaml_path):
    # instantiate sweep objects
    from striqt.sensor.lib import util, bindings, specs
    from striqt import sensor
    from pprint import pprint
    from striqt.sensor.specs.helpers import get_path_fields
    from pathlib import Path
    import pandas as pd
    import itertools

    util.show_messages(util.logging.WARNING)

    spec = sensor.read_yaml_spec(yaml_path)
    print(f'Opened a bound specification for bindings {type(spec).__name__!r}')
    manager = sensor.open_resources(spec, spec_path=yaml_path, skip_warmup=True)

    print(f'Opening sensor resources...')
    with manager as res:
        assert isinstance(spec.__bindings__, bindings.SensorBinding)
        schema = spec.__bindings__.schema

        print(f'source_id: {res["source"].id!r}')

        print('\nCalibration info')
        print(60 * '=')
        if res['calibration'] is None:
            print('Configured for uncalibrated operation')
        else:
            summary = calibration.summarize_calibration(res['calibration'])
            with pd.option_context('display.max_rows', None):
                print(summary.sort_index(axis=1).sort_index(axis=0))

        print('\nPaths')
        print(60 * '=')
        alias_func = res['alias_func']
        expanded_paths = {
            'sink.path': spec.sink.path,
            'extensions.import_path': spec.extensions.import_path,
        }
        if isinstance(spec.source, specs.SoapySource):
            expanded_paths['source.calibration'] = spec.source.calibration
        for name, p in expanded_paths.items():
            print(f'{name}:')
            print(f'\tRaw spec: ', repr(p))
            if p is None:
                continue
            if alias_func is not None:
                pf = alias_func(p)
                print(f'\tFormatted: ', repr(pf))
            else:
                pf = p
            print('\tExists: ', 'yes' if Path(pf).exists() else 'no')

        kws = {'sweep': spec, 'source_id': res['source'].id, 'spec_path': yaml_path}
        field_sets = {}
        splits = (helpers.split_capture_ports(c) for c in spec.captures)
        for c in itertools.chain(*splits):
            fields = get_path_fields(c, **kws)
            for k, v in fields.items():
                if k in schema.capture.__struct_fields__ and k != 'start_time':
                    continue
                field_sets.setdefault(k, set()).add(v)

        print('\n\nUnique alias field coordinates in output:')
        print(60 * '=')
        cfields = set(schema.capture.__struct_fields__)
        afields = set(field_sets.keys()) - cfields
        pprint({k: field_sets[k] for k in afields}, width=40)

        print('\n\nUnique capture field coordinates in output:')
        print(60 * '=')
        omit = {'start_time', 'delay'}
        pprint(
            {k: field_sets[k] for k in (cfields - omit) if k in field_sets}, width=40
        )


if __name__ == '__main__':
    run()
