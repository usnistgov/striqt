#!/usr/bin/env python

import click

from striqt.sensor.lib import calibration, resources


@click.command('runtime information about running a sweep')
@click.argument('yaml_path', type=click.Path(exists=True, dir_okay=False))
def cli(yaml_path):
    # instantiate sweep objects
    import striqt.analysis as sa

    sa.util.show_messages(sa.util.logging.WARNING)
    run(yaml_path)


def run(yaml_path: str):
    # instantiate sweep objects
    import striqt.sensor as ss
    from pprint import pformat
    from pathlib import Path
    import pandas as pd

    spec = ss.read_yaml_spec(yaml_path)
    print(f'Opened a bound specification for {type(spec).__name__!r} bindings')

    print(f'Opening sensor resources...')
    import sys

    spec = spec.replace(
        extensions=spec.extensions.replace(sink='striqt.sensor.sinks.NoSink')
    )

    manager = ss.open_resources(spec, yaml_path, test_only=True)

    with manager as res:
        assert isinstance(spec.sensor, ss.lib.bindings.SensorBinding)
        source_id = res['source'].backend.get_id()

        print(f'source_id: {source_id!r}')

        print('\nCalibration info')
        print(80 * '▀')
        if res['calibration'] is None:
            print('Configured for uncalibrated operation')
        else:
            summary = calibration.summarize_calibration(res['calibration'])
            with pd.option_context('display.max_rows', None):
                print(summary.sort_index(axis=1).sort_index(axis=0))

        print('\nPaths')
        print(80 * '▀')
        format_path = res['format_path']
        expanded_paths = {
            'sink.path': spec.sink.path,
            'extensions.import_path': spec.extensions.import_path,
        }
        if isinstance(spec.source, ss.specs.SoapySource):
            expanded_paths['source.calibration'] = spec.source.calibration
        for name, p in expanded_paths.items():
            print(f'{name}:')
            print(f'  Input: ', repr(p))
            if p is None:
                continue
            if format_path is not None:
                pf = format_path(p)
                print(f'  Formatted: ', repr(pf))
            else:
                pf = p
            print('  Exists: ', 'yes' if Path(pf).exists() else 'no')

        print('\n\nFormat fields available for use in paths:')
        print(80 * '▀')
        afields = ss.specs.helpers.get_path_fields(
            spec, source_id=source_id, spec_path=yaml_path
        )
        afields = {f'{{{k}}}': v for k, v in afields.items()}
        afields_repr = pformat(afields, indent=2, sort_dicts=False)
        print(f' {afields_repr[1:-1]}')

        print('\n\nRoundoff error budget (worst case over captures):')
        print(80 * '▀')
        budget = ss.lib.compute.worst_case_tolerances(
            ss.lib.compute.sweep_tolerances(spec, source_id=source_id)
        )
        table = pd.DataFrame({name: tol.to_dict() for name, tol in budget.items()}).T
        print(table.to_string())

        print('\n\nUnique capture field coordinates in output:')
        labels = ss.specs.helpers.list_capture_adjustments(spec, source_id)

        if len(labels) == 0:
            return

        labels_repr = pformat(labels, indent=2, sort_dicts=False)
        print(80 * '▀')
        print(f' {labels_repr[1:-1]}')


if __name__ == '__main__':
    cli()
