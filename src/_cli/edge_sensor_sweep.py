"""this is installed into the shell PATH environment as configured by pyproject.toml"""

import click
from pathlib import Path
import contextlib


@contextlib.contextmanager
def prepare_gpu(radio, captures, spec, swept_fields):
    """warm up the gpu by running analysis while the SDR connects"""
    from edge_sensor.radio import soapy
    from channel_analysis import waveform
    from edge_sensor import actions
    import labbench as lb
    from iqwaveform import fourier

    with lb.stopwatch('priming gpu'):
        sizes = [c.duration * c.sample_rate for c in captures]
        big_capture = captures[sizes.index(max(sizes))]

        # populate the buffer if necessary, and analyze
        iq = soapy.empty_capture(radio, big_capture)

        coords = actions.capture_to_coords(
            big_capture, tuple(swept_fields), timestamp=None
        )
        for i in range(3):
            waveform.analyze_by_spec(iq, big_capture, spec=spec).assign_coords(coords)

    yield None


@click.command(
    'Run a radio spectrum sensor acquisition sweep according to a configuration file.'
)
@click.argument('yaml_path', type=click.Path(exists=True, dir_okay=False))
@click.option(
    '--output-path',
    default=None,
    type=click.Path(),
    help='output file path; if unspecified, follows yaml file name',
)
@click.option(
    '--force/',
    '-f',
    is_flag=True,
    show_default=True,
    default=False,
    help='overwrite an existing output; otherwise, attempt append on existing data',
)
def run(yaml_path: Path, output_path, force):
    if output_path is None:
        output_path = Path(yaml_path).with_suffix('.zarr.zip')

    # defer imports to here to make the command line --help snappier
    from edge_sensor.actions import sweep
    from edge_sensor.structs import read_yaml_sweep, RadioCapture

    run_spec, sweep_fields = read_yaml_sweep(yaml_path)

    from edge_sensor.radio import airt
    import labbench as lb
    from channel_analysis import dump

    lb.show_messages('info')

    sdr = airt.AirT7201B()
    prep = prepare_gpu(sdr, run_spec.captures, run_spec.channel_analysis, sweep_fields)

    with lb.concurrently(sdr, prep):
        data = sweep(sdr, run_spec, sweep_fields)

    if force:
        mode = 'w'
    else:
        mode = 'a'

    dump(output_path, data, mode)

    click.echo(f'wrote to {output_path}')
