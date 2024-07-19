"""this is installed into the shell PATH environment as configured by pyproject.toml"""

import click
from pathlib import Path
import contextlib


def set_cuda_mem_limit(fraction=0.5):
    import cupy
    import psutil

    available = psutil.virtual_memory().available

    cupy.get_default_memory_pool().set_limit(fraction=fraction)


def warm_resampler_design_cache(radio, captures):
    """warm up the cache of resampler designs"""
    from iqwaveform import fourier

    for c in captures:
        _ = fourier.design_cola_resampler(
            fs_base=radio.MASTER_CLOCK_RATE,
            fs_target=c.sample_rate,
            bw=c.analysis_bandwidth,
            bw_lo=0.75e6,
            shift=c.lo_shift,
        )


@contextlib.contextmanager
def prepare_gpu(radio, captures, spec, swept_fields):
    """perform analysis imports and warm up the gpu evaluation graph"""
    from edge_sensor.radio import soapy
    from channel_analysis import waveform
    from edge_sensor import actions
    import labbench as lb

    with lb.stopwatch('priming gpu'):
        warm_resampler_design_cache(radio, tuple(captures))

        sizes = [c.duration * c.sample_rate for c in captures]
        c = captures[sizes.index(max(sizes))]

        iq = soapy.empty_capture(radio, c)
        coords = actions.capture_to_coords(c, tuple(swept_fields), timestamp=None)
        waveform.analyze_by_spec(iq, c, spec=spec).assign_coords(coords)

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

    set_cuda_mem_limit
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
