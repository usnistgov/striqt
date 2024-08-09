"""this is installed into the shell PATH environment as configured by pyproject.toml"""

import click
from pathlib import Path
import contextlib


def warm_resampler_design_cache(radio, captures):
    """warm up the cache of resampler designs"""
    from iqwaveform import fourier

    for c in captures:
        _ = fourier.design_cola_resampler(
            fs_base=type(radio).backend_sample_rate.max,
            fs_target=c.sample_rate,
            bw=c.analysis_bandwidth,
            bw_lo=0.75e6,
            shift=c.lo_shift,
        )


@contextlib.contextmanager
def prepare_gpu(radio, captures, spec, swept_fields):
    """perform analysis imports and warm up the gpu evaluation graph"""

    try:
        import cupy
    except ModuleNotFoundError:
        # skip priming if a gpu is unavailable
        yield None
        return
    
    from edge_sensor.radio import util
    from edge_sensor import actions
    import labbench as lb

    analyzer = actions._RadioCaptureAnalyzer(radio, analysis_spec=spec, remove_attrs=swept_fields)

    with lb.stopwatch('priming gpu'):
        # select the capture with the largest size
        capture = util.find_largest_capture(radio, captures)
        iq = util.empty_capture(radio, capture)
        analyzer(iq, timestamp=None, capture=capture)
        # soapy.free_cuda_memory()

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
@click.option(
    '--verbose/',
    '-v',
    is_flag=True,
    show_default=True,
    default=False,
    help='print debug',
)
def run(yaml_path: Path, output_path, force, verbose):
    if output_path is None:
        output_path = Path(yaml_path).with_suffix('.zarr.zip')

    # defer imports to here to make the command line --help snappier
    from edge_sensor.actions import concat_sweeps, sweep_iterator
    from edge_sensor.structs import read_yaml_sweep

    sweep_spec, sweep_fields = read_yaml_sweep(yaml_path)

    from edge_sensor.radio import find_radio_cls_by_name
    from edge_sensor.util import set_cuda_mem_limit

    import labbench as lb
    from channel_analysis import dump

    if verbose:
        lb.show_messages('debug')
    else:
        lb.show_messages('info')

    try:
        set_cuda_mem_limit()
    except ModuleNotFoundError:
        pass

    radio_type = find_radio_cls_by_name(sweep_spec.radio_setup.driver)
    radio = radio_type()
    prep = prepare_gpu(radio, sweep_spec.captures, sweep_spec.channel_analysis, sweep_fields)

    with lb.concurrently(radio, prep):
        radio.setup(sweep_spec.radio_setup)
        sweep_it = sweep_iterator(radio, sweep_spec, sweep_fields)
        data = concat_sweeps(sweep_it, radio, sweep_spec, sweep_fields)

    if force:
        mode = 'w'
    else:
        mode = 'a'

    dump(output_path, data, mode)

    click.echo(f'wrote to {output_path}')
