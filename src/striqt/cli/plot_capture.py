#!/usr/bin/env python

import click
import typing

if typing.TYPE_CHECKING:
    import striqt.figures as sf
    import xarray as xr

    class WorkerData(typing.TypedDict):
        data: xr.Dataset
        plotter: sf.CapturePlotter
        opts: sf.specs.PlotOptions

else:
    WorkerData = dict


worker_ctx: WorkerData | None = None


def load_data(zarr_path: str, opts: 'sf.specs.PlotOptions', index=True) -> 'xr.Dataset':
    import striqt.analysis as sa
    import xarray as xr

    dataset = sa.load(zarr_path)
    # data_var_coords = [n for n, c in dataset.coords.items() if 'capture' not in c.dims]

    if not isinstance(dataset, xr.Dataset):
        raise TypeError(f'file contents are {type(dataset).__name__!r} a dataset')

    for func_name in opts.variables.keys():
        if func_name not in dataset.data_vars:
            raise KeyError(
                f'yaml specifies a {func_name!r} plot, but no such variable is in the dataset'
            )

    if 'channel' in dataset.coords:
        # legacy name
        dataset = dataset.rename_vars({'channel': 'port'})

    dataset = _query_match_at_index(
        dataset, 'capture', 'sweep_start_time', opts.data.sweep_index
    )

    if opts.data.query is not None:
        dataset = dataset.query({'capture': opts.data.query})

    if index:
        idx_coords = _get_index_coords(dataset, opts) + [opts.plotter.col]
        dataset = dataset.set_xindex(idx_coords)

    return dataset


def worker_init(zarr_path, opts: 'sf.specs.PlotOptions', interactive: bool, no_save):
    from pathlib import Path
    import warnings

    import striqt.figures as sf
    import matplotlib as mpl
    from matplotlib import pyplot as plt

    if not interactive:
        plt.ioff()
        mpl.use('agg')
    else:
        plt.ion()
    if opts.plotter.style is not None:
        plt.style.use(opts.plotter.style)

    warning_ctx = warnings.catch_warnings()
    warning_ctx.__enter__()
    warnings.filterwarnings(
        'ignore', category=UserWarning, message=r'.*figure layout has changed.*'
    )
    warnings.filterwarnings(
        'ignore', category=UserWarning, message='.*artists with labels.*'
    )

    dataset = load_data(zarr_path, opts)

    if no_save:
        output_path = None
    else:
        output_path = Path(zarr_path).parent / Path(zarr_path).name.split('.', 1)[0]
        output_path.mkdir(exist_ok=True)
    plotter = sf.CapturePlotter(
        opts.plotter, output_dir=output_path, interactive=interactive
    )

    for name in _get_looped_coords(dataset):
        if name in (opts.plotter.col, opts.plotter.row):
            continue
        if f'{{{name}}}' in opts.plotter.filename_fmt:
            continue
        p = Path(opts.plotter.filename_fmt)
        opts.plotter.filename_fmt = f'{p.stem} {name}={{{name}}}{p.suffix}'

    global worker_ctx
    worker_ctx = WorkerData(data=dataset, plotter=plotter, opts=opts)


def worker_plot(variable: str, sel: dict[str, typing.Any]):
    import striqt.figures as sf

    ctx = worker_ctx
    if ctx is None:
        raise TimeoutError('no data to plot')

    kwargs = ctx['opts'].variables[variable]
    func = sf.plots.data_var_plotters[variable]

    return func(ctx['plotter'], ctx['data'].sel(**sel), **kwargs)


@click.command('plot signal analysis from zarr or zarr.zip files')
@click.argument('zarr_path', type=click.Path(exists=True, dir_okay=True))
@click.argument('yaml_path', type=click.Path(exists=True, dir_okay=False))
@click.option(
    '--interactive/',
    '-i',
    is_flag=True,
    show_default=True,
    default=False,
    help='',
)
@click.option(
    '--no-save',
    is_flag=True,
    show_default=True,
    default=False,
    help="don't save the resulting plots",
)
def run(zarr_path: str, yaml_path: str, interactive=False, no_save=False):
    """generic plots"""

    # yaml first, since it fails fastest
    import msgspec
    from striqt import figures as sf

    yaml_text = open(yaml_path, 'rb').read()
    opts = msgspec.yaml.decode(yaml_text, type=sf.specs.PlotOptions, strict=False)

    # spawn processes
    from concurrent import futures
    import os

    ncores = os.process_cpu_count()
    assert ncores is not None
    executor = futures.ProcessPoolExecutor(
        ncores,
        initializer=worker_init,
        initargs=(zarr_path, opts, interactive, no_save),
    )

    # then the heavier data
    dataset = load_data(zarr_path, opts, index=False)

    # run
    import itertools

    gb_fields = _get_groupby_fields(dataset, opts)
    groups = dataset.groupby(gb_fields)
    group_sel = [dict(zip(gb_fields, idx)) for idx, _ in groups]
    combos = list(itertools.product(opts.variables.keys(), group_sel))

    for _ in executor.map(worker_plot, *zip(*combos)):
        pass

    if interactive:
        import striqt.analysis as sa

        sa.util.blocking_input('press enter to quit')


if __name__ == '__main__':
    run()  # type: ignore


def _ordered_union(*args: typing.Iterable[typing.Any]):
    result = []
    for seq in args:
        result += list(dict.fromkeys(seq))
    return result


def _get_looped_coords(ds: 'xr.Dataset|xr.DataArray'):
    return [l['field'] for l in ds.attrs['loops'] if l['kind'] != 'repeat']


def _get_groupby_fields(ds: 'xr.Dataset', opts: 'sf.specs.PlotOptions'):
    from striqt import figures as sf

    loops = _get_looped_coords(ds)
    fields = sf.specs.get_format_fields(opts.plotter.filename_fmt, exclude=('name',))
    return _ordered_union(opts.data.groupby_dims, fields, loops)


def _get_index_coords(ds: 'xr.Dataset', opts: 'sf.specs.PlotOptions'):
    idx_coords = _get_groupby_fields(ds, opts) + [opts.plotter.col]
    if opts.plotter.row:
        idx_coords = idx_coords + [opts.plotter.row]
    return idx_coords


def _query_match_at_index(ds: 'xr.Dataset', dim: str, var_name: str, index: int):
    return ds.query({dim: f'{var_name} == {var_name}[{index}]'})  # .squeeze(var_name)
