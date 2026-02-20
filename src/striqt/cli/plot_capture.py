#!/usr/bin/env python

from __future__ import annotations
import click
import typing

if typing.TYPE_CHECKING:
    import striqt.figures as sf
    import xarray as xr

    class WorkerData(typing.TypedDict):
        data: xr.Dataset
        plotter: sf.backend.PlotBackend
        opts: sf.specs.PlotOptions

else:
    WorkerData = dict


worker_ctx: WorkerData | None = None


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

    if hasattr(os, 'process_cpu_count'):
        ncores = os.process_cpu_count() or 1
    else:
        ncores = os.cpu_count() or 1

    assert ncores is not None
    executor = futures.ProcessPoolExecutor(
        max(1,ncores-1),
        initializer=worker_init,
        initargs=(zarr_path, opts, interactive, no_save),
    )

    # then the heavier data
    dataset = load_data(zarr_path, opts, index=False)

    # run
    import itertools

    gb_fields = sf.util.get_groupby_fields(dataset, opts)
    indexed = dataset.reset_index(list(dataset.indexes.keys())).set_xindex(gb_fields)#.unstack(gb_fields)
    idx_values = list(dict.fromkeys(indexed.indexes['capture']))
    group_sel = [dict(zip(gb_fields, v)) for v in idx_values]

    combos = list(itertools.product(opts.variables.keys(), group_sel))

    for _ in executor.map(worker_plot, *zip(*combos)):
        pass

    if interactive:
        import striqt.analysis as sa

        sa.util.blocking_input('press enter to quit')


def load_data(zarr_path: str, opts: 'sf.specs.PlotOptions', index=True) -> 'xr.Dataset':
    import striqt.analysis as sa
    import striqt.figures as sf
    import xarray as xr

    dataset = sa.load(zarr_path)

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

    dataset = sf.util.query_match_at_index(
        dataset, 'capture', 'sweep_start_time', opts.data.sweep_index
    )

    if opts.data.query is not None:
        dataset = dataset.query({'capture': opts.data.query})

    if index:
        idx_coords = sf.util.guess_index_coords(dataset, opts) + [opts.plotter.col]
        dataset = dataset.set_xindex(idx_coords)

    return dataset


def worker_init(zarr_path, opts: 'sf.specs.PlotOptions', interactive: bool, no_save):
    from pathlib import Path
    from warnings import filterwarnings

    import striqt.figures as sf
    import striqt.sensor as ss
    import matplotlib as mpl
    from matplotlib import pyplot as plt

    if not interactive:
        plt.ioff()
        mpl.use('agg')
    else:
        plt.ion()
    if opts.plotter.style is not None:
        plt.style.use(opts.plotter.style)

    filterwarnings('ignore', r'.*figure layout has changed.*', UserWarning)
    filterwarnings('ignore', '.*artists with labels.*', UserWarning)

    dataset = load_data(zarr_path, opts)
    dataset = dataset.sel(**opts.data.select)

    if no_save:
        output_path = None
    else:
        output_path = Path(zarr_path).parent / Path(zarr_path).name.split('.', 1)[0]
        output_path.mkdir(exist_ok=True)

    plotter = sf.backend.PlotBackend(
        opts.plotter, output_dir=output_path, interactive=interactive
    )

    for name in ss.lib.compute.get_looped_coords(dataset):
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
    func = sf.data_vars._data_plots[variable]

    return func(ctx['data'].sel(**sel), ctx['plotter'], **kwargs)


if __name__ == '__main__':
    run()  # type: ignore
