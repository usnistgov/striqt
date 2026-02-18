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


def load_data(zarr_path: str, opts: 'sf.specs.PlotOptions') -> 'xr.Dataset':
    import striqt.analysis as sa
    import xarray as xr

    dataset = sa.load(zarr_path)

    if not isinstance(dataset, xr.Dataset):
        raise TypeError(f'file contents are {type(dataset).__name__!r} a dataset')

    for func_name, kwargs in opts.variables.items():
        if func_name not in dataset.data_vars:
            raise KeyError(
                f'yaml specifies a {func_name!r} plot, but no such variable is in the dataset'
            )

    if 'channel' in dataset.coords:
        # legacy coord name
        dataset = dataset.rename_vars({'channel': 'port'})

    if (
        'sweep_start_time' in dataset.coords
        and 'sweep_start_time' not in opts.data.index_dims
    ):
        index_dims = ('sweep_start_time',) + opts.data.index_dims
    else:
        index_dims = opts.data.index_dims

    dataset = dataset.set_xindex(index_dims)
    if 'sweep_start_time' in dataset.coords:
        sweep_idx = dataset.coords['sweep_start_time'][opts.data.sweep_index]
        dataset = dataset.sel(sweep_start_time=sweep_idx, drop=True)

    if opts.data.query is not None:
        dataset = dataset.query({'capture': opts.data.query})

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

    global worker_ctx
    worker_ctx = WorkerData(data=dataset, plotter=plotter, opts=opts)


def worker_plot(variable: str, start_time: typing.Any):
    import striqt.figures as sf
    import striqt.analysis as sa

    ctx = worker_ctx
    if ctx is None:
        raise TimeoutError('no data to plot')

    kwargs = ctx['opts'].variables[variable]
    data = ctx['data'].sel(start_time=start_time)
    func = sf.plots.data_var_plotters[variable]

    return func(ctx['plotter'], data, **kwargs)


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
    dataset = load_data(zarr_path, opts)

    # run
    import itertools

    if 'start_time' in dataset.coords:
        groups = dataset.groupby('start_time')
    else:
        groups = [(dataset.start_time[0], dataset)]

    combos = list(itertools.product(opts.variables.keys(), (g[0] for g in groups)))

    for _ in executor.map(
        worker_plot, *zip(*combos)
    ):  # , chunksize=len(combos)//ncores):
        pass

    if interactive:
        import striqt.analysis as sa

        sa.util.blocking_input('press enter to quit')


if __name__ == '__main__':
    run()  # type: ignore
