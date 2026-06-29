#!/usr/bin/env python

from __future__ import annotations
import click
import typing

if typing.TYPE_CHECKING:
    import dask.array
    import striqt.figures as sf
    import xarray as xr

    class WorkerData(typing.TypedDict):
        data: xr.Dataset
        plotter: sf.backend.PlotBackend
        opts: sf.specs.PlotOptions

else:
    WorkerData = dict


worker_ctx: WorkerData | None = None


@click.command('plot signal analysis from .zarr or .zarr.zip files')
@click.argument('zarr_path', type=click.Path(exists=True, dir_okay=True))
@click.argument(
    'yaml_path',
    type=click.Path(exists=True, dir_okay=False),
    required=False,
    default=None,
)
@click.option(
    '--interactive/',
    '-i',
    type=click.Choice([None, 'sixel', 'kitty']),
    default=None,
    show_default=True,
    help='',
)
@click.option(
    '--no-save/',
    '-n',
    is_flag=True,
    show_default=True,
    default=False,
    help="don't save the resulting plots",
)
def cli(zarr_path: str, yaml_path: str, interactive=None, no_save=False):
    run(**locals())


def run(
    zarr_path: str, yaml_path: str | None, interactive: str | None = None, no_save=False
):
    import msgspec
    import multiprocessing
    from striqt import figures as sf

    if yaml_path is None:
        import striqt.analysis as sa

        attrs = sa.io.load_attrs(zarr_path)
        plot_hint = attrs.get('plot_hint', None)
        if plot_hint is None:
            raise click.ClickException(
                'this dataset has no plot settings, specify them via YAML_PATH'
            )
        opts = sf.specs.PlotOptions.from_dict(plot_hint)
    else:
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
    manager = multiprocessing.Manager()
    sf.backend.term_graphics_notice(interactive)

    executor = futures.ProcessPoolExecutor(
        max(1, ncores - 1),
        initializer=worker_init,
        initargs=(zarr_path, opts, interactive, no_save, manager.Lock()),
    )

    # then the heavier data
    dataset = load_data(zarr_path, opts, index=False)

    # run
    import itertools

    gb_fields = sf.util.get_groupby_fields(dataset, opts)
    if len(gb_fields) > 0:
        # the .set_index(...) is for compatibility with xarray < 2024.9.0
        groups = dataset.set_index(capture=gb_fields).groupby('capture')
        selects = [
            {k: v for k, v in zip(gb_fields, _listify(values))} for values, _ in groups
        ]
    else:
        selects = [{}]

    combos = list(itertools.product(opts.variables.keys(), selects))

    for _ in executor.map(worker_plot, *zip(*combos)):
        pass


def load_data(zarr_path: str, opts: 'sf.specs.PlotOptions', index=True) -> 'xr.Dataset':
    import striqt.analysis as sa
    import striqt.figures as sf
    import xarray as xr
    import dask.array

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
        dataset, 'capture', 'sweep_index', opts.data.sweep_index
    )

    if opts.data.query is not None:
        dataset = dataset.query({'capture': opts.data.query})

    if index:
        idx_coords = sf.util.guess_index_coords(dataset, opts) + [opts.plotter.col]
        dataset = dataset.set_xindex(idx_coords)

    return dataset


def worker_init(
    zarr_path,
    opts: 'sf.specs.PlotOptions',
    interactive: typing.Literal['sixel', 'kitcat'] | None,
    no_save: bool,
    lock,
):
    from pathlib import Path
    from warnings import filterwarnings

    import striqt.figures as sf
    import striqt.analysis as sa
    import striqt.sensor as ss

    sf.backend.select_mpl_backend(opts.plotter.style, interactive)

    filterwarnings('ignore', r'.*figure layout has changed.*', UserWarning)
    filterwarnings('ignore', '.*artists with labels.*', UserWarning)

    dataset = load_data(zarr_path, opts)
    dataset = dataset.sel(**opts.data.select)

    if no_save:
        output_path = None
    else:
        output_name = Path(zarr_path).name.split('.', 1)[0]
        output_path = (Path(zarr_path).parent / output_name).with_suffix('.plots')
        output_path.mkdir(exist_ok=True)

    plotter = sf.backend.PlotBackend(
        opts.plotter, output_dir=output_path, interactive=interactive, lock=lock
    )

    for name in ss.lib.compute.get_looped_coords(dataset):
        if name in (opts.plotter.col, opts.plotter.row):
            continue
        if f'{{{name}}}' in opts.plotter.filename_fmt:
            continue
        p = Path(opts.plotter.filename_fmt)
        plotter_options = opts.plotter.replace(
            filename_fmt=f'{p.stem} {name}={{{name}}}{p.suffix}'
        )
        opts_dict = dict(
            data=opts.data,
            variables=sa.specs.helpers.unfreeze(opts.variables),
            plotter=plotter_options,
        )
        opts = opts.from_dict(sa.specs.helpers.freeze(opts_dict, 10))

    global worker_ctx
    worker_ctx = WorkerData(data=dataset, plotter=plotter, opts=opts)


def _listify(values):
    if isinstance(values, (tuple, list)):
        return values
    else:
        return [values]


def worker_plot(variable: str, sel: dict[str, typing.Any]):
    import striqt.figures as sf

    ctx = worker_ctx
    if ctx is None:
        raise TimeoutError('no data to plot')

    kwargs = ctx['opts'].variables[variable]
    func = sf.data_vars._data_plots[variable]

    return func(ctx['data'].sel(**sel), ctx['plotter'], **kwargs)


if __name__ == '__main__':
    cli()  # pyright: ignore
