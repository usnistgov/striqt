#!/usr/bin/env python

import click


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

    # then the heavier data
    import striqt.analysis as sa

    dataset = sa.load(zarr_path)

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

    # the actual runner
    from striqt import sensor as ss
    from concurrent import futures
    import xarray as xr
    import os
    from pathlib import Path

    if no_save:
        output_path = None
    else:
        output_path = Path(zarr_path).parent / Path(zarr_path).name.split('.', 1)[0]
        output_path.mkdir(exist_ok=True)

    ncores = os.process_cpu_count()
    plotter = sf.CapturePlotter(opts.plotter, output_path, interactive=interactive)
    executor = futures.ProcessPoolExecutor(ncores, initializer=plotter.setup)
    exc = ss.util.ExceptionStack('plots')

    # with executor, exc:
    pending = []
    if 'start_time' in dataset.coords:
        groups = dataset.groupby('start_time')
    else:
        groups = [(None, dataset)]

    new_index = [
        n
        for n in dataset.capture.indexes
        if n not in ('capture', 'start_time', 'sweep_start_time')
    ]

    # queue
    for _, data in groups:
        assert isinstance(data, xr.Dataset)

        if len(new_index) > 0:
            data = data.reset_index(list(data.capture.indexes.keys())).set_xindex(
                new_index
            )

        for func_name, kwargs in opts.variables.items():
            pending += [
                executor.submit(
                    sf.analysis.data_var_plotters[func_name], plotter, data, **kwargs
                )
            ]

    with exc:
        for future in futures.as_completed(pending):
            with exc.defer():
                future.result()

    if interactive:
        sa.util.blocking_input('press enter to quit')


if __name__ == '__main__':
    run()  # type: ignore
