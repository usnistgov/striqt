#!/usr/bin/env python

from __future__ import annotations
import click
import functools


@click.command('archive a .zarr directory into a .zarr.zip file')
@click.argument('zarr_input', type=click.Path(exists=True, dir_okay=True))
@click.argument('zarr_output', type=click.Path(exists=False), required=False)
@click.option('--remove/-r', type=bool, is_flag=True, default=False)
@click.option('--force/-f', type=bool, is_flag=True, default=False)
def cli(zarr_input: str, zarr_output: str | None, remove: bool, force: bool):
    import warnings

    warnings.filterwarnings('ignore', message='.*may change without warning.*')
    zip_zarr(zarr_input, zarr_output, remove=remove, force=force)


def zip_zarr(
    zarr_input: str, zarr_output: str | None, remove: bool = False, force: bool = False
):
    from pathlib import Path
    import striqt.analysis as sa
    from striqt.sensor.lib.sinks import Zipper

    path_in = Path(zarr_input)

    if zarr_output is None:
        data = sa.load(path_in)
        if path_in.name.endswith('.zarr'):
            timestamp = generate_timestamp_suffix(data)
            if timestamp[:8] in path_in.name:
                path_out = path_in
            else:
                path_out = path_in.with_stem(f'{path_in.stem}_{timestamp}')
        else:
            path_out = path_in.with_suffix('').with_suffix('.zarr')
        path_out = str(path_out) + '.zip'
    else:
        path_out = Path(zarr_output)

    zipper = Zipper.from_zarr(path_in, path_out, force=force)
    zipper.archive(remove)

    print(f'wrote "{path_out!s}"')


def generate_timestamp_suffix(data) -> str:
    from datetime import datetime

    if 'start_time' not in data.variables or len(data.start_time) == 0:
        raise click.ClickException(
            'the data contained no timestamp to autogenerate an output file'
        )

    ts = float(data['start_time'][0])

    return datetime.fromtimestamp(ts / 1e9).strftime('%Y%m%d-%Hh%Mm%S')


if __name__ == '__main__':
    cli()  # pyright: ignore
