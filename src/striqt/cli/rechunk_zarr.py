#!/usr/bin/env python

from __future__ import annotations
import click


def try_zarrs_input():
    try:
        import zarrs
    except ImportError:
        print('not accelerating with zarrs because it could not be imported')
    else:
        import zarr
        zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})


def generate_timestamp_suffix(data) -> str:
    from datetime import datetime

    if 'start_time' not in data.variables or len(data.start_time) == 0:
        raise click.ClickException('the data contained no timestamp to autogenerate an output file')

    ts=data['start_time'][0]
    
    return datetime.fromtimestamp(ts/1e9).strftime('%Y%m%d-%Hh%Mm%S')



@click.command('plot signal analysis from .zarr or .zarr.zip files')
@click.argument('zarr_input', type=click.Path(exists=True, dir_okay=True))
@click.argument('zarr_output', type=click.Path(exists=False, dir_okay=True), required=False)
def run(zarr_input: str, zarr_output: str|None):
    # yaml first, since it fails fastest
    import striqt.analysis as sa
    from pathlib import Path

    path_in = Path(zarr_input)

    if path_in.name.endswith('.zarr'):
        try_zarrs_input()

    data = sa.load(path_in)

    if zarr_output is None:
        if path_in.name.endswith('.zarr'):
            timestamp = generate_timestamp_suffix(data)
            path_out = path_in.with_stem(f'{path_in.stem}_{timestamp}')
        else:
            path_out = path_in.with_suffix('').with_suffix('.zarr')
    else:
        path_out = Path(zarr_output)

    if path_out.exists():
        raise click.ClickException(f'file or directory already exists at output {str(path_out)!r}')

    store = sa.open_store(path_out, mode='w')
    print(f'rechunking input into {str(path_out)}')
    sa.dump(store, data)


if __name__ == '__main__':
    run()  # type: ignore
