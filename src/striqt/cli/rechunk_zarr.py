#!/usr/bin/env python

from __future__ import annotations
import click
import functools


@functools.cache
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
@click.argument('zarr_output', type=click.Path(exists=False), required=False)
@click.option('--chunk-size', type=int, required=False, show_default=True, default=50)
@click.option('--compression', type=int, required=False, show_default=True, default=1, help='compression level (0-9)')
def run(zarr_input: str, zarr_output: str|None, chunk_size, compression):
    # yaml first, since it fails fastest
    import striqt.analysis as sa
    from pathlib import Path
    import warnings
    warnings.filterwarnings('ignore', message='.*may change without warning.*')

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
    print(f'rechunking input into {str(path_out)} (compression={compression}, chunk_size={chunk_size} MB)')

    if compression > 1:
        try:
            from zarr import codecs  # type: ignore
            # zarr v3
            shuffle = codecs.BloscShuffle.shuffle
            c = codecs.BloscCodec(cname='zstd', clevel=compression, shuffle=shuffle)
        except ImportError, AttributeError:
            # zarr v2
            import numcodecs
            c = numcodecs.Blosc('zstd', clevel=1)
    else:
        c = compression

    sa.dump(store, data, chunk_bytes=1_000_000 * chunk_size, compression=c)


if __name__ == '__main__':
    run()  # type: ignore
