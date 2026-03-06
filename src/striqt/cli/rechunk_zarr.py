#!/usr/bin/env python

from __future__ import annotations
import click


@click.command('plot signal analysis from .zarr or .zarr.zip files')
@click.argument('zarr_input', type=click.Path(exists=True, dir_okay=True))
@click.argument('zarr_output', type=click.Path(exists=True, dir_okay=True), required=False)
def run(zarr_input: str, zarr_output: str|None):
    # yaml first, since it fails fastest
    import striqt.analysis as sa
    from pathlib import Path

    path_in = Path(zarr_input)
    if zarr_output is None:
        if path_in.name.endswith('.zarr'):
            raise ValueError('must pass in zarr_output for .zarr directory inputs')
        else:
            path_out = path_in.with_suffix('').with_suffix('.zarr')
    else:
        path_out = Path(zarr_output)

    data = sa.load(path_in)
    store = sa.open_store(path_out, mode='w')
    sa.dump(store, data)


if __name__ == '__main__':
    run()  # type: ignore
