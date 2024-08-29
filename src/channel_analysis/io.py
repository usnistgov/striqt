from __future__ import annotations

from pathlib import Path
from collections import defaultdict

import math
import typing

import labbench as lb
import zarr.storage

from . import waveform, type_stubs

if typing.TYPE_CHECKING:
    import numpy as np
    import numcodecs
    import xarray as xr
    import zarr
else:
    np = lb.util.lazy_import('numpy')
    numcodecs = lb.util.lazy_import('numcodecs')
    xr = lb.util.lazy_import('xarray')
    zarr = lb.util.lazy_import('zarr')


def open_store(path: str|Path, *, mode: str):
    if isinstance(path, zarr.storage.Store):
        store = path
    elif not isinstance(path, (str, Path)):
        raise ValueError('must pass a string or Path savefile or zarr.Store object')
    elif str(path).endswith('.zip'):
        store = zarr.ZipStore(path, mode=mode, compression=0)
    else:
        store = zarr.DirectoryStore(path, mode=mode)
    
    return store


def dump(
    path_or_store: str | Path,
    data: typing.Optional[type_stubs.DataArrayType | type_stubs.DatasetType]=None,
    mode='a',
    compression=None,
    filter=True
) -> zarr.storage.Store:
    """serialize a dataset into a zarr directory structure"""

    if hasattr(data, waveform.IQ_WAVEFORM_INDEX_NAME):
        if 'sample_rate' in data.attrs:
            # sample rate is metadata
            sample_rate = data.attrs['sample_rate']
        else:
            # sample rate is a variate
            sample_rate = data.sample_rate.values.flatten()[0]

        chunks = {waveform.IQ_WAVEFORM_INDEX_NAME: round(sample_rate * 10e-3)}
    else:
        chunks = {}

    if compression is None:
        compressor = numcodecs.Blosc('zlib', clevel=6)
    elif compression is False:
        compressor = None
    else:
        compressor = compression

    if isinstance(filter, (list,tuple)):
        filters = filter
    elif filter:
        # round in dBs, tolerate max error +/- 0.005 dB
        filters = [numcodecs.Quantize(3, dtype='float32')]
    else:
        filters = None

    for name in dict(data.coords).keys():
        if data[name].dtype == np.dtype('object'):
            data = data.assign(name=data[name].astype('str'))

    encodings = defaultdict(dict)
    for name in data.data_vars.keys():
    # skip compression of iq waveforms, which is slow and
    # ineffective due to high entropy
        if name != waveform.iq_waveform.__name__:
            if compressor is not None:
                encodings[name]['compressor'] = compressor

        if data[name].attrs.get('units', '').startswith('dB'):
            if filters is not None:
                encodings[name]['filters'] = filters
            encodings[name]['dtype'] = 'float32'

    if isinstance(path_or_store, zarr.storage.Store):
        # write/append only
        data.chunk(chunks).to_zarr(path_or_store, encoding=encodings)
    else:
        # open, write/append, and close
        with open_store(path_or_store, mode=mode) as store:
            data.chunk(chunks).to_zarr(store, encoding=encodings)

def load(path: str | Path) -> type_stubs.DataArrayType | type_stubs.DatasetType:
    """load a dataset or data array"""

    if str(path).endswith('.zip'):
        store = zarr.storage.ZipStore(path, mode='r')
    else:
        store = zarr.storage.DirectoryStore(path, mode='r')

    return xr.open_dataset(store, engine='zarr')
