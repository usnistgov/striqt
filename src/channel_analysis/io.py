from __future__ import annotations

import typing
from pathlib import Path
from collections import defaultdict
import warnings

from . import type_stubs, util, xarray_wrappers
from .xarray_wrappers._iq_waveform import IQSampleIndexAxis

if typing.TYPE_CHECKING:
    import numpy as np
    import numcodecs
    import xarray as xr
    import zarr
else:
    np = util.lazy_import('numpy')
    numcodecs = util.lazy_import('numcodecs')
    xr = util.lazy_import('xarray')
    zarr = util.lazy_import('zarr')


warnings.filterwarnings('ignore', category=FutureWarning, message='is deprecated and will be removed in a Zarr-Python version 3')

IQ_WAVEFORM_INDEX_NAME = typing.get_args(IQSampleIndexAxis)[0]


def open_store(path: str | Path, *, mode: str):
    if isinstance(path, zarr.storage.Store):
        store = path
    elif not isinstance(path, (str, Path)):
        raise ValueError('must pass a string or Path savefile or zarr.Store object')
    elif str(path).endswith('.zip'):
        store = zarr.ZipStore(path, mode=mode, compression=0)
    elif str(path).endswith('.db'):
        if mode == 'a':
            flag = 'c'
        elif mode == 'w':
            flag = 'n'
        else:
            flag = mode
        warnings.simplefilter('ignore')
        store = zarr.DBMStore(path, flag=flag, write_lock=False)
        warnings.resetwarnings()
    else:
        store = zarr.DirectoryStore(path, mode=mode)

    return store


def _build_encodings(data, compression=None, filter: bool = True):
    if compression is None:
        compressor = numcodecs.Blosc('zlib', clevel=6)
    elif compression is False:
        compressor = None
    else:
        compressor = compression

    if isinstance(filter, (list, tuple)):
        filters = filter
    elif filter:
        # round in dBs, tolerate max error +/- 0.005 dB
        filters = [numcodecs.Quantize(3, dtype='float32')]
    else:
        filters = None

    encodings = defaultdict(dict)

    for name in data.data_vars.keys():
        # skip compression of iq waveforms, which is slow and
        # ineffective due to high entropy
        if name != xarray_wrappers.iq_waveform.__name__:
            if compressor is not None:
                encodings[name]['compressor'] = compressor

        if data[name].attrs.get('units', '').startswith('dB'):
            if filters is not None:
                encodings[name]['filters'] = filters
            encodings[name]['dtype'] = 'float32'

    return encodings


def dump(
    store: zarr.storage.Store,
    data: typing.Optional[type_stubs.DataArrayType | type_stubs.DatasetType] = None,
    append_dim=None,
    compression=None,
    filter=True,
) -> zarr.storage.Store:
    """serialize a dataset into a zarr directory structure"""

    # if not isinstance(store, zarr.storage.Store):
    #     raise TypeError('must pass a zarr store object')

    if hasattr(data, IQ_WAVEFORM_INDEX_NAME):
        if 'sample_rate' in data.attrs:
            # sample rate is metadata
            sample_rate = data.attrs['sample_rate']
        else:
            # sample rate is a variate
            sample_rate = data.sample_rate.values.flatten()[0]

        chunks = {IQ_WAVEFORM_INDEX_NAME: round(sample_rate * 10e-3)}
    else:
        chunks = {}

    # take object dtypes to mean variable length strings for coordinates
    # and make fixed length now

    for name in dict(data.coords).keys():
        if data[name].dtype == np.dtype('object'):
            data = data.assign({name: data[name].astype('str')})

    if append_dim is None:
        append_dim = 'capture'

    # write/append only
    if len(store) > 0:
        return data.chunk(chunks).to_zarr(store, mode='a', append_dim=append_dim)
    else:
        encodings = _build_encodings(data, compression=compression, filter=filter)
        return data.chunk(chunks).to_zarr(store, encoding=encodings, mode='w')


def load(path: str | Path) -> type_stubs.DataArrayType | type_stubs.DatasetType:
    """load a dataset or data array"""

    if isinstance(path, (str, Path)):
        store = open_store(path, mode='r')

    return xr.open_dataset(store, engine='zarr')
