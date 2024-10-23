from __future__ import annotations

import functools
import typing
import warnings

from pathlib import Path
from collections import defaultdict

from . import util

if typing.TYPE_CHECKING:
    import numpy as np
    import numcodecs
    import xarray as xr
    import zarr
    import pandas as pd
else:
    np = util.lazy_import('numpy')
    pd = util.lazy_import('pandas')
    numcodecs = util.lazy_import('numcodecs')
    xr = util.lazy_import('xarray')
    zarr = util.lazy_import('zarr')


warnings.filterwarnings(
    'ignore',
    category=FutureWarning,
    message='is deprecated and will be removed in a Zarr-Python version 3',
)



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
        store = zarr.DirectoryStore(path)

    return store


@functools.cache
def _get_iq_index_name():
    from . import measurements
    return typing.get_args(measurements.IQSampleIndexAxis)[0]


def _build_encodings(data, compression=None, filter: bool = True):
    from . import measurements

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
        if name != measurements.iq_waveform.__name__:
            if compressor is not None:
                encodings[name]['compressor'] = compressor

        if data[name].attrs.get('units', '').startswith('dB'):
            if filters is not None:
                encodings[name]['filters'] = filters
            encodings[name]['dtype'] = 'float32'

    return encodings


def dump(
    store: 'zarr.storage.Store',
    data: typing.Optional['xr.DataArray' | 'xr.Dataset'] = None,
    append_dim=None,
    compression=None,
    filter=True,
) -> 'zarr.storage.Store':
    """serialize a dataset into a zarr directory structure"""

    # if not isinstance(store, zarr.storage.Store):
    #     raise TypeError('must pass a zarr store object')

    if hasattr(data, _get_iq_index_name()):
        if 'sample_rate' in data.attrs:
            # sample rate is metadata
            sample_rate = data.attrs['sample_rate']
        else:
            # sample rate is a variate
            sample_rate = data.sample_rate.values.flatten()[0]

        chunks = {_get_iq_index_name(): round(sample_rate * 100e-3)}
    else:
        chunks = {}

    # take object dtypes to mean variable length strings for coordinates
    # and make fixed length now

    for name in dict(data.coords).keys():
        if data[name].size == 0:
            continue

        if isinstance(data[name].values[0], str):
            # avoid potential truncation due to fixed-length strings
            target_dtype = 'str'
        elif isinstance(data[name].values[0], pd.Timestamp):
            target_dtype = 'datetime64[ns]'
        else:
            continue

        data = data.assign({name: data[name].astype(target_dtype)})

    if append_dim is None:
        append_dim = 'capture'

    # write/append only
    if len(store) > 0:
        return data.chunk(chunks).to_zarr(store, mode='a', append_dim=append_dim)
    else:
        encodings = _build_encodings(data, compression=compression, filter=filter)
        return data.chunk(chunks).to_zarr(store, encoding=encodings, mode='w')


def load(path: str | Path) -> 'xr.DataArray' | 'xr.Dataset':
    """load a dataset or data array"""

    if isinstance(path, (str, Path)):
        store = open_store(path, mode='r')

    return xr.open_dataset(store, engine='zarr')
