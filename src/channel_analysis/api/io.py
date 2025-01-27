from __future__ import annotations

import functools
import typing
import warnings

from pathlib import Path
from collections import defaultdict
import numcodecs

from . import util

if typing.TYPE_CHECKING:
    import numpy as np
    import xarray as xr
    import zarr
    import pandas as pd

    if hasattr(zarr.storage, 'Store'):
        # zarr 2.x
        StoreBase = zarr.storage.Store
        LocalStore = zarr.storage.DirectoryStore
    else:
        # zarr 3.x
        StoreBase = zarr.abc.store.Store
        LocalStore = zarr.storage.LocalStore

else:
    np = util.lazy_import('numpy')
    pd = util.lazy_import('pandas')
    xr = util.lazy_import('xarray')
    zarr = util.lazy_import('zarr')

warnings.filterwarnings(
    'ignore',
    category=FutureWarning,
    message='.*is deprecated and will be removed in a Zarr-Python version 3.*',
)

warnings.filterwarnings(
    'ignore',
    category=UserWarning,
    message='.*Duplicate name.*'
)

def open_store(path: str | Path, *, mode: str):
    if hasattr(zarr.storage, 'Store'):
        # zarr 2.x
        StoreBase = zarr.storage.Store
        LocalStore = zarr.storage.DirectoryStore
    else:
        # zarr 3.x
        StoreBase = zarr.abc.store.Store
        LocalStore = zarr.storage.LocalStore

    if isinstance(path, StoreBase):
        store = path
    elif not isinstance(path, (str, Path)):
        raise ValueError('must pass a string or Path savefile or zarr Store')
    elif str(path).endswith('.zip'):
        store = zarr.storage.ZipStore(path, mode=mode, compression=0)
    else:
        store = LocalStore(path)

    return store


@functools.cache
def _get_iq_index_name():
    from .. import measurements

    return typing.get_args(measurements.IQSampleIndexAxis)[0]


def _build_encodings(data, compression=None, filter: bool = True):
    # todo: this will need to be updated to work with zarr 3

    from .. import measurements

    if compression is None:
        compressor = numcodecs.Blosc('zlib', clevel=6)
    elif compression is False:
        compressor = None

    encodings = defaultdict(dict)

    for name in data.data_vars.keys():
        # skip compression of iq waveforms, which is slow and
        # ineffective due to high entropy
        if name == measurements.iq_waveform.__name__:
            encodings[name]['compressor'] = None
        else:
            encodings[name]['compressor'] = compressor

    return encodings


def dump(
    store: 'StoreBase',
    data: typing.Optional['xr.DataArray' | 'xr.Dataset'] = None,
    append_dim=None,
    compression=None,
    filter=True,
) -> 'StoreBase':
    """serialize a dataset into a zarr directory structure"""

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

    data = data.chunk(chunks)

    # write/append only
    path = store.path if hasattr(store, 'path') else store.root

    if zarr.__version__.startswith('2'):
        exists = len(store) > 0
        kws = {}
    else:
        exists = Path(path).exists()
        kws = {'zarr_format': 2}

    if exists:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            return data.to_zarr(store, mode='a', append_dim=append_dim, **kws)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', xr.SerializationWarning)
            encodings = _build_encodings(data, compression=compression, filter=filter)
            return data.to_zarr(store, encoding=encodings, mode='w', **kws)


def load(path: str | Path) -> 'xr.DataArray' | 'xr.Dataset':
    """load a dataset or data array"""

    if isinstance(path, (str, Path)):
        store = open_store(path, mode='r')

    return xr.open_dataset(store, engine='zarr')
