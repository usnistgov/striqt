from __future__ import annotations

from pathlib import Path
import typing

import labbench as lb

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


def dump(path_or_store: str | Path, data: type_stubs.DataArrayType | type_stubs.DatasetType, mode='a'):
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

    names = data.coords.keys() | data.keys() | data.indexes.keys()
    compressor = numcodecs.Blosc('zstd', clevel=6)

    for name in data.coords.keys():
        if data[name].dtype == np.dtype('object'):
            data[name] = data[name].astype('str')

    if mode == 'a':
        # follow existing encodings if appending
        encodings = None
    else:
        # skip compression of iq waveforms, which is slow and
        # ineffective due to high entropy
        encodings = {
            name: {'compressor': compressor}
            for name in names
            if name != waveform.iq_waveform.__name__
        }

    if isinstance(path_or_store, zarr.storage.Store):
        data.chunk(chunks).to_zarr(path_or_store, encoding=encodings)
    else:
        with zarr.storage.ZipStore(path_or_store, mode=mode, compression=0) as store:
            data.chunk(chunks).to_zarr(store, encoding=encodings)


def load(path: str | Path) -> type_stubs.DataArrayType|type_stubs.DatasetType:
    """load a dataset or data array"""

    return xr.open_dataset(zarr.storage.ZipStore(path, mode='r'), engine='zarr')
