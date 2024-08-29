from __future__ import annotations

from pathlib import Path
from collections import defaultdict

import math
import typing

import labbench as lb
import zarr.storage

from . import waveform, type_stubs
import iqwaveform

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


class QuantizeTodB(zarr.abc.Codec):
    """Lossy filter to reduce the precision of floating point data.

    Parameters
    ----------
    digits : int
        Desired precision (number of decimal digits).
    dtype : dtype
        Data type to use for decoded data.
    astype : dtype, optional
        Data type to use for encoded data.

    Examples
    --------
    >>> import numcodecs
    >>> import numpy as np
    >>> x = np.linspace(0, 1, 10, dtype='f8')
    >>> x
    array([0.        , 0.11111111, 0.22222222, 0.33333333, 0.44444444,
           0.55555556, 0.66666667, 0.77777778, 0.88888889, 1.        ])
    >>> codec = numcodecs.Quantize(digits=1, dtype='f8')
    >>> codec.encode(x)
    array([0.    , 0.125 , 0.25  , 0.3125, 0.4375, 0.5625, 0.6875,
           0.75  , 0.875 , 1.    ])
    >>> codec = numcodecs.Quantize(digits=2, dtype='f8')
    >>> codec.encode(x)
    array([0.       , 0.109375 , 0.21875  , 0.3359375, 0.4453125,
           0.5546875, 0.6640625, 0.78125  , 0.890625 , 1.       ])
    >>> codec = numcodecs.Quantize(digits=3, dtype='f8')
    >>> codec.encode(x)
    array([0.        , 0.11132812, 0.22265625, 0.33300781, 0.44433594,
           0.55566406, 0.66699219, 0.77734375, 0.88867188, 1.        ])

    See Also
    --------
    numcodecs.fixedscaleoffset.FixedScaleOffset

    """

    codec_id = 'quantizetodB'

    def __init__(self, digits, dtype, astype=None):
        self.digits = digits
        self.dtype = np.dtype(dtype)
        if astype is None:
            self.astype = self.dtype
        else:
            self.astype = np.dtype(astype)
        if self.dtype.kind != 'f' or self.astype.kind != 'f':
            raise ValueError('only floating point data types are supported')

    def encode(self, buf):
        # normalise input
        arr = zarr.compat.ensure_ndarray(buf).view(self.dtype)
        arr = iqwaveform.powtodB(arr)

        # apply scaling
        precision = 10.0**-self.digits
        exp = math.log(precision, 10)
        if exp < 0:
            exp = int(math.floor(exp))
        else:
            exp = int(math.ceil(exp))
        bits = math.ceil(math.log(10.0**-exp, 2))
        scale = 2.0**bits
        enc = np.around(scale * arr) / scale

        # cast dtype
        enc = enc.astype(self.astype, copy=False)

        return enc


    def decode(self, buf, out=None):
        # filter is lossy, decoding is no-op
        dec = zarr.compat.ensure_ndarray(buf).view(self.astype)
        dec = iqwaveform.dBtopow(dec)
        dec = dec.astype(self.dtype, copy=False)
        return zarr.compat.ndarray_copy(dec, out)


    def get_config(self):
        # override to handle encoding dtypes
        return dict(
            id=self.codec_id,
            digits=self.digits,
            dtype=self.dtype.str,
            astype=self.astype.str,
        )


    def __repr__(self):
        r = '%s(digits=%s, dtype=%r' % (
            type(self).__name__,
            self.digits,
            self.dtype.str,
        )
        if self.astype != self.dtype:
            r += ', astype=%r' % self.astype.str
        r += ')'
        return r

numcodecs.registry.register_codec(QuantizeTodB)


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
        filters = [QuantizeTodB(3, dtype='float32')]
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

        if data[name].attrs.get('units', '').startswith('mW'):
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
