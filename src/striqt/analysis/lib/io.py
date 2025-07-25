from __future__ import annotations

import threading
import typing
import warnings

from pathlib import Path
from collections import defaultdict

import numcodecs
import zarr.storage

from . import dataarrays, specs, util

if typing.TYPE_CHECKING:
    import numpy as np
    import xarray as xr
    import zarr
    import pandas as pd
    import yaml

    if hasattr(zarr.storage, 'Store'):
        # zarr 2.x
        StoreType = typing.TypeVar('StoreType', bound=zarr.storage.Store)
    else:
        # zarr 3.x
        StoreType = typing.TypeVar('StoreType', bound=zarr.abc.store.Store)

else:
    np = util.lazy_import('numpy')
    pd = util.lazy_import('pandas')
    xr = util.lazy_import('xarray')
    zarr = util.lazy_import('zarr')
    yaml = util.lazy_import('yaml')

warnings.filterwarnings(
    'ignore',
    category=FutureWarning,
    message='.*is deprecated and will be removed in a Zarr-Python version 3.*',
)

warnings.filterwarnings(
    'ignore', category=UserWarning, module='.*zipfile.*', message='.*Duplicate name.*'
)


def open_store(
    path: str | Path, *, mode: str
) -> 'zarr.storage.Store|zarr.abc.store.Store':
    if hasattr(zarr.storage, 'Store'):
        # zarr 2.x
        StoreBase = zarr.storage.Store
        DirectoryStore = zarr.storage.DirectoryStore
    else:
        # zarr 3.x
        StoreBase = zarr.abc.store.Store
        DirectoryStore = zarr.storage.LocalStore

    if isinstance(path, StoreBase):
        store = path
    elif not isinstance(path, (str, Path)):
        raise ValueError('must pass a string or Path savefile or zarr Store')
    elif str(path).endswith('.zip'):
        store = zarr.storage.ZipStore(path, mode=mode, compression=0)
    else:
        store = DirectoryStore(path)

    return store


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


def _get_store_info(store: StoreType) -> tuple[bool, dict]:
    path = store.path if hasattr(store, 'path') else store.root

    if zarr.__version__.startswith('2'):
        exists = len(store) > 0
        kws = {'zarr_version': 2}
    else:
        exists = Path(path).exists()
        kws = {'zarr_format': 2}

    return exists, kws


def dump(
    store: 'StoreType',
    data: typing.Optional['xr.DataArray' | 'xr.Dataset'] = None,
    append_dim=None,
    compression=None,
    filter=True,
    overwrite=False,
) -> 'StoreType':
    """serialize a dataset into a zarr directory structure"""

    from ..measurements._iq_waveform import iq_index

    if hasattr(data, iq_index.__name__):
        if 'sample_rate' in data.attrs:
            # sample rate is metadata
            sample_rate = data.attrs['sample_rate']
        else:
            # sample rate is a variate
            sample_rate = data.sample_rate.values.flatten()[0]

        chunks = {iq_index.__name__: round(sample_rate * 100e-3)}
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
        append_dim = dataarrays.CAPTURE_DIM

    data = data.chunk(chunks)

    exists, kws = _get_store_info(store)

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


class _YAMLIncludeConstructor:
    _lock = threading.RLock()

    def __init__(self, path):
        self.nested_paths: list[Path] = [Path(path)]

    def __enter__(self):
        self._lock.acquire()
        yaml.add_constructor('!include', self, Loader=yaml.CSafeLoader)

    def __exit__(self, *args):
        self._lock.release()

    def get_include_path(self, s: str):
        s = Path(s)
        if s.is_absolute():
            path = s
        else:
            path = self.nested_paths[-1].parent / s
        self.nested_paths.append(path)

        return path

    def pop_include_path(self):
        self.nested_paths.pop()

    def __call__(self, _, node):
        path = self.get_include_path(node.value)
        with open(path, 'rb') as stream:
            content = yaml.load(stream, yaml.CSafeLoader)

        self.pop_include_path()
        return content


def decode_from_yaml_file(path: str | Path, *, type=typing.Any):
    """Deserialize an object from YAML.

    Parameters
    ----------
    buf : path
        Path to the YAML file.
    type : type, optional
        A type that is a subclass of `striqt.analysis.specs.SpecBase`
        to decode the object as. If provided, the message will be type checked
        and decoded as the specified type. Defaults to `Any`, in which case
        the message will be decoded using the default YAML types.

    Returns
    -------
    obj : Any
        The deserialized object.

    See Also
    --------
    `msgspec.yaml.decode`
    """

    with open(path) as f, _YAMLIncludeConstructor(path):
        obj = yaml.load(f, yaml.CSafeLoader)

    if type is typing.Any:
        return obj
    elif issubclass(type, specs.SpecBase):
        return type.fromdict(obj)
    else:
        raise TypeError(f'unsupported type {repr(type)}')


class _FileStreamBase:
    def __init__(
        self,
        path,
        *,
        rx_channel_count=1,
        skip_samples=0,
        dtype='complex64',
        xp=np,
        **meta,
    ):
        self._rx_channel_count = rx_channel_count
        self._leftover = None
        self._position = None
        self._skip_samples = skip_samples
        self.dtype = dtype
        self._xp = xp
        self._meta = meta
        self.seek(0)

    def seek(self, pos):
        self._leftover = None
        self._position = pos

    def get_metadata(self) -> dict:
        return self._meta


class MATNewFileStream(_FileStreamBase):
    def __init__(
        self,
        path,
        sample_rate: float,
        rx_channel_count=1,
        skip_samples=0,
        dtype='complex64',
        input_dtype='complex128',
        loop=False,
        xp=np,
        **meta,
    ):
        kws = {
            'sample_rate': sample_rate,
            'rx_channel_count': rx_channel_count,
            'skip_samples': skip_samples,
            'dtype': dtype,
            'input_dtype': input_dtype,
            'xp': xp,
        }

        import h5py  # noqa

        self._fd = h5py.File(path, 'r')
        self._input_dtype = input_dtype
        self._loop = loop

        super().__init__(path, **(kws | meta))

    def close(self):
        self._fd.close()

    def read(self, count=None):
        if count == 0:
            return

        xp = self._xp

        if self._leftover is None:
            tally = 0
            array_list = []
        else:
            tally = self._leftover.shape[1]
            array_list = [self._leftover]

        all_refs = list(self._refs)

        while tally < count:
            try:
                ref = all_refs.pop(0)
            except IndexError:
                if count is None:
                    break
                elif self._loop and len(self._refs) > 0:
                    all_refs = list(self._refs)
                    ref = all_refs.pop(0)
                else:
                    raise ValueError('too few samples in the file')

            if not hasattr(ref, 'shape') or ref.ndim != 2:
                continue

            x = xp.asarray(ref, ref.dtype).view(self._input_dtype).astype(self.dtype)
            array_list.append(x)
            tally += x.shape[1]

        iq = xp.concat(array_list, axis=1)
        self._meta[dataarrays.CHANNEL_DIM] = list(range(iq.shape[0]))

        if count is None:
            self._leftover = None
            return iq

        self._leftover = iq[:, count:]
        self._position += count
        return iq[:, :count]

    def seek(self, pos):
        if pos == self._position:
            return

        super().seek(pos)

        self._refs = list(self._fd['#refs#'].values())
        self.read(self._skip_samples + pos)


class MATLegacyFileStream(_FileStreamBase):
    def __init__(
        self,
        path,
        sample_rate: float,
        key: str = 'waveform',
        rx_channel_count=1,
        skip_samples=0,
        dtype='complex64',
        xp=np,
        loop=False,
        **meta,
    ):
        from scipy import io as sio

        available = self.list_variables(path)

        if key not in available:
            raise KeyError(
                f'key {key!r} does not point to array data. valid array keys: {available!r}'
            )

        self._fd = sio.loadmat(path, variable_names=[key], squeeze_me=True)
        self._key = key
        self._loop = loop

        super().__init__(
            path,
            rx_channel_count=rx_channel_count,
            skip_samples=skip_samples,
            dtype=dtype,
            xp=xp,
            sample_rate=sample_rate,
            **meta,
        )

    @staticmethod
    def list_variables(path: str) -> list[str]:
        from scipy import io as sio

        return [name for (name, shape, _) in sio.whosmat(path) if len(shape) > 0]

    def close(self):
        pass

    def read(self, count=None):
        if count == 0:
            return

        xp = self._xp

        if self._leftover is None:
            tally = 0
            array_list = []
        else:
            tally = self._leftover.shape[1]
            array_list = [self._leftover]

        all_refs = list(self._refs)

        while count is None or tally < count:
            try:
                ref = all_refs.pop(0)
            except IndexError:
                if count is None:
                    break
                elif self._loop and len(self._refs) > 0:
                    all_refs = list(self._refs)
                    ref = all_refs.pop(0)
                else:
                    raise ValueError('too few samples in the file')

            if not hasattr(ref, 'shape') or ref.ndim != 2:
                continue

            x = xp.asarray(ref, ref.dtype).astype(self.dtype)
            array_list.append(x)
            tally += x.shape[1]

        iq = xp.concat(array_list, axis=1)
        self._meta[dataarrays.CHANNEL_DIM] = list(range(iq.shape[0]))

        if count is None:
            self._leftover = None
            return iq

        self._leftover = iq[:, count:]
        self._position += count
        return iq[:, :count]

    def seek(self, pos):
        if pos == self._position:
            return

        super().seek(pos)

        iq = np.atleast_2d(self._fd[self._key])
        self._refs = [iq]
        self.read(self._skip_samples + pos)


class NPYFileStream(_FileStreamBase):
    def __init__(
        self,
        path,
        sample_rate: float,
        rx_channel_count=1,
        skip_samples=0,
        dtype='complex64',
        xp=np,
        **meta,
    ):
        self._data = xp.asarray(np.atleast_2d(np.load(path))).astype(dtype)

        if rx_channel_count <= self._data.shape[-2]:
            self._data = self._data[..., :rx_channel_count, :]
        else:
            raise ValueError(
                f'rx_channel_count exceeds input data channel dimension size ({self._data.shape[-2]})'
            )

        super().__init__(
            path,
            rx_channel_count=rx_channel_count,
            skip_samples=skip_samples,
            dtype=dtype,
            xp=xp,
            sample_rate=sample_rate,
            **meta,
        )

    def close(self):
        pass

    def read(self, count=None):
        if count == 0:
            return

        xp = self._xp

        if self._leftover is None:
            tally = 0
            array_list = []
        else:
            tally = self._leftover.shape[1]
            array_list = [self._leftover]

        while tally < count:
            try:
                ref = self._refs.pop(0)
            except IndexError:
                if count is not None:
                    raise ValueError('too few samples in the file')
                else:
                    break

            if not hasattr(ref, 'shape') or ref.ndim < 2:
                continue

            array_list.append(ref)
            tally += ref.shape[1]

        iq = xp.concat(array_list, axis=1)
        self._meta[dataarrays.CHANNEL_DIM] = list(range(iq.shape[0]))

        if count is None:
            self._leftover = None
            return iq

        self._leftover = iq[:, count:]
        self._position += count
        return iq[:, :count]

    def seek(self, pos):
        if pos == self._position:
            return

        super().seek(pos)

        self._refs = [self._data]
        self.read(self._skip_samples + pos)


class TDMSFileStream(_FileStreamBase):
    def __init__(
        self,
        path,
        rx_channel_count=1,
        skip_samples=0,
        dtype='complex64',
        loop=False,
        xp=np,
        **meta,
    ):
        from nptdms import TdmsFile  # noqa

        self._fd = TdmsFile.read(self.path)
        self._header_fd, self._iq_fd = self._fd.groups()

        super().__init__(
            path,
            rx_channel_count=rx_channel_count,
            skip_samples=skip_samples,
            dtype=dtype,
            xp=xp,
            **meta,
        )

    def close(self):
        self._fd.close()

    def read(self, count=None):
        xp = self._xp

        offset = self._position

        size = int(self.backend['header_fd']['total_samples'][0])
        ref_level = self.backend['header_fd']['reference_level_dBm'][0]

        if size < count:
            raise ValueError(
                f'requested {count} samples but file capture length is {size} samples'
            )

        scale = 10 ** (float(ref_level) / 20.0) / np.iinfo(xp.int16).max
        i, q = self.backend['iq_fd'].channels()
        iq = xp.empty((2 * count,), dtype=xp.int16)
        iq[offset * 2 :: 2] = xp.asarray(i[offset : count + offset])
        iq[1 + offset * 2 :: 2] = xp.asarray(q[offset : count + offset])

        float_dtype = np.finfo(np.dtype(self.dtype)).dtype

        self._position += count

        iq = (iq * float_dtype(scale)).view(self.dtype).copy()

        return iq[np.newaxis, :]

    def get_metadata(self):
        fs = self._header_fd['IQ_samples_per_second'][0]
        fc = self._header_fd['carrier_frequency'][0]
        duration = self._header_fd['header_fd']['total_samples'][0] * fs

        return dict(self.meta, sample_rate=fs, center_frequency=fc, duration=duration)


def open_bare_iq(
    path,
    *args,
    format='auto',
    skip_samples=0,
    rx_channel_count=1,
    dtype='complex64',
    loop: bool = False,
    xp=np,
    **kws,
) -> _FileStreamBase:
    if format in ('auto', None):
        format = Path(path).suffix

    if format == '.tdms':
        cls = TDMSFileStream
    elif format == '.mat':
        try:
            MATLegacyFileStream.list_variables(path)
        except NotImplementedError:
            cls = MATNewFileStream
        else:
            cls = MATLegacyFileStream
    elif format == '.npy':
        cls = NPYFileStream
    else:
        raise ValueError(f'unsupported file format "{format}"')

    return cls(
        path,
        *args,
        loop=loop,
        skip_samples=skip_samples,
        rx_channel_count=rx_channel_count,
        dtype=dtype,
        xp=xp,
        **kws,
    )
