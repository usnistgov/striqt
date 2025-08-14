from __future__ import annotations

import functools
import threading
import typing
import warnings
from pathlib import Path
from collections import defaultdict

import numcodecs
import zarr.storage

from . import dataarrays, register, specs, util

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


@functools.cache
def _zarr_version() -> tuple[int, ...]:
    return tuple(int(n) for n in zarr.__version__.split('.'))


@functools.cache
def _xarray_version() -> tuple[int, ...]:
    return tuple(int(n) for n in xr.__version__.split('.'))


def _get_store_info(store, zarr_format='auto') -> tuple[bool, dict]:
    path = store.path if hasattr(store, 'path') else store.root

    if isinstance(zarr_format, str):
        zarr_format = zarr_format.lower()

    if zarr_format in (2, 3):
        pass
    elif zarr_format == 'auto':
        if _zarr_version() < (3, 0, 0):
            zarr_format = 2
        else:
            zarr_format = 3
    else:
        raise TypeError('zarr_format must be one of (2,3,"auto")')

    if zarr_format == 2:
        exists = len(store) > 0
        kws = {'zarr_version': 2}
    else:
        exists = Path(path).exists()
        kws = {'zarr_format': 3}

    return exists, kws


def _choose_chunk_and_shard(
    data, data_bytes=100_000_000, dim='capture'
) -> tuple[int, dict[str, int]]:
    """pick chunk and shard sizing for each data variable in data"""
    if data_bytes is None:
        return None, {}

    count = data.capture.size

    target_shards = {
        name: min(count, round(count * data_bytes / da.nbytes))
        for name, da in data.variables.items()
        if dim in da.dims
    }

    chunk_size = max(min(target_shards.values()), 1)

    shards = {name: chunk_size for name in target_shards}

    return chunk_size, shards


def _build_encodings_zarr_v3(
    data, shards: dict[str, int], compression=True, dim='capture'
):
    if isinstance(compression, zarr.core.codec_pipeline.Codec):
        compressors = [compression]
    elif compression:
        shuffle = zarr.codecs.BloscShuffle.shuffle
        compressors = [zarr.codecs.BloscCodec(cname='zstd', clevel=1, shuffle=shuffle)]
    else:
        compressors = None

    encodings = defaultdict(dict)
    info_map = {info.name: info for info in register.measurement.values()}

    for name, var in data.variables.items():
        meas_info = info_map.get(name, None)
        if dim in var.dims and name in shards:
            shape = list(var.shape)
            shape[var.dims.index(dim)] = shards[name]
            encodings[name]['shards'] = shape
        if meas_info is None or not meas_info.store_compressed:
            encodings[name]['compressors'] = None
        elif issubclass(var.dtype.type, np.str_):
            encodings[name]['compressors'] = None
        else:
            encodings[name]['compressors'] = compressors

    return encodings


def _build_encodings_zarr_v2(data, compression=True):
    if isinstance(compression, numcodecs.abc.Codec):
        compressor = compression
    elif compression:
        compressor = numcodecs.Blosc('lz4', clevel=3)
    else:
        compressor = None

    encodings = defaultdict(dict)
    info_map = {info.name: info for info in register.measurement.values()}

    for name in data.data_vars.keys():
        meas_info = info_map.get(name, None)
        if meas_info is None or not meas_info.store_compressed:
            encodings[name]['compressor'] = None
        else:
            encodings[name]['compressor'] = compressor

    return encodings


def open_store(
    path: str | Path, *, mode: str
) -> 'zarr.storage.Store|zarr.abc.store.Store':
    if _zarr_version() < (3, 0, 0):
        StoreBase = zarr.storage.Store
        DirectoryStore = zarr.storage.DirectoryStore
    else:
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


def dump(
    store: 'StoreType',
    data: typing.Optional['xr.DataArray' | 'xr.Dataset'] = None,
    append_dim: str = 'capture',
    compression: bool = True,
    zarr_format: str | typing.Literal[2] | typing.Literal[3] = 'auto',
    compute: bool = True,
    chunk_bytes: int = 50_000_000,
    max_threads: int | None = None,
) -> 'StoreType':
    """serialize a dataset into a zarr directory or zipfile"""

    if max_threads is not None:
        numcodecs.blosc.set_nthreads(max_threads)

    # prefer the variable-length string dtype from numpy 2, if available
    string_dtype = getattr(np.dtype, 'StrDType', 'str')

    for name in dict(data.coords).keys():
        if data[name].size == 0:
            continue

        if isinstance(data[name].values[0], pd.Timestamp):
            # ensure nanosecond resolution
            target_dtype = 'datetime64[ns]'
        elif _xarray_version() < (2025, 7, 1) and isinstance(data[name].values[0], str):
            # avoid potential truncation bug due to fixed-length strings
            target_dtype = string_dtype
        else:
            continue

        data = data.assign({name: data[name].astype(target_dtype)})

    exists, kws = _get_store_info(store, zarr_format)
    kws['compute'] = compute

    if exists:
        kws['mode'] = 'a'
        kws['append_dim'] = append_dim

    else:
        # establish the chunking and encodings for this and any subsequent writes
        kws['mode'] = 'w'

        chunk_size, shards = _choose_chunk_and_shard(
            data, dim=append_dim, data_bytes=chunk_bytes
        )

        if chunk_size is not None:
            data = data.chunk(dict(data.sizes, **{append_dim: chunk_size}))

        if _zarr_version() >= (3, 0, 0):
            kws['encoding'] = _build_encodings_zarr_v3(
                data, shards, compression=compression, dim=append_dim
            )
        else:
            kws['encoding'] = _build_encodings_zarr_v2(data, compression=compression)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', xr.SerializationWarning)
        warnings.simplefilter('ignore', UserWarning)

        return data.to_zarr(store, **kws)


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
