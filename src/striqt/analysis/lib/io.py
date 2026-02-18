from __future__ import annotations as __

import functools
import math
import threading
import typing
import warnings
from collections import defaultdict
from pathlib import Path

from yaml import SequenceNode

from .. import specs

from . import dataarrays, register, util

if typing.TYPE_CHECKING:
    import numcodecs
    import numpy as np
    import pandas as pd
    import xarray as xr
    import yaml
    import zarr
    import zarr.storage
    from typing_extensions import TypeAlias

    from striqt.waveform._typing import ArrayType

    if hasattr(zarr.storage, 'Store'):
        # zarr 2.x
        StoreType: TypeAlias = zarr.storage.Store  # type: ignore
    else:
        # zarr 3.x
        StoreType: TypeAlias = zarr.abc.store.Store  # type: ignore

else:
    numcodecs = util.lazy_import('numcodecs')
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


def _get_store_info(
    store, zarr_format: str | typing.Literal[2, 3] = 'auto'
) -> tuple[bool, dict]:
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


def _choose_chunk_sizes(
    ds: 'xr.Dataset|xr.DataArray', data_bytes=100_000_000, dim: str = 'capture'
) -> dict[str, int]:
    """pick chunk and chunk sizing for each data variable in data"""
    chunks = {dim: ds.sizes[dim]}

    if data_bytes is None:
        return chunks

    for da in ds.data_vars.values():
        if da.nbytes < data_bytes:
            continue
        else:
            reduction = math.ceil(da.nbytes / data_bytes)

        reduce_dim = str(da.dims[-1])
        if reduce_dim == dim:
            continue

        chunk_size = max(1, da.sizes[reduce_dim] // reduction)

        if reduce_dim not in chunks or chunk_size < chunks[reduce_dim]:
            chunks[reduce_dim] = chunk_size

    return chunks


def _build_encodings_zarr_v3(
    data, registry: register.AnalysisRegistry, compression=True
):
    if isinstance(compression, zarr.core.codec_pipeline.Codec):  # type: ignore
        compressors = [compression]
    elif compression:
        from zarr import codecs  # type: ignore

        shuffle = codecs.BloscShuffle.shuffle
        compressors = [codecs.BloscCodec(cname='zstd', clevel=1, shuffle=shuffle)]
    else:
        compressors = None

    encodings = defaultdict(dict)
    info_map = {info.name: info for info in registry.values()}

    for name, var in data.variables.items():
        meas_info = info_map.get(name, None)
        if meas_info is None or not meas_info.store_compressed:
            encodings[name]['compressors'] = None
        elif issubclass(var.dtype.type, np.str_):
            encodings[name]['compressors'] = None
        else:
            encodings[name]['compressors'] = compressors

    return encodings


def _build_encodings_zarr_v2(
    data, registry: register.AnalysisRegistry, compression=True
):
    if isinstance(compression, numcodecs.abc.Codec):  # type: ignore
        compressor = compression
    elif compression:
        compressor = numcodecs.Blosc('lz4', clevel=3)
    else:
        compressor = None

    encodings = defaultdict(dict)
    info_map = {info.name: info for info in registry.values()}

    for name in data.data_vars.keys():
        meas_info = info_map.get(name, None)
        if meas_info is None or not meas_info.store_compressed:
            encodings[name]['compressor'] = None
        else:
            encodings[name]['compressor'] = compressor

    return encodings


def open_store(path: str | Path, *, mode: str) -> StoreType:
    import zarr.storage

    if _zarr_version() < (3, 0, 0):
        StoreBase = zarr.storage.Store  # type: ignore
        DirectoryStore = zarr.storage.DirectoryStore  # type: ignore
    else:
        StoreBase = zarr.abc.store.Store  # type: ignore
        DirectoryStore = zarr.storage.LocalStore  # type: ignore

    if isinstance(path, StoreBase):
        store = path
    elif not isinstance(path, (str, Path)):
        raise ValueError('must pass a string or Path savefile or zarr Store')
    elif str(path).endswith('.zip'):
        if mode == 'a' and Path(path).exists():
            raise IOError('zip store does support appends')
        else:
            assert mode in ('r', 'w', 'a')
        store = zarr.storage.ZipStore(path, mode=mode, compression=0)
    else:
        store = DirectoryStore(path)

    return store


def dump(
    store: 'StoreType',
    data: 'xr.DataArray | xr.Dataset',
    *,
    append_dim: str = 'capture',
    compression: bool = True,
    zarr_format: str | typing.Literal[2, 3] = 'auto',
    compute: bool = True,
    chunk_bytes: int | typing.Literal['auto'] = 50_000_000,
    max_threads: int | None = None,
    **kwargs,
) -> 'StoreType':
    """serialize a dataset into a zarr directory or zipfile"""

    if max_threads is not None:
        numcodecs.blosc.set_nthreads(max_threads)

    # prefer the variable-length string dtype from numpy 2, if available
    string_dtype = getattr(np.dtype, 'StrDType', 'str')

    from ..measurements.registry import measurements as registry

    for name in dict(data.coords).keys():
        if data[name].size == 0:
            continue

        if isinstance(data[name].data[0], pd.Timestamp):
            # ensure nanosecond resolution
            target_dtype = 'datetime64[ns]'
        elif _xarray_version() < (2025, 7, 1) and isinstance(data[name].data[0], str):
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

        if isinstance(chunk_bytes, int):
            chunks = _choose_chunk_sizes(data, dim=append_dim, data_bytes=chunk_bytes)
            data = data.chunk(chunks)
        elif chunk_bytes == 'auto':
            data = data.chunk('auto')
        else:
            raise TypeError(f'invalid chunk_bytes argument {chunk_bytes!r}')

        if _zarr_version() >= (3, 0, 0):
            kws['encoding'] = _build_encodings_zarr_v3(
                data, registry=registry, compression=compression
            )
        else:
            kws['encoding'] = _build_encodings_zarr_v2(
                data, registry=registry, compression=compression
            )

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', xr.SerializationWarning)
        warnings.simplefilter('ignore', UserWarning)

        return data.to_zarr(store, **kws, **kwargs)


def load(
    path: str | Path,
    chunks: str | int | typing.Literal['auto'] | tuple[int, ...] | None = None,
    **kwargs,
) -> 'xr.DataArray | xr.Dataset':
    """load a dataset or data array.

    Args:
        path: location of the data store
        chunks: None to load the file without dask, or 'auto' to return a dask
            array with automatically selected chunk sizes
    """

    if isinstance(path, (str, Path)):
        store = open_store(path, mode='r')

    result = xr.open_dataset(store, chunks=chunks, engine='zarr', **kwargs)

    if chunks is None:
        return result
    else:
        return result.unify_chunks()


def _deep_update(dict1, dict2):
    """nested merge of two dictionaries"""
    for key, value in dict2.items():
        if key not in dict1:
            continue
        if isinstance(dict1[key], dict) and isinstance(value, dict):
            # If both values are dicts, merge them recursively
            _deep_update(dict1[key], value)
        else:
            # Otherwise, the value from dict2 overwrites the one from dict1
            dict1[key] = value


class _YAMLFrozenLoader(yaml.SafeLoader):
    def _construct_sequence(
        self, node: SequenceNode, deep: bool = False
    ) -> tuple[typing.Any, ...]:
        return tuple(super().construct_sequence(node, deep))


_YAMLFrozenLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_SEQUENCE_TAG,
    _YAMLFrozenLoader._construct_sequence,
)


def _expand_paths(node: yaml.Node, root_dir: Path) -> list[str]:
    def glob_path(s: str):
        p = Path(s)
        if p.is_absolute():
            rel = p.parent
            s = p.name
        else:
            rel = root_dir
        paths = list(str(g.relative_to(root_dir)) for g in rel.glob(s))
        if len(paths) == 0:
            raise FileNotFoundError(s)
        return paths

    if isinstance(node.value, str):
        values = glob_path(node.value)
    elif isinstance(node.value, (list, tuple)):
        values = []
        for n in node.value:
            values.extend(glob_path(n.value))
    else:
        raise TypeError(f'invalid tag type {type(node.value)!r} in !import')

    return sorted(values)


class _YAMLIncludeConstructor(yaml.Loader):
    _lock = threading.RLock()

    def __init__(self, path):
        self.nested_paths: list[Path] = [Path(path)]

    def __enter__(self):
        self._lock.acquire()
        yaml.add_constructor('!include', self, Loader=_YAMLFrozenLoader)  # type: ignore

    def __exit__(self, *args):
        self._lock.release()

    def get_include_path(self, s: str):
        p = Path(s)
        if p.is_absolute():
            pass
        else:
            p = self.nested_paths[-1].parent / p
        self.nested_paths.append(p)

        return p

    def pop_include_path(self):
        self.nested_paths.pop()

    def __call__(self, loader: yaml.Loader, node: yaml.Node):
        if not node.tag.startswith('!include'):
            raise ValueError(f'unknown tag {node.tag!r}')

        values = _expand_paths(node, root_dir=self.nested_paths[0].parent)

        content = []
        for v in values:
            path = self.get_include_path(v)
            with open(path, 'rb') as stream:
                content.append(yaml.load(stream, _YAMLFrozenLoader))
            self.pop_include_path()

        if len(content) == 1:
            return content[0]

        if all(isinstance(c, dict) for c in content):
            result = dict(content[0])
            for d in content[1:]:
                result.update(d)
        elif all(isinstance(c, (tuple, list)) for c in content):
            result = list(content[0])
            for l in content[1:]:
                result.extend(l)
        else:
            raise TypeError(
                'contents of multiple !include files be '
                'either all mappings or all sequences'
            )

        return result


def decode_from_yaml_file(
    path: str | Path, *, type: type[specs.SpecBase] | type[dict] = dict
):
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
        obj = yaml.load(f, _YAMLFrozenLoader)

    if issubclass(type, dict):
        return obj
    elif issubclass(type, specs.SpecBase):
        return type.from_dict(obj)
    else:
        raise TypeError(f'unsupported type {repr(type)}')


class _FileStreamProtocol(typing.Protocol):
    def close(self):
        pass

    def read(self, count: int) -> ArrayType:
        pass


class _FileStreamBase(_FileStreamProtocol):
    def __init__(
        self,
        path,
        *,
        skip_samples=0,
        dtype='complex64',
        xp=np,
        **capture_dict: typing.Any,
    ):
        self._leftover = None
        self._position = None
        self._skip_samples = skip_samples
        self.dtype = dtype
        self._xp = xp
        self._capture_dict = capture_dict
        self.seek(0)

    def seek(self, pos):
        self._leftover = None
        self._position = pos

    def get_capture_fields(self) -> dict:
        return self._capture_dict


class MATNewFileStream(_FileStreamBase):
    def __init__(
        self,
        path,
        backend_sample_rate: float,
        skip_samples=0,
        dtype='complex64',
        input_dtype='complex128',
        loop=False,
        xp=np,
        **meta,
    ):
        kws = {
            'backend_sample_rate': backend_sample_rate,
            'skip_samples': skip_samples,
            'dtype': dtype,
            'input_dtype': input_dtype,
            'xp': xp,
        }

        import h5py  # noqa # type: ignore

        self._fd = h5py.File(path, 'r')
        self._input_dtype = input_dtype
        self._loop = loop

        super().__init__(path, **(kws | meta))

    def close(self):
        self._fd.close()

    def read(self, count: int):
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
        self._capture_dict[dataarrays.PORT_DIM] = list(range(iq.shape[0]))

        if count is None:
            self._leftover = None
            return iq

        self._leftover = iq[:, count:]
        if self._position is None:
            self._position = count
        else:
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
        backend_sample_rate: float,
        key: str = 'waveform',
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
            skip_samples=skip_samples,
            dtype=dtype,
            xp=xp,
            backend_sample_rate=backend_sample_rate,
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

        iq = xp.concatenate(array_list, axis=1)
        self._capture_dict[dataarrays.PORT_DIM] = list(range(iq.shape[0]))

        if count is None:
            self._leftover = None
            return iq

        self._leftover = iq[:, count:]
        if self._position is None:
            self._position = count
        else:
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
        backend_sample_rate: float,
        skip_samples=0,
        dtype='complex64',
        xp=np,
        **meta,
    ):
        self._data = xp.asarray(np.atleast_2d(np.load(path))).astype(dtype)

        self.meta = meta

        super().__init__(
            path,
            skip_samples=skip_samples,
            dtype=dtype,
            xp=xp,
            backend_sample_rate=backend_sample_rate,
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
        self._capture_dict[dataarrays.PORT_DIM] = list(range(iq.shape[0]))

        if count is None:
            self._leftover = None
            return iq

        self._leftover = iq[:, count:]
        if self._position is None:
            self._position = count
        else:
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
        skip_samples=0,
        dtype='complex64',
        loop=False,
        xp=np,
        **meta,
    ):
        from nptdms import TdmsFile  # type: ignore

        self._fd = TdmsFile.read(path)
        self._header_fd, self._iq_fd = self._fd.groups()
        self._position = 0

        super().__init__(
            path,
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

        size = int(self._fd['header_fd']['total_samples'][0])
        ref_level = self._fd['header_fd']['reference_level_dBm'][0]

        if size < count:
            raise ValueError(
                f'requested {count} samples but file capture length is {size} samples'
            )

        scale = 10 ** (float(ref_level) / 20.0) / np.iinfo(xp.int16).max
        i, q = self._fd['iq_fd'].channels()
        iq = xp.empty((2 * count,), dtype=xp.int16)
        iq[offset * 2 :: 2] = xp.asarray(i[offset : count + offset])
        iq[1 + offset * 2 :: 2] = xp.asarray(q[offset : count + offset])

        self._position += count

        iq = (iq * float_dtype(scale)).view(self.dtype)  # type: ignore

        return iq[np.newaxis, :].copy()

    def get_capture_fields(self):
        fs = self._header_fd['IQ_samples_per_second'][0]
        fc = self._header_fd['carrier_frequency'][0]
        duration = self._header_fd['header_fd']['total_samples'][0] * fs

        return dict(
            self._capture_dict,
            backend_sample_rate=fs,
            center_frequency=fc,
            duration=duration,
        )


def open_bare_iq(
    path,
    *args,
    format='auto',
    skip_samples=0,
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
        dtype=dtype,
        xp=xp,
        **kws,
    )
