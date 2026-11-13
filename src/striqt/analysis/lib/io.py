from __future__ import annotations as __

import contextlib
import dataclasses
import functools
import importlib
import math
import os
import re
import shutil
import tempfile
import threading
import typing
import warnings
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

from yaml import SequenceNode

from .. import specs
from . import dataarrays, register, util
from .util import np, pd, xr

if typing.TYPE_CHECKING:
    import numcodecs
    import ruamel.yaml
    import ruamel.yaml.comments
    import ruamel.yaml.constructor
    import ruamel.yaml.representer
    import ruamel.yaml.resolver
    import ruamel.yaml.scalarbool
    import yaml
    import yaml.resolver
    import zarr
    from typing_extensions import TypeIs

    from .typing import Array, ChunksSize, FileStream, ZarrFormat, ZarrStore

else:
    numcodecs = util.lazy_import('numcodecs')
    zarr = util.lazy_import('zarr')
    yaml = util.lazy_import('yaml')
    util.lazy_import('ruamel.yaml')
    import ruamel

warnings.filterwarnings(
    'ignore', '.*removed in a Zarr-Python version 3.*', FutureWarning
)


def open_store(target: str | Path, *, mode: typing.Literal['r', 'w', 'a']) -> ZarrStore:
    """return the zarr store at `target`.

    `mode` applies to a ``.zip`` path: ``'r'`` opens the archive read-only, ``'w'``
    replaces the archive, and ``'a'`` creates or extends the archive. A directory
    path is opened read-write in every mode. A store passed as `target` is returned
    unchanged. With a URL that contains ``simplecache``, the fetched data is cached
    under ``~/.cache/fsspec``, and the directory is created when absent.

    Args:
        target: a directory path, a path ending in ``.zip``, an fsspec URL, or an
            open store
        mode: the access mode of a ``.zip`` path

    Returns:
        a `ZipStore` for a ``.zip`` path, an fsspec mapper for a URL, `target` for
        a store, and otherwise a directory store (`DirectoryStore` on zarr 2,
        `LocalStore` on zarr 3)

    Raises:
        ValueError: when `target` is not a string, `Path`, or store
    """
    # the zarr 2 and zarr 3 stubs each lack the names of the other version
    storage: typing.Any = importlib.import_module('zarr.storage')
    if _zarr_version() < (3, 0, 0):
        StoreBase = storage.Store
        DirectoryStore = storage.DirectoryStore
    else:
        StoreBase = importlib.import_module('zarr.abc.store').Store
        DirectoryStore = storage.LocalStore

    if isinstance(target, StoreBase):
        store = target
    elif not isinstance(target, (str, Path)):
        raise ValueError('must pass a string or Path savefile or zarr Store')
    elif _is_fsspec_url(target):
        import fsspec

        if 'simplecache' in target:
            p = Path.home() / '.cache' / 'fsspec'
            p.mkdir(parents=True, exist_ok=True)
            kws = {'cache_storage': str(p)}
        else:
            kws = {}
        fs, _ = fsspec.url_to_fs(target, asynchronous=True, **kws)
        return fs.get_mapper('')
    elif str(target).endswith('.zip'):
        store = storage.ZipStore(target, mode=mode, compression=0)
    else:
        store = DirectoryStore(target)

    return store


def dump(
    store: ZarrStore,
    data: xr.DataArray | xr.Dataset,
    *,
    append_dim: str = 'capture',
    compression: bool = True,
    zarr_format: ZarrFormat = 'auto',
    compute: bool = True,
    chunk_bytes: ChunksSize = 50_000_000,
    max_threads: int | None = None,
    **kwargs,
) -> ZarrStore:
    """serialize a dataset into a zarr directory or zipfile"""

    if max_threads is not None:
        numcodecs.blosc.set_nthreads(max_threads)

    # prefer the variable-length string dtype from numpy 2, if available
    string_dtype = getattr(np.dtype, 'StrDType', 'str')

    from .register import registry

    for name in dict(data.coords):
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


def allocate_store(
    store: ZarrStore,
    template: xr.Dataset,
    *,
    rows: int,
    chunks: Mapping[str, tuple[int, ...]],
    encoding: Mapping[str, Mapping[str, typing.Any]] = {},
    compression: bool = True,
    registry: register.AnalysisRegistry | None = None,
) -> dict[str, typing.Any]:
    """create the arrays of `template` in `store` with `rows` along the `capture` dimension.

    No chunk of a variable with the `capture` dimension is written. The coordinates
    without that dimension are written in full. A variable already in `store` is
    replaced. A second call adds variables to the group. A returned array writes a
    chunk equal to the fill value of the array instead of deleting the chunk.

    Args:
        store: the open store of the group
        template: a dataset with ``sizes['capture'] == 0`` and the other coordinates
            populated
        rows: the length of the `capture` dimension in the allocated arrays
        chunks: the chunk shape of each variable, keyed by name, in dimension order
        encoding: xarray encoding entries per variable, applied over the compression
            and chunk entries
        compression: `True` compresses the measurements registered with
            ``store_compressed`` with Blosc zstd (clevel 1), `False` stores every
            variable uncompressed
        registry: the measurement registry that selects the compressed variables, or
            `None` for the live registry

    Returns:
        the zarr array of each variable with the `capture` dimension, keyed by name

    Raises:
        ValueError: when `template` has no `capture` dimension or a nonzero length
            along it
    """
    if template.sizes.get('capture', None) != 0:
        raise ValueError('template must have length 0 along the capture dimension')

    if registry is None:
        registry = register.registry

    if _zarr_version() >= (3, 0, 0):
        encodings = _build_encodings_zarr_v3(
            template, registry=registry, compression=compression
        )
    else:
        encodings = _build_encodings_zarr_v2(
            template, registry=registry, compression=compression
        )

    for name, shape in chunks.items():
        encodings[name]['chunks'] = tuple(shape)
    for name, entries in encoding.items():
        encodings[name].update(entries)

    kws: dict[str, typing.Any] = {
        'mode': 'a',
        'consolidated': False,
        'encoding': dict(encodings),
        **zarr_format_kws(),
    }

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', xr.SerializationWarning)
        warnings.simplefilter('ignore', UserWarning)

        template.to_zarr(store, **kws)

    arrays: dict[str, typing.Any] = {}
    with _write_empty_chunks():
        for name, var in template.variables.items():
            if 'capture' not in var.dims:
                continue
            array = zarr.open_array(store=store, path=str(name), mode='r+')
            shape = list(array.shape)
            shape[var.dims.index('capture')] = rows
            array.resize(tuple(shape))
            arrays[str(name)] = array

    return arrays


def _write_empty_chunks() -> typing.ContextManager[typing.Any]:
    """return a context in which an opened zarr array writes its all-fill chunks."""
    if _zarr_version() < (3, 0, 0):
        # zarr 2 defaults to write_empty_chunks=True; zarr 3 defaults to False
        return contextlib.nullcontext()

    config = importlib.import_module('zarr.core.config').config
    return config.set({'array.write_empty_chunks': True})


def load(path: str | Path, chunks: ChunksSize = None, **kwargs) -> xr.Dataset:
    """load and return a dataset or data array.

    Args:
        path: location of the data store, potentially an fsspec string
        chunks: None to load the file without dask, or 'auto' to return a dask
            array with automatically selected chunk sizes
        kwargs: Additional arguments passed into `xr.open_dataset`
    Returns:
        The opened dataset object.
    """

    # the xarray annotation does not admit the store objects of the zarr engine
    store: typing.Any = open_store(path, mode='r')

    result = xr.open_dataset(store, chunks=chunks, engine='zarr', **kwargs)
    result = repair_legacy_attrs(result)

    if chunks is None:
        return result
    else:
        return result.unify_chunks()


_BANDWIDTH_UNITS = re.compile(r'^(dBm|dB|√mW)/(.+)$')
_BANDWIDTH_SUFFIX = re.compile(r' \((dBm|dBW|dB|√mW)/[^)]+\)$')
_POWER_BIN_SUFFIX = '_power_bin'
_CAPTURE_BANDWIDTH_NAMES = frozenset({
    'channel_power_time_series',
    'channel_power_histogram',
    'cyclic_channel_power',
    'channel_power_bin',
    'iq_waveform',
})


def _owner_noise_bandwidth(data: xr.Dataset, coord_name: str) -> float | None:
    """return the `noise_bandwidth` attr of a data variable on the dimension of `coord_name`.

    Returns:
        the attr of the first such data variable, in Hz, or None when `coord_name`
        is a scalar coordinate or no data variable on the dimension has the attr
    """
    coord = data[coord_name]
    if coord.ndim == 0:
        return None
    dim = coord.dims[0]
    for var in data.data_vars.values():
        if dim in var.dims and 'noise_bandwidth' in var.attrs:
            return float(var.attrs['noise_bandwidth'])
    return None


def repair_legacy_attrs(data: xr.Dataset) -> xr.Dataset:
    """return `data` with the power-unit and bandwidth attrs rewritten by the rules below.

    The attrs are rewritten in place. The data values are unchanged. After the
    rewrite, `units` is the unit of the stored values, `noise_bandwidth` is the noise
    bandwidth in Hz, and `long_name` and `standard_name` are noun phrases without
    units. The rewrite applies to every variable and coordinate that matches one of
    these conditions:

    * `channel_power_time_series`, `channel_power_histogram`, `cyclic_channel_power`,
      `iq_waveform`, and the `channel_power_bin` coordinate: `noise_bandwidth` is
      removed, and `long_name` is removed when the text ends in
      ``' (<unit>/<bandwidth>)'``.
    * `long_name` or `standard_name` ending in ``' (<unit>/<bandwidth>)'``: the
      suffix is removed.
    * A variable with `units` starting in ``'√mW'``, a `subcarrier_spacing` attr, and
      equal `noise_bandwidth` and `sample_rate` attrs: `noise_bandwidth` is set to
      the `subcarrier_spacing` attr of the variable.
    * A ``*_power_bin`` coordinate without `noise_bandwidth` and with `units` not
      starting in ``'dB/'``: `noise_bandwidth` is copied from a data variable on the
      dimension of the coordinate.
    * `units` of the form ``'<unit>/<bandwidth>'`` with ``<unit>`` one of ``dBm``,
      ``dB``, or ``√mW`` and ``<bandwidth>`` other than ``Hz``: `units` is set to
      ``'<unit>'``.
    * A data variable on a ``*_power_bin`` dimension with a coordinate with `units`
      equal to ``'dB'``: `noise_bandwidth` is removed.
    * `system_noise` without `standard_name`: `standard_name` is set to
      ``'System noise'``.

    A dataset that matches none of the conditions is returned unchanged.

    Returns:
        the same dataset object
    """
    for name in _CAPTURE_BANDWIDTH_NAMES.intersection(map(str, data.variables)):
        attrs = data[name].attrs
        attrs.pop('noise_bandwidth', None)
        long_name = attrs.get('long_name')
        if isinstance(long_name, str) and _BANDWIDTH_SUFFIX.search(long_name):
            del attrs['long_name']

    for var in data.variables.values():
        for key in ('long_name', 'standard_name'):
            text = var.attrs.get(key)
            if isinstance(text, str):
                var.attrs[key] = _BANDWIDTH_SUFFIX.sub('', text)

    for var in data.variables.values():
        attrs = var.attrs
        units = str(attrs.get('units', ''))
        is_correlator = (
            units.startswith('√mW')
            and 'subcarrier_spacing' in attrs
            and 'sample_rate' in attrs
            and attrs.get('noise_bandwidth') == attrs['sample_rate']
        )
        if is_correlator:
            attrs['noise_bandwidth'] = float(attrs['subcarrier_spacing'])

    for name in data.coords:
        if (
            str(name).endswith(_POWER_BIN_SUFFIX)
            and 'noise_bandwidth' not in data[name].attrs
        ):
            if str(data[name].attrs.get('units', '')).startswith('dB/'):
                continue
            bandwidth = _owner_noise_bandwidth(data, str(name))
            if bandwidth is not None:
                data[name].attrs['noise_bandwidth'] = bandwidth

    for var in data.variables.values():
        match = _BANDWIDTH_UNITS.match(str(var.attrs.get('units', '')))
        if match is None or match.group(2) == 'Hz':
            # a unit per Hz denotes a spectral density and is kept
            continue
        var.attrs['units'] = match.group(1)

    for var in data.data_vars.values():
        for dim in var.dims:
            if not str(dim).endswith(_POWER_BIN_SUFFIX) or dim not in data.coords:
                continue
            if data[dim].attrs.get('units') == 'dB':
                var.attrs.pop('noise_bandwidth', None)

    if 'system_noise' in data.variables:
        data['system_noise'].attrs.setdefault('standard_name', 'System noise')

    return data


def load_attrs(path: str | Path) -> dict[str, typing.Any]:
    """load and return the attrs metadata dictionary from the given zarr store"""

    with open_store(path, mode='r') as store:
        array = zarr.open(store)
        return array.attrs.asdict()


_S = typing.TypeVar('_S', bound=specs.SpecBase)


@typing.overload
def decode_from_yaml_file(
    path: str | Path, *, type: type[dict] = ...
) -> dict[str, typing.Any]: ...


@typing.overload
def decode_from_yaml_file(path: str | Path, *, type: type[_S]) -> _S: ...


def decode_from_yaml_file(
    path: str | Path, *, type: type[specs.SpecBase | dict] = dict
) -> dict[str, typing.Any] | specs.SpecBase:
    """decode a YAML file into a dict or a spec struct.

    An ``!include`` tag is expanded relative to the directory of the file that
    contains the tag. YAML sequences decode as tuples.

    Args:
        path: the YAML file
        type: `dict` for the mapping as decoded, or a `specs.SpecBase` subclass to
            convert the mapping into

    Returns:
        the decoded dict, or an instance of `type`

    Raises:
        TypeError: when `type` is neither `dict` nor a `specs.SpecBase` subclass
        FileNotFoundError: when `path` or an ``!include`` pattern matches no file
        yaml.YAMLError: when the document is malformed
        msgspec.ValidationError: when the document does not match `type`
    """

    with open(path) as f, _YAMLIncludeConstructor(path):
        obj = yaml.load(f, _YAMLFrozenLoader)

    if issubclass(type, dict):
        return obj
    elif issubclass(type, specs.SpecBase):
        return type.from_dict(obj)
    else:
        raise TypeError(f'unsupported type {type!r}')


class _FileStreamBase:
    _refs: list[typing.Any]
    _pending: list[typing.Any]
    _loop: bool = False

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

    def close(self):
        pass

    def _cast_ref(self, ref: typing.Any) -> Array:
        """convert one array reference from the file into a (port, sample) array"""
        raise NotImplementedError

    def read(self, count: int | None = None) -> Array | None:
        if count == 0:
            return None

        if self._leftover is None:
            tally = 0
            array_list: list[Array] = []
        else:
            tally = self._leftover.shape[1]
            array_list = [self._leftover]

        while count is None or tally < count:
            try:
                ref = self._pending.pop(0)
            except IndexError:
                if count is None:
                    break
                elif self._loop and len(self._refs) > 0:
                    self._pending = list(self._refs)
                    ref = self._pending.pop(0)
                else:
                    raise ValueError('too few samples in the file')

            if not hasattr(ref, 'shape') or ref.ndim != 2:
                continue

            x = self._cast_ref(ref)
            array_list.append(x)
            tally += x.shape[1]

        iq = self._xp.concatenate(array_list, axis=1)
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
        self._leftover = None
        self._position = pos

    def get_capture_fields(self) -> dict:
        return self._capture_dict


class MATNewFileStream(_FileStreamBase):
    def __init__(
        self,
        path,
        *,
        backend_sample_rate: float,
        key: str | None = None,
        skip_samples=0,
        dtype='complex64',
        loop=False,
        xp=np,
        **meta,
    ):
        kws = {
            'backend_sample_rate': backend_sample_rate,
            'skip_samples': skip_samples,
            'dtype': dtype,
            'xp': xp,
        }

        import h5py  # type: ignore

        self._fd = h5py.File(path, 'r')
        self._loop = loop
        self._key = key

        super().__init__(path, **(kws | meta))

    def close(self):
        self._fd.close()

    def _cast_ref(self, ref: typing.Any) -> Array:
        raw = np.asarray(ref)
        input_dtype = np.result_type(raw.dtype[0], complex)
        return raw.view(input_dtype).astype(self.dtype)

    def seek(self, pos):
        if pos == self._position:
            return

        super().seek(pos)

        if self._key:
            iq = np.atleast_2d(self._fd[self._key])
            self._refs = [iq]
        elif '#refs#' in self._fd:
            self._refs = list(self._fd['#refs#'].values())
        else:
            raise OSError('no data references found in file')

        self._pending = list(self._refs)
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

    def _cast_ref(self, ref: typing.Any) -> Array:
        return self._xp.asarray(ref, ref.dtype).astype(self.dtype)

    def seek(self, pos):
        if pos == self._position:
            return

        super().seek(pos)

        iq = np.atleast_2d(self._fd[self._key])
        self._refs = [iq]
        self._pending = list(self._refs)
        self.read(self._skip_samples + pos)


class NPYFileStream(_FileStreamBase):
    def __init__(
        self,
        path,
        backend_sample_rate: float,
        skip_samples=0,
        dtype='complex64',
        loop=False,
        xp=np,
        **meta,
    ):
        self._data = xp.asarray(np.atleast_2d(np.load(path))).astype(dtype, copy=False)
        self._loop = loop

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

    def read(self, count: int | None = 0) -> Array | None:
        if count == 0:
            return None

        xp = self._xp

        if self._leftover is None:
            tally = 0
            array_list = []
        else:
            tally = self._leftover.shape[1]
            array_list = [self._leftover]

        while count is None or tally < count:
            try:
                ref = self._refs.pop(0)
            except IndexError:
                if count is None:
                    break
                elif self._loop:
                    ref = self._data
                else:
                    raise ValueError('too few samples in the file')

            if not hasattr(ref, 'shape') or ref.ndim < 2:
                continue

            array_list.append(ref)
            tally += ref.shape[1]

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
        try:
            from nptdms import TdmsFile
        except ImportError:
            raise ImportError('install nptdms to open TDMS file streams')

        self._fd = TdmsFile.read(path)
        self._header_fd, self._iq_fd = self._fd.groups()

        super().__init__(
            path,
            skip_samples=skip_samples,
            dtype=dtype,
            xp=xp,
            **meta,
        )

        self._position = 0

    def close(self):
        self._fd.close()

    def read(self, count: int | None = None):
        xp = self._xp

        offset = self._position or 0

        size = int(self._fd['header_fd']['total_samples'][0])
        ref_level = self._fd['header_fd']['reference_level_dBm'][0]

        if count is None:
            count = size

        if size < count:
            raise ValueError(
                f'requested {count} samples but file capture length is {size} samples'
            )

        scale = 10 ** (float(ref_level) / 20.0) / np.iinfo(xp.int16).max
        i, q = self._fd['iq_fd'].channels()
        iq = xp.empty((2 * count,), dtype=xp.int16)
        iq[offset * 2 :: 2] = xp.asarray(i[offset : count + offset])
        iq[1 + offset * 2 :: 2] = xp.asarray(q[offset : count + offset])

        self._position = (self._position or 0) + count

        float_dtype = np.finfo(np.dtype(self.dtype)).dtype
        iq = (iq * float_dtype.type(scale)).view(self.dtype)

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
) -> FileStream:
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


def _is_fsspec_url(url_string: Path | str) -> TypeIs[str]:
    if isinstance(url_string, Path):
        return False

    import fsspec.core

    protocol, _ = fsspec.core.split_protocol(url_string)
    return protocol is not None


@functools.cache
def _zarr_version() -> tuple[int, ...]:
    return tuple(int(n) for n in zarr.__version__.split('.'))


@functools.cache
def _xarray_version() -> tuple[int, ...]:
    return tuple(int(n) for n in xr.__version__.split('.'))


def zarr_format_kws(zarr_format: ZarrFormat = 'auto') -> dict[str, int]:
    """return the `to_zarr` keyword argument that selects the zarr format.

    Args:
        zarr_format: 2, 3, or ``'auto'`` for the major version of the installed zarr

    Returns:
        ``{'zarr_version': 2}`` or ``{'zarr_format': 3}``

    Raises:
        TypeError: when `zarr_format` is not 2, 3, or ``'auto'``
    """
    if isinstance(zarr_format, str):
        zarr_format = zarr_format.lower()

    if zarr_format == 'auto':
        zarr_format = 2 if _zarr_version() < (3, 0, 0) else 3
    elif zarr_format not in (2, 3):
        raise TypeError('zarr_format must be one of (2,3,"auto")')

    if zarr_format == 2:
        return {'zarr_version': 2}
    else:
        return {'zarr_format': 3}


def _get_store_info(store, zarr_format: ZarrFormat = 'auto') -> tuple[bool, dict]:
    kws = zarr_format_kws(zarr_format)

    if _zarr_version() < (3, 0, 0):
        exists = len(store) > 0
    else:
        sync = importlib.import_module('zarr.core.sync').sync
        exists = not sync(store.is_empty(''))

    return exists, kws


def _choose_chunk_sizes(
    ds: xr.Dataset | xr.DataArray, data_bytes=100_000_000, dim: str = 'capture'
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
    # zarr 3 modules, absent from the zarr 2 stubs
    codec_pipeline: typing.Any = importlib.import_module('zarr.core.codec_pipeline')
    if isinstance(compression, codec_pipeline.Codec):
        compressors = [compression]
    elif compression:
        codecs: typing.Any = importlib.import_module('zarr.codecs')
        compressors = [codecs.BloscCodec(cname='zstd', clevel=1, shuffle='shuffle')]
    else:
        compressors = None

    encodings = defaultdict(dict)
    info_map = {info.name: info for info in registry.values()}

    for name, var in data.variables.items():
        meas_info = info_map.get(name, None)
        if (
            meas_info is None
            or not meas_info.store_compressed
            or issubclass(var.dtype.type, np.str_)
        ):
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
        compressor = numcodecs.Blosc('zstd', clevel=1)
    else:
        compressor = None

    encodings = defaultdict(dict)
    info_map = {info.name: info for info in registry.values()}

    for name in data.data_vars:
        meas_info = info_map.get(name, None)
        if meas_info is None or not meas_info.store_compressed:
            encodings[name]['compressor'] = None
        else:
            encodings[name]['compressor'] = compressor

    return encodings


class _YAMLFrozenLoader(yaml.SafeLoader):
    def _construct_sequence(
        self, node: SequenceNode, deep: bool = False
    ) -> tuple[typing.Any, ...]:
        return tuple(super().construct_sequence(node, deep))


_YAMLFrozenLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_SEQUENCE_TAG,
    _YAMLFrozenLoader._construct_sequence,
)


def _glob_include_patterns(patterns: str | Sequence[str], root_dir: Path) -> list[str]:
    """expand ``!include`` glob patterns into paths, sorted within each pattern.

    Args:
        patterns: one glob pattern, or a sequence of patterns
        root_dir: the directory under which relative patterns are globbed; an
            absolute pattern is globbed under its own parent directory

    Raises:
        FileNotFoundError: when a pattern matches no file
    """

    def glob_path(s: str) -> list[str]:
        p = Path(s)
        if p.is_absolute():
            rel = p.parent
            s = p.name
        else:
            rel = root_dir
        paths = [str(g) for g in rel.glob(s)]
        if len(paths) == 0:
            raise FileNotFoundError(s)
        return sorted(paths)

    if isinstance(patterns, str):
        return glob_path(patterns)

    values: list[str] = []
    for pattern in patterns:
        values.extend(glob_path(pattern))
    return values


def _expand_paths(node: yaml.Node, root_dir: Path) -> list[str]:
    """expand the ``!include`` patterns in `node` into sorted paths.

    Args:
        node: a scalar node with one glob pattern, or a sequence node of patterns
        root_dir: the directory under which relative patterns are globbed

    Raises:
        FileNotFoundError: when a pattern matches no file
        TypeError: when `node` is neither a scalar nor a sequence
    """

    if isinstance(node.value, str):
        patterns: str | list[str] = node.value
    elif isinstance(node.value, (list, tuple)):
        patterns = [n.value for n in node.value]
    else:
        raise TypeError(f'invalid tag type {type(node.value)!r} in !import')

    return _glob_include_patterns(patterns, root_dir)


class _YAMLIncludeConstructor(yaml.Loader):
    _lock = threading.RLock()

    def __init__(self, path):
        self.nested_paths: list[Path] = [Path(path)]

    def __enter__(self):
        self._lock.acquire()
        yaml.add_constructor('!include', self, Loader=_YAMLFrozenLoader)

    def __exit__(self, *args):
        self._lock.release()

    def get_include_path(self, s: str) -> Path:
        """push and return the path of an included file.

        A relative `s` is kept relative to the working directory, not to the
        including file.
        """
        p = Path(s)
        self.nested_paths.append(p)
        return p

    def pop_include_path(self):
        self.nested_paths.pop()

    def __call__(self, loader: _YAMLFrozenLoader, node: yaml.Node):
        if not node.tag.startswith('!include'):
            raise ValueError(f'unknown tag {node.tag!r}')

        values = _expand_paths(node, root_dir=self.nested_paths[-1].parent)

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


_INCLUDE_TAG = '!include'
_EDIT_OPS = ('set', 'delete', 'insert')


@dataclasses.dataclass
class YAMLSource:
    """one YAML file read by `load_yaml_provenance`, as a round-trip document"""

    path: Path
    """absolute path of the file"""

    doc: typing.Any
    """the round-trip document (`CommentedMap`, `CommentedSeq`, or a scalar), which
    keeps comments, anchors, quotes, key order, and tags"""


@dataclasses.dataclass
class ProvenanceTree:
    """the merged document of a YAML spec file and the file position of every node.

    JSON pointers (RFC 6901) address nodes in `tree`; a pointer token is the `str`
    of the mapping key or the sequence index. `origin` maps each key and element
    of `tree` to the file that holds its value and the pointer within that
    file's document; `regions` maps each node produced by an ``!include`` tag to
    the files merged there, in merge order.
    """

    root: Path
    """absolute path of the top file"""

    tree: typing.Any
    """the merged document as `dict`, `list`, and scalars, equal to the
    `decode_from_yaml_file` result with tuples read as lists"""

    files: dict[Path, YAMLSource]
    """every file read, keyed by absolute path, in first-read order"""

    origin: dict[str, tuple[Path, str]]
    """JSON pointer in `tree` to the file that holds the value and the JSON
    pointer within that file's document; a key set by several files of a merged
    include maps to the last of them"""

    regions: dict[str, tuple[Path, ...]]
    """JSON pointer of each node that an ``!include`` tag produced to the files
    merged there, in merge order"""


@dataclasses.dataclass(frozen=True)
class YAMLEdit:
    """one change to the merged document of a `ProvenanceTree`"""

    pointer: str
    """JSON pointer (RFC 6901) in the merged tree; ``insert`` takes the pointer
    of the sequence, with or without a trailing ``/-``"""

    op: str
    """``'set'``, ``'delete'``, or ``'insert'``"""

    value_text: typing.Optional[str] = None
    """YAML text of the new value for ``set`` and ``insert``; `None` for ``delete``"""


@dataclasses.dataclass
class _FileLoad:
    """the converted document of one file, with pointers relative to the file root"""

    tree: typing.Any
    origin: dict[str, tuple[Path, str]]
    regions: dict[str, tuple[Path, ...]]


def load_yaml_provenance(path: str | Path) -> ProvenanceTree:
    """load a YAML spec file and record the file that holds each value.

    ``!include`` tags are expanded as in `decode_from_yaml_file`: a pattern is
    globbed relative to the directory of the file that contains the tag, several
    matches of mappings are merged with later keys overriding, and several
    matches of sequences are concatenated. Merge keys (``<<``) are resolved into
    `tree` with the file of the mapping as origin. Scalars resolve under the
    PyYAML rules (YAML 1.1: ``3750e6`` is a string, ``yes`` is a bool). When a
    key repeats in one mapping, the last value is kept.

    Args:
        path: the YAML file

    Returns:
        the merged tree with the origin of every key and element, and the
        round-trip document of every file read, for `apply_yaml_edits`

    Raises:
        FileNotFoundError: when `path` or an ``!include`` pattern matches no file
        ValueError: when a tag other than ``!include`` is present
        TypeError: when the files of one ``!include`` mix mappings and sequences,
            or an ``!include`` tag is on a mapping
    """
    rt = _round_trip_yaml()
    files: dict[Path, YAMLSource] = {}
    root = Path(path).resolve()
    loaded = _load_yaml_source(root, files, rt)
    return ProvenanceTree(
        root=root,
        tree=loaded.tree,
        files=files,
        origin=loaded.origin,
        regions=loaded.regions,
    )


def apply_yaml_edits(
    prov: ProvenanceTree, edits: Sequence[YAMLEdit]
) -> tuple[Path, ...]:
    """write edits to the merged document back into the files that hold the values.

    Each edit is applied to the round-trip document of the file named by
    `prov.origin` for its pointer, and the touched files are rewritten with
    comments, anchors, quotes, key order, and untouched ``!include`` tags kept.
    A value is parsed from `value_text` as YAML, so ``3750e6`` is written as
    ``3750e6`` and ``[0, 1]`` as a flow sequence. A ``set`` on a key that does
    not exist appends the key to the mapping of the parent pointer; when the
    parent is a merged ``!include``, the last file of the region receives the
    key. An ``insert`` appends to the sequence at the pointer, into the last
    file of the region when the sequence came from a merge. A ``delete`` removes
    the key or element. Each file is replaced atomically by `os.replace` of a
    temporary file in the same directory. `prov` is not updated, so the caller
    reloads with `load_yaml_provenance` before further edits.

    Args:
        prov: the tree returned by `load_yaml_provenance`
        edits: the edits, applied in order

    Returns:
        the paths written, in sorted order

    Raises:
        KeyError: when a pointer resolves to no node, or ``delete`` names a key
            that a merge key (``<<``) supplies rather than the mapping itself
        ValueError: for an `op` other than ``'set'``, ``'delete'``, or
            ``'insert'``, or a ``set``/``insert`` without `value_text`
        TypeError: when ``insert`` targets a node that is not a sequence
    """
    for edit in edits:
        if edit.op not in _EDIT_OPS:
            raise ValueError(f'unknown op {edit.op!r}; expected one of {_EDIT_OPS}')
        if edit.op != 'delete' and edit.value_text is None:
            raise ValueError(f'value_text is required for op {edit.op!r}')

    rt = _round_trip_yaml()
    touched = {_apply_yaml_edit(prov, edit, rt) for edit in edits}

    written = tuple(sorted(touched))
    for path in written:
        _write_yaml_atomic(prov.files[path].doc, path)
    return written


def _load_yaml_source(
    path: Path, files: dict[Path, YAMLSource], rt: ruamel.yaml.YAML
) -> _FileLoad:
    """load `path` once into `files` and convert its document into a `_FileLoad`"""
    if path not in files:
        with open(path, encoding='utf-8') as f:
            files[path] = YAMLSource(path=path, doc=rt.load(f))
    source = files[path]
    load = _FileLoad(tree=None, origin={}, regions={})
    load.tree = _convert_node(source.doc, '', source, load, files, rt)
    return load


def _convert_node(
    node: typing.Any,
    pointer: str,
    source: YAMLSource,
    load: _FileLoad,
    files: dict[Path, YAMLSource],
    rt: ruamel.yaml.YAML,
) -> typing.Any:
    """convert a round-trip node into plain Python, recording origins into `load`"""
    comments = ruamel.yaml.comments

    if isinstance(node, comments.TaggedScalar) or (
        isinstance(node, (comments.CommentedMap, comments.CommentedSeq))
        and node.tag.value is not None
    ):
        return _convert_include(node, pointer, source, load, files, rt)

    if isinstance(node, comments.CommentedMap):
        mapping: dict[typing.Any, typing.Any] = {}
        for key, value in node.items():
            child = f'{pointer}/{_escape_pointer_token(_key_token(key))}'
            load.origin[child] = (source.path, child)
            mapping[_plain_key(key)] = _convert_node(
                value, child, source, load, files, rt
            )
        return mapping

    if isinstance(node, comments.CommentedSeq):
        items: list[typing.Any] = []
        for i, value in enumerate(node):
            child = f'{pointer}/{i}'
            load.origin[child] = (source.path, child)
            items.append(_convert_node(value, child, source, load, files, rt))
        return items

    return _plain_scalar(node)


def _convert_include(
    node: typing.Any,
    pointer: str,
    source: YAMLSource,
    load: _FileLoad,
    files: dict[Path, YAMLSource],
    rt: ruamel.yaml.YAML,
) -> typing.Any:
    """expand an ``!include`` node and merge the files into `load` at `pointer`"""
    comments = ruamel.yaml.comments

    tag = node.tag.value
    if tag != _INCLUDE_TAG:
        raise ValueError(f'unknown tag {tag!r}')

    if isinstance(node, comments.TaggedScalar):
        patterns: str | list[str] = str(node.value)
    elif isinstance(node, comments.CommentedSeq):
        patterns = [str(v) for v in node]
    else:
        raise TypeError(f'invalid tag type {type(node)!r} in {_INCLUDE_TAG}')

    paths = [
        Path(p).resolve() for p in _glob_include_patterns(patterns, source.path.parent)
    ]
    loads = [_load_yaml_source(p, files, rt) for p in paths]

    if len(loads) == 1:
        result = loads[0].tree
        _adopt_subtree(load, loads[0], '', pointer)
    elif all(isinstance(sub.tree, dict) for sub in loads):
        result = {}
        for sub in loads:
            for key, value in sub.tree.items():
                token = _escape_pointer_token(_key_token(key))
                _adopt_subtree(load, sub, f'/{token}', f'{pointer}/{token}')
                result[key] = value
    elif all(isinstance(sub.tree, list) for sub in loads):
        result = []
        for sub in loads:
            for i, value in enumerate(sub.tree):
                _adopt_subtree(load, sub, f'/{i}', f'{pointer}/{len(result)}')
                result.append(value)
    else:
        raise TypeError(
            f'contents of multiple {_INCLUDE_TAG} files must be '
            'either all mappings or all sequences'
        )

    load.regions[pointer] = tuple(paths)
    return result


def _adopt_subtree(load: _FileLoad, sub: _FileLoad, old: str, new: str) -> None:
    """move the origins and regions of `sub` under `old` into `load` under `new`,
    replacing the entries of an earlier file below `new`"""
    for entries in (load.origin, load.regions):
        for key in [k for k in entries if k.startswith(new + '/')]:
            del entries[key]
    load.origin.update(_rebase_pointers(sub.origin, old, new))
    load.regions.update(_rebase_pointers(sub.regions, old, new))


_T = typing.TypeVar('_T')


def _rebase_pointers(entries: dict[str, _T], old: str, new: str) -> dict[str, _T]:
    """return the entries at or below pointer `old` with the prefix replaced by `new`"""
    return {
        new + k[len(old) :]: v
        for k, v in entries.items()
        if k == old or k.startswith(old + '/')
    }


def _plain_scalar(value: typing.Any) -> typing.Any:
    """return the builtin equivalent of a round-trip scalar"""
    if isinstance(value, ruamel.yaml.scalarbool.ScalarBoolean):
        return bool(value)
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    if isinstance(value, str):
        return str(value)
    return value


def _plain_key(key: typing.Any) -> typing.Any:
    """return a mapping key as a builtin, with a sequence key as a tuple"""
    if isinstance(key, ruamel.yaml.comments.CommentedKeySeq):
        return tuple(_plain_key(k) for k in key)
    return _plain_scalar(key)


def _key_token(key: typing.Any) -> str:
    """return the JSON pointer token of a mapping key, before escaping"""
    if isinstance(key, ruamel.yaml.comments.CommentedKeySeq):
        return str(tuple(_plain_key(k) for k in key))
    return str(_plain_scalar(key))


def _escape_pointer_token(token: str) -> str:
    return token.replace('~', '~0').replace('/', '~1')


def _unescape_pointer_token(token: str) -> str:
    return token.replace('~1', '/').replace('~0', '~')


def _split_pointer(pointer: str) -> tuple[str, str]:
    """return the parent pointer and the unescaped last token of `pointer`.

    Raises:
        KeyError: when `pointer` is empty or does not start with ``/``
    """
    if not pointer.startswith('/'):
        raise KeyError(pointer)
    parent, _, token = pointer.rpartition('/')
    return parent, _unescape_pointer_token(token)


def _mapping_key(node: typing.Any, token: str) -> typing.Any:
    """return the key of the round-trip mapping `node` with pointer token `token`.

    Raises:
        KeyError: when no key of `node` has that token
    """
    for key in node:
        if _key_token(key) == token:
            return key
    raise KeyError(token)


def _child_node(node: typing.Any, token: str) -> typing.Any:
    """return the child of a round-trip mapping or sequence at `token`.

    Raises:
        KeyError: when `node` is a scalar or has no child at `token`
    """
    if isinstance(node, ruamel.yaml.comments.CommentedSeq):
        try:
            return node[int(token)]
        except (ValueError, IndexError):
            raise KeyError(token) from None
    if isinstance(node, ruamel.yaml.comments.CommentedMap):
        return node[_mapping_key(node, token)]
    raise KeyError(token)


def _resolve_pointer(doc: typing.Any, pointer: str) -> typing.Any:
    """return the node of the round-trip document `doc` at the JSON pointer"""
    node = doc
    for token in pointer.split('/')[1:]:
        node = _child_node(node, _unescape_pointer_token(token))
    return node


def _locate_container(prov: ProvenanceTree, pointer: str) -> tuple[Path, typing.Any]:
    """return the file and round-trip node that receive a new child of `pointer`.

    A node produced by a merged ``!include`` resolves to the root of the last
    file of the region.

    Raises:
        KeyError: when `pointer` names no node of `prov.tree`
    """
    if pointer in prov.regions:
        path = prov.regions[pointer][-1]
        return path, prov.files[path].doc
    if pointer == '':
        return prov.root, prov.files[prov.root].doc
    if pointer in prov.origin:
        path, local = prov.origin[pointer]
        return path, _resolve_pointer(prov.files[path].doc, local)
    raise KeyError(pointer)


def _locate_existing(
    prov: ProvenanceTree, pointer: str
) -> tuple[Path, typing.Any, str]:
    """return the file, round-trip parent node, and token of the node at `pointer`.

    Raises:
        KeyError: when `pointer` names no key or element of `prov.tree`
    """
    try:
        path, local = prov.origin[pointer]
    except KeyError:
        raise KeyError(pointer) from None
    parent_local, token = _split_pointer(local)
    return path, _resolve_pointer(prov.files[path].doc, parent_local), token


def _apply_yaml_edit(
    prov: ProvenanceTree, edit: YAMLEdit, rt: ruamel.yaml.YAML
) -> Path:
    """apply one edit to the round-trip documents of `prov` and return its file"""
    comments = ruamel.yaml.comments

    if edit.op == 'delete':
        path, parent, token = _locate_existing(prov, edit.pointer)
        if isinstance(parent, comments.CommentedSeq):
            del parent[int(token)]
        else:
            key = _mapping_key(parent, token)
            literal_keys = [k for k, _ in parent.non_merged_items()]
            if not any(k == key for k in literal_keys):
                raise KeyError(
                    f'{edit.pointer!r} is supplied by a merge key (<<) of '
                    f'{path}, not by the mapping itself'
                )
            del parent[key]
        return path

    value = rt.load(edit.value_text)

    if edit.op == 'set':
        if edit.pointer in prov.origin:
            path, parent, token = _locate_existing(prov, edit.pointer)
            if isinstance(parent, comments.CommentedSeq):
                parent[int(token)] = value
            else:
                parent[_mapping_key(parent, token)] = value
            return path

        parent_pointer, token = _split_pointer(edit.pointer)
        path, parent = _locate_container(prov, parent_pointer)
        if not isinstance(parent, comments.CommentedMap):
            raise KeyError(edit.pointer)
        parent[token] = value
        return path

    pointer = edit.pointer.removesuffix('/-')
    path, node = _locate_container(prov, pointer)
    if not isinstance(node, comments.CommentedSeq):
        raise TypeError(f'{pointer!r} is not a sequence')
    node.append(value)
    return path


def _write_yaml_atomic(doc: typing.Any, path: Path) -> None:
    """dump the round-trip `doc` to a temporary file beside `path`, then replace `path`"""
    rt = _round_trip_yaml(indent=_detect_indent(doc))
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f'.{path.name}.', suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='') as f:
            rt.dump(doc, f)
        shutil.copymode(path, tmp)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def _detect_indent(doc: typing.Any) -> tuple[int, int, int]:
    """return the (mapping, sequence, offset) indents of `doc` from its node columns.

    The first block mapping nested in a mapping sets the mapping indent, and the
    first block sequence sets the sequence indent and dash offset; a style
    absent from `doc` takes the default (2, 4, 2).
    """
    comments = ruamel.yaml.comments
    mapping: int | None = None
    sequence: int | None = None
    offset: int | None = None

    def walk(node: typing.Any, col: int) -> None:
        nonlocal mapping, sequence, offset
        if isinstance(node, comments.CommentedMap):
            for key, value in node.non_merged_items():
                key_col = _child_col(node, key)
                if key_col is None:
                    continue
                if mapping is None and _is_block(value, comments.CommentedMap):
                    mapping = value.lc.col - key_col
                walk(value, key_col)
        elif isinstance(node, comments.CommentedSeq):
            if offset is None and _is_block(node, comments.CommentedSeq):
                offset = node.lc.col - col
                item_col = _child_col(node, 0)
                if item_col is not None:
                    sequence = item_col - col
            for value in node:
                walk(value, node.lc.col)

    def _child_col(node: typing.Any, key: typing.Any) -> int | None:
        # a key or element added since the load has no line and column record
        data = node.lc.data
        if data is None or key not in data:
            return None
        return data[key][1]

    def _is_block(value: typing.Any, cls: type) -> bool:
        return isinstance(value, cls) and not value.fa.flow_style()

    walk(doc, 0)
    return (
        2 if mapping is None or mapping < 1 else mapping,
        4 if sequence is None or sequence < 2 else sequence,
        2 if offset is None or offset < 0 else offset,
    )


@functools.cache
def _round_trip_classes() -> tuple[type, type, type]:
    """return the resolver, constructor, and representer classes of the round-trip loader"""
    # PyYAML resolves with its YAML 1.1 tables, under which 3750e6 is a string;
    # ruamel resolves 1.1 floats without a mantissa dot, so the tables are
    # copied from PyYAML to keep the merged tree equal to decode_from_yaml_file
    implicit = {
        first: list(resolvers)
        for first, resolvers in yaml.resolver.Resolver.yaml_implicit_resolvers.items()
    }

    class Resolver(ruamel.yaml.resolver.VersionedResolver):
        def __init__(
            self,
            version: typing.Any = None,
            loader: typing.Any = None,
            loadumper: typing.Any = None,
        ) -> None:
            super().__init__((1, 1), loader, loadumper)

        @property
        def versioned_resolver(self) -> dict[typing.Any, list[typing.Any]]:
            return implicit

    class Constructor(ruamel.yaml.constructor.RoundTripConstructor):
        def check_mapping_key(
            self,
            node: typing.Any,
            key_node: typing.Any,
            mapping: typing.Any,
            key: typing.Any,
            value: typing.Any,
        ) -> bool:
            # PyYAML keeps the last value of a repeated key
            return True

    class Representer(ruamel.yaml.representer.RoundTripRepresenter):
        pass

    Representer.add_representer(
        type(None),
        lambda representer, data: representer.represent_scalar(
            'tag:yaml.org,2002:null', 'null'
        ),
    )

    return Resolver, Constructor, Representer


def _round_trip_yaml(indent: tuple[int, int, int] = (2, 4, 2)) -> ruamel.yaml.YAML:
    """return a round-trip `YAML` that resolves scalars as PyYAML does.

    Args:
        indent: the (mapping, sequence, offset) indents used on dump
    """
    Resolver, Constructor, Representer = _round_trip_classes()
    rt = ruamel.yaml.YAML(typ='rt', pure=True)
    rt.Resolver = Resolver
    rt.Constructor = Constructor
    rt.Representer = Representer
    rt.preserve_quotes = True
    rt.width = 4096
    mapping, sequence, offset = indent
    rt.indent(mapping=mapping, sequence=sequence, offset=offset)
    return rt
