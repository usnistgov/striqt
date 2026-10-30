"""zarr stores that stage metadata documents in memory and pass chunk keys through
to an inner store.

A zip archive cannot replace an entry, and the ``ZipStore`` of zarr 2 and zarr 3
each append a duplicate entry on a repeated key. zarr rewrites the metadata
document of a node on every attribute update, resize, and consolidation. The
wrappers here keep the metadata documents in memory until `flush_metadata` or
`close` and record each chunk key written to the inner store.
`staged_metadata_store` selects the class for the installed zarr version.
"""

from __future__ import annotations as __

import typing
from typing import Any, Union

import zarr

from . import util

if typing.TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterable, Iterator
    from types import ModuleType
    from typing import Any as _V2Base, Any as _V3Base, TypeAlias

    _store_abc: ModuleType
    _zbuffer: ModuleType
    _zsync: ModuleType

    if hasattr(zarr, 'abc'):
        Buffer: TypeAlias = zarr.core.buffer.Buffer
        BufferPrototype: TypeAlias = zarr.core.buffer.BufferPrototype
        ByteRequest: TypeAlias = zarr.abc.store.ByteRequest
        InnerStoreV3: TypeAlias = zarr.abc.store.Store
    else:
        Buffer: TypeAlias = Any
        BufferPrototype: TypeAlias = Any
        ByteRequest: TypeAlias = Any
        InnerStoreV3: TypeAlias = Any
elif int(zarr.__version__.split('.')[0]) < 3:
    from zarr.storage import Store as _V2Base

    _V3Base = object
else:
    import zarr.abc.store as _store_abc
    import zarr.core.buffer as _zbuffer
    import zarr.core.sync as _zsync
    from zarr.storage import WrapperStore as _V3Base

    _V2Base = object


_ZARR_MAJOR = int(zarr.__version__.split('.')[0])
_V2_METADATA_NAMES = frozenset({'.zarray', '.zgroup', '.zattrs', '.zmetadata'})


def is_metadata_key(key: str) -> bool:
    """return whether `key` names a zarr metadata document under the installed zarr.

    Args:
        key: a store key, with ``/`` separators

    Returns:
        ``True`` for a basename of ``.zarray``, ``.zgroup``, ``.zattrs``, or
        ``.zmetadata`` under zarr 2, and for a basename of ``zarr.json`` or one
        of the zarr 2 names under zarr 3.
    """
    name = key.rsplit('/', 1)[-1]
    if _ZARR_MAJOR >= 3 and name == 'zarr.json':
        return True
    return name in _V2_METADATA_NAMES


def _warn_repeated_write(key: str) -> None:
    util.get_logger('analysis').warning(
        'store key %r written more than once (duplicate zip entry)', key
    )


def _child_names(keys: Iterable[str], prefix: str) -> set[str]:
    """return the first path component below `prefix` of every key with that prefix"""
    prefix = prefix.strip('/')
    if prefix:
        prefix += '/'
    children = set()
    for key in keys:
        if key.startswith(prefix) and len(key) > len(prefix):
            children.add(key[len(prefix) :].split('/')[0])
    return children


class _StagedStoreV2(_V2Base):
    """a zarr 2 store that keeps metadata documents in memory until flushed.

    A staged key is read from memory; every other key is read from the inner store.
    A chunk key is written to the inner store on assignment. A chunk key assigned a
    second time logs a WARNING on the ``analysis`` logger.
    """

    path: str

    def __init__(self, store: Any) -> None:
        """wrap `store`; nothing is read or written."""
        self._inner = store
        self._staged: dict[str, Any] = {}
        self._dirty: set[str] = set()
        self.written_keys: set[str] = set()
        """chunk keys written to the inner store since construction"""
        self._erasable = store.is_erasable()
        self._writeable = store.is_writeable()
        self._listable = store.is_listable()
        path = getattr(store, 'path', None)
        if path is not None:
            self.path = path

    def __getitem__(self, key: str) -> Any:
        if key in self._staged:
            return self._staged[key]
        return self._inner[key]

    def __setitem__(self, key: str, value: Any) -> None:
        if is_metadata_key(key):
            self._staged[key] = value
            self._dirty.add(key)
            return
        if key in self.written_keys:
            _warn_repeated_write(key)
        self.written_keys.add(key)
        self._inner[key] = value

    def __delitem__(self, key: str) -> None:
        found = key in self._staged
        self._staged.pop(key, None)
        self._dirty.discard(key)
        if key in self._inner:
            del self._inner[key]
        elif not found:
            raise KeyError(key)

    def __contains__(self, key: object) -> bool:
        return key in self._staged or key in self._inner

    def __iter__(self) -> Iterator[str]:
        seen = set(self._inner.keys())
        yield from seen
        for key in list(self._staged):
            if key not in seen:
                yield key

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def listdir(self, path: str = '') -> list[str]:
        return sorted(_child_names(self, path))

    def rmdir(self, path: str = '') -> None:
        if not self._erasable:
            raise NotImplementedError(f'{type(self._inner)} is not erasable')
        prefix = path.strip('/')
        if prefix:
            prefix += '/'
        for key in list(self._staged):
            if key.startswith(prefix):
                del self._staged[key]
                self._dirty.discard(key)
        self._inner.rmdir(path)

    def getsize(self, path: str = '') -> int:
        path = path.strip('/')
        if path and path in self:
            return memoryview(self[path]).nbytes
        prefix = path + '/' if path else ''
        size = 0
        for child in self.listdir(path):
            key = prefix + child
            if key in self:
                size += memoryview(self[key]).nbytes
        return size

    def flush_metadata(self) -> int:
        """write every staged metadata document changed since the last flush.

        Returns:
            the number of keys written to the inner store
        """
        count = 0
        for key in sorted(self._dirty):
            self._inner[key] = self._staged[key]
            count += 1
        self._dirty.clear()
        return count

    def close(self) -> None:
        """flush the staged metadata, then close the inner store."""
        self.flush_metadata()
        close = getattr(self._inner, 'close', None)
        if close is not None:
            close()


class _StagedStoreV3(_V3Base):
    """a zarr 3 store that keeps metadata documents in memory until flushed.

    A staged key is read from memory; every other key is read from the inner store.
    A chunk key is written to the inner store on `set`. A chunk key set a second
    time logs a WARNING on the ``analysis`` logger. A read-only copy made through
    `with_read_only` shares the staged documents of the original.
    """

    def __init__(self, store: InnerStoreV3) -> None:
        """wrap `store`; nothing is read or written."""
        self._store = store
        self._staged: dict[str, bytes] = {}
        self._dirty: set[str] = set()
        self.written_keys: set[str] = set()
        """chunk keys written to the inner store since construction"""

    def _with_store(self, store: InnerStoreV3) -> _StagedStoreV3:
        new = type(self)(store=store)
        new._staged = self._staged
        new._dirty = self._dirty
        new.written_keys = self.written_keys
        return new

    def _record_chunk_write(self, key: str) -> None:
        if key in self.written_keys:
            _warn_repeated_write(key)
        self.written_keys.add(key)

    def _staged_under(self, prefix: str) -> list[str]:
        return [key for key in list(self._staged) if key.startswith(prefix)]

    @staticmethod
    def _slice(data: bytes, byte_range: ByteRequest | None) -> bytes:
        if byte_range is None:
            return data
        if isinstance(byte_range, _store_abc.RangeByteRequest):
            return data[byte_range.start : byte_range.end]
        if isinstance(byte_range, _store_abc.OffsetByteRequest):
            return data[byte_range.offset :]
        if isinstance(byte_range, _store_abc.SuffixByteRequest):
            return data[max(0, len(data) - byte_range.suffix) :]
        raise TypeError(f'unexpected byte_range {byte_range!r}')

    @property
    def supports_consolidated_metadata(self) -> bool:
        return True

    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        data = self._staged.get(key)
        if data is None:
            return await self._store.get(key, prototype, byte_range)
        return prototype.buffer.from_bytes(self._slice(data, byte_range))

    async def get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: Iterable[tuple[str, ByteRequest | None]],
    ) -> list[Buffer | None]:
        return [await self.get(key, prototype, r) for key, r in key_ranges]

    async def _get_many(
        self, requests: Iterable[tuple[str, BufferPrototype, ByteRequest | None]]
    ) -> AsyncIterator[tuple[str, Buffer | None]]:
        for key, prototype, byte_range in requests:
            yield (key, await self.get(key, prototype, byte_range))

    async def exists(self, key: str) -> bool:
        return key in self._staged or await self._store.exists(key)

    async def set(self, key: str, value: Buffer) -> None:
        if is_metadata_key(key):
            self._staged[key] = value.to_bytes()
            self._dirty.add(key)
            return
        self._record_chunk_write(key)
        await self._store.set(key, value)

    async def set_if_not_exists(self, key: str, value: Buffer) -> None:
        if not await self.exists(key):
            await self.set(key, value)

    async def _set_many(self, values: Iterable[tuple[str, Buffer]]) -> None:
        for key, value in values:
            await self.set(key, value)

    async def delete(self, key: str) -> None:
        self._staged.pop(key, None)
        self._dirty.discard(key)
        if await self._store.exists(key):
            await self._store.delete(key)

    async def delete_dir(self, prefix: str) -> None:
        if prefix != '' and not prefix.endswith('/'):
            prefix += '/'
        for key in self._staged_under(prefix):
            del self._staged[key]
            self._dirty.discard(key)
        await self._store.delete_dir(prefix)

    async def clear(self) -> None:
        self._staged.clear()
        self._dirty.clear()
        await self._store.clear()

    async def is_empty(self, prefix: str) -> bool:
        if prefix != '' and not prefix.endswith('/'):
            prefix += '/'
        if self._staged_under(prefix):
            return False
        return await self._store.is_empty(prefix)

    async def list(self) -> AsyncIterator[str]:
        seen = set()
        async for key in self._store.list():
            seen.add(key)
            yield key
        for key in self._staged_under(''):
            if key not in seen:
                yield key

    async def list_prefix(self, prefix: str) -> AsyncIterator[str]:
        seen = set()
        async for key in self._store.list_prefix(prefix):
            seen.add(key)
            yield key
        for key in self._staged_under(prefix):
            if key not in seen:
                yield key

    async def list_dir(self, prefix: str) -> AsyncIterator[str]:
        seen = set()
        async for child in self._store.list_dir(prefix):
            seen.add(child)
            yield child
        for child in _child_names(self._staged_under(''), prefix):
            if child not in seen:
                yield child

    def get_sync(
        self,
        key: str,
        *,
        prototype: BufferPrototype | None = None,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        data = self._staged.get(key)
        if data is None:
            return self._store.get_sync(key, prototype=prototype, byte_range=byte_range)
        if prototype is None:
            prototype = _zbuffer.default_buffer_prototype()
        return prototype.buffer.from_bytes(self._slice(data, byte_range))

    def set_sync(self, key: str, value: Buffer) -> None:
        if is_metadata_key(key):
            self._staged[key] = value.to_bytes()
            self._dirty.add(key)
            return
        self._record_chunk_write(key)
        self._store.set_sync(key, value)

    def delete_sync(self, key: str) -> None:
        self._staged.pop(key, None)
        self._dirty.discard(key)
        self._store.delete_sync(key)

    async def _flush(self, keys: list[str]) -> None:
        prototype = _zbuffer.default_buffer_prototype()
        for key in keys:
            await self._store.set(key, prototype.buffer.from_bytes(self._staged[key]))

    def flush_metadata(self) -> int:
        """write every staged metadata document changed since the last flush.

        Returns:
            the number of keys written to the inner store
        """
        keys = sorted(self._dirty)
        if keys:
            _zsync.sync(self._flush(keys))
        self._dirty.clear()
        return len(keys)

    def close(self) -> None:
        """flush the staged metadata, then close the inner store."""
        self.flush_metadata()
        self._store.close()


StagedMetadataStore = Union[_StagedStoreV2, _StagedStoreV3]


def staged_metadata_store(store: Any) -> StagedMetadataStore:
    """return a wrapper of `store` that stages metadata documents until flushed.

    Args:
        store: an open, writable store of the installed zarr version

    Returns:
        the wrapper for the installed zarr version, with `flush_metadata`,
        `close`, and `written_keys`
    """
    if _ZARR_MAJOR < 3:
        return _StagedStoreV2(store)
    return _StagedStoreV3(store)
