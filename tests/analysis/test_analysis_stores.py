"""striqt.analysis.lib.stores: metadata staging over directory and zip stores on the
installed zarr version"""

from __future__ import annotations

import logging
import zipfile

import numpy as np
import pytest
import zarr

from striqt.analysis.lib import stores

ZARR_3 = int(zarr.__version__.split('.')[0]) >= 3

SHAPE = (4, 4)
CHUNKS = (2, 2)
# every chunk differs from the fill value: zarr 3 (write_empty_chunks=False) skips
# an all-fill chunk
VALUES = np.arange(1, 17, dtype='float32').reshape(SHAPE)

# the key layouts of the zarr v2 spec (one dot-separated chunk key per array, one
# `.zarray`/`.zattrs`/`.zgroup` document per node) and the zarr v3 spec (one
# `zarr.json` per node, chunk keys under `c/`)
if ZARR_3:
    METADATA_KEYS = {'zarr.json', 'x/zarr.json'}
    CHUNK_KEYS = {f'x/c/{i}/{j}' for i in range(2) for j in range(2)}
    CONSOLIDATED_KEYS: set[str] = set()
    ROOT_CHILDREN = {'zarr.json', 'x'}
    ARRAY_CHILDREN = {'zarr.json', 'c'}
else:
    METADATA_KEYS = {'.zgroup', '.zattrs', 'x/.zarray', 'x/.zattrs'}
    CHUNK_KEYS = {f'x/{i}.{j}' for i in range(2) for j in range(2)}
    CONSOLIDATED_KEYS = {'.zmetadata'}
    ROOT_CHILDREN = {'.zgroup', '.zattrs', 'x'}
    ARRAY_CHILDREN = {'.zarray', '.zattrs', '0.0', '0.1', '1.0', '1.1'}

ALL_KEYS = METADATA_KEYS | CHUNK_KEYS


# %% helpers for the installed zarr version


def open_inner(path, kind: str, mode: str):
    if kind == 'zip':
        return zarr.storage.ZipStore(path, mode=mode)
    if ZARR_3:
        return zarr.storage.LocalStore(path, read_only=(mode == 'r'))
    return zarr.storage.DirectoryStore(path)


def listed_keys(store) -> list[str]:
    """every key the store lists, duplicates included"""
    if ZARR_3:
        from zarr.core.sync import collect_aiterator

        return list(collect_aiterator(store.list()))
    return list(store)


def children(store, prefix: str) -> list[str]:
    if ZARR_3:
        from zarr.core.sync import collect_aiterator

        return list(collect_aiterator(store.list_dir(prefix)))
    return store.listdir(prefix)


def write_group(store):
    """a group with attrs and one 4x4 float32 array in 2x2 chunks, fully written"""
    group = zarr.open_group(store, mode='w')
    group.attrs['a'] = 1
    if ZARR_3:
        x = group.create_array(
            'x', shape=SHAPE, chunks=CHUNKS, dtype='float32', dimension_names=('i', 'j')
        )
    else:
        x = group.create_dataset('x', shape=SHAPE, chunks=CHUNKS, dtype='float32')
        x.attrs['_ARRAY_DIMENSIONS'] = ['i', 'j']
    x.attrs['units'] = 'dB'
    x[:] = VALUES
    return x


def open_consolidated(target):
    if ZARR_3:
        return zarr.open_group(target, mode='r', use_consolidated=True)
    return zarr.open_consolidated(target)


@pytest.fixture(params=['directory', 'zip'])
def inner_store(request, tmp_path):
    kind = request.param
    path = tmp_path / ('store.zarr.zip' if kind == 'zip' else 'store.zarr')
    return kind, path, open_inner(path, kind, 'w')


# %% is_metadata_key


@pytest.mark.parametrize('key', sorted(METADATA_KEYS | {'.zmetadata', 'g/x/.zattrs'}))
def test_is_metadata_key_accepts_metadata_documents(key):
    assert stores.is_metadata_key(key)


@pytest.mark.parametrize('key', sorted(CHUNK_KEYS | {'0', 'x/c/0', 'x/zarr.jsonx'}))
def test_is_metadata_key_rejects_chunk_keys(key):
    assert not stores.is_metadata_key(key)


@pytest.mark.skipif(ZARR_3, reason='zarr.json is the zarr 3 document name')
def test_is_metadata_key_rejects_zarr_json_under_zarr_2():
    assert not stores.is_metadata_key('zarr.json')


# %% staged_metadata_store


def test_metadata_is_staged_until_flush_and_chunks_pass_through(inner_store):
    _, _, inner = inner_store
    staged = stores.staged_metadata_store(inner)
    write_group(staged)

    assert set(listed_keys(staged)) == ALL_KEYS
    assert set(listed_keys(inner)) == CHUNK_KEYS
    assert staged.written_keys == CHUNK_KEYS

    assert staged.flush_metadata() == len(METADATA_KEYS)
    assert set(listed_keys(inner)) == ALL_KEYS
    assert staged.flush_metadata() == 0


def test_repeated_chunk_write_warns_once_per_repeat(tmp_path, caplog):
    inner = open_inner(tmp_path / 'store.zarr', 'directory', 'w')
    staged = stores.staged_metadata_store(inner)
    caplog.set_level(logging.WARNING, logger='striqt.analysis')
    x = write_group(staged)
    assert caplog.records == []

    x[0:2, 0:2] = 99.0
    repeated = 'x/c/0/0' if ZARR_3 else 'x/0.0'
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert repeated in warnings[0].getMessage()
    assert staged.written_keys == CHUNK_KEYS


def test_listing_merges_staged_and_inner_keys_without_duplicates(inner_store):
    _, _, inner = inner_store
    staged = stores.staged_metadata_store(inner)
    write_group(staged)

    for _ in range(2):
        keys = listed_keys(staged)
        assert len(keys) == len(set(keys))
        assert set(keys) == ALL_KEYS
        assert set(children(staged, '')) == ROOT_CHILDREN
        assert len(children(staged, '')) == len(ROOT_CHILDREN)
        assert set(children(staged, 'x')) == ARRAY_CHILDREN
        assert len(children(staged, 'x')) == len(ARRAY_CHILDREN)
        staged.flush_metadata()


@pytest.mark.skipif(not ZARR_3, reason='list_prefix is a zarr 3 store method')
def test_list_prefix_merges_staged_and_inner_keys(tmp_path):
    from zarr.core.sync import collect_aiterator

    inner = open_inner(tmp_path / 'store.zarr', 'directory', 'w')
    staged = stores.staged_metadata_store(inner)
    write_group(staged)
    under_x = {k for k in ALL_KEYS if k.startswith('x/')}

    assert set(collect_aiterator(staged.list_prefix('x/'))) == under_x
    staged.flush_metadata()
    listed = list(collect_aiterator(staged.list_prefix('x/')))
    assert len(listed) == len(under_x)
    assert set(listed) == under_x


@pytest.mark.filterwarnings('ignore:Consolidated metadata is currently not part')
def test_closed_store_reads_back_with_consolidated_metadata(inner_store):
    kind, path, inner = inner_store
    staged = stores.staged_metadata_store(inner)
    write_group(staged)
    zarr.consolidate_metadata(staged)
    assert set(listed_keys(inner)) == CHUNK_KEYS
    staged.close()

    target = open_inner(path, kind, 'r') if kind == 'zip' else str(path)
    group = open_consolidated(target)
    assert group.attrs['a'] == 1
    assert group['x'].attrs['units'] == 'dB'
    np.testing.assert_array_equal(group['x'][:], VALUES)

    if kind == 'directory':
        import xarray as xr

        with xr.open_dataset(path, engine='zarr') as ds:
            assert ds['x'].dims == ('i', 'j')
            np.testing.assert_array_equal(ds['x'].values, VALUES)


@pytest.mark.filterwarnings('ignore:Consolidated metadata is currently not part')
def test_zip_store_has_one_entry_per_key(tmp_path):
    path = tmp_path / 'store.zarr.zip'
    staged = stores.staged_metadata_store(open_inner(path, 'zip', 'w'))
    write_group(staged)
    zarr.consolidate_metadata(staged)
    staged.close()

    assert zipfile.is_zipfile(path)
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
    assert len(names) == len(set(names))
    assert set(names) == ALL_KEYS | CONSOLIDATED_KEYS


@pytest.mark.skipif(ZARR_3, reason='the zarr 2 wrapper mirrors the `path` attribute')
def test_zarr_2_wrapper_exposes_inner_path(tmp_path):
    path = tmp_path / 'store.zarr'
    staged = stores.staged_metadata_store(open_inner(path, 'directory', 'w'))
    assert staged.path == str(path)
