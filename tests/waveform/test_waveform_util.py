"""striqt.waveform.lib.util: lazy imports and the disk-backed call cache"""

from __future__ import annotations

import importlib
import json
import pathlib
import sys
import threading
from collections import UserDict

import pytest

from striqt.waveform.lib import util


class _Shelf(UserDict):
    """an in-memory shelf that raises from the operations named in `faults`"""

    def __init__(self, *faults: str):
        self.faults = set(faults)
        self.synced = 0
        super().__init__()

    def _check(self, op: str):
        if op in self.faults:
            # what the macOS dbm backend raises once the file is corrupt.
            # UnicodeDecodeError is a ValueError subclass, so a caller mistakes the
            # fault for a bad argument
            raise UnicodeDecodeError('utf-8', b'\x80', 0, 1, 'invalid start byte')

    def keys(self):
        self._check('keys')
        return super().keys()

    def __setitem__(self, key, value):
        self._check('setitem')
        super().__setitem__(key, value)

    def sync(self):
        self._check('sync')
        self.synced += 1


@pytest.fixture
def shelf(monkeypatch, tmp_path):
    """install a fresh shelf with files under `tmp_path`, never the real cache.

    `_discard_cache_shelf` unlinks the file that `_cache_shelf_path` names, so without
    this redirection a test that reaches `_discard_cache_shelf` would delete the cache
    of the developer.
    """
    fake = _Shelf()
    monkeypatch.setattr(util, '_cache_shelf_path', lambda: str(tmp_path / 'calls.db'))
    monkeypatch.setattr(util, '_cache_shelf_disabled', False)
    monkeypatch.setattr(util, '_get_cache_shelf', lambda: (fake, threading.Lock()))
    return fake


# %% cache_dir


def test_cache_dir_follows_the_environment_override(monkeypatch, tmp_path):
    monkeypatch.setenv('STRIQT_CACHE_DIR', str(tmp_path / 'caches'))

    assert util.cache_dir() == tmp_path / 'caches'
    assert (tmp_path / 'caches').is_dir()


def test_cache_dir_defaults_to_the_platformdirs_user_cache(monkeypatch):
    import platformdirs

    monkeypatch.setenv('STRIQT_CACHE_DIR', '')

    assert str(util.cache_dir()) == platformdirs.user_cache_dir('striqt')


def test_shelf_path_is_per_python_version_under_cache_dir(monkeypatch, tmp_path):
    monkeypatch.setenv('STRIQT_CACHE_DIR', str(tmp_path))

    path = util._cache_shelf_path()

    assert isinstance(path, str)  # python 3.9 shelve rejects Path
    assert path == str(
        tmp_path / f'calls-py{sys.version_info.major}.{sys.version_info.minor}.db'
    )


def test_the_session_isolates_the_disk_caches_in_a_temporary_directory():
    """the conftest sets ``STRIQT_CACHE_DIR`` before collection, when the strategy
    modules validate sweeps and design windows, so the shelf of the developer stays
    untouched even by code that bypasses the in-memory shelf"""
    import os
    import tempfile

    import platformdirs

    cache_dir = util.cache_dir()

    assert str(cache_dir) == os.environ['STRIQT_CACHE_DIR']
    assert cache_dir != pathlib.Path(platformdirs.user_cache_dir('striqt'))
    assert cache_dir.resolve().is_relative_to(
        pathlib.Path(tempfile.gettempdir()).resolve()
    )


# %% persistent_cache


def test_a_hit_returns_the_cached_value_without_recomputing(shelf):
    calls = []

    @util.persistent_cache()
    def square(x):
        calls.append(x)
        return x * x

    assert square(3) == 9
    assert square(3) == 9
    assert calls == [3]
    # a hit syncs too, because reading an entry puts it in shelve's writeback cache
    assert shelf.synced == 2


def test_eviction_drops_the_oldest_entry(shelf):
    @util.persistent_cache(maxsize=2)
    def square(x):
        return x * x

    for x in (1, 2, 3):
        square(x)

    assert len(shelf) == 2


def test_an_unreadable_shelf_falls_back_to_computing(shelf, tmp_path):
    shelf.faults.add('keys')
    corrupt = tmp_path / 'calls.db'
    corrupt.write_bytes(b'\x80 not a dbm file')

    @util.persistent_cache()
    def square(x):
        return x * x

    with pytest.warns(UserWarning, match='deleted the unusable disk cache'):
        assert square(4) == 16

    # the file contains only recomputable results, so the file is deleted rather
    # than failing again
    assert not corrupt.exists()
    assert util._cache_shelf_disabled
    # later calls skip the shelf entirely instead of warning once per call
    assert square(5) == 25


def test_a_failing_write_still_returns_the_result(shelf):
    shelf.faults.add('setitem')

    @util.persistent_cache()
    def square(x):
        return x * x

    with pytest.warns(UserWarning, match='deleted the unusable disk cache'):
        assert square(6) == 36


def test_a_failing_open_falls_back_to_computing(monkeypatch, shelf):
    def boom():
        raise OSError('no such directory')

    monkeypatch.setattr(util, '_get_cache_shelf', boom)

    @util.persistent_cache()
    def square(x):
        return x * x

    with pytest.warns(UserWarning, match='deleted the unusable disk cache'):
        assert square(7) == 49


def test_the_wrapped_functions_own_error_propagates_unchanged(shelf):
    """`get_window` raises a bare `ValueError` for a bad window name, and callers
    label that error a specification error. A cache fault must not arrive as a
    `ValueError`, and a `ValueError` must not arrive as a cache fault."""

    @util.persistent_cache()
    def get_window(name):
        raise ValueError('Unknown window type.')

    with pytest.raises(ValueError, match='Unknown window type'):
        get_window('nosuchwindow')

    assert not util._cache_shelf_disabled
    assert len(shelf) == 0


# %% lazy_import


def test_lazy_import_binds_a_submodule_onto_its_imported_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`import json.tool` leaves `json.tool` resolvable, so `lazy_import` must too"""
    monkeypatch.delitem(sys.modules, 'json.tool', raising=False)
    monkeypatch.delattr(json, 'tool', raising=False)

    module = util.lazy_import('json.tool')

    assert json.tool is module
    # the binding must not have reified the module
    assert type(module).__name__ == '_LazyModule'
    assert importlib.import_module('json.tool') is module


def test_lazy_import_of_dask_array_leaves_the_dask_attribute_resolvable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """array_api_compat 1.11 tests ``isinstance(x, dask.array.Array)`` after a bare
    ``import dask.array``. That import returns the lazy entry without binding the
    `array` attribute."""
    dask = pytest.importorskip('dask')
    monkeypatch.delitem(sys.modules, 'dask.array', raising=False)
    monkeypatch.delattr(dask, 'array', raising=False)

    module = util.lazy_import('dask.array')

    import dask.array as da

    assert dask.array is module
    assert da is module
