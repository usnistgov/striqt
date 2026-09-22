"""striqt.waveform.lib.util: the disk-backed call cache"""

from __future__ import annotations

import threading
from collections import UserDict

import pytest

from striqt.waveform.lib import util


class _Shelf(UserDict):
    """an in-memory shelf that raises from whichever operations a test names"""

    def __init__(self, *faults: str):
        self.faults = set(faults)
        self.synced = 0
        super().__init__()

    def _check(self, op: str):
        if op in self.faults:
            # what the macOS dbm backend raises once its file is corrupt, and a
            # ValueError subclass, which is how it gets mistaken for a bad argument
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
    """install a fresh shelf whose files are in `tmp_path`, never the real cache.

    `_discard_cache_shelf` unlinks what `_cache_shelf_path` names, so a test that
    reaches it would delete the developer's cache without this redirection.
    """
    fake = _Shelf()
    monkeypatch.setattr(util, '_cache_shelf_path', lambda: str(tmp_path / 'calls.db'))
    monkeypatch.setattr(util, '_cache_shelf_disabled', False)
    monkeypatch.setattr(util, '_get_cache_shelf', lambda: (fake, threading.Lock()))
    return fake


# %% persistent_lru_cache


def test_a_hit_returns_the_cached_value_without_recomputing(shelf):
    calls = []

    @util.persistent_lru_cache()
    def square(x):
        calls.append(x)
        return x * x

    assert square(3) == 9
    assert square(3) == 9
    assert calls == [3]
    # a hit syncs too, because reading an entry puts it in shelve's writeback cache
    assert shelf.synced == 2


def test_eviction_drops_the_oldest_entry(shelf):
    @util.persistent_lru_cache(maxsize=2)
    def square(x):
        return x * x

    for x in (1, 2, 3):
        square(x)

    assert len(shelf) == 2


def test_an_unreadable_shelf_falls_back_to_computing(shelf, tmp_path):
    shelf.faults.add('keys')
    corrupt = tmp_path / 'calls.db'
    corrupt.write_bytes(b'\x80 not a dbm file')

    @util.persistent_lru_cache()
    def square(x):
        return x * x

    with pytest.warns(UserWarning, match='deleted the unusable disk cache'):
        assert square(4) == 16

    # the file holds only recomputable results, so it goes rather than failing again
    assert not corrupt.exists()
    assert util._cache_shelf_disabled
    # later calls skip the shelf entirely instead of warning once per call
    assert square(5) == 25


def test_a_failing_write_still_returns_the_result(shelf):
    shelf.faults.add('setitem')

    @util.persistent_lru_cache()
    def square(x):
        return x * x

    with pytest.warns(UserWarning, match='deleted the unusable disk cache'):
        assert square(6) == 36


def test_a_failing_open_falls_back_to_computing(monkeypatch, shelf):
    def boom():
        raise OSError('no such directory')

    monkeypatch.setattr(util, '_get_cache_shelf', boom)

    @util.persistent_lru_cache()
    def square(x):
        return x * x

    with pytest.warns(UserWarning, match='deleted the unusable disk cache'):
        assert square(7) == 49


def test_the_wrapped_functions_own_error_propagates_unchanged(shelf):
    """the reason the fallback is worth having: `get_window` raises a bare ValueError
    for a bad window name, which callers label as a specification error, so a cache
    fault must never arrive as one of those and vice versa"""

    @util.persistent_lru_cache()
    def get_window(name):
        raise ValueError('Unknown window type.')

    with pytest.raises(ValueError, match='Unknown window type'):
        get_window('nosuchwindow')

    assert not util._cache_shelf_disabled
    assert len(shelf) == 0
