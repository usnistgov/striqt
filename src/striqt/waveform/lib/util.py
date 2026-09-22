from __future__ import annotations as __

import collections
import functools
import importlib.util
from pathlib import Path
import sys
import threading
from types import ModuleType
from typing import Any, Callable, cast, TYPE_CHECKING


_lazy_import_locks = collections.defaultdict(threading.RLock)


def lazy_import(module_name: str, package=None):
    """postponed import of the module with the specified name.

    The import is not performed until the module is accessed in the code. This
    reduces the total time to import labbench by waiting to import the module
    until it is used.
    """
    # see https://docs.python.org/3/library/importlib.html#implementing-lazy-imports

    # in case the lazy_import call itself happens in a thread
    with _lazy_import_locks[module_name]:
        try:
            ret = sys.modules[module_name]
            return ret
        except KeyError:
            pass

        spec = importlib.util.find_spec(module_name, package=package)
        if spec is None or spec.loader is None:
            raise ImportError(f'no module found named "{module_name}"')
        spec.loader = importlib.util.LazyLoader(spec.loader)
        # spec.loader = _LazyLoader(spec.loader)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        spec.loader_state['lock'] = _lazy_import_locks[module_name]

    return module


if TYPE_CHECKING:
    import typing_extensions
    from .typing import LRUWrapped, P, R

    try:
        import cupy as cp  # type: ignore

        TypeIsCupy = typing_extensions.TypeIs[cp.ndarray]
    except ModuleNotFoundError:
        cp = None

    import numpy as np

    from .typing import ArrayLike, Array

else:
    np = lazy_import('numpy')
    try:
        cp = lazy_import('cupy')
    except ImportError:
        cp = None
    pickle = lazy_import('pickle')


# %% function call caching

_caches = {}


@functools.wraps(functools.lru_cache)
def lru_cache(
    maxsize: int | None = 128, typed: bool = False
) -> 'Callable[[Callable[P, R]], LRUWrapped[P, R]]':
    # presuming that the API is designed to accept only hashable types, set
    # the type hint to match the wrapped function
    func = functools.lru_cache(maxsize, typed)

    @functools.wraps(func)
    def wrap(wrapee: Callable[P, R]) -> LRUWrapped[P, R]:
        wrapped = func(wrapee)
        _caches[wrapee] = wrapped
        return wrapped  # type: ignore

    return wrap


def _cache_shelf_path() -> str:
    import platformdirs

    dir = Path(platformdirs.user_cache_dir('striqt'))
    dir.mkdir(parents=True, exist_ok=True)
    # Use version-specific cache file to avoid cross-version pickle issues
    pyver = f'{sys.version_info.major}.{sys.version_info.minor}'
    return str(dir / f'calls-py{pyver}.db')  # python 3.9 shelve wants str


_cache_shelf_disabled = False


def _discard_cache_shelf(ex: BaseException) -> None:
    """stop using the disk cache in this process, and delete its files.

    A fault in the shelf must never fail a call that has a working uncached path,
    and it does happen: the macOS dbm backend corrupts once the cache evicts, and
    the `UnicodeDecodeError` it then raises subclasses `ValueError`, so letting it
    propagate gets a machine-local fault mistaken for a bad argument several layers
    up. The files hold nothing that cannot be recomputed.
    """
    global _cache_shelf_disabled
    import glob
    import warnings

    _cache_shelf_disabled = True
    filename = _cache_shelf_path()

    for path in glob.glob(f'{filename}*'):
        Path(path).unlink(missing_ok=True)

    warnings.warn(f'deleted the unusable disk cache {filename!r} after {ex!r}')


@functools.cache
def _get_cache_shelf():
    import dbm
    import glob
    import shelve
    from threading import Lock

    filename = _cache_shelf_path()

    cache_lock = Lock()

    try:
        shelf = shelve.open(filename, writeback=True)
    except dbm.error:
        # Cache file was created with a different dbm backend (e.g., different
        # Python version). Delete and recreate.
        for path in glob.glob(f'{filename}*'):
            Path(path).unlink(missing_ok=True)
        shelf = shelve.open(filename, writeback=True)

    # sync is forced on each operation already
    shelf.__del__ = lambda: None  # ty: ignore
    return shelf, cache_lock


def _make_lru_key(func, args, kwargs) -> str:
    import base64
    import pickle

    def fix_arg(obj):
        if isinstance(obj, ModuleType):
            return repr(obj)
        else:
            return obj

    p = pickle.dumps((
        (func.__name__,),
        tuple(fix_arg(v) for v in args),
        tuple((k, fix_arg(v)) for k, v in kwargs.items()),
    ))

    # Use base64 encoding to avoid key collisions from decode(errors='replace')
    return base64.b64encode(p).decode('ascii')


def persistent_lru_cache(
    maxsize=128,
) -> Callable[[Callable[P, R]], LRUWrapped[P, R]]:
    """caches a decorated function persistently on disk.

    The cache is best-effort: a fault reading or writing the shelf discards it and
    the call proceeds uncached, so only the wrapped function's own exceptions ever
    reach its caller.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if _cache_shelf_disabled:
                return func(*args, **kwargs)

            try:
                shelf, lock = _get_cache_shelf()
            except Exception as ex:
                _discard_cache_shelf(ex)
                return func(*args, **kwargs)

            key = _make_lru_key(func, args, kwargs)

            with lock:
                try:
                    access_order = collections.OrderedDict().fromkeys(shelf.keys())
                    if key in shelf:
                        access_order.move_to_end(key)
                        hit = shelf[key]
                        # a read populates shelve's writeback cache, so skipping the
                        # sync defers the write to interpreter shutdown, where the
                        # pickle import it needs is no longer available
                        shelf.sync()
                        return hit
                except Exception as ex:
                    _discard_cache_shelf(ex)
                    return func(*args, **kwargs)

                # outside the try, so that its exceptions propagate as themselves
                result = func(*args, **kwargs)

                try:
                    # Add to cache
                    shelf[key] = result
                    access_order[key] = None
                    access_order.move_to_end(key)

                    # Evict if over limit
                    if len(access_order) > maxsize:
                        oldest_key, _ = access_order.popitem(last=False)
                        del shelf[oldest_key]

                    shelf.sync()
                except Exception as ex:
                    _discard_cache_shelf(ex)

            return result

        wrapper.__wrapped__ = func
        return wrapper

    return decorator  # pyright: ignore


def clear_caches():
    for cache in _caches.values():
        cache.cache_clear()


def qualified_name(obj: Any) -> str:
    """return the module-qualified name of a class or function"""
    return f'{obj.__module__}.{obj.__name__}'


def cache_info() -> dict[str, Any]:
    return {qualified_name(f): c.cache_info() for f, c in _caches.items()}


# %% memory management
def except_on_low_memory(threshold_bytes=500_000_000):
    import psutil

    if psutil.virtual_memory().available >= threshold_bytes:
        return

    raise MemoryError('too little memory to proceed')


# %% numbers
def ceildiv(a: int, b: int) -> int:
    """Returns ceil(a/b)."""
    return -(-a // b)


@lru_cache()
def find_float_inds(seq: tuple[str | float, ...]) -> list[bool]:
    """return a list to flag whether each element can be converted to float"""

    ret = []
    for s in seq:
        try:
            float(s)
        except ValueError:
            ret.append(False)
        else:
            ret.append(True)
    return ret
