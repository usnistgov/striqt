from __future__ import annotations as __

import collections
import functools
import importlib.util
import sys
import threading
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
    from .typing import CachedCallable, P, R

    try:
        import cupy as cp  # pyright: ignore[reportMissingImports]

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


_caches = {}


@functools.wraps(functools.lru_cache)
def lru_cache(
    maxsize: int | None = 128, typed: bool = False
) -> Callable[
    [Callable[P, R]],
    CachedCallable[P, R],
]:
    # presuming that the API is designed to accept only hashable types, set
    # the type hint to match the wrapped function
    func = functools.lru_cache(maxsize, typed)

    @functools.wraps(func)
    def wrap(wrapee):
        wrapped = func(wrapee)
        _caches[wrapee] = wrapped
        return wrapped

    return cast(
        'Callable[[Callable[P, R]], CachedCallable[P, R]]',
        wrap,
    )


@functools.cache
def _get_cache_shelf(func):
    import platformdirs
    import shelve
    from threading import Lock

    dir = platformdirs.user_cache_dir('striqt.waveform')
    filename = f'{dir}/{func.__name__}.db'

    cache_lock = Lock()
    return shelve.open(filename, writeback=True), cache_lock


def persistent_lru_cache(
    maxsize=128,
) -> Callable[
    [Callable[P, R]],
    CachedCallable[P, R],
]:
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            shelf, lock = _get_cache_shelf(func)
            access_order = collections.OrderedDict().fromkeys(shelf.keys())

            # Create a unique string key based on arguments
            # We include the function name to avoid collisions if sharing a file
            key = repr(hash((tuple(args), frozenset(kwargs.items()))))

            with lock:
                if key in shelf:
                    access_order.move_to_end(key)
                    return shelf[key]

                # If not in cache, compute the result
                result = func(*args, **kwargs)

                # Add to cache
                shelf[key] = result
                access_order[key] = None
                access_order.move_to_end(key)

                # Evict if over limit
                if len(access_order) > maxsize:
                    oldest_key, _ = access_order.popitem(last=False)
                    del shelf[oldest_key]

                # Ensure data is written to disk
                shelf.sync()
                return result

        wrapper.__wrapped__ = func
        return wrapper

    return decorator


def clear_caches():
    for cache in _caches.values():
        cache.cache_clear()


def cache_info() -> dict[str, Any]:
    return {f'{f.__module__}.{f.__name__}': c.cache_info() for f, c in _caches.items()}


def ceildiv(a: int, b: int) -> int:
    """Returns ceil(a/b)."""
    return -(-a // b)


def except_on_low_memory(threshold_bytes=500_000_000):
    import psutil

    if psutil.virtual_memory().available >= threshold_bytes:
        return

    raise MemoryError('too little memory to proceed')


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