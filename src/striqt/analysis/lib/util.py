from __future__ import annotations
import array_api_compat
import contextlib
import functools
import importlib
import importlib.util
import sys
import threading
import typing
import typing_extensions


if typing.TYPE_CHECKING:
    _P = typing_extensions.ParamSpec('_P')
    _R = typing_extensions.TypeVar('_R')


def lazy_import(module_name: str, package=None):
    """postponed import of the module with the specified name.

    The import is not performed until the module is accessed in the code. This
    reduces the total time to import labbench by waiting to import the module
    until it is used.
    """

    # see https://docs.python.org/3/library/importlib.html#implementing-lazy-imports
    try:
        ret = sys.modules[module_name]
        return ret
    except KeyError:
        pass

    spec = importlib.util.find_spec(module_name, package=package)
    if spec is None:
        raise ImportError(f'no module found named "{module_name}"')
    spec.loader = importlib.util.LazyLoader(spec.loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@functools.wraps(functools.lru_cache)
def lru_cache(
    maxsize: int | None = 128, typed: bool = False
) -> typing.Callable[[typing.Callable[_P, _R]], typing.Callable[_P, _R]]:
    # presuming that the API is designed to accept only hashable types, set
    # the type hint to match the wrapped function
    return functools.lru_cache(maxsize, typed)


def pinned_array_as_cupy(x, stream=None):
    import cupy as cp

    out = cp.empty_like(x)
    out.data.copy_from_host_async(x.ctypes.data, x.data.nbytes, stream=stream)
    return out


def except_on_low_memory(threshold_bytes=500_000_000):
    try:
        import cupy as cp
    except ModuleNotFoundError:
        return
    import psutil

    if psutil.virtual_memory().available >= threshold_bytes:
        return

    raise MemoryError('too little memory to proceed')


def free_cupy_mempool():
    import cupy as cp

    mempool = cp.get_default_memory_pool()
    if mempool is not None:
        mempool.free_all_blocks()


_compute_lock = threading.RLock()


@contextlib.contextmanager
def compute_lock(array=None):
    is_cupy = array_api_compat.is_cupy_array(array)
    get_lock = array is None or is_cupy
    if get_lock:
        _compute_lock.acquire()
    yield
    if get_lock:
        _compute_lock.release()
