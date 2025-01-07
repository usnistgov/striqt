import importlib
import importlib.util
import sys


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


def pinned_array_as_cupy(x, stream=None):
    import cupy as cp
    out = cp.empty_like(x)
    out.data.copy_from_host_async(x.ctypes.data, x.data.nbytes, stream=stream)
    return out


def free_mempool_on_low_memory(threshold_bytes=1_000_000_000):
    import psutil

    if psutil.virtual_memory().available >= threshold_bytes:
        return
    else:
        import labbench as lb
        lb.logger.warning(f'low_memory, freeing GPU caches (threshold {threshold_bytes/1e9:0.1f} GB)')
    
    import cupy as cp
    mempool = cp.get_default_memory_pool()
    mempool.free_all_blocks()