from __future__ import annotations
import importlib
import itertools
import typing
from functools import cache
import sys

TGen = type[typing.Any]


def lazy_import(module_name: str):
    """postponed import of the module with the specified name.

    The import is not performed until the module is accessed in the code. This
    reduces the total time to import the module by waiting to import submodules
    until they are used.
    """

    # see https://docs.python.org/3/library/importlib.html#implementing-lazy-imports
    try:
        ret = sys.modules[module_name]
        return ret
    except KeyError:
        pass

    import importlib.util

    spec = importlib.util.find_spec(module_name)
    if spec is None:
        raise ImportError(f'no module found named "{module_name}"')
    spec.loader = importlib.util.LazyLoader(spec.loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def set_cuda_mem_limit(fraction=0.75):
    try:
        import cupy
    except ModuleNotFoundError:
        return

    # Alternative: select an absolute amount of memory
    #
    # import psutil
    # available = psutil.virtual_memory().available

    cupy.get_default_memory_pool().set_limit(fraction=fraction)


def concurrently_with_fg(
    calls: dict[str, callable] = {}, flatten=True
) -> tuple[typing.Any, typing.Any]:
    """runs foreground() in the current thread, and lb.concurrently(**background) in another thread"""
    from concurrent.futures import ThreadPoolExecutor
    import labbench as lb

    # split to foreground and backround
    pairs = iter(calls.items())
    if len(calls) > 0:
        fg_name, foreground = next(pairs)
        background = dict(pairs)
    else:
        fg_name, foreground = None, None
        background = {}

    executor = ThreadPoolExecutor()
    exc_list = []
    result = {}

    with executor:
        bg_future = executor.submit(lb.concurrently, **background, flatten=flatten)

        try:
            if foreground is not None:
                result[fg_name] = foreground()
        except BaseException as ex:
            if isinstance(ex, lb.util.ConcurrentException):
                exc_list.extend(ex.thread_exceptions)
            else:
                exc_list.append(ex)

        try:
            result.update(bg_future.result())
        except BaseException as ex:
            if isinstance(ex, lb.util.ConcurrentException):
                exc_list.extend(ex.thread_exceptions)
            else:
                exc_list.append(ex)

    if len(exc_list) == 0:
        pass
    elif len(exc_list) == 1:
        raise exc_list[0]
    else:
        ex = lb.util.ConcurrentException('multiple exceptions raised')
        ex.thread_exceptions = exc_list
        raise ex

    return result


def zip_offsets(
    seq: typing.Iterable[TGen],
    shifts: tuple[int, ...] | list[int],
    fill: typing.Any,
    *,
    squeeze=True,
) -> typing.Generator[tuple[TGen, ...]]:
    """a generator that yields from `seq` at multiple index shifts.

    Shifts that would yield an invalid index (i.e., before or beyond the end of `seq`) are
    replaced with `fill`.

    Args:
        seq: iterable of values to shift
        shifts: a sequence of integers indicating the index shifts to apply
        fill: the value to yield before or after `seq` yields values

    Yields:
        tuples of length `shifts` composed of delayed/advanced values from `seq`
    """
    lo = min(shifts)

    if lo < 0:
        # prepend as necessary to allow [None] values
        seq = itertools.chain(-lo * [fill], seq)
    else:
        lo = 0

    iters = []

    for it, shift in zip(itertools.tee(seq, len(shifts)), shifts):
        sl = itertools.islice(it, shift - lo, None)
        iters.append(sl)

    if squeeze and len(shifts) == 1:
        return iters[0]
    else:
        return itertools.zip_longest(*iters, fillvalue=fill)


@cache
def configure_cupy():
    import cupy

    # the FFT plan sets up large caches that don't help us
    cupy.fft.config.get_plan_cache().set_size(0)
    cupy.cuda.set_pinned_memory_allocator(None)
