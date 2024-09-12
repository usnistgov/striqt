from __future__ import annotations
import itertools
from typing import Iterable, Any
from functools import cache


def set_cuda_mem_limit(fraction=0.75):
    try:
        import cupy
    except ModuleNotFoundError:
        return

    import psutil
    from cupy.fft.config import get_plan_cache

    #    cupy.cuda.set_allocator(None)

    available = psutil.virtual_memory().available

    cupy.get_default_memory_pool().set_limit(fraction=fraction)


def zip_offsets(seq: Iterable[Any], shifts: tuple | list, fill: Any, squeeze=True):
    """return an iterator that yields tuples of length `len(shifts)` consisting of delayed or advanced iterations of `seq`.

    Shifts that would yield an invalid index (negative or beyond the end of `seq`) are replaced with the `fill` value.
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
def import_cupy_with_fallback():
    try:
        import cupy as xp
    except ModuleNotFoundError:
        # warn only once due to @cache
        import labbench as lb
        import numpy as xp

        lb.logger.warning('cupy is not installed; falling back to cpu with numpy')

    return xp
