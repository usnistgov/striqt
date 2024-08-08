from __future__ import annotations
import itertools
from typing import Iterable, Any

__all__ = ['zip_offsets']


def zip_offsets(seq: Iterable[Any], shifts: tuple | list, fill: Any):
    """return an iterator that yields tuples of length len(shifts) - shifted values in seq.

    Shifts to indexes below `0`, or greater than the length of `seq` return `fill`.
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

    return itertools.zip_longest(*iters, fillvalue=fill)
