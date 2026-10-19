"""Standard-agnostic binary sequence generators."""

from __future__ import annotations as __

import typing


def m_sequence(
    initial: typing.Sequence[int], taps: typing.Sequence[int], length: int
) -> list[int]:
    """return a binary LFSR sequence of `length` bits from `initial` and feedback `taps`.

    Bit ``i`` is the modulo-2 sum of bits ``i - t`` over `taps`; the first
    ``len(initial)`` bits are `initial`. Values are 0 or 1.
    """
    x = list(initial)
    for i in range(len(initial), length):
        x.append(sum(x[i - t] for t in taps) % 2)
    return x
