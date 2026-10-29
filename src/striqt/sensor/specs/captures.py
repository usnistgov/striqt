"""utilities on individual captures and the ports of a capture"""

from __future__ import annotations as __

import numbers
from collections import Counter
from typing import TYPE_CHECKING, cast, get_args

import msgspec

import striqt.analysis as sa

from . import structs

if TYPE_CHECKING:
    from ..lib.typing import SC, TypeVar

    _T = TypeVar('_T')


@sa.util.lru_cache()
def get_capture_type(
    sweep_cls: type[structs.CapturePlan],
) -> type[structs.SensorCapture]:
    if sweep_cls.sensor is not None:
        return sweep_cls.schema.capture
    else:
        fields = {f.name: f for f in msgspec.structs.fields(sweep_cls)}
        return get_args(fields['captures'].type)[0]


@sa.util.lru_cache()
def split_capture_ports(capture: SC) -> list[SC]:
    """split a multi-port capture into one single-port capture per entry of `port`.

    Each tuple-valued field is distributed by position. A capture with a scalar
    `port` is returned as the only element.

    Raises:
        ValueError: if a tuple-valued field has a length other than the number of
            ports
    """

    if isinstance(capture.port, numbers.Number):
        return [capture]
    else:
        assert isinstance(capture.port, tuple)

    remaps = [{} for i in range(len(capture.port))]

    for field in capture.__struct_fields__:
        values = getattr(capture, field)
        if not isinstance(values, tuple):
            continue
        if len(values) != len(capture.port):
            raise ValueError(
                f'{field!r} has {len(values)} values for {len(capture.port)} ports'
            )

        for remap, value in zip(remaps, values):
            remap[field] = value

    return [capture.replace(**remap) for remap in remaps]


@sa.util.lru_cache()
def pairwise_by_port(c1: SC, c2: SC | None, is_new: bool) -> list[tuple[SC, SC | None]]:
    # a list with 1 capture per port
    c1_split = split_capture_ports(c1)

    # a new capture has no predecessor
    if c2 is None or is_new:
        c2_split = len(c1_split) * [None]
    else:
        c2_split = split_capture_ports(c2)
        if len(c2_split) != len(c1_split):
            raise ValueError(f'captures span {len(c1_split)} and {len(c2_split)} ports')

    pairwise = zip(*(c1_split, c2_split))
    return list(pairwise)


def ensure_tuple(obj: _T | tuple[_T, ...], size: int | None = None) -> tuple[_T, ...]:
    if isinstance(obj, tuple):
        if size is not None and len(obj) == 1:
            return obj * size
        else:
            return obj
    elif size is None:
        return (obj,)
    else:
        return (obj,) * size


@sa.util.lru_cache()
def get_unique_ports(
    captures: tuple[SC, ...],
    loops: tuple[structs.LoopSpec, ...] | None = None,
) -> tuple[int, ...]:
    ports = set()

    if loops is not None:
        for l in loops:
            if l.field != 'port':
                continue
            looped_ports = cast(list[structs.types.Port], l.get_points())
            for p in looped_ports:
                ports |= set(ensure_tuple(p))

    for c in captures:
        ports |= set(ensure_tuple(c.port))

    return tuple(sorted(ports))


def concat_group_sizes(
    captures: tuple[structs.SensorCapture, ...], *, min_size: int = 1
) -> list[int]:
    """return the sizes of capture groups that each include every distinct capture.

    The groups are consecutive runs of `captures`, which compare by the
    `SensorCapture` fields alone. A group closes once the
    group has at least `min_size` captures and both the group and the remaining
    captures include every distinct capture. The last group takes the remainder.

    Returns:
        the group sizes in order, summing to ``len(captures)``
    """

    class C(structs.SensorCapture, frozen=True, forbid_unknown_fields=False):
        """the base capture fields, with subclass fields ignored on conversion"""

    remaining = sa.specs.helpers.convert_spec(captures, type=list[C])
    whole_set = set(remaining)
    counts = Counter(remaining)

    pending = []
    sizes = []
    count = 0

    while len(remaining) > 0:
        if count >= min_size and set(pending) == set(counts) == whole_set:
            # close the group only when both pending and remaining span every
            # distinct capture
            sizes.append(count)
            count = 0
            pending = []

        count += 1
        new = remaining.pop(0)
        pending.append(new)

        counts[new] -= 1
        if counts[new] == 0:
            del counts[new]

    if count > 0:
        sizes.append(count)

    return sizes
