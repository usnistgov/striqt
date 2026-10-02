"""utilities on individual captures and the ports they span, shared across the sensor
package.
"""

from __future__ import annotations as __

from collections import Counter
import numbers
from typing import cast, get_args, TYPE_CHECKING

import msgspec

import striqt.analysis as sa

from . import structs


if TYPE_CHECKING:
    from ..lib.typing import SC, TypeVar

    _T = TypeVar('_T')


@sa.util.lru_cache()
def get_capture_type(sweep_cls: type[structs.Sweep]) -> type[structs.SensorCapture]:
    if sweep_cls.sensor is not None:
        return sweep_cls.schema.capture
    else:
        fields = {f.name: f for f in msgspec.structs.fields(sweep_cls)}
        return get_args(fields['captures'].type)[0]


@sa.util.lru_cache()
def split_capture_ports(capture: SC) -> list[SC]:
    """split a multi-channel capture into a list of single-channel captures.

    If capture is not a multi-channel capture (its channel field is just a number),
    then the returned list will be [capture].
    """

    if isinstance(capture.port, numbers.Number):
        return [capture]
    else:
        assert isinstance(capture.port, tuple)

    remaps = [dict() for i in range(len(capture.port))]

    for field in capture.__struct_fields__:
        values = getattr(capture, field)
        if not isinstance(values, tuple):
            continue

        for remap, value in zip(remaps, values):
            remap[field] = value

    return [capture.replace(**remap) for remap in remaps]


@sa.util.lru_cache()
def pairwise_by_port(c1: SC, c2: SC | None, is_new: bool) -> list[tuple[SC, SC | None]]:
    # a list with 1 capture per port
    c1_split = split_capture_ports(c1)

    # any changes to the port index
    if c2 is None or is_new:
        c2_split = len(c1_split) * [None]
    else:
        c2_split = split_capture_ports(c2)

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
    """return the minimum sizes of groups of captures that can be concatenated.

    This is important, because some channel analysis results produce a different
    shape depending on (sample_rate, analysis_bandwidth, duration).

    Returns:
        The list l of sizes of each group such that sum(l) == len(captures)
    """

    class C(structs.SensorCapture, frozen=True, forbid_unknown_fields=False):
        """minimal capture fields that safely ignore fields from subclasses"""

    remaining = sa.specs.helpers.convert_spec(captures, type=list[C])
    whole_set = set(remaining)
    counts = Counter(remaining)

    pending = []
    sizes = []
    count = 0

    while len(remaining) > 0:
        if count >= min_size and set(pending) == set(counts) == whole_set:
            # make sure that the pending and remaining captures
            # will result in equivalent shapes when concatenated
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
