"""the cache decorators return `LRUWrapped` with the decorated signature intact"""

from __future__ import annotations

from typing_extensions import assert_type

import striqt.analysis as sa
import striqt.waveform as sw
from striqt.analysis.specs.helpers import lru_cache_on_converted


@sw.util.lru_cache()
def cached(a: int, *, b: str = '') -> float:
    return 0.0


@sw.util.persistent_cache(4)
def persisted(a: int, *, b: str = '') -> float:
    return 0.0


@sw.util.lru_cache()
@sw.util.persistent_cache(4)
def stacked(a: int, *, b: str = '') -> float:
    return 0.0


@lru_cache_on_converted(sa.specs.Capture)
def converted(
    capture: sa.specs.Capture, spec: sa.specs.ChannelPowerTimeSeries, n: int = 0
) -> list[float]:
    return []


def probe(cap: sa.specs.Capture, spec: sa.specs.ChannelPowerTimeSeries) -> None:
    assert_type(cached(1, b='x'), float)
    assert_type(cached.__wrapped__(1), float)
    cached(1, c=2)  # ty: ignore[unknown-argument]
    cached('x')  # ty: ignore[invalid-argument-type]

    assert_type(persisted(1), float)
    persisted(1, c=2)  # ty: ignore[unknown-argument]

    assert_type(stacked(1), float)
    stacked('x')  # ty: ignore[invalid-argument-type]

    assert_type(converted(cap, spec), list[float])
    # the converted leading argument is `Any` on purpose: the cache converts it first
    converted(object(), spec)
    converted(cap, cap)  # ty: ignore[invalid-argument-type]
    converted(cap, spec, n='x')  # ty: ignore[invalid-argument-type]
    converted(cap, spec, bogus=1)  # ty: ignore[unknown-argument]
