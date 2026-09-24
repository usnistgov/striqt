"""`retry` returns a wrapper with the decorated signature intact"""

from __future__ import annotations

from typing_extensions import assert_type

import striqt.sensor.lib.util as su


@su.retry(ValueError, 3)
def flaky(a: int, *, b: str = '') -> float:
    return 0.0


assert_type(flaky(1, b='x'), float)
flaky(1, c=2)  # ty: ignore[unknown-argument]
flaky('x')  # ty: ignore[invalid-argument-type]
