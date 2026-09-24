"""an unknown keyword through `**kwargs: Unpack[TypedDict]` should be rejected.

ty 0.0.81 lowers the unpacked TypedDict to `**kwargs: object`, so these lines pass
unflagged and the ignores below are reported unused; `tests/test_typing.py` holds this
probe as a strict xfail until the `ty` pin moves.
"""

from __future__ import annotations

import striqt.sensor as ss
from striqt.sensor.lib import execute, resources
from striqt.sensor.lib.compute import corrections


def probe(cap: ss.specs.SensorCapture, res: resources.Resources) -> None:
    corrections.design_resampler(cap, 1e6, bogus=1)  # ty: ignore[unknown-argument]
    execute.iterate_sweep(res, bogus=1)  # ty: ignore[unknown-argument]
