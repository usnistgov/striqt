"""`**kwargs: Unpack[TypedDict]` forwarding keeps the value types of the target"""

from __future__ import annotations

import striqt.sensor as ss
from striqt.sensor.lib import execute, resources
from striqt.sensor.lib.compute import corrections


def probe(cap: ss.specs.SensorCapture, res: resources.Resources) -> None:
    corrections.design_resampler(cap, 1e6, bw_lo='x')  # ty: ignore[invalid-argument-type]
    corrections.design_resampler(cap, 'x')  # ty: ignore[invalid-argument-type]
    execute.iterate_sweep(res, sink=3)  # ty: ignore[invalid-argument-type]
    execute.iterate_sweep(res, always_yield='x')  # ty: ignore[invalid-argument-type]
