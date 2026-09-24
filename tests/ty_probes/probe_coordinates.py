"""a registered coordinate factory keeps its (capture, spec) parameter types"""

from __future__ import annotations

import numpy as np
from typing_extensions import assert_type

import striqt.analysis as sa
from striqt.analysis.measurements import power


def probe(cap: sa.specs.Capture, spec: sa.specs.ChannelPowerTimeSeries) -> None:
    assert_type(power.time_elapsed(cap, spec), np.ndarray)
    power.time_elapsed(cap, cap)  # ty: ignore[invalid-argument-type]
    power.time_elapsed(cap, spec, 1)  # ty: ignore[too-many-positional-arguments]
    power.time_elapsed(capture=cap, spec=spec)  # ty: ignore[positional-only-parameter-as-kwarg]
