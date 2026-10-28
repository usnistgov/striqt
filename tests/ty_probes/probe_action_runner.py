"""each registered runner class satisfies the `ActionRunner` protocol, including the
read-only `sweep` property"""

from __future__ import annotations

from typing import Any, cast

from striqt.sensor.lib.actions import (
    Cellular5GNRSweepRunner,
    ManualYFactorRunner,
    SweepRunner,
)
from striqt.sensor.lib.actions.core import ActionRunner

AnyRunner = ActionRunner[Any, Any, Any, Any, Any, Any, Any]
spec = cast(Any, None)

sweep_runner: AnyRunner = SweepRunner(spec)
yfactor_runner: AnyRunner = ManualYFactorRunner(spec)
cellular_runner: AnyRunner = Cellular5GNRSweepRunner(spec)
