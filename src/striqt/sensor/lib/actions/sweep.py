"""the action of a plain `Sweep`: its captures through the acquire/analyze/sink pipeline"""

from __future__ import annotations as __

from collections.abc import Generator
from typing import Generic

from striqt.analysis.lib.util import xr

from ... import specs
from .. import compute
from ..resources import Resources
from ..sinks import SinkBase
from ..typing import PC, PS, SC, SP, SS
from . import core
from .registry import registry


@registry.runner(spec_type=specs.Sweep)
class SweepRunner(Generic[SS, SP, SC, PS, PC]):
    """run the captures of a `Sweep` through `iterate_captures`.

    The document is the sweep that runs, so `resources['sweep_spec']` and the
    `on_source_opened` callback of `open_resources` receive the decoded object. The
    sink class is the binding's.
    """

    spec: specs.Sweep[SS, SP, SC]
    sweep: specs.Sweep[SS, SP, SC]
    sink_cls: type[SinkBase[SC]]

    def __init__(self, spec: specs.Sweep[SS, SP, SC]) -> None:
        self.spec = spec
        self.sweep = spec
        self.sink_cls = type(spec).sensor.sink_cls

    def iterate(
        self,
        resources: Resources[SS, SP, SC, PS, PC],
        *,
        always_yield: bool = False,
        yield_values: bool = True,
        loop: bool = False,
    ) -> Generator[xr.Dataset | compute.DelayedDataset | None]:
        return core.iterate_captures(
            resources,
            always_yield=always_yield,
            yield_values=yield_values,
            loop=loop,
        )
