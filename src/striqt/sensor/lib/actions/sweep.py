"""the action that runs a plain `Sweep` as written"""

from __future__ import annotations as __

from collections.abc import Generator
from typing import Generic

from striqt.analysis.lib.util import xr

from ... import specs
from .. import compute
from ..resources import Resources
from ..sinks import NoSink, SinkBase, ZarrCaptureSink
from ..typing import PC, PS, SA, SC, SP, SS
from . import core
from .registry import registry


@registry.runner(spec_type=specs.Sweep)
class SweepRunner(Generic[SS, SP, SC, SA, PS, PC]):
    """run the captures of a `Sweep` through `iterate_captures`.

    `spec` and `sweep` are the same object. ``resources['sweep_spec']`` and the
    `on_source_opened` callbacks of `open_resources` receive the spec unchanged. The
    output is a `ZarrCaptureSink` at `sink.path`. A path of None writes nothing.
    """

    spec: specs.Sweep[SS, SP, SC, SA]

    def __init__(self, spec: specs.Sweep[SS, SP, SC, SA]) -> None:
        self.spec = spec

    @property
    def sweep(self) -> specs.Sweep[SS, SP, SC, SA]:
        return self.spec

    def open_sink(self, format_path: specs.paths.PathFormatter | None) -> SinkBase[SC]:
        if self.sweep.sink.path is None:
            return NoSink(self.sweep)
        return ZarrCaptureSink(self.sweep, format_path)

    def iterate(
        self,
        resources: Resources[SS, SP, SC, SA, PS, PC],
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
