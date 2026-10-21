"""the action of a plain `Sweep`: its captures through the acquire/analyze/sink pipeline"""

from __future__ import annotations as __

from ... import specs
from ..typing import PC, PS, SC, SP, SS
from . import core
from .registry import registry


@registry.runner(spec_type=specs.Sweep)
class SweepRunner(core.ActionRunner[specs.Sweep[SS, SP, SC], SS, SP, SC, PS, PC]):
    """run the captures of a `Sweep` through `iterate_captures`.

    `build_sweep` returns the document itself, so `resources['sweep_spec']` and the
    `on_source_opened` callback of `open_resources` receive the decoded object.
    """

    @classmethod
    def build_sweep(cls, spec: specs.Sweep[SS, SP, SC]) -> specs.Sweep[SS, SP, SC]:
        return spec
