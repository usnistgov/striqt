from __future__ import annotations
import dataclasses
from typing import Any, Callable, Generic, Literal

from . import structs, paths
from ..lib.typing import SS, SC, SP, PS, PC

import striqt.waveform as sw
import striqt.analysis as sa


@dataclasses.dataclass(frozen=True)
class Schema(Generic[SS, SP, SC, PS, PC]):
    """the spec types of a sensor binding.

    `action` is the sweep spec type that a bound sweep class subclasses, and so
    selects which action runs the sweep.
    """

    source: type[SS]
    capture: type[SC]
    peripherals: type[SP]

    # unused at runtime; these carry the parameter types for the type checker
    init_like: Callable[PS, Any]
    arm_like: Callable[PC, Any]

    action: type[structs.Sweep[SS, SP, SC]] = structs.Sweep

    def __post_init__(self):
        assert issubclass(self.source, structs.Source)
        assert issubclass(self.capture, structs.SensorCapture)
        assert issubclass(self.peripherals, structs.Peripherals)
        assert issubclass(self.action, structs.Sweep)


@dataclasses.dataclass
class AcquiredIQ(sa.dataarrays.AcquiredIQ):
    """extra metadata needed for downstream analysis"""

    info: structs.AcquisitionInfo
    extra_data: dict[str, Any]
    source_spec: structs.Source
    resampler: sw.ResamplerDesign
    format_path: paths.PathFormatter | None = None
    voltage_scale: sw.typing.Array | float = 1

    # whether to evaluate the complex conjugate of each port
    conjugate: tuple[bool, ...] | Literal[False] = False
