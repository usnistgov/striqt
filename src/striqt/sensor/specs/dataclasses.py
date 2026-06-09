from __future__ import annotations
import dataclasses
from typing import Any, Callable, Generic

from . import structs, helpers
from ..lib.typing import SS, SC, SP, PS, PC

import striqt.waveform as sw
import striqt.analysis as sa


@dataclasses.dataclass(frozen=True)
class Schema(Generic[SS, SP, SC, PS, PC]):
    source: type[SS]
    capture: type[SC]
    peripherals: type[SP]

    # these aren't actually used; they just set up the type hinting properly
    init_like: Callable[PS, Any]
    arm_like: Callable[PC, Any]

    def __post_init__(self):
        assert issubclass(self.source, structs.Source)
        assert issubclass(self.capture, structs.SensorCapture)
        assert issubclass(self.peripherals, structs.Peripherals)


@dataclasses.dataclass
class AcquiredIQ(sa.dataarrays.AcquiredIQ):
    """extra metadata needed for downstream analysis"""

    info: structs.AcquisitionInfo
    extra_data: dict[str, Any]
    source_spec: structs.Source
    resampler: sw.ResamplerDesign
    format_path: helpers.PathFormatter | None = None
    voltage_scale: sw.typing.Array | float = 1
