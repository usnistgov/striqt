from __future__ import annotations
import dataclasses
from typing import Any

from . import structs, helpers

import striqt.waveform as sw
import striqt.analysis as sa


@dataclasses.dataclass
class AcquiredIQ(sa.dataarrays.AcquiredIQ):
    """extra metadata needed for downstream analysis"""

    info: structs.AcquisitionInfo
    extra_data: dict[str, Any]
    source_spec: structs.Source
    resampler: sw.ResamplerDesign
    alias_func: helpers.PathAliasFormatter | None = None
    voltage_scale: sw.typing.Array | float = 1