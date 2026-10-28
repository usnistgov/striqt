from __future__ import annotations
import dataclasses
from typing import Any, Callable, cast, Generic, Literal, TYPE_CHECKING

from . import structs, paths
from ..lib import util
from ..lib.typing import SS, SC, SP, SA, PS, PC

import striqt.waveform as sw
import striqt.analysis as sa

if TYPE_CHECKING:
    from ..lib.compute import corrections


@dataclasses.dataclass(frozen=True)
class Schema(Generic[SS, SP, SC, SA, PS, PC]):
    """the spec types of a sensor binding.

    `action` is the action spec type that `bind_sensor` subclasses. The
    `ActionRunner` registered for that type runs the bound spec. `analysis` is the
    `AnalysisGroup` subclass that the ``analysis:`` block of the bound `Sweep` is
    decoded against, and the type of the analysis that the other actions generate.
    """

    source: type[SS]
    capture: type[SC]
    peripherals: type[SP]

    # unused at runtime; these carry the parameter types for the type checker
    init_like: Callable[PS, Any]
    arm_like: Callable[PC, Any]

    action: type[structs.ActionSpec[SS, SP, SA]] = structs.Sweep
    analysis: type[SA] = dataclasses.field(
        default=cast('type[SA]', sa.specs.BuiltinAnalyses)
    )

    def __post_init__(self):
        assert issubclass(self.source, structs.Source)
        assert issubclass(self.capture, structs.SensorCapture)
        assert issubclass(self.peripherals, structs.Peripherals)
        assert issubclass(self.action, structs.ActionSpec)
        assert issubclass(self.analysis, sa.specs.AnalysisGroup)


@dataclasses.dataclass
class AcquiredIQ(sa.AcquiredIQ):
    """an acquisition from a sensor source with the metadata and conditioning of the capture.

    `preprocess` conditions `x` through the sensor signal path and memoizes each
    stage. `x` must not change after the first call to `preprocess`.
    """

    info: structs.AcquisitionInfo
    extra_data: dict[str, Any]
    source_spec: structs.Source
    resampler: sw.ResamplerDesign
    format_path: paths.PathFormatter | None = None
    voltage_scale: sw.typing.Array | float = 1

    # whether to evaluate the complex conjugate of each port
    conjugate: tuple[bool, ...] | Literal[False] = False
    signal_trigger: sa.Trigger | None = None
    """the trigger that `preprocess` runs for alignment; None selects
    `info.signal_trigger`"""
    overwrite_x: bool = False
    """True to let the resampling stage consume `x` in place"""

    @util.cached_property
    def preprocess(self) -> corrections.Preprocessor:
        """the sensor `Preprocessor` of this acquisition, one per instance.

        A `dataclasses.replace` copy, including the copy that a controller returns
        for a reused acquisition, starts with a `Preprocessor` of its own and every
        stage empty.
        """
        from ..lib.compute import corrections

        return corrections.Preprocessor(self)
