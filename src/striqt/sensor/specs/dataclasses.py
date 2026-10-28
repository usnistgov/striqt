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
    """the spec types of a sensor binding."""

    source: type[SS]
    """the `Source` subclass of the binding"""
    capture: type[SC]
    """the `SensorCapture` subclass of the binding"""
    peripherals: type[SP]
    """the `Peripherals` subclass of the binding"""

    init_like: Callable[PS, Any]
    """a callable with the keyword signature of `source`; read only by the type
    checker"""
    arm_like: Callable[PC, Any]
    """a callable with the keyword signature of `capture`; read only by the type
    checker"""

    action: type[structs.ActionSpec[SS, SP, SA]] = structs.Sweep
    """the action spec type that `bind_sensor` subclasses"""
    analysis: type[SA] = dataclasses.field(
        default=cast('type[SA]', sa.specs.BuiltinAnalyses)
    )
    """the `AnalysisGroup` subclass of the ``analysis:`` block of a bound `Sweep`"""

    def __post_init__(self):
        assert issubclass(self.source, structs.Source)
        assert issubclass(self.capture, structs.SensorCapture)
        assert issubclass(self.peripherals, structs.Peripherals)
        assert issubclass(self.action, structs.ActionSpec)
        assert issubclass(self.analysis, sa.specs.AnalysisGroup)


@dataclasses.dataclass
class AcquiredIQ(sa.AcquiredIQ):
    """an acquisition from a sensor source with the metadata and conditioning of the capture.

    `x` must not change after the first access of `preprocess`.
    """

    info: structs.AcquisitionInfo
    extra_data: dict[str, Any]
    source_spec: structs.Source
    resampler: sw.ResamplerDesign
    format_path: paths.PathFormatter | None = None
    voltage_scale: sw.typing.Array | float = 1

    conjugate: tuple[bool, ...] | Literal[False] = False
    """one flag per port, True to conjugate the IQ of the port; False conjugates none"""
    signal_trigger: sa.Trigger | None = None
    """the trigger that `preprocess` runs for alignment; None selects
    `info.signal_trigger`"""
    overwrite_x: bool = False
    """True permits the resampling stage to overwrite `x`"""

    @util.cached_property
    def preprocess(self) -> corrections.Preprocessor:
        """return the `Preprocessor` that conditions `x` through the sensor signal path.

        One `Preprocessor` is created per instance. A `dataclasses.replace` copy
        starts with a `Preprocessor` of its own and every stage empty.
        """
        from ..lib.compute import corrections

        return corrections.Preprocessor(self)
