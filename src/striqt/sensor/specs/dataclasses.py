from __future__ import annotations
import dataclasses
from typing import Any, Callable, cast, Generic, Literal

from . import structs, paths
from ..lib.typing import SS, SC, SP, SA, PS, PC

import striqt.waveform as sw
import striqt.analysis as sa


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
class CorrectionStages:
    """the memoized conditioning stages of one acquisition."""

    resampled: tuple[sw.typing.Array, int | None] | None = None
    """the waveform at the capture sample rate with the lead and tail overlaps,
    paired with the lead offset in samples"""
    unfiltered: sw.typing.Array | None = None
    """the resampled waveform from the lead offset, with the tail overlap"""
    filtered: sw.typing.Array | None = None
    """the low-passed waveform from the lead offset, with the tail overlap"""
    shifts: sw.typing.Array | None = None
    """the per-port trigger shift in samples"""
    outputs: dict[tuple[bool, bool], sw.typing.Array] = dataclasses.field(
        default_factory=dict
    )
    """the selected waveforms keyed by the effective ``(filtered, aligned)`` pair"""
    aligning: bool = False
    """True while the trigger measurement runs"""


@dataclasses.dataclass
class AcquiredIQ(sa.AcquiredIQ):
    """an acquisition from a sensor source with the metadata and conditioning of the capture.

    `select` conditions `x` through the sensor signal path and memoizes each stage
    in `stages`. `x` must not change after the first call to `select`.
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
    """the trigger that `select` runs for alignment; None selects
    `info.signal_trigger`"""
    overwrite_x: bool = False
    """True to let the resampling stage consume `x` in place"""
    stages: CorrectionStages = dataclasses.field(
        default_factory=CorrectionStages, init=False, compare=False, repr=False
    )
    """the memoized conditioning stages; every instance, including each
    `dataclasses.replace` copy, starts empty"""

    def select(
        self, *, prefer_filtered: bool = True, prefer_aligned: bool = True
    ) -> sw.typing.Array:
        """return `x` conditioned as `prefer_filtered` and `prefer_aligned` allow.

        The stages, in order: resample to `capture.sample_rate`, conjugate the ports
        in `conjugate`, low-pass to `capture.analysis_bandwidth`, trim to
        `capture.duration`, and shift by the trigger lags. Each stage runs on the
        first call that needs the stage.

        Args:
            prefer_filtered: True for the waveform low-passed to
                `capture.analysis_bandwidth`; the unfiltered waveform is returned
                when that bandwidth is infinite
            prefer_aligned: True for the waveform shifted by the trigger lags; the
                unaligned waveform is returned when no trigger is set

        Returns:
            shape (port, sample) array of ``round(duration * sample_rate)`` samples
            per port at `capture.sample_rate`, in the namespace of `x`. Repeated calls
            with the same effective conditioning return the same object.

        Raises:
            TypeError: if `capture` is not a `SensorCapture`
            RuntimeError: if the trigger measurement selected ``prefer_aligned=True``
            ValueError: if the acquisition is too short for the resampler or the
                trigger shifts
        """
        from ..lib.compute import corrections

        return corrections.select_corrected(
            self, prefer_filtered=prefer_filtered, prefer_aligned=prefer_aligned
        )
