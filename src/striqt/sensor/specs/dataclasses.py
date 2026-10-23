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
    """the memoized conditioning stages of one acquisition.

    `resampled` is the waveform at the capture sample rate with the lead and tail
    overlaps, paired with the lead offset in samples. `unfiltered` and `filtered`
    start at the lead offset and keep the tail. `outputs` is keyed by the
    ``(filtered, aligned)`` pair after the preferences are reduced to what the
    capture and trigger allow. `aligning` is set while the trigger runs.
    """

    resampled: tuple[sw.typing.Array, int | None] | None = None
    unfiltered: sw.typing.Array | None = None
    filtered: sw.typing.Array | None = None
    shifts: sw.typing.Array | None = None
    outputs: dict[tuple[bool, bool], sw.typing.Array] = dataclasses.field(
        default_factory=dict
    )
    aligning: bool = False


@dataclasses.dataclass
class AcquiredIQ(sa.AcquiredIQ):
    """an acquisition from a sensor source with the metadata and conditioning of the capture.

    `select` conditions `x` through the sensor signal path: resample to
    `capture.sample_rate`, conjugate the ports in `conjugate`, low-pass to
    `capture.analysis_bandwidth`, trim to `capture.duration`, and shift by the lags of
    the trigger. Each stage is computed on the first call that needs the stage and
    kept in `stages`, so `x` must not change after the first call to `select`. With
    `overwrite_x` set, the first stage may consume `x` in place.

    `signal_trigger` takes precedence over `info.signal_trigger` when set. `stages`
    is not an argument of the constructor: every instance, including each copy that
    `dataclasses.replace` returns, starts with empty stages.
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
    overwrite_x: bool = False
    stages: CorrectionStages = dataclasses.field(
        default_factory=CorrectionStages, init=False, compare=False, repr=False
    )

    def select(
        self, *, prefer_filtered: bool = True, prefer_aligned: bool = True
    ) -> sw.typing.Array:
        """return `x` conditioned as `prefer_filtered` and `prefer_aligned` allow.

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
            TypeError: `capture` is not a `SensorCapture`
            RuntimeError: the trigger measurement selected ``prefer_aligned=True``
            ValueError: the acquisition is too short for the resampler or the trigger
                shifts
        """
        from ..lib.compute import corrections

        return corrections.select_corrected(
            self, prefer_filtered=prefer_filtered, prefer_aligned=prefer_aligned
        )
