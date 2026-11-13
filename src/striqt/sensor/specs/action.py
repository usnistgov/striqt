"""validation of an action spec after construction.

A spec is validated at the end of `__post_init__`, and a constructed spec is
therefore a validated one. Each error is a `SpecValidationError` located at the
failing field of the spec, with the loop point for an expanded capture.
"""

from __future__ import annotations as __

import contextlib
import math
from collections import Counter
from collections.abc import Generator, Iterator
from typing import TYPE_CHECKING, Any, Callable

import striqt.analysis as sa
import striqt.waveform as sw
from striqt.analysis.specs.helpers import SpecValidationError, validation_path

from . import sequencing, structs, types

if TYPE_CHECKING:
    from ..lib.typing import SC


def validate_action(spec: structs.ActionSpec[Any, Any, Any]) -> None:
    """check that the action runner for `spec` derives a `Sweep` from `spec`.

    An unbound spec, or a spec type with no runner, is not checked.

    Raises:
        msgspec.ValidationError: if a field of `spec` is rejected when the `Sweep` is
            derived
    """
    if type(spec).sweep_cls is None:
        return
    try:
        from ..lib.actions.registry import registry
    except ImportError:
        return
    try:
        runner_cls = registry.lookup(spec)
    except TypeError:
        return
    _ = runner_cls(spec).sweep


def null_sink(
    spec: structs.ActionSpec[Any, Any, Any],
) -> structs.ActionSpec[Any, Any, Any]:
    """return `spec` with ``sink.path`` None, so that a run of the spec writes no store."""
    return spec.replace(sink=spec.sink.replace(path=None))


def analysis_group_type(
    spec: structs.ActionSpec[Any, Any, Any],
) -> type[sa.specs.AnalysisGroup]:
    """return the analysis group type of the binding of `spec`.

    The group is `schema.analysis` of the bound class, or `BuiltinAnalyses` for an
    instance of an unbound spec class.
    """
    schema = type(spec).schema
    if schema is None:
        return structs.BuiltinAnalyses
    return schema.analysis


def validate_sweep(
    sweep: structs.Sweep[Any, Any, SC, Any],
    source_id: types.SourceID | None = None,
) -> None:
    """validate the loops, measurement names, analysis, and signal path of `sweep`.

    The checks, in order:

    * the `loops` entries: a `repeat` only outermost, and at most one loop per field;
    * a loop over a field that the `captures` entries set to different values;
    * a measurement name that shadows a capture field;
    * `options.reuse_iq` together with a capture `burst` above 1, and a capture
      `burst` above 1 with a `duration` that is not a whole number of
      `source.trigger_strobe` periods;
    * each unique (capture, analysis) combination against the registered
      measurements;
    * each unique capture against the conditioning signal path: trigger lookup,
      resampler design, analysis filter, and LO shift.

    The captures checked are those that `loop_captures` expands for `source_id`.

    Args:
        sweep: the sweep to check
        source_id: hex id of the open source, selecting the `adjust_captures` block
            applied on top of ``'defaults'``; `None` applies ``'defaults'`` alone

    Raises:
        SpecValidationError: at the offending path, with an expanded capture located
            by the `sweep.captures` entry and the loop point that produced the
            capture
    """
    _validate_loops(sweep.loops)
    _validate_loop_capture_collisions(sweep.loops, sweep.captures)
    _validate_measurement_names(sweep)
    _validate_bursts(sweep, source_id)
    _validate_sweep_analysis(sweep, source_id)
    _validate_sweep_corrections(sweep, source_id)


@sa.util.lru_cache()
def _validate_loops(loops: tuple[structs.LoopSpec, ...]) -> None:
    if len(loops) == 0:
        return

    # an outermost repeat is legal, so the offset reports the index in the sweep
    # rather than in the slice
    offset = 1 if loops[0].field is None else 0
    fields = [l.field for l in loops[offset:]]

    counts = Counter(fields)

    if None in counts:
        index = offset + fields.index(None)
        raise SpecValidationError(
            'Expected a `repeat` loop only as the outermost (first) entry',
            ('.loops', f'[{index}]'),
        )

    common = counts.most_common(1)

    if len(common) == 0:
        return

    (which, howmany), *_ = common
    if howmany > 1:
        repeated = [offset + i for i, f in enumerate(fields) if f == which]
        first = loops[repeated[0]]
        raise SpecValidationError(
            f'Expected at most one loop over field `{which}`, which is already '
            f'looped in `{first.isin}` at $.loops[{repeated[0]}]',
            ('.loops', f'[{repeated[1]}]'),
        )


_MISSING = object()


@sa.util.lru_cache()
def _validate_loop_capture_collisions(
    loops: tuple[structs.LoopSpec, ...], captures: tuple[structs.SensorCapture, ...]
) -> None:
    """reject a loop over a field that the `captures:` entries set to different values.

    Analysis loops and a loop over a field that the capture class lacks are not
    rejected.

    Raises:
        SpecValidationError: at the loop, locating ``captures[0]`` and the first
            later entry with a different value in the looped field
    """
    if len(captures) < 2:
        return

    for index, loop in enumerate(loops):
        if loop.field is None or loop.isin != 'capture':
            continue

        values = [getattr(c, loop.field, _MISSING) for c in captures]
        if any(v is _MISSING for v in values):
            # _build_loop_points_dict reports an unknown loop field later
            continue

        for j, value in enumerate(values[1:], start=1):
            if value != values[0]:
                raise SpecValidationError(
                    f'Expected field `{loop.field}` to be looped or set per '
                    'capture, not both',
                    ('.loops', f'[{index}]'),
                    ('.captures[0]', f'.captures[{j}]'),
                )


def _validate_measurement_names(sweep: structs.Sweep) -> None:
    """reject a measurement named like a capture field.

    A sweep with neither `captures` nor a sensor binding is not checked.

    Raises:
        SpecValidationError: at ``$.analysis``, for the first such measurement
    """
    if len(sweep.captures) > 0:
        coord_fields = set(sweep.captures[0].__struct_fields__)
    elif sweep.schema is not None:
        coord_fields = set(sweep.schema.capture.__struct_fields__)
    else:
        return

    invalid = set(sweep.analysis.__struct_fields__) & coord_fields
    if len(invalid) > 0:
        raise SpecValidationError(
            f'Object contains measurement `{min(invalid)}`, which '
            'shadows a capture field of the same name',
            ('.analysis',),
        )


def _validate_bursts(
    sweep: structs.Sweep[Any, Any, SC, Any],
    source_id: types.SourceID | None = None,
) -> None:
    """check each expanded capture with `burst` above 1 against the sweep.

    The rules, each located at the first capture that violates the rule:

    * `options.reuse_iq` must be false;
    * with a `source.trigger_strobe`, `duration` must be a whole number of strobe
      periods.

    Raises:
        SpecValidationError: at ``$.options.reuse_iq`` or at the `duration` of the
            capture
    """
    strobe = sweep.source.trigger_strobe or 0
    for capture, origin in sequencing.loop_capture_origins(sweep, source_id).items():
        if capture.burst == 1:
            continue
        locations = sequencing.describe_capture_origin(sweep.loops, capture, origin)
        if sweep.options.reuse_iq:
            raise SpecValidationError(
                'Expected false, since the reuse cache holds one acquisition and a '
                'capture burst produces several',
                ('.options', '.reuse_iq'),
                locations,
            )
        if strobe and not sw.isroundmod(capture.duration, strobe):
            raise SpecValidationError(
                'Expected a whole number of `trigger_strobe` periods, since the '
                'segments of a burst start on the strobe grid',
                ('.duration',),
                locations,
            )


def _validate_sweep_analysis(
    sweep: structs.Sweep[Any, Any, SC, Any],
    source_id: types.SourceID | None = None,
) -> None:
    """validate each unique (capture, analysis) combination run by `sweep`.

    A sweep with no measurement enabled in `analysis` is not expanded or checked.

    Raises:
        SpecValidationError: for the first capture that a measurement rejects,
            located by the `sweep.captures` entry and the loop point that produced
            the capture
    """

    if len(sweep.analysis.to_dict()) == 0:
        return

    origins = _unique_capture_origins(
        sweep, source_id, sa.specs.helpers.to_analysis_capture
    )
    for capture, analysis, origin in origins:
        with _located_in_sweep(sweep.loops, capture, origin):
            sa.registry.validate(capture, analysis)


def _validate_sweep_corrections(
    sweep: structs.Sweep[Any, Any, SC, Any],
    source_id: types.SourceID | None = None,
) -> None:
    """reject each unique capture of `sweep` that conditioning cannot process.

    The checks are the trigger lookup, the analysis filter band of each capture
    with a finite `analysis_bandwidth`, the resampler design, and the LO shift of
    the overlap-add resampler when `USE_OARESAMPLE` is set. The resampler and LO
    shift checks are not run on a `FileCapture`.

    Raises:
        SpecValidationError: at ``$.source.signal_trigger`` when `sweep.analysis`
            lacks the measurement that the trigger runs, or otherwise at the
            `analysis_bandwidth`, `sample_rate`, or `lo_shift` field of the first
            rejected capture, located by the `sweep.captures` entry and the loop
            point that produced the capture
    """
    from ..lib import compute
    from ..lib.compute import corrections

    if sweep.source.signal_trigger is not None:
        with validation_path('.source', '.signal_trigger'):
            compute.get_trigger_from_spec(sweep.source, sweep.analysis)

    def key(capture: SC) -> structs.SensorCapture:
        return sa.specs.helpers.convert_spec_cached(structs.SensorCapture, capture)

    for capture, _, origin in _unique_capture_origins(sweep, source_id, key):
        with _located_in_sweep(sweep.loops, capture, origin):
            if math.isfinite(capture.analysis_bandwidth):
                with validation_path(
                    *_capture_field_path(origin, 'analysis_bandwidth')
                ):
                    corrections.validate_fir_band(capture)

            if isinstance(capture, structs.FileCapture):
                continue

            with validation_path(*_capture_field_path(origin, 'sample_rate')):
                design = compute.design_resampler(
                    capture, sweep.source.master_clock_rate
                )

            if compute.needs_resample(design, capture) and corrections.USE_OARESAMPLE:
                with validation_path(*_capture_field_path(origin, 'lo_shift')):
                    corrections.validate_oaresample_shift(design)


def _unique_capture_origins(
    sweep: structs.Sweep[Any, Any, SC, Any],
    source_id: types.SourceID | None,
    key: Callable[[SC], Any],
) -> Iterator[tuple[SC, structs.AnalysisGroup, sequencing.CaptureOrigin]]:
    """iterate the distinct ``(key(capture), analysis)`` combinations of `sweep`.

    The analysis is `sweep.analysis` with the `adjust_analysis` of the capture
    applied. Nothing is yielded for a sweep with neither `captures` nor a sensor
    binding.

    Yields:
        ``(capture, analysis, origin)`` for the first capture of each combination,
        in expansion order, until the captures are exhausted
    """
    if len(sweep.captures) == 0 and sweep.sensor is None:
        # loop_captures rejects this combination, and such a sweep stays constructible
        return

    seen = set()

    for capture, origin in sequencing.loop_capture_origins(sweep, source_id).items():
        analysis = sequencing.adjust_analysis(sweep.analysis, capture.adjust_analysis)
        dedupe = (key(capture), analysis)

        if dedupe in seen:
            continue

        seen.add(dedupe)
        yield capture, analysis, origin


@contextlib.contextmanager
def _located_in_sweep(
    loops: tuple[structs.LoopSpec, ...],
    capture: structs.SensorCapture,
    origin: sequencing.CaptureOrigin,
) -> Generator[None, None, None]:
    """re-raise a validation failure at the origin entry and loop point of `capture`.

    Only `SpecValidationError` is intercepted, and any other exception propagates
    unchanged.

    Raises:
        SpecValidationError: the caught error with the locations from
            `sequencing.describe_capture_origin` appended
    """
    try:
        yield
    except SpecValidationError as ex:
        raise ex.at(
            *sequencing.describe_capture_origin(loops, capture, origin)
        ) from ex.__cause__


def _capture_field_path(
    origin: sequencing.CaptureOrigin, field: str
) -> tuple[str, ...]:
    """locate `field` in the `captures:` entry of `origin`, as path parts.

    The path is empty for a capture built from loops alone.
    """
    if origin.spec_index is None:
        return ()
    return ('.captures', f'[{origin.spec_index}]', f'.{field}')
