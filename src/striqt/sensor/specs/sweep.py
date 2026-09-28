"""validation of a `Sweep` after construction.

`validate_sweep` runs from `Sweep.__post_init__`, so a constructed sweep is a
validated one: the `loops` entries and their collisions with `captures`,
measurement names that shadow capture fields, and each unique capture against the
registered measurements and the `correct_iq` signal path. Failures are located in
the user's file through the capture origins that `sequencing` records.
"""

from __future__ import annotations as __

from collections import Counter
from collections.abc import Iterator
import contextlib
import math
from typing import Any, Callable, TYPE_CHECKING

import striqt.analysis as sa
from striqt.analysis.specs.helpers import SpecValidationError, validation_path

from . import sequencing, structs, types


if TYPE_CHECKING:
    from ..lib.typing import SC


def validate_sweep(
    sweep: structs.Sweep[Any, Any, SC],
    source_id: types.SourceID | None = None,
) -> None:
    """raise on anything in `sweep` that would fail once the sweep ran.

    `Sweep.__post_init__` calls this, so a constructed `Sweep` is a validated one.
    The checks, in order: the `loops` entries (a `repeat` only outermost, at most one
    loop per field) and their collisions with fields set per entry in `captures`; a
    measurement name that shadows a capture field; each unique (capture, analysis)
    combination against the registered measurements; and each unique capture against
    the signal path of `correct_iq` (trigger lookup, resampler design, analysis
    filter and LO shift). The captures checked are those `loop_captures` expands for
    `source_id`, which is what `iterate_sweep` runs.

    Args:
        sweep: the sweep to check
        source_id: hex id of the open source, selecting its `adjust_captures` block.
            `None` resolves `adjust_captures` through the ``'defaults'`` block alone,
            which needs no hardware but leaves per-source remaps checked only in
            their default form.

    Raises:
        SpecValidationError: (a `msgspec.ValidationError`) at the offending path,
            locating a capture by its `sweep.captures` entry and the loop points that
            produced it
    """
    _validate_loops(sweep.loops)
    _validate_loop_capture_collisions(sweep.loops, sweep.captures)
    _validate_measurement_names(sweep)
    _validate_sweep_analysis(sweep, source_id)
    _validate_sweep_corrections(sweep, source_id)


@sa.util.lru_cache()
def _validate_loops(loops: tuple[structs.LoopSpec, ...]) -> None:
    if len(loops) == 0:
        return

    # an outermost repeat is legal, so index the rest of the list against the
    # position the user wrote rather than against the slice
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
    """reject a loop that erases a distinction written into `captures:`.

    A loop point overrides the same field in every capture, so captures written to
    differ in a looped field would become identical copies at each loop point. A
    loop over a field the capture class lacks is left for `_build_loop_points_dict`
    to report.

    Raises:
        SpecValidationError: at the loop, locating the first two `captures:` entries
            that differ in its field
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
    """reject a measurement whose name shadows a capture field.

    The two would collide as the capture coordinate and data variable of the same
    name in the saved dataset. A sweep with neither `captures` nor a sensor binding
    has no capture class to check against and passes.

    Raises:
        SpecValidationError: at ``$.analysis``, naming the first such measurement
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


def _validate_sweep_analysis(
    sweep: structs.Sweep[Any, Any, SC],
    source_id: types.SourceID | None = None,
) -> None:
    """validate each unique (capture, analysis) combination that `sweep` runs.

    A sweep whose `analysis` block sets no measurement passes without expanding.

    Raises:
        SpecValidationError: locating the first offending capture by its
            `sweep.captures` entry and the loop points that produced it, and naming
            the measurement that rejected it
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
    sweep: structs.Sweep[Any, Any, SC],
    source_id: types.SourceID | None = None,
) -> None:
    """reject each unique capture that `correct_iq` could not process for `sweep`.

    This raises at load what the trigger lookup, `design_resampler`, `design_fir_lpf`
    and the LO shift check of `oaresample` (only when `USE_OARESAMPLE` is set) would
    otherwise raise on the first acquisition. The resampler design is skipped for
    `FileCapture` sweeps, whose source rate comes from the file rather than from
    `sweep.source.master_clock_rate`; the analysis filter, which depends only on the
    capture, is still checked. The parity of the padded acquisition that `resample`
    requires is left to acquisition time: sizing it here would import scipy at spec
    load, and no design from a hamming COLA pads to an odd length.

    Raises:
        SpecValidationError: at ``$.source.signal_trigger`` when `sweep.analysis`
            lacks the measurement the trigger runs, or otherwise at the
            `analysis_bandwidth`, `sample_rate` or `lo_shift` field of the first
            offending capture, located by its `sweep.captures` entry and the loop
            points that produced it
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
    sweep: structs.Sweep[Any, Any, SC],
    source_id: types.SourceID | None,
    key: Callable[[SC], Any],
) -> Iterator[tuple[SC, structs.AnalysisGroup, sequencing.CaptureOrigin]]:
    """iterate the distinct ``(key(capture), analysis)`` combinations of `sweep`.

    The analysis is `sweep.analysis` with the capture's `adjust_analysis` applied.
    A sweep with neither `captures` nor a sensor binding yields nothing rather than
    raising, so that such a sweep stays constructible.

    Yields:
        ``(capture, analysis, origin)`` for the first capture of each combination,
        in expansion order, until the captures are exhausted
    """
    if len(sweep.captures) == 0 and sweep.sensor is None:
        # loop_captures refuses this combination; leave such a sweep constructible
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
) -> Iterator[None]:
    """re-raise a validation failure at the entry and loop points that made `capture`.

    Only `SpecValidationError` is intercepted; any other exception propagates as is.

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
    """locate `field` in the `captures:` entry behind `origin`, as path parts.

    A capture built from loops alone has no entry to point at, so its path is empty;
    the loop point in the ``at ...`` trailer then locates it on its own.
    """
    if origin.spec_index is None:
        return ()
    return ('.captures', f'[{origin.spec_index}]', f'.{field}')
