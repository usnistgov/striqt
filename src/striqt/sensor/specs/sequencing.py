"""expansion of a sweep specification into the captures that the sweep runs.

Loops nest over the listed captures, the `adjust_captures` remaps for the open source
apply on top, and the loop point is applied last. A looped field wins over an
adjustment. Each expanded capture is paired with the entry and loop point that
produced the capture.
"""

from __future__ import annotations as __

import itertools
import math
from collections import ChainMap, defaultdict
from collections.abc import Hashable, Iterable, Mapping
from typing import TYPE_CHECKING, Any, NamedTuple, Optional, cast

import msgspec

import striqt.analysis as sa
from striqt.analysis.specs.helpers import (
    SpecValidationError,
    _dec_hook,
    frozendict,
)

from . import captures, structs, types

if TYPE_CHECKING:
    from ..lib.typing import SC, TypeAlias

    _LoopPointsDict: TypeAlias = dict[tuple[types.IsIn, str], list]


# %% loop expansion


class CaptureOrigin(NamedTuple):
    """the place in a sweep specification that produced an expanded capture."""

    capture_index: int
    """position in the expanded capture tuple"""

    spec_index: Optional[int]
    """index into `sweep.captures`, or None when the sweep lists no captures"""

    loop_points: frozendict
    """the loop point that produced the capture, keyed by ``(isin, field)``"""


def loop_captures(
    sweep: structs.Sweep[Any, Any, SC, Any],
    source_id: types.SourceID | None = None,
    *,
    only_fields: tuple[str, ...] | None = None,
    limit: int | None = None,
) -> tuple[SC, ...]:
    """expand `sweep.loops` into the flat tuple of captures that the sweep runs.

    The loops are nested in declaration order, outermost first, with `sweep.captures`
    innermost: 2 captures under a 3-point loop give 6 captures ordered
    ``(point0, capture0), (point0, capture1), (point1, capture0), ...``. A leading
    `Repeat` is not expanded. When `sweep.captures` is empty, each loop point builds
    a new capture, and the loops must then supply every field the capture class
    requires.

    Each capture is assembled in three passes, the later ones winning:

    1. the entry from `sweep.captures`, with the loop point applied;
    2. `adjust_captures` for `source_id`, evaluated against the result of pass 1. A
       `CaptureRemap` may key on a looped field;
    3. the loop point again. A looped field overrides an adjustment.

    Loops with ``isin='analysis'`` name a measurement parameter rather than a capture
    field. The points of such loops are merged into the `adjust_analysis` mapping of
    the capture. Loop points are coerced to the type of the capture field. A sweep
    decoded from YAML strings expands to the same captures as the numeric equivalent.

    Repeated calls with equal `sweep.captures`, `sweep.loops`,
    `sweep.adjust_captures`, and keyword arguments return the identical tuple.

    Args:
        sweep: the sweep to expand. The capture class is that of `sweep.captures[0]`,
            or `schema.capture` of the sensor binding when no captures are listed.
        source_id: hex id of the open source, selecting which `adjust_captures` block
            applies on top of the ``'defaults'`` block. `None` uses ``'defaults'``
            alone, and the per-source remaps are not applied.
        only_fields: keep only the capture loops naming one of these fields, dropping
            the rest. Analysis loops are unaffected.
        limit: the maximum number of expanded captures; the `loop_only_nyquist`
            filter applies after the limit and may remove more

    Returns:
        the expanded captures, as instances of the capture class of the sweep. Empty
        when the sweep has neither captures nor loops, or when `loop_only_nyquist` is
        set and every point has a finite `analysis_bandwidth` above the `sample_rate`
        of the point.

    Raises:
        msgspec.ValidationError: if a loop names a field the capture class does not
            declare; if the sweep has neither explicit captures nor a sensor
            binding; if a loop point does not convert to the type of the capture
            field; if an expanded capture violates a `Meta` bound or a
            `Capture.__post_init__` rule that no single input violates alone; or if
            the loops left a required capture field unset
    """

    args, kws = _expansion_args(
        sweep, source_id, only_fields, limit, sweep.options.loop_only_nyquist
    )
    return _expand_capture_loops_with_origins(*args, **kws)[0]


def loop_capture_origins(
    sweep: structs.Sweep[Any, Any, SC, Any],
    source_id: types.SourceID | None = None,
    *,
    only_fields: tuple[str, ...] | None = None,
    limit: int | None = None,
) -> frozendict[SC, CaptureOrigin]:
    """map each capture of `loop_captures` to the origin of the capture in the sweep.

    The arguments are those of `loop_captures`.

    Captures that compare equal collapse onto the first origin. Equal captures arise
    when a `List` repeats a value, when two loop points coerce to one value
    (``'1e6'`` and ``1e6`` are one point), and when an `adjust_captures` fixed value
    flattens the field that distinguished two `sweep.captures` entries.

    Returns:
        a `frozendict` from each distinct capture to the `CaptureOrigin` of the first
        occurrence of the capture, keyed in expansion order

    Raises:
        msgspec.ValidationError: as `loop_captures`
        TypeError: if an analysis loop point is unhashable after freezing
    """
    args, kws = _expansion_args(
        sweep, source_id, only_fields, limit, sweep.options.loop_only_nyquist
    )
    captures, origins = _expand_capture_loops_with_origins(*args, **kws)

    # not frozendict(zip(...)), which keeps the last value for a repeated key; the
    # first origin is the one to report
    first: dict[SC, CaptureOrigin] = {}
    for capture, origin in zip(captures, origins):
        first.setdefault(capture, origin)

    return frozendict(first)


@sa.util.lru_cache()
def max_by_frequency(
    field: str,
    sweep_captures: tuple[structs.SoapyCapture, ...],
    loops: tuple[structs.LoopSpec, ...] = (),
) -> dict[types.Port, dict[types.CenterFrequency, Any]]:
    """get the maximum value of a field across looped captures by center frequency"""

    map = {}
    looped_captures = _expand_capture_loops_with_origins(
        sweep_captures, loops, only_fields=(field, 'center_frequency')
    )[0]

    for c in looped_captures:
        for pc in captures.split_capture_ports(c):
            current = map.setdefault(pc.port, {}).setdefault(pc.center_frequency, None)
            v = getattr(pc, field)

            if current is None or v > current:
                map[pc.port][pc.center_frequency] = v

    return map


@sa.util.lru_cache()
def adjust_analysis(
    analyses: structs.AnalysisGroup,
    adjust_analysis: Mapping[str, Any] | None,
) -> structs.AnalysisGroup:
    if adjust_analysis is None or len(adjust_analysis) == 0:
        return analyses

    result = analyses.to_dict(unfreeze=True)

    used_names = set()

    for analysis_kws in result.values():
        matching_names = analysis_kws.keys() & adjust_analysis.keys()
        used_names |= matching_names
        for field in matching_names:
            analysis_kws[field] = adjust_analysis[field]

    unused_names = adjust_analysis.keys() - used_names

    if len(unused_names) > 0:
        logger = sa.util.get_logger('sweep')
        logger.warning(
            f'analysis_adjust keys {unused_names} do not match any analysis parameters'
        )

    return sa.specs.helpers.freeze(type(analyses).from_dict(result))


def _resolve_capture_cls(sweep: structs.CapturePlan[Any, Any, SC, Any]) -> type[SC]:
    if len(sweep.captures) > 0:
        return type(sweep.captures[0])
    elif sweep.sensor is None:
        raise SpecValidationError(
            'Expected a non-empty `captures` list, unless the sweep is bound to a '
            'sensor with striqt.sensor.bind_sensor',
            ('.captures',),
        )
    else:
        from .dataclasses import Schema

        assert isinstance(sweep.schema, Schema)
        return sweep.schema.capture


def _expansion_args(
    plan: structs.CapturePlan[Any, Any, SC, Any],
    source_id: types.SourceID | None,
    only_fields: tuple[str, ...] | None,
    limit: int | None,
    loop_only_nyquist: bool,
) -> tuple[tuple, dict]:
    """marshal `plan` into the arguments of `_expand_capture_loops_with_origins`.

    Args:
        plan: the spec with the `captures`, `loops`, and `adjust_captures` to expand
        source_id: as `loop_captures`
        only_fields: as `loop_captures`
        limit: as `loop_captures`
        loop_only_nyquist: True drops each expanded capture with a finite
            `analysis_bandwidth` above the capture `sample_rate`

    Returns:
        ``(args, kws)`` to splat into `_expand_capture_loops_with_origins`, with the
        keys of `kws` in one fixed order
    """

    args = (plan.captures, plan.loops, plan.adjust_captures)
    # lru_cache keys on keyword insertion order; one fixed order here keeps equal
    # plans on one cache entry, so expanded captures stay identical, not just equal
    kws = {
        'source_id': source_id,
        'cls': _resolve_capture_cls(plan),
        'only_fields': only_fields,
        'loop_only_nyquist': loop_only_nyquist,
        'limit': limit,
    }
    return args, kws


@sa.util.lru_cache()
def _expand_capture_loops_with_origins(
    captures: tuple[SC, ...],
    loops: tuple[structs.LoopSpec, ...],
    adjust: structs.AdjustCapturesType | None = None,
    *,
    source_id: types.SourceID | None = None,
    cls: type[SC] | None = None,
    only_fields: tuple[str, ...] | None = None,
    loop_only_nyquist: bool = False,
    limit: int | None = None,
) -> tuple[tuple[SC, ...], tuple[CaptureOrigin, ...]]:
    """expand `loops` over `captures` into captures paired with the origin of each.

    The precedence rules and the arguments are those of `loop_captures`. `cls` is
    the capture class to instantiate, required when `captures` is empty.

    Returns:
        ``(captures, origins)`` of equal length, with each `CaptureOrigin.capture_index`
        the position in the returned tuple after the `loop_only_nyquist` filter

    Raises:
        SpecValidationError: as `loop_captures`
    """
    if len(captures) == 0 and len(loops) == 0:
        return (), ()
    if cls is None:
        assert len(captures) > 0
        cls = type(captures[0])
    assert issubclass(cls, structs.Capture)

    loop_points = _build_loop_points_dict(loops, cls, only_fields)

    if len(captures) == 0 and len(loop_points) == 0:
        # no captures and no loop points: do not build a default capture
        return (), ()

    loop_starts = {k: v[0] for k, v in loop_points.items() if len(v) > 0}
    defaults = _merge_analysis_loops(loop_starts)
    loop_combos = itertools.product(*loop_points.values())

    cdicts = cast(tuple[dict, ...], _to_builtins(captures))
    spec_indexes: list[int | None] = (
        list(range(len(cdicts))) if len(cdicts) > 0 else [None]
    )

    result = []
    origins = []
    for i, values in enumerate(loop_combos):
        if limit is not None and i * len(captures) >= limit:
            break

        point = dict(zip(loop_points.keys(), values))
        loop_point = _merge_analysis_loops(point)

        if len(cdicts) > 0:
            # merge into the specified captures, if any
            new = (_merge_capture_updates(defaults, c, loop_point) for c in cdicts)
        else:
            # otherwise, instances are new captures
            new = [_merge_capture_updates(defaults, loop_point)]

        if adjust is not None:
            # adjust_captures keys on the looped capture, so the loop point is
            # applied before it; applying the loop point again last keeps a loop
            # ahead of any adjustment
            new = (
                _merge_capture_updates(
                    c, adjust_captures(c, adjust, source_id), loop_point
                )
                for c in new
            )

        # analysis loop points skip the coercion in _build_loop_points_dict and may
        # still be an unhashable list or dict from yaml
        frozen_point = frozendict({
            k: sa.specs.helpers.freeze(v) for k, v in point.items()
        })

        result += list(new)
        origins += [
            CaptureOrigin(0, spec_index, frozen_point) for spec_index in spec_indexes
        ]

    if limit is not None:
        result = result[:limit]
        origins = origins[:limit]

    if len(result) == 0:
        # there were no loops
        return (), ()
    else:
        try:
            expanded = msgspec.convert(
                result, tuple[cls, ...], strict=False, dec_hook=_dec_hook
            )
        except msgspec.ValidationError as ex:
            raise _expanded_capture_error(loops, result, origins, cls, ex) from ex

    if loop_only_nyquist:
        keep = [
            (c, o)
            for c, o in zip(expanded, origins)
            if not math.isfinite(c.analysis_bandwidth)
            or c.sample_rate >= c.analysis_bandwidth
        ]
        expanded = tuple(c for c, _ in keep)
        origins = [o for _, o in keep]

    # the position in the returned tuple is known only after the nyquist filter
    numbered = tuple(o._replace(capture_index=i) for i, o in enumerate(origins))

    return expanded, numbered


@sa.util.lru_cache()
def _build_loop_points_dict(
    loops: tuple[structs.LoopBase, ...],
    capture_cls: type[SC],
    only_fields: tuple[str, ...] | None = None,
) -> _LoopPointsDict:
    """map ``(isin, field)`` to the loop points, keeping only `only_fields` if given.

    Capture loop points are coerced to the capture field type, and analysis loop
    points are left as decoded. `only_fields` drops capture loops only.

    Returns:
        the points of each loop with a `field`, in `loops` order, keyed by
        ``(loop.isin, loop.field)``; `repeat` loops are omitted

    Raises:
        SpecValidationError: at ``$.loops`` for a capture loop over a field that
            `capture_cls` does not declare, or at ``$.loops[i]`` for a point that
            does not convert to the type of the field
    """
    loop_points: _LoopPointsDict = {}
    loop_indexes: dict[tuple[types.IsIn, str], int] = {}

    for index, loop in enumerate(loops):
        if loop.field is None:
            continue
        loop_points[loop.isin, loop.field] = loop.get_points()
        loop_indexes[loop.isin, loop.field] = index

    if only_fields is not None:
        loop_points = {
            (isin, name): points
            for (isin, name), points in loop_points.items()
            if isin == 'analysis' or name in only_fields
        }

    field_types = sa.specs.helpers.get_capture_field_types(capture_cls)
    available = {name for isin, name in loop_points if isin == 'capture'}

    extra = available - set(field_types)
    if len(extra) > 0:
        raise SpecValidationError(
            f'Object contains unknown field `{min(extra)}` for capture type '
            f'`{sa.util.qualified_name(capture_cls)}`',
            ('.loops',),
        )

    for (isin, name), points in loop_points.items():
        if isin != 'capture':
            continue
        try:
            loop_points[isin, name] = msgspec.convert(
                points, list[field_types[name]], strict=False, dec_hook=_dec_hook
            )
        except msgspec.ValidationError as ex:
            raise _loop_point_error(
                loop_indexes[isin, name], name, points, field_types[name], ex
            ) from ex

    return loop_points


def _locate_conversion_failure(
    values: Iterable[Any], element_type: Any
) -> tuple[int, msgspec.ValidationError] | None:
    """find which element of a failed bulk `msgspec.convert` fails alone.

    Returns:
        ``(index, error)`` of the first element that fails to convert to
        `element_type`, or None when no single element reproduces the failure
    """
    for index, value in enumerate(values):
        try:
            msgspec.convert(value, element_type, strict=False, dec_hook=_dec_hook)
        except msgspec.ValidationError as element_ex:
            return index, element_ex

    return None


def _loop_point_error(
    index: int,
    name: str,
    points: list,
    field_type: Any,
    ex: msgspec.ValidationError,
) -> SpecValidationError:
    """build the error for the loop at ``$.loops[index]`` with unconvertible points.

    Returns:
        a `SpecValidationError` for the first point that fails alone and the field
        `name`, or one with the message of `ex` when no single point reproduces the
        failure
    """
    path = ('.loops', f'[{index}]')
    located = _locate_conversion_failure(points, field_type)

    if located is None:
        return SpecValidationError(str(ex), path)

    point_index, point_ex = located
    point = points[point_index]
    return SpecValidationError(
        f'{point_ex} for loop point {point!r} over field `{name}`', path
    )


def _expanded_capture_error(
    loops: tuple[structs.LoopBase, ...],
    captures: list[dict],
    origins: list[CaptureOrigin],
    capture_cls: type[SC],
    ex: msgspec.ValidationError,
) -> msgspec.ValidationError:
    """build the error for an invalid expanded capture, located in the sweep.

    Returns:
        a `SpecValidationError` at the locations `describe_capture_origin` gives for
        the first capture that fails alone, or `ex` unchanged when no single
        capture reproduces the failure
    """
    located = _locate_conversion_failure(captures, capture_cls)

    if located is None:
        return ex

    index, capture_ex = located
    locations = describe_capture_origin(loops, captures[index], origins[index])

    if isinstance(capture_ex, SpecValidationError):
        return capture_ex.at(*locations)

    return SpecValidationError(str(capture_ex), locations=locations)


def _merge_analysis_loops(points: _LoopPointsDict) -> dict[str, Any]:
    """regroup `points` by field, with analysis points under ``'adjust_analysis'``"""

    updates: dict[str, Any] = {}
    analysis_updates = {}
    for (target, field), value in points.items():
        if target == 'capture':
            updates[field] = value
        else:
            analysis_updates[field] = value
    if analysis_updates:
        updates['adjust_analysis'] = analysis_updates

    return updates


def _merge_capture_updates(*dicts: dict[str, Any]) -> dict[str, Any]:
    """merge `dicts` left to right, with the later value of a repeated key winning.

    The ``'adjust_analysis'`` mappings of the dicts are merged the same way into
    one mapping.
    """
    capture = {}
    adjust_analysis = {}
    for d in dicts:
        adjust_analysis.update(d.get('adjust_analysis', {}))
        capture.update(d)
    if adjust_analysis:
        capture['adjust_analysis'] = adjust_analysis
    return capture


def _to_builtins(obj: Any) -> Any:
    return msgspec.to_builtins(obj, enc_hook=sa.specs.helpers._enc_hook)


# %% adjust_captures


def adjust_captures(
    capture: Mapping[str, Any],
    adjust_spec: structs.AdjustCapturesType,
    source_id: types.SourceID | None,
) -> dict[str, Any]:
    """evaluate the field values"""

    if not isinstance(capture, (dict, frozendict)):
        raise TypeError(
            f'Expected `capture` as a mapping, got `{type(capture).__name__}`'
        )

    fields = _get_capture_adjust_fields(adjust_spec, source_id)

    ret = {}
    key_lookup = ChainMap(ret, capture)  # type: ignore

    def get_key(fields: str | tuple[str, ...], name: str):
        if not isinstance(fields, tuple):
            field = fields
        elif len(fields) == 1:
            field = fields[0]
        else:
            # a tuple key in lookup_spec
            values = [get_key(f, name) for f in fields]
            size = max(len(obj) if isinstance(obj, tuple) else 1 for obj in values)
            values = (captures.ensure_tuple(v, size) for v in values)
            return tuple(zip(*values))
        try:
            value = key_lookup.get(field)
        except KeyError:
            raise KeyError(f'no such key {field!r} for field {name!r}')
        return value

    def do_lookup(
        lookup_spec: structs.CaptureRemap,
        key,
        field_default: str | structs.CaptureRemap | None,
    ):
        def lookup_one(k: Hashable) -> Any:
            try:
                return lookup_spec.lookup[k]
            except KeyError:
                if lookup_spec.default != msgspec.UNSET:
                    return lookup_spec.default

                if isinstance(field_default, structs.CaptureRemap):
                    default = field_default.default
                    required = field_default.required
                else:
                    default = msgspec.UNSET
                    required = False

                if default != msgspec.UNSET:
                    return default
                elif not lookup_spec.required or required:
                    return msgspec.UNSET
                else:
                    raise SpecValidationError(
                        f'Object missing a lookup entry for key `{k!r}`',
                        (
                            '.adjust_captures',
                            f'[{source_id!r}]',
                            f'.{field}',
                            '.lookup',
                        ),
                    )

        if isinstance(key, tuple) and len(key) > 1:
            # per-port value in the capture: the field is a full-length tuple or
            # absent, so an optional miss on any port omits the whole field
            values = tuple(lookup_one(k) for k in key)
            if any(v == msgspec.UNSET for v in values):
                return msgspec.UNSET
            return values
        return lookup_one(key[0] if isinstance(key, tuple) else key)

    defaults = adjust_spec.get('defaults', {})
    default_fields = {
        k: v for k, v in defaults.items() if isinstance(v, structs.CaptureRemap)
    }
    for field, lookup_spec in fields.items():
        if not isinstance(lookup_spec, structs.CaptureRemap):
            # no lookup - use value
            ret[field] = lookup_spec
            continue

        key = get_key(lookup_spec.key, field)
        v = do_lookup(lookup_spec, key, field_default=default_fields.get(field, None))
        if v != msgspec.UNSET:
            ret[field] = v

    return ret


@sa.util.lru_cache()
def list_capture_adjustments(
    sweep: structs.Sweep[Any, Any, SC, Any], source_id: str
) -> dict[str, tuple[Any, ...]]:
    """list the unique values of each adjusted capture field across the sweep.

    The values are those of the expanded captures. A loop over an adjusted field
    wins over the adjustment (see `loop_captures`), and the listing shows the loop
    values in loop order. A `CaptureRemap` miss that leaves the field unset
    contributes nothing for that capture.

    Args:
        sweep: the sweep with the `adjust_captures` block to list
        source_id: hex id of the source; the block of that source applies on top of
            ``'defaults'``

    Returns:
        the unique values of each adjusted field in first-seen order, keyed by
        field name. Empty when `adjust_captures` touches no fields.
    """
    adjust = sweep.adjust_captures
    adjusted_fields = tuple(_get_capture_adjust_fields(adjust, source_id))
    lookup_fields = _list_capture_adjustments(adjust, source_id=source_id)
    only_fields = adjusted_fields + lookup_fields

    captures = loop_captures(sweep, only_fields=only_fields, source_id=source_id)
    cdicts = cast(tuple[dict[str, Any], ...], _to_builtins(captures))
    result: defaultdict[str, dict[Any, None]] = defaultdict(dict)

    for capture, cdict in zip(captures, cdicts):
        # `adjust_captures` decides which remaps resolved; the value itself comes
        # from the capture so that a loop over the same field is reported correctly
        resolved = adjust_captures(cdict, adjust, source_id=source_id)
        for name in resolved:
            result[name][getattr(capture, name)] = None

    return {name: tuple(v.keys()) for name, v in result.items()}


def _convert_label_lookup_keys(
    sweep: structs.CapturePlan,
) -> structs.AdjustCapturesType:
    """coerce the `adjust_captures` lookup keys of `sweep` to the keyed field types.

    Raises:
        SpecValidationError: if a source key is neither ``'defaults'`` nor a hex id,
            if a block is not a mapping, if a field is reserved or unknown for the
            capture type, or if a lookup key does not convert to the type of the
            keyed field
    """

    result = {}
    capture_cls = captures.get_capture_type(type(sweep))

    field_types = sa.specs.helpers.get_capture_field_types(capture_cls)
    adjust_map = sweep.adjust_captures

    cls_repr = sa.util.qualified_name(capture_cls)

    for source_id, lookup_map in adjust_map.items():
        at_source = ('.adjust_captures', f'[{source_id!r}]')

        if not isinstance(lookup_map, (dict, frozendict)):
            raise SpecValidationError(
                'Expected `object` mapping capture field names to values', at_source
            )

        if source_id != 'defaults':
            try:
                bytes.fromhex(source_id)
            except ValueError:
                raise SpecValidationError(
                    'Expected `defaults` or a hex source id', at_source
                )

        result[source_id] = {}
        lookup_types = dict(field_types)
        for field, v in lookup_map.items():
            if field == 'port' or field in structs.Capture.__struct_fields__:
                raise SpecValidationError(
                    f'Object contains reserved capture field `{field}`', at_source
                )
            if field not in field_types:
                raise SpecValidationError(
                    f'Object contains unknown field `{field}` for capture type '
                    f'`{cls_repr}`',
                    at_source,
                )
            elif not isinstance(v, structs.CaptureRemap):
                # defines a fixed value
                result[source_id][field] = v
                lookup_types[field] = str
                continue
            elif not isinstance(v.key, tuple):
                # defines lookup on a single field
                if v.key not in lookup_types:
                    # prune refs with invalid lookups
                    continue
                    raise msgspec.ValidationError(
                        f'no metadata capture lookup with key {v.key!r} in source {source_id!r}'
                    )
                key_type = lookup_types[v.key]
            elif len(v.key) == 1:
                key_type = lookup_types[v.key[0]]
            elif all(kc in lookup_types for kc in v.key):
                key_type = tuple[tuple(lookup_types[kc] for kc in v.key)]
            else:
                # prune refs with invalid lookups
                continue
                invalid = set(v.key) - set(lookup_types) - set(field_types)
                raise msgspec.ValidationError(
                    f'no such capture fields {invalid!r} for metadata field {field!r}'
                )
            lookup = {}
            try:
                for k, value in v.lookup.items():
                    lookup_key = msgspec.convert(k, key_type, strict=False)
                    lookup[lookup_key] = value
            except msgspec.ValidationError as ex:
                keys = captures.ensure_tuple(v.key)
                names = ', '.join(f'`{k}`' for k in keys)
                plural = 'fields' if len(keys) > 1 else 'field'
                raise SpecValidationError(
                    f'Expected lookup keys matching the type of capture '
                    f'{plural} {names}',
                    at_source + (f'.{field}', '.lookup'),
                ) from ex

            result[source_id][field] = structs.CaptureRemap(
                key=v.key, lookup=lookup, default=v.default, required=v.required
            )
            lookup_types[field] = str

    depth = sa.specs.helpers.inspect_freeze_depths(type(sweep))['adjust_captures']
    fixed = msgspec.convert(result, structs.AdjustCapturesType, strict=False)
    return sa.specs.helpers.freeze(fixed, depth)  # ty: ignore[invalid-return-type]


@sa.util.lru_cache()
def _get_capture_adjust_fields(
    spec: structs.AdjustCapturesType, source_id: str | None
) -> dict[str, str | structs.CaptureRemap | float | None]:
    fields = {}
    map = spec

    # the globals spec may use the source-specific spec
    for name, value in map.get('defaults', {}).items():
        if name not in fields:
            fields[name] = value

    if isinstance(source_id, str):
        source_fields = map.get(source_id, {})
        fields.update(source_fields)

    return fields


@sa.util.lru_cache()
def _get_capture_adjust_dependencies(
    spec: structs.AdjustCapturesType, source_id: str | None
) -> dict[str, str]:
    adjust_specs = _get_capture_adjust_fields(spec, source_id)
    deps = {}
    for name, s in adjust_specs.items():
        if not isinstance(s, structs.CaptureRemap):
            continue
        for k in captures.ensure_tuple(s.key):
            deps.setdefault(k, name)
    return deps


def _get_source_capture_adjustments(
    spec: structs.AdjustCapturesType,
    source_id: str | None,
) -> dict[str, Any]:
    """get a map of capture adjustments that do not require lookups"""

    ret = {}
    fields = _get_capture_adjust_fields(spec, source_id)

    for field, lookup_spec in fields.items():
        if isinstance(lookup_spec, str):
            ret[field] = lookup_spec

    return ret


@sa.util.lru_cache()
def _list_capture_adjustments(
    spec: structs.AdjustCapturesType, source_id: str | None
) -> tuple[str, ...]:
    ret = set()
    fields = _get_capture_adjust_fields(spec, source_id)
    for lookup_spec in fields.values():
        if not isinstance(lookup_spec, structs.CaptureRemap):
            continue

        for name in captures.ensure_tuple(lookup_spec.key):
            if name not in fields:
                ret.add(name)

    return tuple(ret)


# %% describing captures


@sa.util.lru_cache()
def describe_capture(
    capture: structs.Capture,
    fields: tuple[str, ...],
    *,
    adjust_spec: structs.AdjustCapturesType | None = None,
    source_id: structs.types.SourceID | None,
    join: str = ', ',
) -> str:
    """generate a description of a capture"""
    diffs = []

    if adjust_spec is not None:
        deps = _get_capture_adjust_dependencies(adjust_spec, source_id)
        use_fields = [deps.get(name, name) for name in fields]
    else:
        use_fields = fields

    for name in use_fields:
        desc = sa.lib.dataarrays.describe_field(capture, name, sep=': ')
        diffs.append(desc)

    return join.join(diffs)


def describe_capture_origin(
    loops: tuple[structs.LoopBase, ...],
    capture: structs.SensorCapture | Mapping[str, Any],
    origin: CaptureOrigin,
) -> tuple[str, ...]:
    """locate an expanded capture in the sweep, as sibling paths for an error message.

    The loop point is rendered as one mapping in `loops` declaration order. A
    `repeat` is rendered as the tag of the loop at pass 0. Capture field values are
    those of `capture`, after coercion. Two loops that name the same field in
    different `isin` blocks collapse onto one entry.

    Args:
        loops: the `loops` of the sweep, in declaration order
        capture: the expanded capture, or the mapping the expansion assembled for a
            capture that failed to convert to the capture class
        origin: the origin `loop_capture_origins` recorded for `capture`

    Returns:
        up to two locations for `SpecValidationError.at`: ``'.loops: {...}'`` when
        any loop point survives in `origin`, then ``'.captures[i]'`` when the
        capture came from a `captures:` entry
    """
    points = {}

    for loop in loops:
        if loop.field is None:
            # a Repeat is not expanded here, so only its first pass is ever validated
            points[type(loop).__struct_config__.tag] = 0
        elif (loop.isin, loop.field) not in origin.loop_points:
            # dropped by only_fields
            continue
        elif loop.isin == 'capture':
            # the capture value is the coerced value
            points[loop.field] = _read_capture_field(capture, loop.field)
        else:
            points[loop.field] = origin.loop_points[loop.isin, loop.field]

    locations = []

    if len(points) > 0:
        locations.append(f'.loops: {points!r}')

    if origin.spec_index is not None:
        locations.append(f'.captures[{origin.spec_index}]')

    return tuple(locations)


def _read_capture_field(
    capture: structs.SensorCapture | Mapping[str, Any], field: str
) -> Any:
    if isinstance(capture, Mapping):
        return capture[field]
    else:
        return getattr(capture, field)
