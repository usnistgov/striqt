"""expansion of a sweep specification into the sequence of captures it runs.

`loop_captures` nests `Sweep.loops` over `Sweep.captures` and applies the
`adjust_captures` remaps for the open source; `loop_capture_origins` and
`describe_capture_origin` trace each expanded capture back to the entry and loop
point in the user's file, which the validators in `sweep` use to locate errors.
"""

from __future__ import annotations as __

from collections import defaultdict, ChainMap
from collections.abc import Hashable, Iterable, Mapping
import itertools
import math
from typing import Any, cast, NamedTuple, Optional, Tuple, TYPE_CHECKING

import msgspec

import striqt.analysis as sa
from striqt.analysis.specs.helpers import (
    _dec_hook,
    frozendict,
    SpecValidationError,
)

from . import captures, structs, types


if TYPE_CHECKING:
    from ..lib.typing import SC, TypeAlias

    _LoopPointsDict: TypeAlias = dict[tuple[types.IsIn, str], list]


# %% loop expansion


class CaptureOrigin(NamedTuple):
    """locate an expanded capture in the sweep specification that produced it.

    An error message needs this to name a place in the user's file: the expanded
    index alone cannot, since `sweep.captures` and `sweep.loops` are both flattened
    away by the expansion.
    """

    capture_index: int
    """position in the expanded capture tuple"""

    spec_index: Optional[int]
    """index into `sweep.captures`, or None when the sweep lists no captures"""

    loop_points: frozendict
    """the loop point that produced this capture, keyed by (isin, field)"""


def loop_captures(
    sweep: structs.Sweep[Any, Any, SC],
    source_id: types.SourceID | None = None,
    *,
    only_fields: tuple[str, ...] | None = None,
    limit: int | None = None,
) -> tuple[SC, ...]:
    """expand `sweep.loops` into the flat tuple of captures that the sweep will run.

    The loops are nested in declaration order, outermost first, with `sweep.captures`
    innermost, so 2 captures under a 3-point loop give 6 captures ordered
    ``(point0, capture0), (point0, capture1), (point1, capture0), ...``. A leading
    `Repeat` is not expanded here, since its captures would be identical; the sweep
    runner multiplies this tuple by `Repeat.count` instead. When `sweep.captures` is
    empty, each loop point instead builds a new capture, and the loops must then
    supply every field the capture class requires.

    Each capture is assembled in three passes, the later ones winning:

    1. the entry from `sweep.captures`, with the loop point applied;
    2. `adjust_captures` for `source_id`, evaluated against that result, so a
       `CaptureRemap` may key on a looped field;
    3. the loop point again, which therefore always overrides an adjustment.

    Loops with ``isin='analysis'`` name a measurement parameter rather than a capture
    field; they accumulate into the capture's `adjust_analysis` mapping, merged with
    whatever that capture already carries. Loop points are coerced to the capture's
    field type, so a sweep decoded from YAML strings expands to the same captures as
    the equivalent numeric one.

    Results are cached on `sweep.captures`, `sweep.loops`, `sweep.adjust_captures`
    and the keyword arguments rather than on the sweep object, so repeated calls
    from the acquire, analyze and write stages return the identical tuple.

    Args:
        sweep: the sweep to expand. Its capture class is that of `sweep.captures[0]`,
            or the sensor binding's `schema.capture` when no captures are listed.
        source_id: hex id of the open source, selecting which `adjust_captures` block
            applies on top of the `'defaults'` block. `None` uses `'defaults'` alone,
            which lets a sweep be expanded with no hardware open; per-source remaps
            are then not applied.
        only_fields: keep only the capture loops naming one of these fields, dropping
            the rest. Analysis loops are unaffected. This expands only the
            dimensions a lookup depends on, as `max_by_frequency` does.
        limit: stop expanding after this many captures. Applied before the
            `options.loop_only_nyquist` filter, so fewer may come back.

    Returns:
        the expanded captures, as instances of the sweep's capture class. Empty when
        the sweep has neither captures nor loops, or when `loop_only_nyquist` is set
        and every point has a finite `analysis_bandwidth` above its `sample_rate`.

    Raises:
        msgspec.ValidationError: a loop names a field the capture class does not
            declare; the sweep has neither explicit captures nor a sensor binding,
            leaving no capture class to instantiate; a loop point does not convert to
            the type of its capture field; or an expanded capture is invalid -- it
            violates a `Meta` bound or a `Capture.__post_init__` rule that no single
            input violates on its own, or the loops left a required capture field
            unset.
    """

    args, kws = _expansion_args(sweep, source_id, only_fields, limit)
    return _expand_capture_loops_with_origins(*args, **kws)[0]


def loop_capture_origins(
    sweep: structs.Sweep[Any, Any, SC],
    source_id: types.SourceID | None = None,
    *,
    only_fields: tuple[str, ...] | None = None,
    limit: int | None = None,
) -> frozendict[SC, CaptureOrigin]:
    """map each capture of `loop_captures` to its origin in the sweep specification.

    The arguments mean what they do in `loop_captures`, which shares this function's
    one cached expansion pass.

    Captures that compare equal collapse onto the first of their origins, since a
    label only ever needs to name one of them. They are not rare: a `List` may repeat
    a value, loop points are coerced with ``strict=False`` so ``'1e6'`` and ``1e6``
    are one point, and an `adjust_captures` fixed value can flatten the field that
    distinguished two `sweep.captures` entries. Use `loop_captures` wherever the
    multiplicity matters.

    Returns:
        a `frozendict` from each distinct capture to the `CaptureOrigin` of its
        first occurrence, keyed in expansion order

    Raises:
        msgspec.ValidationError: as `loop_captures`.
        TypeError: an analysis loop point that survived freezing unhashable, which no
            mapping can key on.
    """
    args, kws = _expansion_args(sweep, source_id, only_fields, limit)
    captures, origins = _expand_capture_loops_with_origins(*args, **kws)

    # not frozendict(zip(...)): dict() from pairs keeps the last value for a repeated
    # key, and the first origin is the one worth reporting
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

    return sa.specs.helpers.freeze(structs.BundledAnalysis.from_dict(result))


def _resolve_capture_cls(sweep: structs.Sweep[Any, Any, SC]) -> type[SC]:
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
    sweep: structs.Sweep[Any, Any, SC],
    source_id: types.SourceID | None,
    only_fields: tuple[str, ...] | None,
    limit: int | None,
) -> tuple[tuple, dict]:
    """marshal `sweep` into the arguments of `_expand_capture_loops_with_origins`.

    Every caller builds the keywords here because `lru_cache` keys on their insertion
    order: the same values passed in two orders miss the cache, and the sweep expands
    twice into captures that are equal but not identical.

    Returns:
        ``(args, kws)`` to splat into `_expand_capture_loops_with_origins`, with the
        keys of `kws` in one fixed order
    """

    args = (sweep.captures, sweep.loops, sweep.adjust_captures)
    kws = {
        'source_id': source_id,
        'cls': _resolve_capture_cls(sweep),
        'only_fields': only_fields,
        'loop_only_nyquist': sweep.options.loop_only_nyquist,
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
    """expand `loops` over `captures` into captures paired with their origins.

    This is the cached expansion pass behind `loop_captures`, whose docstring gives
    the precedence rules and the meaning of each argument. `cls` is the capture class
    to instantiate, required when `captures` is empty.

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
        # nothing is left to build a capture from, so don't build a default one
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
            # we needed to get a first cut at the loops so that adjust_captures
            # can work on the looped capture. here we adjust the captures, but
            # make sure loops have the highest priority at the end
            new = (
                _merge_capture_updates(
                    c, adjust_captures(c, adjust, source_id), loop_point
                )
                for c in new
            )

        # analysis loop points skip the coercion in _build_loop_points_dict, so they
        # can still be a bare list or dict from yaml, which no mapping could key on
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

    # only now is the position in the returned tuple known
    numbered = tuple(o._replace(capture_index=i) for i, o in enumerate(origins))

    return expanded, numbered


@sa.util.lru_cache()
def _build_loop_points_dict(
    loops: tuple[structs.LoopBase, ...],
    capture_cls: type[SC],
    only_fields: tuple[str, ...] | None = None,
) -> _LoopPointsDict:
    """map ``(isin, field)`` to the loop points, keeping only `only_fields` if given.

    Capture loop points are coerced to the capture field type so that remaps keyed
    on a looped field see typed values rather than JSON/YAML decoded strings;
    analysis loop points are left as decoded. `only_fields` drops capture loops
    only, and is applied here rather than by the caller so that the indexes in error
    messages stay positions in the loop list the user wrote.

    Returns:
        the points of each loop with a `field`, in `loops` order, keyed by
        ``(loop.isin, loop.field)``; `repeat` loops are omitted

    Raises:
        SpecValidationError: at ``$.loops`` for a capture loop over a field
            `capture_cls` does not declare, or at ``$.loops[i]`` for a point that
            does not convert to the field's type
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
            f'Object contains unknown field `{sorted(extra)[0]}` for capture type '
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
    """find which element of a failed bulk `msgspec.convert` fails on its own.

    Converting a list at once gets msgspec's own message, but with a path rooted at
    that list rather than at the sweep. Re-converting one element at a time recovers
    which one it was, at a cost paid only on the failing path.

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
    """build the error for a loop at ``$.loops[index]`` whose points did not convert.

    Returns:
        a `SpecValidationError` whose message names the first point that fails on
        its own and its field, or repeats `ex` when no single point reproduces the
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
    """build the error for an invalid expanded capture, located in its sweep.

    msgspec locates the failure by its index in the expanded tuple, which is not a
    place in the user's specification; the origin is.

    Returns:
        a `SpecValidationError` at the locations `describe_capture_origin` gives for
        the first capture that fails on its own, or `ex` unchanged when no single
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
    """apply any loops where isin == 'analysis' into capture.adjust_captures"""

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
    """merge the given dicts as `dicts[0] | dicts[1] | ... | dicts[len(dicts)]`.

    Handles 'adjust_analysis' field similarly across dicts, if present, is
    is merged separately and set in the return.
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
            # There is probably a clearer way to to this
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
                elif not lookup_spec.required:
                    return msgspec.UNSET
                elif required:
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
    sweep: structs.Sweep[Any, Any, SC], source_id: str
) -> dict[str, tuple[Any, ...]]:
    """list the unique values of each adjusted capture field across the sweep.

    The values are read from the expanded captures, so they are what the runner
    produces: a loop over an adjusted field wins over the adjustment (see
    `loop_captures`), and the listing shows the loop values in loop order. A
    `CaptureRemap` miss that leaves the field unset contributes nothing for that
    capture, as in `adjust_captures`.

    Args:
        sweep: the sweep whose `adjust_captures` block is listed
        source_id: hex id of the source whose block applies on top of ``'defaults'``

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


def _convert_label_lookup_keys(sweep: structs.Sweep) -> structs.AdjustCapturesType:
    """convert label lookup keys types to match corresponding capture fields"""

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
                key_type = Tuple[tuple(lookup_types[kc] for kc in v.key)]
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
    return sa.specs.helpers.freeze(fixed, depth)  # type: ignore


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
    """locate an expanded capture in its sweep, as sibling paths for an error message.

    The loop point is rendered as one mapping in `loops` declaration order, with a
    `repeat` rendered as its tag at pass 0 and capture fields read back from
    `capture` so that the value shown is the coerced one it ran with. Two loops that
    name the same field in different `isin` blocks collapse onto one entry, keeping
    the rendering readable at the cost of that distinction.

    Args:
        loops: the sweep's `loops`, in the order the user wrote them
        capture: the expanded capture, or the mapping the expansion assembled for a
            capture that failed to convert to its class
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
            # from the capture, so that the value is the coerced one that it ran with
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
