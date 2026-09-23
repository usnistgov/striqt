"""helper functions for specification data structures and their aliases"""

from __future__ import annotations as __

from collections import Counter, defaultdict, ChainMap
import itertools
import math
import numbers
import string
from typing import (
    Any,
    Callable,
    cast,
    get_args,
    get_type_hints,
    Iterable,
    Literal,
    Mapping,
    NamedTuple,
    Optional,
    Tuple,
    TYPE_CHECKING,
    Union,
)
from datetime import datetime
from pathlib import Path

import msgspec

import striqt.analysis as sa
from striqt.analysis.specs.helpers import (
    _dec_hook,
    frozendict,
    SpecValidationError,
    validation_path,
)

from . import structs
from . import types
from .structs import _AdjustSourceCapturesMap


if TYPE_CHECKING:
    from ..lib.typing import SC, TypeAlias, TypeVar

    _T = TypeVar('_T')
    _LoopPointsDict: TypeAlias = dict[tuple[types.IsIn, str], list]


@sa.util.lru_cache()
def pairwise_by_port(c1: SC, c2: SC | None, is_new: bool) -> list[tuple[SC, SC | None]]:
    # a list with 1 capture per port
    c1_split = split_capture_ports(c1)

    # any changes to the port index
    if c2 is None or is_new:
        c2_split = len(c1_split) * [None]
    else:
        c2_split = split_capture_ports(c2)

    pairwise = zip(*(c1_split, c2_split))
    return list(pairwise)


class CaptureOrigin(NamedTuple):
    """where an expanded capture came from in the sweep specification.

    This is what lets an error name a place in the user's file: the expanded index
    alone cannot, since `sweep.captures` and `sweep.loops` are both flattened away
    by the expansion.
    """

    capture_index: int
    """position in the expanded capture tuple"""

    spec_index: Optional[int]
    """index into `sweep.captures`, or None when the sweep lists no captures"""

    loop_points: frozendict
    """the loop point that produced this capture, keyed by (isin, field)"""


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
    """marshal `sweep` into arguments for `_expand_capture_loops_with_origins`.

    There is one call site for these keywords because `functools.lru_cache` keys on
    their insertion order, so passing the same values in two different orders misses
    the cache: the sweep would expand twice and hand back captures that are equal but
    not identical.
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

    Results are cached on the loop-determining fields rather than on the sweep, so
    the acquire, analyze and write stages can each call this without recomputing.

    Args:
        sweep: the sweep to expand. Its capture class is that of `sweep.captures[0]`,
            or the sensor binding's `schema.capture` when no captures are listed.
        source_id: hex id of the open source, selecting which `adjust_captures` block
            applies on top of the `'defaults'` block. `None` uses `'defaults'` alone,
            which is what lets a sweep be expanded with no hardware open; per-source
            remaps are then not applied.
        only_fields: keep only the capture loops naming one of these fields, dropping
            the rest. Analysis loops are unaffected. Used to expand just the
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
    """map each capture `loop_captures` would return to where it came from.

    The keys are in expansion order, and the arguments mean what they do in
    `loop_captures`, which shares this function's one cached expansion pass.

    Captures that compare equal collapse onto the first of their origins, since a
    label only ever needs to name one of them. They are not rare: a `List` may repeat
    a value, loop points are coerced with ``strict=False`` so `'1e6'` and `1e6` are
    one point, and an `adjust_captures` fixed value can flatten the field that
    distinguished two `sweep.captures` entries. Use `loop_captures` wherever the
    multiplicity matters.

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


def validate_sweep(
    sweep: structs.Sweep[Any, Any, SC],
    source_id: types.SourceID | None = None,
) -> None:
    """raise on anything in `sweep` that would fail once it started to run.

    `Sweep.__post_init__` calls this, so a constructed `Sweep` is a validated one.
    The checks, in order: the `loops` entries (a `repeat` only outermost, at most one
    loop per field) and their collisions with fields set per entry in `captures`; a
    measurement name that shadows a capture field; each unique (capture, analysis)
    combination against the registered measurements; and each unique capture against
    the signal path of `correct_iq` (trigger lookup, resampler design, analysis
    filter and resampler length). The capture chain mirrors the one `iterate_sweep`
    follows, so what is checked here is what will run. `source_id=None` resolves
    `adjust_captures` through its 'defaults' block, which needs no hardware but also
    leaves per-source remaps checked only in their default form.

    Raises:
        `SpecValidationError` (a `msgspec.ValidationError`) at the offending path,
        locating a capture by its `sweep.captures` entry and the loop points that
        produced it
    """
    _validate_loops(sweep.loops)
    _validate_loop_capture_collisions(sweep.loops, sweep.captures)
    _validate_measurement_names(sweep)
    _validate_sweep_analysis(sweep, source_id)
    _validate_sweep_corrections(sweep, source_id)


@sa.util.lru_cache()
def get_capture_type(sweep_cls: type[structs.Sweep]) -> type[structs.SensorCapture]:
    if sweep_cls.sensor is not None:
        return sweep_cls.schema.capture
    else:
        captures_type = get_type_hints(sweep_cls)['captures']
        return get_args(captures_type)[0]


@sa.util.lru_cache()
def split_capture_ports(capture: SC) -> list[SC]:
    """split a multi-channel capture into a list of single-channel captures.

    If capture is not a multi-channel capture (its channel field is just a number),
    then the returned list will be [capture].
    """

    if isinstance(capture.port, numbers.Number):
        return [capture]
    else:
        assert isinstance(capture.port, tuple)

    remaps = [dict() for i in range(len(capture.port))]

    for field in capture.__struct_fields__:
        values = getattr(capture, field)
        if not isinstance(values, tuple):
            continue

        for remap, value in zip(remaps, values):
            remap[field] = value

    return [capture.replace(**remap) for remap in remaps]


@sa.util.lru_cache()
def max_by_frequency(
    field: str,
    captures: tuple[structs.SoapyCapture, ...],
    loops: tuple[structs.LoopSpec, ...] = (),
) -> dict[types.Port, dict[types.CenterFrequency, Any]]:
    """get the maximum value of a field across looped captures by center frequency"""

    map = {}
    looped_captures = _expand_capture_loops_with_origins(
        captures, loops, only_fields=(field, 'center_frequency')
    )[0]

    for c in looped_captures:
        for pc in split_capture_ports(c):
            current = map.setdefault(pc.port, {}).setdefault(pc.center_frequency, None)
            v = getattr(pc, field)

            if current is None or v > current:
                map[pc.port][pc.center_frequency] = v

    return map


_AdjustCaptureMap = dict[
    Union[types.SourceID, Literal['defaults']], _AdjustSourceCapturesMap
]


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
    capture: SC | Mapping[str, Any],
    origin: CaptureOrigin,
) -> tuple[str, ...]:
    """locate an expanded capture in its sweep, as sibling paths for an error message.

    `capture` may still be the mapping the expansion assembled, since a capture that
    fails to convert to its class never becomes an instance.

    The loop point is rendered as one mapping in `loops` declaration order. Two
    loops that name the same field in different `isin` blocks therefore collapse onto
    one entry, keeping the rendering readable at the cost of that distinction.
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
            values = (ensure_tuple(v, size) for v in values)
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
        def lookup_one(k):
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

    defaults = _get_capture_adjust_map(adjust_spec).get('defaults', {})
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
def get_path_fields(
    sweep: structs.Sweep,
    *,
    source_id: str | Callable[[], str],
    spec_path: Path | str | None = None,
) -> dict[str, str]:
    """return a mapping for string `'{field_name}'.format()` style mapping values"""

    assert isinstance(sweep, structs.Sweep)

    if isinstance(source_id, str):
        id_ = source_id
    else:
        id_ = source_id()

    fields = {}
    fields['start_time'] = datetime.now().strftime('%Y%m%d-%Hh%Mm%S')
    fields['sensor_binding'] = type(sweep).__name__
    if spec_path is not None:
        fields['spec_name'] = Path(spec_path).stem
        fields['parent_name'] = Path(spec_path).parent.absolute().name
    fields['source_id'] = id_

    labels = _get_source_capture_adjustments(sweep.adjust_captures, id_)
    fields.update(labels)

    return fields


def ensure_tuple(obj: _T | tuple[_T, ...], size: int | None = None) -> tuple[_T, ...]:
    if isinstance(obj, tuple):
        if size is not None and len(obj) == 1:
            return obj * size
        else:
            return obj
    elif size is None:
        return (obj,)
    else:
        return (obj,) * size


@sa.util.lru_cache()
def get_unique_ports(
    captures: tuple[SC, ...],
    loops: tuple[structs.LoopSpec, ...] | None = None,
) -> tuple[int, ...]:
    ports = set()

    if loops is not None:
        for l in loops:
            if l.field != 'port':
                continue
            looped_ports = cast(list[structs.types.Port], l.get_points())
            for p in looped_ports:
                ports |= set(ensure_tuple(p))

    for c in captures:
        ports |= set(ensure_tuple(c.port))

    return tuple(sorted(ports))


@sa.util.lru_cache()
def get_format_fields(s: str):
    """
    Extracts and returns a list of formatting field names from a given format string.
    """
    formatter = string.Formatter()
    fields = []
    for _, field_name, *_ in formatter.parse(s):
        if field_name is not None:
            fields.append(field_name)
    return fields


@sa.util.lru_cache()
def list_capture_adjustments(
    sweep: structs.Sweep[Any, Any, SC], source_id: str
) -> dict[str, tuple[str, ...]]:
    lookup_fields = _list_capture_adjustments(
        sweep.adjust_captures, source_id=source_id
    )
    captures = loop_captures(sweep, only_fields=lookup_fields, source_id=source_id)
    cdicts = cast(tuple[dict[str, Any], ...], _to_builtins(captures))
    result = defaultdict(dict)

    for c in cdicts:
        changes = adjust_captures(c, sweep.adjust_captures, source_id=source_id)
        for name, value in changes.items():
            result[name][value] = None

    return {name: tuple(v.keys()) for name, v in result.items()}


class PathFormatter:
    def __init__(
        self,
        sweep: structs.Sweep,
        spec_path: Path | str | None = None,
        id_timeout: float = 5,
    ):
        self.sweep_spec = sweep
        if spec_path is None:
            self.spec_path = spec_path
        else:
            self.spec_path = Path(spec_path).resolve()
        self.id_timeout = id_timeout

    def __call__(self, path: str | Path) -> str:
        path_fields = get_format_fields(str(path))
        if len(path_fields) == 0:
            return str(path)

        from ..lib.controller import lookup

        id_ = lookup.id(self.sweep_spec.source, timeout=self.id_timeout)
        path = Path(path).expanduser()

        fields = get_path_fields(
            self.sweep_spec, source_id=id_, spec_path=self.spec_path
        )

        try:
            path = Path(str(path).format(**fields))
        except KeyError as ex:
            key = ex.args[0]
            available = tuple(fields.keys())
            raise KeyError(
                f'sink path format field {key!r}, only {available!r} are allowed'
            ) from ex

        return str(path)


def concat_group_sizes(
    captures: tuple[structs.SensorCapture, ...], *, min_size: int = 1
) -> list[int]:
    """return the minimum sizes of groups of captures that can be concatenated.

    This is important, because some channel analysis results produce a different
    shape depending on (sample_rate, analysis_bandwidth, duration).

    Returns:
        The list l of sizes of each group such that sum(l) == len(captures)
    """

    class C(structs.SensorCapture, frozen=True, forbid_unknown_fields=False):
        """minimal capture fields that safely ignore fields from subclasses"""

    remaining = sa.specs.helpers.convert_spec(captures, type=list[C])
    whole_set = set(remaining)
    counts = Counter(remaining)

    pending = []
    sizes = []
    count = 0

    while len(remaining) > 0:
        if count >= min_size and set(pending) == set(counts) == whole_set:
            # make sure that the pending and remaining captures
            # will result in equivalent shapes when concatenated
            sizes.append(count)
            count = 0
            pending = []

        count += 1
        new = remaining.pop(0)
        pending.append(new)

        counts[new] -= 1
        if counts[new] == 0:
            del counts[new]

    if count > 0:
        sizes.append(count)

    return sizes


# %% module-local helpers


def _capture_field_path(origin: CaptureOrigin, field: str) -> tuple[str, ...]:
    """the path of `field` in the `captures:` entry behind `origin`, if there is one.

    A capture built from loops alone has no entry to point at; the loop point in
    the `at ...` trailer then locates it on its own.
    """
    if origin.spec_index is None:
        return ()
    return ('.captures', f'[{origin.spec_index}]', f'.{field}')


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
    differ in a looped field silently become identical copies at each loop point.
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
    """reject a measurement named like a capture field, which would collide with the
    capture coordinate of the same name in the saved dataset"""
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
    """validate each unique (capture, analysis) combination that `sweep` will run.

    Raises:
        `msgspec.ValidationError` locating the first offending capture by the
        `sweep.captures` entry and the loop points that produced it, and naming the
        measurement that rejected it
    """

    if len(sweep.analysis.to_dict()) == 0:
        return

    if len(sweep.captures) == 0 and sweep.sensor is None:
        # loop_captures refuses this combination; leave such a sweep constructible
        return

    seen = set()

    for capture, origin in loop_capture_origins(sweep, source_id).items():
        analysis = adjust_analysis(sweep.analysis, capture.adjust_analysis)
        key = (sa.specs.helpers.to_analysis_capture(capture), analysis)

        if key in seen:
            continue

        seen.add(key)

        try:
            sa.registry.validate(capture, analysis)
        except SpecValidationError as ex:
            raise ex.at(*describe_capture_origin(sweep.loops, capture, origin)) from (
                ex.__cause__
            )


def _validate_sweep_corrections(
    sweep: structs.Sweep[Any, Any, SC],
    source_id: types.SourceID | None = None,
) -> None:
    """reject each unique capture that `correct_iq` could not process for `sweep`.

    This raises at load what the trigger lookup, `design_resampler`, `design_fir_lpf`
    and `oaresample`'s LO shift check would otherwise raise on the first acquisition.
    The resampler design is skipped for `FileCapture` sweeps, whose source rate comes
    from the file rather than from `sweep.source.master_clock_rate`; the analysis
    filter, which depends only on the capture, is still checked. The parity of the
    padded acquisition that `resample` requires is left to acquisition time: sizing it
    here would import scipy at spec load, and no design from a hamming COLA pads to an
    odd length.

    Raises:
        `SpecValidationError` at `$.source.signal_trigger` when `sweep.analysis` lacks
        the measurement the trigger runs, or otherwise at the field of the first
        offending capture, located by its `sweep.captures` entry and the loop points
        that produced it
    """
    from ..lib import compute
    from ..lib.compute import corrections

    if sweep.source.signal_trigger is not None:
        with validation_path('.source', '.signal_trigger'):
            compute.get_trigger_from_spec(sweep.source, sweep.analysis)

    if len(sweep.captures) == 0 and sweep.sensor is None:
        # loop_captures refuses this combination; leave such a sweep constructible
        return

    rate_from_file = issubclass(_resolve_capture_cls(sweep), structs.FileCapture)
    seen = set()

    for capture, origin in loop_capture_origins(sweep, source_id).items():
        analysis = adjust_analysis(sweep.analysis, capture.adjust_analysis)
        key = (
            sa.specs.helpers.convert_spec_cached(structs.SensorCapture, capture),
            analysis,
        )

        if key in seen:
            continue

        seen.add(key)

        try:
            if math.isfinite(capture.analysis_bandwidth):
                with validation_path(
                    *_capture_field_path(origin, 'analysis_bandwidth')
                ):
                    corrections.validate_fir_band(capture)

            if rate_from_file:
                continue

            with validation_path(*_capture_field_path(origin, 'sample_rate')):
                design = compute.design_resampler(
                    capture, sweep.source.master_clock_rate
                )

            if compute.needs_resample(design, capture) and corrections.USE_OARESAMPLE:
                with validation_path(*_capture_field_path(origin, 'lo_shift')):
                    corrections.validate_oaresample_shift(design)
        except SpecValidationError as ex:
            raise ex.at(*describe_capture_origin(sweep.loops, capture, origin)) from (
                ex.__cause__
            )


def _read_capture_field(capture: SC | Mapping[str, Any], field: str) -> Any:
    if isinstance(capture, Mapping):
        return capture[field]
    else:
        return getattr(capture, field)


@sa.util.lru_cache()
def _build_loop_points_dict(
    loops: tuple[structs.LoopBase, ...],
    capture_cls: type[SC],
    only_fields: tuple[str, ...] | None = None,
) -> _LoopPointsDict:
    """map (isin, field) to the loop points, keeping only `only_fields` if given.

    Capture loop points are coerced to the capture field type here so that remaps
    keyed on a looped field see typed values rather than JSON/YAML decoded strings.

    `only_fields` is applied here rather than by the caller so that the indexes in
    error messages stay positions in the loop list the user wrote.
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
    """find which element of a bulk conversion failed, as (index, its error).

    Converting a list at once gets msgspec's own message, but with a path rooted at
    that list rather than at the sweep. Re-converting one element at a time recovers
    which one it was, and runs only on the already-failing path.

    Returns None when no single element reproduces the failure.
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
    """re-raise a failed loop point conversion against the point that failed"""
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
    """re-raise an invalid expanded capture against the sweep that produced it.

    msgspec locates the failure by its index in the expanded tuple, which is not a
    place in the user's specification; the origin is.
    """
    located = _locate_conversion_failure(captures, capture_cls)

    if located is None:
        return ex

    index, capture_ex = located
    locations = describe_capture_origin(loops, captures[index], origins[index])

    if isinstance(capture_ex, SpecValidationError):
        return capture_ex.at(*locations)

    return SpecValidationError(str(capture_ex), locations=locations)


def _convert_label_lookup_keys(sweep: structs.Sweep) -> structs.AdjustCapturesType:
    """convert label lookup keys types to match corresponding capture fields"""

    result = {}
    capture_cls = get_capture_type(type(sweep))

    field_types = sa.specs.helpers.get_capture_field_types(capture_cls)
    adjust_map = _get_capture_adjust_map(sweep.adjust_captures)

    cls_repr = sa.util.qualified_name(capture_cls)

    for source_id, lookup_map in adjust_map.items():
        at_source = ('.adjust_captures', f'[{source_id!r}]')

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
                keys = v.key if isinstance(v.key, tuple) else (v.key,)
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
    """evaluate the loop specification into captures paired with their origins"""
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
def _get_capture_adjust_dependencies(
    spec: structs.AdjustCapturesType, source_id: str | None
) -> dict[str, str]:
    adjust_specs = _get_capture_adjust_fields(spec, source_id)
    deps = {}
    for name, s in adjust_specs.items():
        if not isinstance(s, structs.CaptureRemap):
            continue
        if isinstance(s.key, str):
            keys = ((s.key),)
        else:
            keys = s.key
        for k in keys:
            deps.setdefault(k, name)
    return deps


@sa.util.lru_cache()
def _get_capture_adjust_fields(
    spec: structs.AdjustCapturesType, source_id: str | None
) -> dict[str, str | structs.CaptureRemap | float | None]:
    fields = {}
    map = _get_capture_adjust_map(spec)

    # the globals spec may use the source-specific spec
    for name, value in map.get('defaults', {}).items():
        if name not in fields:
            fields[name] = value

    if isinstance(source_id, str):
        source_fields = map.get(source_id, {})
        fields.update(source_fields)

    return fields


def _get_capture_adjust_map(spec: structs.AdjustCapturesType) -> _AdjustCaptureMap:
    if isinstance(spec, tuple):
        return dict(zip(source_fields))  # type: ignore
    else:
        return spec  # type: ignore


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

        if isinstance(lookup_spec.key, tuple):
            names = lookup_spec.key
        else:
            names = (lookup_spec.key,)

        for name in names:
            if name not in fields:
                ret.add(name)

    return tuple(ret)


def _merge_analysis_loops(points: _LoopPointsDict) -> dict[str, Any]:
    """apply any loops where isin == 'analysis' into capture.adjust_captures"""

    updates = {}
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
