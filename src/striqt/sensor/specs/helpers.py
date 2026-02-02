"""helper functions for specification data structures and their aliases"""

from __future__ import annotations as __

from collections import Counter, defaultdict
import itertools
import numbers
import string
import typing
from datetime import datetime
from pathlib import Path

import msgspec
from msgspec import UNSET, UnsetType

from striqt.analysis.specs.helpers import convert_spec, convert_dict, _deep_freeze

from . import structs as specs
from . import types
from .structs import _TS, _TC, _TP
from ..lib import util

if typing.TYPE_CHECKING:
    from immutabledict import immutabledict

    _T = typing.TypeVar('_T')


@util.lru_cache()
def _check_fields(
    cls: type[specs.SpecBase], names: tuple[str, ...], new_instance=False
):
    fields = msgspec.structs.fields(cls)
    available = set(names)

    if new_instance:
        required = {f.name for f in fields if f.required}
        missing = required - available
        if len(missing) > 0:
            raise TypeError(f'missing required loop fields {missing!r}')

    extra = available - {f.name for f in fields}
    if len(extra) > 0:
        raise TypeError(f'invalid capture fields {extra!r} specified in loops')


@util.lru_cache()
def pairwise_by_port(
    c1: _TC, c2: _TC | None, is_new: bool
) -> list[tuple[_TC, _TC | None]]:
    # a list with 1 capture per port
    c1_split = split_capture_ports(c1)

    # any changes to the port index
    if c2 is None or is_new:
        c2_split = len(c1_split) * [None]
    else:
        c2_split = split_capture_ports(c2)

    pairwise = zip(*(c1_split, c2_split))
    return list(pairwise)


@util.lru_cache()
def _expand_capture_loops(
    captures: tuple[_TC, ...],
    loops: tuple[specs.LoopSpec, ...],
    adjust: specs.AdjustCapturesType | None = None,
    *,
    source_id: types.SourceID | None = None,
    cls: type[_TC] | None = None,
    only_fields: tuple[str, ...] | None = None,
    loop_only_nyquist: bool = False,
) -> tuple[_TC, ...]:
    """evaluate the loop specification, and flatten into one list of loops"""

    defaults = {l.field: l.get_points()[0] for l in loops if len(l.get_points()) > 0}
    if only_fields is not None:
        loops = tuple(l for l in loops if l.field in only_fields)

    loop_fields = tuple([loop.field for loop in loops])

    if len(captures) == 0 and len(loops) == 0:
        return ()
    if cls is None:
        assert len(captures) > 0
        cls = type(captures[0])
    _check_fields(cls, loop_fields, False)
    assert issubclass(cls, specs.Capture)

    loop_points = [loop.get_points() for loop in loops]
    combinations = itertools.product(*loop_points)

    result = []
    for values in combinations:
        updates = defaults | dict(zip(loop_fields, values))
        if len(captures) > 0:
            # iterate specified captures if avialable
            new = (c.replace(**updates) for c in captures)
        else:
            # otherwise, instances are new captures
            new = (cls.from_dict(updates) for _ in range(1))

        if loop_only_nyquist:
            new = (c for c in new if c.sample_rate >= c.analysis_bandwidth)

        if adjust is not None:
            new = [adjust_captures(c, adjust, source_id) for c in new]

        result += list(new)

    if len(result) == 0:
        # there were no loops
        return captures
    else:
        return tuple(result)


def loop_captures(
    sweep: specs.Sweep[typing.Any, typing.Any, _TC],
    source_id: types.SourceID | None = None,
    *,
    only_fields: tuple[str, ...] | None = None,
) -> tuple[_TC, ...]:
    """evaluate the loop specification, and flatten into one list of loops"""

    if len(sweep.captures) > 0:
        cls = type(sweep.captures[0])
    elif sweep.__bindings__ is None:
        raise TypeError(
            'loops may apply only to explicit capture lists unless the sweep '
            'is bound to a sensor with striqt.sensor.bind_sensor'
        )
    else:
        from ..lib import bindings

        assert isinstance(sweep.__bindings__, bindings.SensorBinding)
        cls = sweep.__bindings__.schema.capture

    return _expand_capture_loops(
        sweep.captures,
        sweep.loops,
        sweep.adjust_captures,
        source_id=source_id,
        cls=cls,
        loop_only_nyquist=sweep.options.loop_only_nyquist,
        only_fields=only_fields,
    )


@util.lru_cache()
def adjust_analysis(
    analyses: specs.AnalysisGroup,
    adjust_analysis: immutabledict[str, typing.Any] | None,
) -> specs.AnalysisGroup:
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
        logger = util.get_logger('sweep')
        logger.warning(
            f'no analysis parameters match analysis_adjust keys {unused_names}'
        )

    return _deep_freeze(specs.BundledAnalysis.from_dict(result))


def varied_capture_fields(
    captures: tuple[specs.SensorCapture, ...], loops: tuple[specs.LoopSpec, ...]
) -> list[str]:
    """generate a list of capture fields with at least 2 values in the specified sweep"""

    inner_values = (c.to_dict().values() for c in captures)
    inner_counts = [len(Counter(v)) for v in zip(*inner_values)]
    fields = captures[0].to_dict().keys()
    inner_counts = dict(zip(fields, inner_counts))
    outer_counts = {loop.field: len(loop.get_points()) for loop in loops}
    totals = {
        field: max(inner_counts[field], outer_counts.get(field, 0)) for field in fields
    }
    return [f for f, c in totals.items() if c > 1]


@util.lru_cache()
def get_capture_type(sweep_cls: type[specs.Sweep]) -> type[specs.SensorCapture]:
    if sweep_cls.__bindings__ is not None:
        return sweep_cls.__bindings__.schema.capture
    else:
        captures_type = typing.get_type_hints(sweep_cls)['captures']
        return typing.get_args(captures_type)[0]


Capture = typing.TypeVar('Capture', bound=specs.SensorCapture)


@util.lru_cache()
def split_capture_ports(capture: Capture) -> list[Capture]:
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


@util.lru_cache()
def max_by_frequency(
    field: str,
    captures: tuple[specs.SoapyCapture, ...],
    loops: tuple[specs.LoopSpec, ...] = (),
) -> dict[types.Port, dict[types.CenterFrequency, typing.Any]]:
    """get the maximum value of a field across looped captures by center frequency"""

    map = {}
    looped_captures = _expand_capture_loops(
        captures, loops, only_fields=(field, 'center_frequency')
    )

    for c in looped_captures:
        for pc in split_capture_ports(c):
            current = map.setdefault(pc.port, {}).setdefault(pc.center_frequency, None)
            v = getattr(pc, field)

            if current is None or v > current:
                map[pc.port][pc.center_frequency] = v

    return map


@util.lru_cache()
def _get_capture_adjust_fields(
    spec: specs.AdjustCapturesType, source_id: str | None
) -> dict[str, str | specs.CaptureRemap]:
    fields = {}

    # the globals spec may use the source-specific spec
    for name, value in spec.get('defaults', {}).items():
        if name not in fields:
            fields[name] = value

    if isinstance(source_id, str):
        source_fields = spec.get(source_id, {})
        fields.update(source_fields)

    return fields


@util.lru_cache()
def _list_capture_adjustments(
    spec: specs.AdjustCapturesType, source_id: str | None
) -> tuple[str, ...]:
    ret = set()
    fields = _get_capture_adjust_fields(spec, source_id)
    for lookup_spec in fields.values():
        if isinstance(lookup_spec, str):
            continue

        if isinstance(lookup_spec.key, tuple):
            names = lookup_spec.key
        else:
            names = (lookup_spec.key,)

        for name in names:
            if name not in fields:
                ret.add(name)

    return tuple(ret)


@util.lru_cache()
def adjust_captures(
    capture: _TC,
    adjust_spec: specs.AdjustCapturesType,
    source_id: types.SourceID | None,
) -> _TC:
    """evaluate the field values"""

    if not isinstance(capture, specs.SensorCapture):
        raise TypeError('capture must be a specs.SensorCapture instance')

    ret = {}
    fields = _get_capture_adjust_fields(adjust_spec, source_id)

    def get_key(name: str, field: str):
        if name in ret:
            value = ret[name]
        elif hasattr(capture, name):
            value = getattr(capture, name)
        else:
            raise KeyError(f'no such key {name!r} for field {field!r}')

        return value

    def do_lookup(lookup_spec, key):
        if isinstance(key, tuple):
            return tuple(do_lookup(lookup_spec, k) for k in key)
        try:
            return lookup_spec.lookup[key]
        except KeyError:
            raise KeyError(
                f'adjust_captures[{field!r}] is missing a lookup with key {key!r} '
                f'for source {source_id!r}'
            )

    for field, lookup_spec in fields.items():
        if not isinstance(lookup_spec, specs.CaptureRemap):
            # no lookup
            ret[field] = lookup_spec
            continue
        elif capture is None:
            continue

        if isinstance(lookup_spec.key, tuple):
            # lookup on multiple fields
            key = tuple(get_key(k, field) for k in lookup_spec.key)
        else:
            # lookup on single field
            key = get_key(lookup_spec.key, field)

        ret[field] = do_lookup(lookup_spec, key)

    return capture.replace(**ret)


def _get_source_capture_adjustments(
    spec: specs.AdjustCapturesType,
    source_id: str | None,
) -> dict[str, typing.Any]:
    """get a map of capture adjustments that do not require lookups"""

    ret = {}
    fields = _get_capture_adjust_fields(spec, source_id)

    for field, lookup_spec in fields.items():
        if isinstance(lookup_spec, str):
            ret[field] = lookup_spec

    return ret


# @util.lru_cache()
# def apply_capture_adjustments(
#     capture: _TC,
#     spec: specs.AdjustCapturesType,
#     *,
#     source_id: str | None = None,
# ) -> _TC:
#     remap_by_port = defaultdict(list)
#     splits = split_capture_ports(capture)

#     for c in splits:
#         map = _adjust_captures_single_port(c, spec, source_id)
#         for k, v in map.items():
#             remap_by_port[k].append(v)

#     remap = {}
#     for field, values in remap_by_port.items():
#         counts = len(set(values))

#         if counts > 1:
#             remap[field] = tuple(values)
#             continue

#         capture_value = getattr(capture, field)
#         if isinstance(capture_value, tuple):
#             input_len = len(capture_value)
#         else:
#             input_len = 0

#         if input_len == 0:
#             remap[field] = values[0]
#         elif input_len == 1:
#             remap[field] = (values[0],)
#         else:
#             remap[field] = len(splits) * (values[0],)

#     return capture.replace(**remap)


@util.lru_cache()
def get_path_fields(
    sweep: specs.Sweep,
    *,
    source_id: str | typing.Callable[[], str],
    spec_path: Path | str | None = None,
) -> dict[str, str]:
    """return a mapping for string `'{field_name}'.format()` style mapping values"""

    assert isinstance(sweep, specs.Sweep)

    if callable(source_id):
        id_ = source_id()
    else:
        id_ = source_id

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


def ensure_tuple(obj: _T | tuple[_T, ...]) -> tuple[_T, ...]:
    if isinstance(obj, tuple):
        return obj
    else:
        return (obj,)


@util.lru_cache()
def get_unique_ports(
    captures: tuple[_TC, ...],
    loops: tuple[specs.LoopSpec, ...] | None = None,
) -> tuple[int, ...]:
    ports = set()

    if loops is not None:
        for l in loops:
            if l.field != 'port':
                continue
            looped_ports = typing.cast(list[specs.types.Port], l.get_points())
            for p in looped_ports:
                ports |= set(ensure_tuple(p))

    for c in captures:
        ports |= set(ensure_tuple(c.port))

    return tuple(sorted(ports))


@util.lru_cache()
def _get_format_fields(s: str):
    """
    Extracts and returns a list of formatting field names from a given format string.
    """
    formatter = string.Formatter()
    fields = []
    for _, field_name, *_ in formatter.parse(s):
        if field_name is not None:
            fields.append(field_name)
    return fields


def _convert_label_lookup_keys(sweep: specs.Sweep) -> specs.AdjustCapturesType:
    """convert label lookup keys types to match corresponding capture fields"""

    result = {}
    capture_cls = get_capture_type(type(sweep))

    field_types = {f.name: f.type for f in msgspec.structs.fields(capture_cls)}

    for source_id, lookup_map in sweep.adjust_captures.items():
        if source_id != 'defaults':
            try:
                bytes.fromhex(source_id)
            except ValueError:
                raise msgspec.ValidationError(
                    f'label source key {source_id!r} is not "global" or a hex string'
                )

        result[source_id] = {}
        lookup_types = dict(field_types)
        for field, v in lookup_map.items():
            if field == 'port' or field in specs.Capture.__struct_fields__:
                raise msgspec.ValidationError(
                    f'capture field {field!r} is not allowed by adjust_captures'
                )
            if field not in field_types:
                cls = get_capture_type(type(sweep))
                cls_repr = f'{cls.__module__}.{cls.__name__}'
                raise msgspec.ValidationError(
                    f'adjust_captures field {field!r} was not defined '
                    f'in capture class {cls_repr!r}'
                )
            elif not isinstance(v, specs.CaptureRemap):
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
            elif all(kc in lookup_types for kc in v.key):
                # defines lookup across multiple capture fields
                key_type = typing.Tuple[tuple(lookup_types[kc] for kc in v.key)]
            else:
                # prune refs with invalid lookups
                continue
                invalid = set(v.key) - set(lookup_types) - set(field_types)
                raise msgspec.ValidationError(
                    f'no such capture fields {invalid!r} for metadata field {field!r}'
                )
            try:
                lookup = {
                    msgspec.convert(k, key_type, strict=False): v
                    for k, v in v.lookup.items()
                }
            except msgspec.ValidationError as ex:
                raise msgspec.ValidationError(
                    f'keys must match type of {v.key!r} field(s) in lookup '
                    f'for {field!r} in label for {source_id!r} source'
                ) from ex

            result[source_id][field] = specs.CaptureRemap(key=v.key, lookup=lookup)
            lookup_types[field] = str

    fixed = msgspec.convert(result, specs.AdjustCapturesType, strict=False)
    return _deep_freeze(fixed)  # type: ignore


@util.lru_cache()
def list_capture_adjustments(
    sweep: specs.Sweep, source_id: str
) -> dict[str, tuple[str, ...]]:
    lookup_fields = _list_capture_adjustments(
        sweep.adjust_captures, source_id=source_id
    )
    captures = loop_captures(sweep, only_fields=lookup_fields, source_id=source_id)

    result = {}

    for c in captures:
        labels = adjust_captures(c, sweep.adjust_captures, source_id=source_id)
        for fields in labels:
            for name, value in fields.items():
                result.setdefault(name, {})[value] = None

    return {name: tuple(v.keys()) for name, v in result.items()}


class PathAliasFormatter:
    def __init__(
        self,
        sweep: specs.Sweep,
        spec_path: Path | str | None = None,
        alias_timeout: float = 5,
    ):
        self.sweep_spec = sweep
        self.spec_path = spec_path
        self.alias_timeout = alias_timeout

    def __call__(self, path: str | Path) -> str:
        path_fields = _get_format_fields(str(path))
        if len(path_fields) == 0:
            return str(path)

        from ..lib.sources._base import get_source_id

        id_ = get_source_id(self.sweep_spec.source, timeout=self.alias_timeout)
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
    captures: tuple[specs.SensorCapture, ...], *, min_size: int = 1
) -> list[int]:
    """return the minimum sizes of groups of captures that can be concatenated.

    This is important, because some channel analysis results produce a different
    shape depending on (sample_rate, analysis_bandwidth, duration).

    Returns:
        The list l of sizes of each group such that sum(l) == len(captures)
    """

    class C(specs.SensorCapture, frozen=True, forbid_unknown_fields=False):
        """minimal capture fields that safely ignore fields from subclasses"""

    remaining = convert_spec(captures, type=list[C])
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
