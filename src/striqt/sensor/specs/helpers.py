"""helper functions for specification data structures and their aliases"""

from __future__ import annotations as __

from collections import Counter
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
def loop_captures_from_fields(
    captures: tuple[_TC, ...],
    loops: tuple[specs.LoopSpec, ...],
    *,
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

        result += list(new)

    if len(result) == 0:
        # there were no loops
        return captures
    else:
        return tuple(result)


def loop_captures(
    sweep: specs.Sweep[typing.Any, typing.Any, _TC],
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

    return loop_captures_from_fields(
        sweep.captures,
        sweep.loops,
        cls=cls,
        loop_only_nyquist=sweep.options.loop_only_nyquist,
        only_fields=only_fields,
    )


def varied_capture_fields(
    captures: tuple[specs.ResampledCapture, ...], loops: tuple[specs.LoopSpec, ...]
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
def get_capture_type(sweep_cls: type[specs.Sweep]) -> type[specs.ResampledCapture]:
    captures_type = typing.get_type_hints(sweep_cls)['captures']
    return typing.get_args(captures_type)[0]


Capture = typing.TypeVar('Capture', bound=specs.ResampledCapture)


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
    looped_captures = loop_captures_from_fields(
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
def _get_label_fields(
    label_spec: specs.LabelDictType,
    source_id: str | UnsetType | None = UNSET,
) -> dict[str, str | specs.LabelLookup]:
    fields = {}
    if isinstance(source_id, str):
        source_fields = label_spec.get(source_id, {})
        fields.update(source_fields)

    # the globals spec may use the source-specific spec
    for name, value in label_spec.get('defaults', {}).items():
        if name not in fields:
            fields[name] = value

    return fields


@util.lru_cache()
def _list_label_capture_fields(
    label_spec: specs.LabelDictType,
    source_id: str | UnsetType | None = UNSET,
) -> tuple[str, ...]:
    ret = set()
    fields = _get_label_fields(label_spec, source_id)
    for field, lookup_spec in fields.items():
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


def _get_label(
    capture: specs.ResampledCapture | None,
    label_spec: specs.LabelDictType,
    source_id: str | UnsetType | None = UNSET,
) -> dict[str, typing.Any]:
    """evaluate the field values"""

    ret = {}
    fields = _get_label_fields(label_spec, source_id)

    def get_key(capture, name: str, field: str):
        if hasattr(capture, name):
            return getattr(capture, name)
        elif name in ret:
            return ret[name]
        else:
            raise KeyError(f'no such key {name!r} for field {field!r}')

    for field, lookup_spec in fields.items():
        if isinstance(lookup_spec, str):
            ret[field] = lookup_spec
            continue
        elif capture is None:
            continue

        if isinstance(lookup_spec.key, tuple):
            key = tuple(get_key(capture, k, field) for k in lookup_spec.key)
        else:
            key = get_key(capture, lookup_spec.key, field)

        try:
            ret[field] = lookup_spec.lookup[key]
        except KeyError:
            raise KeyError(
                f'label {field!r} is missing a lookup with key {key!r} '
                f'for source {source_id!r}'
            )

    return ret


@typing.overload
def get_labels(
    capture: None, spec: specs.LabelDictType, *, source_id: str | None = None
) -> immutabledict[str, str]:
    pass


@typing.overload
def get_labels(
    capture: specs.ResampledCapture,
    spec: specs.LabelDictType,
    *,
    source_id: str | None = None,
) -> tuple[immutabledict[str, str], ...]:
    pass


@util.lru_cache()
def get_labels(
    capture: specs.ResampledCapture | None,
    spec: specs.LabelDictType,
    *,
    source_id: str | None = None,
) -> tuple[immutabledict[str, str], ...] | immutabledict[str, str]:
    if capture is None:
        labels = _get_label(None, spec, source_id)
        return _deep_freeze(labels)

    labels = []
    for c in split_capture_ports(capture):
        labels.append(_get_label(c, spec, source_id))
    return _deep_freeze(labels)


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

    labels = get_labels(None, source_id=id_, spec=sweep.labels)
    fields.update(labels)

    return fields


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


def _convert_label_lookup_keys(sweep: specs.Sweep) -> specs.LabelDictType:
    """convert label lookup keys types to match corresponding capture fields"""

    result = {}
    capture_cls = get_capture_type(type(sweep))

    field_types = {f.name: f.type for f in msgspec.structs.fields(capture_cls)}

    for source_id, lookup_map in sweep.labels.items():
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
            if field in field_types:
                raise msgspec.ValidationError(
                    f'lookup field name {field!r} conflicts with a capture field'
                )
            elif not isinstance(v, specs.LabelLookup):
                # defines a fixed value
                result[source_id][field] = v
                lookup_types[field] = str
                continue
            elif not isinstance(v.key, tuple):
                # defines lookup on a single field
                if v.key not in lookup_types:
                    raise msgspec.ValidationError(
                        f'no metadata capture lookup with key {v.key!r} in source {source_id!r}'
                    )
                key_type = lookup_types[v.key]
            elif all(kc in lookup_types for kc in v.key):
                # defines lookup across multiple capture fields
                key_type = tuple[*tuple(lookup_types[kc] for kc in v.key)]
            else:
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

            result[source_id][field] = specs.LabelLookup(key=v.key, lookup=lookup)
            lookup_types[field] = str

    fixed = msgspec.convert(result, specs.LabelDictType, strict=False)
    return _deep_freeze(fixed)  # type: ignore


@util.lru_cache()
def list_all_labels(sweep: specs.Sweep, source_id: str) -> dict[str, tuple[str, ...]]:
    lookup_fields = _list_label_capture_fields(sweep.labels, source_id=source_id)
    captures = loop_captures(sweep, only_fields=lookup_fields)

    result = {}

    for c in captures:
        labels = get_labels(c, sweep.labels, source_id=source_id)
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
                f'invalid format field {key!r} in path {path!r}\n'
                f'available fields: {available!r}'
            ) from ex

        return str(path)


def concat_group_sizes(
    captures: tuple[specs.ResampledCapture, ...], *, min_size: int = 1
) -> list[int]:
    """return the minimum sizes of groups of captures that can be concatenated.

    This is important, because some channel analysis results produce a different
    shape depending on (sample_rate, analysis_bandwidth, duration).

    Returns:
        The list l of sizes of each group such that sum(l) == len(captures)
    """

    class C(specs.ResampledCapture, frozen=True, forbid_unknown_fields=False):
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
