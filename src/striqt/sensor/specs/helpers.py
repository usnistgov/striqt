"""helper functions for specification data structures and their aliases"""

from __future__ import annotations as __

from collections import Counter, defaultdict, ChainMap
from fractions import Fraction
import functools
import itertools
import numbers
import string
import typing
from datetime import datetime
from pathlib import Path

import msgspec

import striqt.analysis as sa
from striqt.analysis.specs.helpers import _dec_hook, frozendict

from . import structs as specs
from . import types
from .structs import _TS, _TC, _TP
from ..lib import util


if typing.TYPE_CHECKING:
    _T = typing.TypeVar('_T')


@sa.util.lru_cache()
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


@sa.util.lru_cache()
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


@sa.util.lru_cache()
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

    if only_fields is not None:
        loops = tuple(l for l in loops if l.field in only_fields)

    if len(captures) == 0 and len(loops) == 0:
        return ()
    if cls is None:
        assert len(captures) > 0
        cls = type(captures[0])
    assert issubclass(cls, specs.Capture)

    loop_points = {
        loop.field: loop.get_points() for loop in loops if loop.field is not None
    }
    _check_fields(cls, tuple(loop_points.keys()), False)
    defaults = {k: v[0] for k, v in loop_points.items() if len(v) > 0}
    combinations = itertools.product(*loop_points)

    cdicts = typing.cast(tuple[dict, ...], to_builtins(captures))

    result = []
    for values in combinations:
        updates = dict(zip(loop_points.keys(), values))
        if len(cdicts) > 0:
            # iterate specified captures if available
            new = (defaults | c | updates for c in cdicts)
        else:
            # otherwise, instances are new captures
            new = [defaults | updates]

        if adjust is not None:
            new = (c | adjust_captures(c, adjust, source_id) for c in new)

        result += list(new)

    if len(result) == 0:
        # there were no loops
        return tuple()
    else:
        captures = msgspec.convert(
            result, tuple[cls, ...], strict=False, dec_hook=_dec_hook
        )

    if loop_only_nyquist:
        captures = tuple(c for c in captures if c.sample_rate >= c.analysis_bandwidth)

    return tuple()


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


@sa.util.lru_cache()
def adjust_analysis(
    analyses: specs.AnalysisGroup,
    adjust_analysis: typing.Mapping[str, typing.Any] | None,
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
        logger = sa.util.get_logger('sweep')
        logger.warning(
            f'no analysis parameters match analysis_adjust keys {unused_names}'
        )

    return sa.specs.helpers.freeze(specs.BundledAnalysis.from_dict(result))


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


@sa.util.lru_cache()
def get_capture_type(sweep_cls: type[specs.Sweep]) -> type[specs.SensorCapture]:
    if sweep_cls.__bindings__ is not None:
        return sweep_cls.__bindings__.schema.capture
    else:
        captures_type = typing.get_type_hints(sweep_cls)['captures']
        return typing.get_args(captures_type)[0]


Capture = typing.TypeVar('Capture', bound=specs.SensorCapture)


@sa.util.lru_cache()
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


@sa.util.lru_cache()
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


@sa.util.lru_cache()
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


@sa.util.lru_cache()
def _get_capture_adjust_dependencies(
    spec: specs.AdjustCapturesType, source_id: str | None
) -> dict[str, str]:
    adjust_specs = _get_capture_adjust_fields(spec, source_id)
    deps = {}
    for name, s in adjust_specs.items():
        if not isinstance(s, specs.CaptureRemap):
            continue
        if isinstance(s.key, str):
            keys = ((s.key),)
        else:
            keys = s.key
        for k in keys:
            deps.setdefault(k, name)
    return deps


@sa.util.lru_cache()
def describe_capture(
    capture: specs.Capture,
    fields: tuple[str, ...],
    *,
    adjust_spec: specs.AdjustCapturesType | None = None,
    source_id: specs.types.SourceID | None,
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


@sa.util.lru_cache()
def _list_capture_adjustments(
    spec: specs.AdjustCapturesType, source_id: str | None
) -> tuple[str, ...]:
    ret = set()
    fields = _get_capture_adjust_fields(spec, source_id)
    for lookup_spec in fields.values():
        if not isinstance(lookup_spec, specs.CaptureRemap):
            continue

        if isinstance(lookup_spec.key, tuple):
            names = lookup_spec.key
        else:
            names = (lookup_spec.key,)

        for name in names:
            if name not in fields:
                ret.add(name)

    return tuple(ret)


def adjust_captures(
    capture: typing.Mapping[str, typing.Any],
    adjust_spec: specs.AdjustCapturesType,
    source_id: types.SourceID | None,
) -> dict[str, typing.Any]:
    """evaluate the field values"""

    if not isinstance(capture, (dict, frozendict)):
        raise TypeError('capture must be a dict or mapping')

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
        lookup_spec: specs.CaptureRemap,
        key,
        field_default: str | specs.CaptureRemap | None,
    ):
        if not isinstance(key, tuple):
            k = key
        elif len(key) == 1:
            k = key[0]
        else:
            # per-port value in the capture
            return tuple(lookup_spec.lookup[k] for k in key)
        try:
            return lookup_spec.lookup[k]
        except KeyError:
            has_default = isinstance(field_default, specs.CaptureRemap)
            if lookup_spec.default != msgspec.UNSET:
                return lookup_spec.default
            elif has_default and field_default.default != msgspec.UNSET:
                return field_default.default
            elif not lookup_spec.required:
                return msgspec.UNSET
            elif has_default and not field_default.required:
                return msgspec.UNSET
            else:
                raise KeyError(
                    f'adjust_captures[{field!r}] is missing a lookup for key {k!r} '
                    f'for source {source_id!r}'
                )

    defaults = adjust_spec.get('defaults', {})
    default_fields = {
        k: v for k, v in defaults.items() if isinstance(v, specs.CaptureRemap)
    }
    for field, lookup_spec in fields.items():
        if not isinstance(lookup_spec, specs.CaptureRemap):
            # no lookup - use value
            ret[field] = lookup_spec
            continue

        key = get_key(lookup_spec.key, field)
        v = do_lookup(lookup_spec, key, field_default=default_fields.get(field, None))
        if v != msgspec.UNSET:
            ret[field] = v

    return ret


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


@sa.util.lru_cache()
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


@sa.util.lru_cache()
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
            elif len(v.key) == 1:
                key_type = lookup_types[v.key[0]]
            elif all(kc in lookup_types for kc in v.key):
                key_type = typing.Tuple[tuple(lookup_types[kc] for kc in v.key)]
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
                raise msgspec.ValidationError(
                    f'keys must match type of {v.key!r} field(s) in lookup '
                    f'for {field!r} in label for {source_id!r} source'
                ) from ex

            result[source_id][field] = specs.CaptureRemap(
                key=v.key, lookup=lookup, default=v.default, required=v.required
            )
            lookup_types[field] = str

    depth = sa.specs.helpers.inspect_freeze_depths(type(sweep))['adjust_captures']
    fixed = msgspec.convert(result, specs.AdjustCapturesType, strict=False)
    return sa.specs.helpers.freeze(fixed, depth)  # type: ignore


def to_builtins(obj: typing.Any) -> typing.Any:
    return msgspec.to_builtins(obj, enc_hook=sa.specs.helpers._enc_hook)


@sa.util.lru_cache()
def list_capture_adjustments(
    sweep: specs.Sweep[typing.Any, typing.Any, _TC], source_id: str
) -> dict[str, tuple[str, ...]]:
    lookup_fields = _list_capture_adjustments(
        sweep.adjust_captures, source_id=source_id
    )
    captures = loop_captures(sweep, only_fields=lookup_fields, source_id=source_id)
    cdicts = typing.cast(tuple[dict[str, typing.Any], ...], to_builtins(captures))
    result = defaultdict(dict)

    for c in cdicts:
        changes = adjust_captures(c, sweep.adjust_captures, source_id=source_id)
        for name, value in changes.items():
            result[name][value] = None

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


# %% msgspec field type introspection
def _infer_field_template(type_: msgspec.inspect.Type) -> tuple[dict, typing.Any]:
    """returns an (attrs, default_value) pair for the given msgspec field type"""
    from msgspec import inspect as mi
    import pandas as pd

    BUILTINS = {
        mi.FloatType: 0.0,
        mi.BoolType: False,
        mi.IntType: 0,
        mi.StrType: '',
        mi.DictType: {},
    }

    if not isinstance(type_, mi.Type):
        type_ = mi.type_info(type_)

    if isinstance(type_, tuple(BUILTINS.keys())):
        # dicey if subclasses show up
        return {}, BUILTINS[type(type_)]
    elif isinstance(type_, mi.CustomType):
        if issubclass(type_.cls, pd.Timestamp):
            return {}, pd.Timestamp(0)
        else:
            try:
                return {}, type_.cls()
            except Exception as ex:
                name = type_.cls.__qualname__
                raise TypeError(f'failed to make default for type {name!r}') from ex
    elif isinstance(type_, mi.Metadata):
        return type_.extra or {}, _infer_field_template(type_.type)[1]
    elif isinstance(type_, mi.LiteralType):
        return {}, type(type_.values[0])
    elif isinstance(type_, mi.UnionType):
        UNION_SKIP = (mi.NoneType, mi.VarTupleType)
        types = [t for t in type_.types if not isinstance(t, UNION_SKIP)]
        if len(types) == 1:
            return _infer_field_template(types[0])
        else:
            names = tuple(type(t).__qualname__ for t in types)
            raise TypeError(
                f'cannot determine xarray type for union of msgspec types {names!r}'
            )
    else:
        raise TypeError(f'unsupported msgspec field type {type_!r}')


@sa.util.lru_cache()
def field_template_values(
    spec_cls: type[msgspec.Struct],
) -> tuple[dict[str, typing.Any], dict[str, typing.Any]]:
    """returns a cached xr.Coordinates object to use as a template for data results"""

    attrs = {}
    defaults = {}

    for field in msgspec.structs.fields(spec_cls):
        if field.name.startswith('_') or field.name == 'adjust_analysis':
            continue

        attrs[field.name], defaults[field.name] = _infer_field_template(field.type)

    return attrs, defaults


@sa.util.lru_cache()
def get_field_metadata(struct: type[specs.SpecBase], field: str) -> dict[str, str]:
    """introspect an attrs dict for xarray from the specified field in `struct`"""
    hints = typing.get_type_hints(struct, include_extras=True)

    try:
        metas = hints[field].__metadata__
    except (AttributeError, KeyError):
        return {}

    if len(metas) == 0:
        return {}
    elif len(metas) == 1 and isinstance(metas[0], msgspec.Meta):
        return metas[0].extra
    else:
        raise TypeError(
            'Annotated[] type hints must contain exactly one msgspec.Meta object'
        )
