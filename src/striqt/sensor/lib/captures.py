"""utility functions for structs.CaptureBase data structures and their aliases"""

from __future__ import annotations
from collections import Counter
from datetime import datetime
import functools
import numbers
from pathlib import Path
import typing

from msgspec import UNSET, UnsetType

from . import specs, util
from striqt.analysis import dataarrays


@functools.lru_cache
def get_capture_type(sweep_cls: type[specs.SweepSpec]) -> type[specs.CaptureSpec]:
    captures_type = typing.get_type_hints(sweep_cls)['captures']
    return typing.get_args(captures_type)[0]


def _single_match(
    fields: dict[str, typing.Any],
    capture: specs.CaptureSpec | None,
    **extras: typing.Any,
) -> bool:
    """return True if all fields match in the specified fields.

    For each `{key: value}` in `fields`, a match requires that
    either `extras[key] == value` or `getattr(capture, key) == value`.
    """

    if capture is None:
        converted_fields = fields
    elif isinstance(capture.port, tuple):
        raise ValueError('split the capture to evaluate alias matches')
    else:
        all_fields = frozenset(capture.__struct_fields__)

        # type conversion for any search fields that are in 'capture'
        valid_fields = {k: v for k, v in fields.items() if k in all_fields}
        converted_fields = fields | capture.replace(**valid_fields).todict()

    for name, value in converted_fields.items():
        hits = (
            getattr(capture, name, UNSET),
            extras.get(name, UNSET),
        )

        if value in hits:
            continue
        elif isinstance(hits[0], tuple) and hits[0][0] == value:
            # is this special case still necessary?
            continue
        else:
            return False

    return True


def _match_fields(
    multi_fields: typing.Iterable[dict[str, typing.Any]],
    capture: specs.CaptureSpec | None,
    **extras: typing.Any,
) -> bool:
    """return True if all fields match in the specified fields.

    For each `{key: value}` in `fields`, a match requires that
    either `extras[key] == value` or `getattr(capture, key) == value`.
    """

    return any(_single_match(f, capture=capture, **extras) for f in multi_fields)


@util.lru_cache()
def evaluate_aliases(
    capture: specs.CaptureSpec | None,
    *,
    source_id: str | UnsetType | None = UNSET,
    output: specs.SinkSpec,
) -> dict[str, typing.Any]:
    """evaluate the field values"""

    ret = {}

    for coord_name, coord_spec in output.coord_aliases.items():
        for alias_value, field_spec in coord_spec.items():
            if isinstance(field_spec, dict):
                # "or" across the list of field specs
                field_spec = [field_spec]

            if not isinstance(field_spec, (list, tuple)):
                raise TypeError(
                    'match specification for field {alias_value!r} must be list or dict'
                )

            if _match_fields(field_spec, capture=capture, source_id=source_id, **ret):
                ret[coord_name] = alias_value
                break
    return ret


Capture = typing.TypeVar('Capture', bound=specs.CaptureSpec)


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


def capture_fields_with_aliases(
    capture: specs.CaptureSpec | None = None,
    *,
    source_id: str | None = None,
    sink_spec: specs.SinkSpec,
) -> dict:
    if capture is None:
        attrs = {}
        c = None
    else:
        attrs = capture.todict(skip_private=True)
        c = split_capture_ports(capture)[0]
    aliases = evaluate_aliases(c, source_id=source_id, output=sink_spec)

    return dict(attrs, **aliases)


def get_field_value(
    name: str,
    capture: specs.CaptureSpec,
    info: specs.AcquisitionInfo,
    alias_hits: dict,
):
    """get the value of a field in `capture`, injecting values for aliases"""
    if isinstance(capture.port, tuple):
        raise ValueError('split the capture before the call to get_capture_field')

    if hasattr(capture, name):
        value = getattr(capture, name)
        if isinstance(value, tuple):
            value = value[0]
    elif hasattr(info, name):
        value = getattr(info, name)
    elif name in alias_hits:
        value = alias_hits[name]
    else:
        raise KeyError
    return value


@util.lru_cache()
def _get_path_fields(
    sweep: specs.SweepSpec,
    *,
    source_id: str | typing.Callable[[], str],
    spec_path: Path | str | None = None,
    json_path: Path | str | None = None,
) -> dict[str, str]:
    """return a mapping for string `'{field_name}'.format()` style mapping values"""

    if callable(source_id):
        id_ = source_id()
    else:
        id_ = source_id

    fields = capture_fields_with_aliases(source_id=id_, sink_spec=sweep.sink)

    fields['start_time'] = datetime.now().strftime('%Y%m%d-%Hh%Mm%S')
    fields['sensor_bindings'] = type(sweep).__name__
    if spec_path is not None:
        fields['spec_name'] = Path(spec_path).stem
    fields['source_id'] = id_

    return fields


class PathAliasFormatter:
    def __init__(self, sweep: specs.SweepSpec, spec_path: Path | str | None = None):
        self.sweep_spec = sweep
        self.spec_path = spec_path

    def __call__(self, path: str | Path) -> str:
        from .sources.base import get_source_id

        id_ = get_source_id(self.sweep_spec.source)
        path = Path(path).expanduser()

        fields = _get_path_fields(
            self.sweep_spec, source_id=id_, spec_path=self.spec_path
        )

        try:
            path = Path(str(path).format(**fields))
        except KeyError as ex:
            valid_fields = ', '.join(fields.keys())
            raise ValueError(f'valid formatting fields are {valid_fields!r}') from ex

        return str(path)


class _NoSweep(specs.SweepSpec, frozen=True):
    # a sweep with captures that express only the parameters that impact capture shape
    source: specs.NullSourceSpec = specs.NullSourceSpec(
        base_clock_rate=0, num_rx_ports=1
    )
    captures: tuple[specs.analysis.CaptureBase, ...] = ()


def concat_group_sizes(
    captures: tuple[specs.CaptureSpec, ...], *, min_size: int = 1
) -> list[int]:
    """return the minimum sizes of groups of captures that can be concatenated.

    This is important, because some channel analysis results produce a different
    shape depending on (sample_rate, analysis_bandwidth, duration).

    Returns:
        The list l of sizes of each group such that sum(l) == len(captures)
    """

    remaining = list(_NoSweep(captures=captures).validate().captures)
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
