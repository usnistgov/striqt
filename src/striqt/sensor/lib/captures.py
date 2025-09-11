"""utility functions for structs.RadioCapture data structures"""

from __future__ import annotations
import numbers
from collections import Counter

from . import specs, util


@util.lru_cache(10000)
def broadcast_to_ports(
    ports: int | tuple[int, ...], *params, allow_mismatch=False
) -> list[tuple[int, ...]]:
    """broadcast sequences in each element in `params` up to the
    length of capture.port.
    """

    res = []
    if isinstance(ports, numbers.Number):
        count = 1
    else:
        count = len(ports)

    for p in params:
        if not isinstance(p, (tuple, list)):
            res.append((p,) * count)
        elif len(p) == count:
            res.append(tuple(p))
        elif allow_mismatch:
            res.append(tuple(p[:1]) * count)
        else:
            raise ValueError(
                f'cannot broadcast tuple of length {len(p)} to {count} ports'
            )

    return res


class _UNDEFINED_FIELD:
    pass


def _single_match(
    fields: dict[str],
    capture: specs.RadioCapture,
    **extras: dict[str],
) -> bool:
    """return True if all fields match in the specified fields.

    For each `{key: value}` in `fields`, a match requires that
    either `extras[key] == value` or `getattr(capture, key) == value`.
    """
    if isinstance(capture.port, tuple):
        raise ValueError('split the capture to evaluate alias matches')

    # type conversion for any search fields that are in 'capture'
    valid_fields = {k: v for k, v in fields.items() if k in capture.__struct_fields__}
    converted_fields = fields | capture.replace(**valid_fields).todict()

    for name, value in converted_fields.items():
        hits = (
            getattr(capture, name, _UNDEFINED_FIELD),
            extras.get(name, _UNDEFINED_FIELD),
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
    multi_fields: list[dict[str]],
    capture: specs.RadioCapture,
    **extras: dict[str],
) -> bool:
    """return True if all fields match in the specified fields.

    For each `{key: value}` in `fields`, a match requires that
    either `extras[key] == value` or `getattr(capture, key) == value`.
    """

    return any(_single_match(f, capture=capture, **extras) for f in multi_fields)


@util.lru_cache()
def evaluate_aliases(
    capture: specs.RadioCapture,
    *,
    radio_id: str | None = _UNDEFINED_FIELD,
    output: specs.Output,
) -> dict[str]:
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

            if _match_fields(field_spec, capture=capture, radio_id=radio_id, **ret):
                ret[coord_name] = alias_value
                break
    return ret


@util.lru_cache()
def split_capture_ports(capture: specs.RadioCapture) -> list[specs.RadioCapture]:
    """split a multi-channel capture into a list of single-channel captures.

    If capture is not a multi-channel capture (its channel field is just a number),
    then the returned list will be [capture].
    """

    if isinstance(capture.port, numbers.Number):
        return [capture]

    remaps = [dict() for i in range(len(capture.port))]

    for field in capture.__struct_fields__:
        values = getattr(capture, field)
        if not isinstance(values, tuple):
            continue

        for remap, value in zip(remaps, values):
            remap[field] = value

    return [capture.replace(**remap) for remap in remaps]


def capture_fields_with_aliases(
    capture: specs.RadioCapture, *, radio_id: str = None, output: specs.Output
) -> dict:
    attrs = capture.todict(skip_private=True)

    c = split_capture_ports(capture)[0]
    aliases = evaluate_aliases(c, radio_id=radio_id, output=output)

    return dict(attrs, **aliases)


def get_field_value(
    name: str,
    capture: specs.RadioCapture,
    radio_id: str,
    alias_hits: dict,
    extra_field_values: dict = {},
):
    """get the value of a field in `capture`, injecting values for aliases"""
    if isinstance(capture.port, tuple):
        raise ValueError('split the capture before the call to get_capture_field')

    if hasattr(capture, name):
        value = getattr(capture, name)
        if isinstance(value, tuple):
            value = value[0]
    elif name in alias_hits:
        value = alias_hits[name]
    elif name == 'radio_id':
        value = radio_id
    elif name in extra_field_values:
        value = extra_field_values[name]
    else:
        raise KeyError
    return value


class _MinSweep(specs.Sweep):
    # a sweep with captures that express only the parameters that impact data shape
    captures: list[specs.analysis.Capture]


def concat_group_sizes(
    captures: tuple[specs.RadioCapture, ...], *, min_size: int = 1
) -> list[int]:
    """return the minimum sizes of groups of captures that can be concatenated.

    This is important, because some channel analysis results produce a different
    shape depending on (sample_rate, analysis_bandwidth, duration).

    Returns:
        The list l of sizes of each group such that sum(l) == len(captures)
    """

    remaining = list(_MinSweep(captures=captures).validate().captures)
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
