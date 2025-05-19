"""utility functions for structs.RadioCapture data structures"""

from __future__ import annotations
import numbers

from . import specs, util


@util.lru_cache(10000)
def broadcast_to_channels(
    channels: int | tuple[int, ...], *params, allow_mismatch=False
) -> list[tuple[int, ...]]:
    """broadcast sequences in each element in `params` up to the
    length of capture.channel.
    """

    res = []
    if isinstance(channels, numbers.Number):
        count = 1
    else:
        count = len(channels)

    for p in params:
        if not isinstance(p, (tuple, list)):
            res.append((p,) * count)
        elif len(p) == count:
            res.append(tuple(p))
        elif allow_mismatch:
            res.append(tuple(p[:1]) * count)
        else:
            raise ValueError(
                f'cannot broadcast tuple of length {len(p)} to {count} channels'
            )

    return res


def _match_capture_fields(
    capture: specs.RadioCapture, fields: dict[str], radio_id: str | None = None
) -> bool:
    if isinstance(capture.channel, tuple):
        raise ValueError('split the capture to evaluate alias matches')

    for name, value in fields.items():
        if name == 'radio_id' and radio_id is not None and value == radio_id:
            continue

        if not hasattr(capture, name):
            return False

        capture_value = getattr(capture, name)

        if isinstance(capture_value, tuple):
            capture_value = capture_value[0]

        if capture_value != value:
            return False

    return True


@util.lru_cache()
def evaluate_aliases(
    capture: specs.RadioCapture,
    *,
    radio_id: str | None = None,
    output: specs.Output,
) -> dict[str]:
    """evaluate the field values"""

    ret = {}

    for coord_name, coord_spec in output.coord_aliases.items():
        for alias_value, field_spec in coord_spec.items():
            if _match_capture_fields(capture, field_spec, radio_id):
                ret[coord_name] = alias_value
                break
    return ret


@util.lru_cache()
def split_capture_channels(capture: specs.RadioCapture) -> list[specs.RadioCapture]:
    """split a multi-channel capture into a list of single-channel captures.

    If capture is not a multi-channel capture (its channel field is just a number),
    then the returned list will be [capture].
    """

    if isinstance(capture.channel, numbers.Number):
        return [capture]

    remaps = [dict() for i in range(len(capture.channel))]

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
    attrs = capture.todict()
    c = split_capture_channels(capture)[0]
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
    if isinstance(capture.channel, tuple):
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

    remaining = _MinSweep(captures=captures).validate().captures
    whole_set = set(remaining)

    pending = []
    sizes = []
    count = 0

    while len(remaining) > 0:
        if count >= min_size and set(pending) == set(remaining) == whole_set:
            # make sure that the pending and remaining captures
            # will result in equivalent shapes when concatenated
            sizes.append(count)
            count = 0
            pending = []

        count += 1
        pending.append(remaining.pop(0))

    if count > 0:
        sizes.append(count)

    return sizes
