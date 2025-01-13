"""utility functions for structs.RadioCapture data structures"""

from __future__ import annotations
import functools
import math
import msgspec
import numbers
import typing

from . import structs, util

if typing.TYPE_CHECKING:
    import numpy as np
    import channel_analysis
else:
    np = util.lazy_import('numpy')
    channel_analysis = util.lazy_import('channel_analysis')


_ENG_PREFIXES = {
    -30: 'q',
    -27: 'r',
    -24: 'y',
    -21: 'z',
    -18: 'a',
    -15: 'f',
    -12: 'p',
    -9: 'n',
    -6: '\N{MICRO SIGN}',
    -3: 'm',
    0: '',
    3: 'k',
    6: 'M',
    9: 'G',
    12: 'T',
    15: 'P',
    18: 'E',
    21: 'Z',
    24: 'Y',
    27: 'R',
    30: 'Q',
}


@functools.lru_cache
def describe_capture(
    this: structs.RadioCapture | None,
    prev: structs.RadioCapture | None = None,
    *,
    index: int,
    count: int,
) -> str:
    """generate a description of a capture"""
    if this is None:
        if prev is None:
            return 'saving last analysis'
        else:
            return 'performing last analysis'

    diffs = []

    for name in type(this).__struct_fields__:
        if name == 'start_time':
            continue
        value = getattr(this, name)
        if prev is None or value != getattr(prev, name):
            diffs.append(_describe_field(this, name))

    capture_diff = ', '.join(diffs)

    if index is not None:
        progress = str(index + 1)

        if count is not None:
            progress = f'{progress}/{count}'

        progress = progress + ' '
    else:
        progress = ''

    return progress + capture_diff


@functools.lru_cache()
def format_units(value, unit='', places=None, sep=' ') -> str:
    """Format a number with SI unit prefixes"""

    sign = 1
    fmt = 'g' if places is None else f'.{places:d}f'

    if value < 0:
        sign = -1
        value = -value

    if value != 0:
        pow10 = int(math.floor(math.log10(value) / 3) * 3)
    else:
        pow10 = 0
        # Force value to zero, to avoid inconsistencies like
        # format_eng(-0) = "0" and format_eng(0.0) = "0"
        # but format_eng(-0.0) = "-0.0"
        value = 0.0

    pow10 = np.clip(pow10, min(_ENG_PREFIXES), max(_ENG_PREFIXES))

    mant = sign * value / (10.0**pow10)
    # Taking care of the cases like 999.9..., which may be rounded to 1000
    # instead of 1 k.  Beware of the corner case of values that are beyond
    # the range of SI prefixes (i.e. > 'Y').
    if abs(float(format(mant, fmt))) >= 1000 and pow10 < max(_ENG_PREFIXES):
        mant /= 1000
        pow10 += 3

    unit_prefix = _ENG_PREFIXES[int(pow10)]
    if unit or unit_prefix:
        suffix = f'{sep}{unit_prefix}{unit}'
    else:
        suffix = ''

    return f'{mant:{fmt}}{suffix}'


def _describe_field(capture: structs.RadioCapture, name: str):
    meta = channel_analysis.structs.get_capture_type_attrs(type(capture))
    attrs = meta[name]
    value = getattr(capture, name)

    if value is None:
        value_str = 'None'
    elif attrs.get('units', None) is not None and np.isfinite(value):
        if isinstance(value, tuple):
            value_tup = [format_units(v, attrs['units']) for v in value]
            value_str = f'({", ".join(value_tup)})'
        else:
            value_str = format_units(value, attrs['units'])
    else:
        value_str = repr(value)

    return f'{name}={value_str}'


@functools.lru_cache(10000)
def broadcast_to_channels(
    channels: int | tuple[int, ...], *params, allow_mismatch=False
) -> list[list]:
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
    capture: structs.RadioCapture, fields: dict[str], radio_id: str
):
    if isinstance(capture.channel, tuple):
        raise ValueError('split the capture to evaluate alias matches')

    for name, value in fields.items():
        if name == 'radio_id' and value == radio_id:
            continue

        if not hasattr(capture, name):
            return False

        capture_value = getattr(capture, name)

        if isinstance(capture_value, tuple):
            capture_value = capture_value[0]

        if capture_value != value:
            return False

    return True


@functools.lru_cache()
def evaluate_aliases(
    capture: structs.RadioCapture, radio_id: str, output: structs.Output
):
    """evaluate the field values"""

    ret = {}

    for coord_name, coord_spec in output.coord_aliases.items():
        for alias_value, field_spec in coord_spec.items():
            if _match_capture_fields(capture, field_spec, radio_id):
                ret[coord_name] = alias_value
                break
    return ret


@functools.lru_cache()
def split_capture_channels(capture: structs.RadioCapture) -> list[structs.RadioCapture]:
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

    return [msgspec.structs.replace(capture, **remap) for remap in remaps]


def capture_fields_with_aliases(
    capture: structs.RadioCapture, radio_id: str, output: structs.Output
) -> dict:
    attrs = structs.struct_to_builtins(capture)
    c = split_capture_channels(capture)[0]
    aliases = evaluate_aliases(c, radio_id, output)

    return dict(attrs, **aliases)


def get_field_value(
    name,
    capture: structs.RadioCapture,
    radio_id: str,
    alias_hits: dict,
    extra_field_values: dict = {},
):
    """get the value of a field in `capture`, injecting values for aliases"""
    if isinstance(capture.channel, tuple):
        raise ValueError('split the capture before the call to get_capture_field')

    # aliases = output.coord_aliases
    # if len(aliases) > 0:
    #     alias_hits = _evaluate_aliases(capture, radio_id, output)

    if hasattr(capture, name):
        value = getattr(capture, name)
        if isinstance(value, tuple):
            value = value[0]
    elif name in alias_hits:
        # default_type = type(next(iter(aliases[name].values())))
        value = alias_hits[name]
    elif name == 'radio_id':
        value = radio_id
    elif name in extra_field_values:
        value = extra_field_values[name]
    else:
        raise KeyError
    return value
