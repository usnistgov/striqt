"""text labels of measured quantities from dataset attrs and capture field metadata.

A label is composed from `standard_name`, `long_name`, `units`, and
`noise_bandwidth` by each call. No attr carries rendered text.
"""

from __future__ import annotations as __

import dataclasses
import math
from collections.abc import Mapping
from typing import Any, Literal, Union

import msgspec

import striqt.waveform as sw
from striqt.waveform.lib.units import SIPrefix

from .. import specs
from . import util


@dataclasses.dataclass(frozen=True)
class Measurand:
    """the facts that name a measured quantity, as a hashable key for the labels."""

    name: str
    """the variable or field name, the label when the 2 name attrs are None"""
    standard_name: Union[str, None] = None
    """the short noun phrase of the quantity, without units"""
    long_name: Union[str, None] = None
    """the long noun phrase of the quantity, without units"""
    units: Union[str, None] = None
    """the unit of the stored values, such as ``'dBm'`` or ``'Hz'``"""
    noise_bandwidth: Union[float, None] = None
    """the noise bandwidth (Hz) of a power quantity, or None"""


def measurand_from_attrs(
    attrs: Mapping[str, Any], name: Union[str, None] = None
) -> Measurand:
    """return the `Measurand` named by the `attrs` of a variable or coordinate.

    Args:
        attrs: the attrs of a variable or coordinate, of which `standard_name`,
            `long_name`, `units`, and `noise_bandwidth` are read
        name: the variable or coordinate name, or None for ``''``
    """
    bandwidth = attrs.get('noise_bandwidth', None)
    return Measurand(
        name='' if name is None else str(name),
        standard_name=_text_or_none(attrs.get('standard_name', None)),
        long_name=_text_or_none(attrs.get('long_name', None)),
        units=_text_or_none(attrs.get('units', None)),
        noise_bandwidth=None if bandwidth is None else float(bandwidth),
    )


def measurand_from_field(capture_cls: type[msgspec.Struct], field: str) -> Measurand:
    """return the `Measurand` of the `Meta` metadata of `field` in `capture_cls`.

    Raises:
        KeyError: when `capture_cls` has no field `field`
    """
    attrs = specs.helpers.get_capture_type_attrs(capture_cls)[field]
    return measurand_from_attrs(attrs, field)


def _text_or_none(value: Any) -> Union[str, None]:
    return None if value is None else str(value)


@util.lru_cache()
def units_label(m: Measurand, *, prefix: str = '') -> Union[str, None]:
    """return the units of `m` with `prefix` on the base unit and the bandwidth appended.

    The noise bandwidth is appended as in ``'dBm/360 kHz'`` when `m.noise_bandwidth`
    is set and `m.units` has no ``/``. `prefix` is dropped for units that
    `striqt.waveform.units.is_prefixable` rejects, such as a dB unit.

    Args:
        m: the measurand
        prefix: an SI prefix symbol placed before the base unit, ``''`` for none

    Returns:
        the units text, or None when `m.units` is None
    """
    if m.units is None:
        return None
    if not sw.units.is_prefixable(m.units):
        prefix = ''
    units = f'{prefix}{m.units}'
    if m.noise_bandwidth is None or '/' in m.units:
        return units
    return f'{units}/{sw.units.format_quantity(m.noise_bandwidth, "Hz")}'


@util.lru_cache()
def quantity_label(
    m: Measurand, *, short: bool = False, with_units: bool = True, prefix: str = ''
) -> str:
    """return the text that names `m`, with the units in parentheses.

    The text is the first of `m.long_name`, `m.standard_name`, and `m.name` that is
    not None. `short` puts `m.standard_name` first.

    Args:
        m: the measurand
        short: True selects `standard_name` over `long_name`
        with_units: True appends ``' (units)'`` from `units_label` when the measurand
            has units
        prefix: the SI prefix symbol placed before the base unit, ``''`` for none
    """
    names = (m.standard_name, m.long_name) if short else (m.long_name, m.standard_name)
    text = next((n for n in names if n is not None), m.name)
    units = units_label(m, prefix=prefix) if with_units else None
    return text if units is None else f'{text} ({units})'


def describe_value(
    value: Any,
    m: Union[Measurand, str, None],
    *,
    prefix: Union[SIPrefix, str, Literal['auto']] = 'auto',
) -> str:
    """return `value` as text in the units of `m`.

    Args:
        value: a number, a tuple of numbers, a string, or None
        m: the measurand, a units string, or None for no units
        prefix: the SI prefix of the numbers: ``'auto'`` selects one prefix shared
            across a tuple, a symbol selects that prefix

    Returns:
        ``'None'`` for None, the members of a tuple in parentheses, a finite number
        in engineering notation with the units when `m` has units, and otherwise
        ``repr(value)``
    """
    units = m.units if isinstance(m, Measurand) else m
    if value is None:
        return 'None'
    if isinstance(value, tuple):
        if prefix == 'auto':
            numbers = [v for v in value if isinstance(v, (int, float))]
            prefix = sw.units.si_prefix(numbers, units)
        return f'({", ".join(describe_value(v, units, prefix=prefix) for v in value)})'
    if units is None or isinstance(value, (str, bool)):
        return repr(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return repr(value)
    if not math.isfinite(number):
        return repr(value)
    return sw.units.format_quantity(number, units, prefix=prefix)


def describe_field(capture: specs.Capture, name: str, *, sep: str = '=') -> str:
    """return ``'<name><sep><value>'`` for the field `name` of `capture`.

    The value is written in the units of the field.

    Raises:
        KeyError: when `capture` has no field `name`
    """
    m = measurand_from_field(type(capture), name)
    return f'{name}{sep}{describe_value(getattr(capture, name), m)}'
