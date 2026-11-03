"""units strings and SI prefixes of the quantities that striqt measures and labels.

A units string is the unit of stored values, such as ``'dBm'``, ``'mW'``, ``'√mW'``,
``'Hz'``, ``'S/s'``, or ``'dBm/Hz'``. The text left of the first ``/`` is the base
unit and the text right of it the denominator. A prefix applies to the base unit
alone, and never to a dB unit or to a unit that already carries a prefix.
"""

from __future__ import annotations as __

import dataclasses
import math
from collections.abc import Iterable
from typing import Literal, Union

from .util import lru_cache

PREFIXES: dict[int, str] = dict(
    zip(
        range(-15, 16, 3),
        ('f', 'p', 'n', '\N{MICRO SIGN}', 'm', '', 'k', 'M', 'G', 'T', 'P'),
    )
)
"""the SI prefix symbol of each decimal exponent, from femto to peta"""

_EXPONENTS = {symbol: exponent for exponent, symbol in PREFIXES.items()} | {'u': -6}
_WAVE = '√'
_DB_TO_LINEAR = {'dBm': 'mW', 'dBW': 'W', 'dB': 'unitless'}
_TO_DB = {v: k for k, v in _DB_TO_LINEAR.items()}
_TO_LINEAR = {**_DB_TO_LINEAR, **{v: v for v in _DB_TO_LINEAR.values()}}
_TO_WAVE = {k: _WAVE + v for k, v in _TO_LINEAR.items()}
_UNPREFIXED = frozenset({'mW', 'unitless'})


@dataclasses.dataclass(frozen=True)
class Units:
    """a units string split at its first ``/``."""

    base: str
    """the unit left of the first ``/``: ``'dBm'``, ``'mW'``, ``'√mW'``, ``'Hz'``, ``'S'``"""
    per: Union[str, None] = None
    """the denominator right of the first ``/``, or None without one"""

    def __str__(self) -> str:
        return self.base if self.per is None else f'{self.base}/{self.per}'


@dataclasses.dataclass(frozen=True)
class SIPrefix:
    """a decimal exponent that is a multiple of 3 and its SI symbol."""

    exponent: int
    """the decimal exponent, a multiple of 3 in the range of `PREFIXES`"""
    symbol: str
    """the prefix symbol, ``''`` at exponent 0"""


@lru_cache()
def parse_units(units: str) -> Units:
    """return `units` split into the base unit and the denominator.

    Args:
        units: a units string such as ``'dBm/Hz'`` or ``'S/s'``

    Returns:
        the `Units` with `base` the text before the first ``/`` and `per` the text
        after it, or None without a ``/``
    """
    base, slash, per = units.partition('/')
    return Units(base, per if slash else None)


def is_prefixable(units: Union[str, None]) -> bool:
    """return whether an SI prefix may scale the base unit of `units`.

    A dB unit (any base that begins ``dB``), a base that already carries a prefix
    (``'mW'``, ``'√mW'``), ``'unitless'``, and None are not prefixable. The empty
    string is a bare number that takes a prefix.
    """
    if units is None:
        return False
    base = parse_units(units).base.lstrip(_WAVE)
    return not (base.startswith('dB') or base in _UNPREFIXED)


def _rewrite_base(units: str, table: dict[str, str]) -> str:
    """return `units` with the base unit, less a ``√``, mapped through `table`."""
    parsed = parse_units(units)
    base = parsed.base.lstrip(_WAVE)
    if base not in table:
        return units
    return str(dataclasses.replace(parsed, base=table[base]))


def to_dB(units: str) -> str:
    """return the dB form of a linear power or amplitude `units` string.

    ``'mW'`` and ``'√mW'`` give ``'dBm'``, ``'W/Hz'`` gives ``'dBW/Hz'``, and
    ``'unitless'`` gives ``'dB'``. A string without a known linear base is returned
    unchanged.
    """
    return _rewrite_base(units, _TO_DB)


def to_linear(units: str) -> str:
    """return the linear power form of a dB or amplitude `units` string.

    ``'dBm'`` and ``'√mW'`` give ``'mW'``, ``'dBW/Hz'`` gives ``'W/Hz'``, and
    ``'dB'`` gives ``'unitless'``. A string without a known base is returned
    unchanged.
    """
    return _rewrite_base(units, _TO_LINEAR)


def to_wave(units: str) -> str:
    """return the amplitude (square root of power) form of a `units` string.

    ``'dBm'`` and ``'mW'`` give ``'√mW'``, and ``'dB'`` gives ``'√unitless'``. A
    string without a known base is returned unchanged.
    """
    return _rewrite_base(units, _TO_WAVE)


def _floor3(value: float) -> int:
    return int(math.floor(math.log10(value) / 3) * 3)


def _magnitude_exponent(magnitude: float, places: Union[int, None]) -> int:
    """return the exponent of `magnitude`, carried up when the mantissa rounds to 1000."""
    exponent = max(min(PREFIXES), min(max(PREFIXES), _floor3(magnitude)))
    fmt = 'g' if places is None else f'.{places}f'
    if float(format(magnitude / 10.0**exponent, fmt)) >= 1000 and exponent < max(
        PREFIXES
    ):
        exponent += 3
    return exponent


def si_prefix(
    values: Union[float, Iterable[float]],
    units: Union[str, None] = None,
    *,
    places: Union[int, None] = None,
) -> SIPrefix:
    """return one SI prefix for every value of `values` in `units`.

    The prefix is that of the largest finite nonzero magnitude, with a mantissa that
    rounds to 1000 under `places` carried to the next prefix. Two or more distinct
    magnitudes lower the prefix to the step between the closest pair, by at most one
    prefix, so that a sequence such as 3.7 and 3.75 GHz reads in MHz while every
    mantissa stays below 10⁶. Units that are not `is_prefixable`, and values with no
    finite nonzero magnitude, give the exponent 0.

    Args:
        values: one value, or the values to share the prefix; zero, NaN, and
            infinite values are ignored
        units: the units string of the values, or None for a bare number
        places: the decimal places the mantissas are shown with, or None for the
            shortest representation

    Returns:
        the `SIPrefix` with an exponent in the range of `PREFIXES`
    """
    if isinstance(values, (int, float)):
        values = (values,)
    magnitudes = sorted({
        abs(float(v)) for v in values if math.isfinite(v) and float(v)
    })
    if not magnitudes or not is_prefixable(units):
        return SIPrefix(0, '')
    exponent = _magnitude_exponent(magnitudes[-1], places)
    if len(magnitudes) > 1:
        step = min(b - a for a, b in zip(magnitudes, magnitudes[1:]))
        exponent = max(min(_floor3(step), exponent), exponent - 3, min(PREFIXES))
    return SIPrefix(exponent, PREFIXES[exponent])


@lru_cache()
def _format(
    value: float,
    units: Union[str, None],
    exponent: int,
    places: Union[int, None],
    sep: str,
) -> str:
    fmt = 'g' if places is None else f'.{places}f'
    mantissa = value / 10.0**exponent or 0.0  # one spelling for 0, -0, and 0.0
    text = format(mantissa, fmt)
    if units is None:
        return text
    suffix = f'{PREFIXES[exponent]}{units}'
    return f'{text}{sep}{suffix}' if suffix else text


def format_quantity(
    value: float,
    units: Union[str, None],
    *,
    prefix: Union[SIPrefix, str, Literal['auto']] = 'auto',
    places: Union[int, None] = None,
    sep: str = ' ',
) -> str:
    """return `value` in engineering notation with a prefixed `units` suffix.

    Args:
        value: the value in `units`; a numpy or cupy scalar is converted with
            `float`
        units: the units string of `value`, ``''`` for a bare prefix, or None to
            return the scaled mantissa alone
        prefix: ``'auto'`` selects `si_prefix` of `value`; a prefix symbol such as
            ``'M'`` or ``'u'`` forces that prefix; an `SIPrefix` is applied as given
        places: the decimal places of the mantissa, or None for the shortest
            representation (``'g'``)
        sep: the text between the mantissa and the suffix

    Returns:
        the mantissa, then `sep` and the prefix symbol and `units` when `units` is not
        None; ``'nan'``, ``'∞'``, or ``'-∞'`` for a value that is not finite, with the
        unprefixed `units` suffix

    Raises:
        ValueError: if `prefix` is a symbol that `PREFIXES` does not define
    """
    value = float(value)
    if not math.isfinite(value):
        text = 'nan' if math.isnan(value) else ('-∞' if value < 0 else '∞')
        return f'{text}{sep}{units}' if units else text
    if prefix == 'auto':
        exponent = si_prefix(value, units, places=places).exponent
    elif isinstance(prefix, SIPrefix):
        exponent = prefix.exponent
    elif prefix in _EXPONENTS:
        exponent = _EXPONENTS[prefix]
    else:
        raise ValueError(f'unknown SI prefix symbol {prefix!r}')
    return _format(value, units, exponent, places, sep)
