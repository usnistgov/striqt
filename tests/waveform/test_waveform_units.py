"""striqt.waveform.lib.units: units strings, dB and linear forms, the shared SI prefix,
and engineering notation
"""

from __future__ import annotations

import math

import pytest
from hypothesis import given, strategies as st

from striqt.waveform import units as U

# %% parse_units / is_prefixable


@pytest.mark.parametrize(
    'text, base, per',
    [
        ('dBm', 'dBm', None),
        ('dBm/Hz', 'dBm', 'Hz'),
        ('S/s', 'S', 's'),
        ('√mW', '√mW', None),
        ('', '', None),
        ('a/b/c', 'a', 'b/c'),
    ],
)
def test_parse_units_splits_at_the_first_slash(text, base, per):
    parsed = U.parse_units(text)
    assert (parsed.base, parsed.per) == (base, per)
    assert str(parsed) == text


@pytest.mark.parametrize(
    'text, expected',
    [
        ('Hz', True),
        ('S/s', True),
        ('s', True),
        ('W', True),
        ('√W', True),
        ('K', True),
        ('B', True),
        ('dBm', False),
        ('dB', False),
        ('dBm/Hz', False),
        ('dB power', False),
        ('dBfs', False),
        ('mW', False),
        ('√mW', False),
        ('unitless', False),
        ('', True),
        (None, False),
    ],
)
def test_is_prefixable(text, expected):
    assert U.is_prefixable(text) is expected


# %% to_dB / to_linear / to_wave

UNIT_PAIRS = [
    ('dBm', 'mW'),
    ('dBW', 'W'),
    ('dB', 'unitless'),
    ('dBm/Hz', 'mW/Hz'),
    ('dBW/Hz', 'W/Hz'),
]
WAVE_PAIRS = [('dBm', '√mW'), ('dBW', '√W'), ('dB', '√unitless')]


@pytest.mark.parametrize('dB_unit, linear_unit', UNIT_PAIRS)
def test_dB_and_linear_forms_are_inverses(dB_unit, linear_unit):
    assert U.to_linear(dB_unit) == linear_unit
    assert U.to_dB(linear_unit) == dB_unit
    assert U.to_dB(U.to_linear(dB_unit)) == dB_unit
    assert U.to_linear(U.to_dB(linear_unit)) == linear_unit


@pytest.mark.parametrize('dB_unit, wave_unit', WAVE_PAIRS)
def test_wave_form_round_trips(dB_unit, wave_unit):
    assert U.to_wave(dB_unit) == wave_unit
    assert U.to_dB(wave_unit) == dB_unit
    assert U.to_linear(wave_unit) == U.to_linear(dB_unit)
    assert U.to_wave(U.to_linear(dB_unit)) == wave_unit


def test_conversions_are_idempotent_on_their_own_form():
    assert U.to_dB('dBm/Hz') == 'dBm/Hz'
    assert U.to_linear('mW') == 'mW'
    assert U.to_wave('√mW') == '√mW'


@pytest.mark.parametrize('convert', [U.to_dB, U.to_linear, U.to_wave])
def test_unknown_units_pass_through(convert):
    assert convert('counts') == 'counts'
    assert convert('counts/s') == 'counts/s'


# %% si_prefix


@pytest.mark.parametrize(
    'values, units, exponent',
    [
        ([3.7e9, 3.75e9], 'Hz', 6),
        ([3.75e9], 'Hz', 9),
        ([1e9], 'Hz', 9),
        ([1e6, 2e6, 4e6], 'S/s', 6),
        ([1e-3], 's', -3),
        ([0.0005], 's', -6),
        ([-120.0, -60.0], 'dBm', 0),
        ([0.0, 1.0], None, 0),
        ([0.0, 1.0], '', 0),
        ([], 'Hz', 0),
        ([float('nan'), float('inf')], 'Hz', 0),
        ([-2.5e3, 2.5e3], 'Hz', 3),
        ([1e-30], 'Hz', -15),
        ([1e30], 'Hz', 15),
    ],
    ids=lambda v: repr(v) if not isinstance(v, list) else str(v)[:24],
)
def test_si_prefix_rule(values, units, exponent):
    prefix = U.si_prefix(values, units)
    assert prefix.exponent == exponent
    assert prefix.symbol == U.PREFIXES[exponent]


def test_si_prefix_of_a_scalar_equals_that_of_the_singleton():
    assert U.si_prefix(3.75e9, 'Hz') == U.si_prefix([3.75e9], 'Hz')


def test_si_prefix_carries_a_mantissa_that_rounds_to_1000():
    assert U.si_prefix(999.99e6, 'Hz', places=1).exponent == 9
    assert U.si_prefix(999.99e6, 'Hz', places=3).exponent == 6


def test_si_prefix_step_rule_is_floored_one_prefix_below_the_magnitude():
    """a 10 kHz step at 1 GHz reads in MHz, not kHz, so the mantissas stay below 1e6"""
    assert U.si_prefix([1e9, 1.00001e9], 'Hz').exponent == 6
    assert U.si_prefix([1e9, 1.5e9], 'Hz').exponent == 6


# %% format_quantity


@pytest.mark.parametrize(
    'value, units, kws, expected',
    [
        (3.75e9, 'Hz', {}, '3.75 GHz'),
        (3.75e9, 'Hz', {'prefix': 'M'}, '3750 MHz'),
        (3.75e9, 'Hz', {'prefix': 'u'}, '3.75e+15 µHz'),
        (2e6, 'S/s', {}, '2 MS/s'),
        (1e-3, 's', {}, '1 ms'),
        (-100.0, 'dBm', {}, '-100 dBm'),
        (-100.0, 'dBm/Hz', {}, '-100 dBm/Hz'),
        (1e-6, 'mW', {}, '1e-06 mW'),
        (1500.0, '', {}, '1.5 k'),
        (1500.0, None, {}, '1500'),
        (1500.0, None, {'prefix': 'k'}, '1.5'),
        (999.99e6, 'Hz', {'places': 1}, '1.0 GHz'),
        (0.0, 'Hz', {}, '0 Hz'),
        (-0.0, 'Hz', {}, '0 Hz'),
        (2.5e3, 'Hz', {'sep': ''}, '2.5kHz'),
        (float('inf'), 'dBm', {}, '∞ dBm'),
        (float('-inf'), None, {}, '-∞'),
        (float('nan'), 'Hz', {}, 'nan Hz'),
    ],
)
def test_format_quantity(value, units, kws, expected):
    assert U.format_quantity(value, units, **kws) == expected


def test_format_quantity_accepts_an_si_prefix_object():
    prefix = U.si_prefix([3.7e9, 3.75e9], 'Hz')
    assert U.format_quantity(3.7e9, 'Hz', prefix=prefix) == '3700 MHz'


def test_format_quantity_rejects_an_unknown_symbol():
    with pytest.raises(ValueError, match='prefix'):
        U.format_quantity(1.0, 'Hz', prefix='Q')


def test_format_quantity_accepts_array_scalars(xp):
    value = xp.asarray(2.5e6)
    assert U.format_quantity(value, 'Hz') == '2.5 MHz'


@given(
    st.floats(min_value=1e-15, max_value=1e18, exclude_max=True, allow_nan=False),
    st.sampled_from(['Hz', 's', 'W', '']),
)
def test_format_quantity_round_trips_the_value(value, units):
    text = U.format_quantity(value, units)
    mantissa, _, suffix = text.partition(' ')
    symbol = suffix[: len(suffix) - len(units)] if units else suffix
    exponent = U.si_prefix(value, units or 'Hz').exponent
    assert symbol == U.PREFIXES[exponent]
    assert math.isclose(float(mantissa) * 10.0**exponent, value, rel_tol=1e-5)
