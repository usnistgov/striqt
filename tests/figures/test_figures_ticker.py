"""striqt.figures.ticker: one SI prefix per axis in `EngFormatter`"""

from __future__ import annotations

from striqt.figures import ticker

# %% EngFormatter

LOCS = [8e8, 9e8, 1e9, 1.1e9]


def test_bare_ticks_share_the_prefix_of_the_step() -> None:
    """the 100 MHz step between ticks at 1 GHz lowers the prefix to M, so the ticks
    are bare mantissas in MHz and the axis label carries the prefixed unit"""
    formatter = ticker.EngFormatter('Hz', unit_in_tick=False)

    assert formatter.format_ticks(LOCS) == ['800', '900', '1000', '1100']
    assert formatter.axis_units_label() == 'MHz'


def test_frozen_exponent_overrides_the_tick_locations() -> None:
    formatter = ticker.EngFormatter('Hz', unit_in_tick=False, exponent=9)

    assert formatter.format_ticks(LOCS) == ['0.8', '0.9', '1', '1.1']
    assert formatter.axis_units_label() == 'GHz'


def test_unit_in_tick_puts_one_prefixed_unit_on_every_tick() -> None:
    formatter = ticker.EngFormatter('Hz', unit_in_tick=True)

    ticks = formatter.format_ticks(LOCS)

    assert ticks == ['800 MHz', '900 MHz', '1000 MHz', '1100 MHz']
    assert formatter.axis_units_label() == ''


def test_dB_units_take_no_prefix() -> None:
    """a dB unit is never prefixed, and the minus sign follows the matplotlib
    ``axes.unicode_minus`` setting through `fix_minus`"""
    formatter = ticker.EngFormatter('dBm', unit_in_tick=False)

    ticks = formatter.format_ticks([-120, -60])

    assert ticks == [formatter.fix_minus('-120'), formatter.fix_minus('-60')]
    assert formatter.axis_units_label() == 'dBm'


def test_no_units_gives_a_bare_mantissa() -> None:
    formatter = ticker.EngFormatter(None, unit_in_tick=False)

    assert formatter.format_ticks([0.5, 1.0]) == ['0.5', '1']
    assert formatter.axis_units_label() == ''


def test_empty_units_give_a_bare_prefix() -> None:
    formatter = ticker.EngFormatter('', unit_in_tick=False)

    assert formatter.format_ticks([0.5, 1.0]) == ['500', '1000']
    assert formatter.axis_units_label() == 'm'
