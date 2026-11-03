"""striqt.figures.labels: axis, legend, and facet text from dataset attrs"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from striqt.figures import labels

PSD_ATTRS = {
    'standard_name': 'PSD',
    'long_name': 'Power spectral density',
    'units': 'dBm',
    'noise_bandwidth': 360e3,
}
FREQUENCY_ATTRS = {'standard_name': 'Baseband frequency', 'units': 'Hz'}


@pytest.fixture
def ax():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    yield ax
    plt.close(fig)


def tick_texts(axis) -> list[str]:
    axis.axes.figure.canvas.draw()
    return [t.get_text() for t in axis.get_ticklabels()]


def facet_array() -> xr.DataArray:
    """return an array over 3 captures with one coordinate of each kind."""
    start = np.array(
        ['2026-01-01T00:00:00', '2026-01-01T00:00:01', '2026-01-01T00:00:02'],
        dtype='datetime64[ns]',
    )
    coords = {
        'center_frequency': (
            'capture',
            np.array([1e9, 1.1e9, 2e9]),
            {'standard_name': 'Center frequency', 'units': 'Hz'},
        ),
        'port': ('capture', np.array([0, 1, 2])),
        'start_time': ('capture', start),
        'channel_name': ('capture', np.array(['n77_a', 'n77_b', 'n78'])),
        'flag': ('capture', np.array([True, False, True])),
        'band': (
            'capture',
            np.array([(1e6, 2e6), (1e6, 2e6), None], dtype=object),
            {'units': 'Hz'},
        ),
    }
    return xr.DataArray(np.zeros((3, 4)), dims=('capture', 't'), coords=coords)


# %% label_axis


@pytest.mark.parametrize(
    ('short', 'expected'),
    [(False, 'Power spectral density (dBm/360 kHz)'), (True, 'PSD (dBm/360 kHz)')],
)
def test_psd_label_names_the_bandwidth_once(ax, short: bool, expected: str) -> None:
    """the units text of a power spectral density is composed from the `units` and
    `noise_bandwidth` attrs once, after a `long_name` without units"""
    da = xr.DataArray(np.linspace(-120, -60, 5), dims='f', name='psd', attrs=PSD_ATTRS)
    ax.plot(da.values)

    labels.label_axis('y', da, ax=ax, tick_units=False, short=short)

    assert ax.get_ylabel() == expected
    assert ax.get_ylabel().count('(') == 1


def frequency_axis(ax) -> xr.DataArray:
    freq = xr.DataArray(
        np.linspace(800e6, 1.1e9, 7), dims='f', name='f', attrs=FREQUENCY_ATTRS
    )
    ax.plot(freq.values, np.zeros(freq.size))
    ax.set_xlim(800e6, 1.1e9)
    return freq


def test_units_in_the_label_leave_bare_ticks(ax) -> None:
    """the extent 0.8 to 1.1 GHz spans 300 MHz, so the shared prefix is M: it goes
    in the axis label, and every tick is a bare mantissa"""
    freq = frequency_axis(ax)

    labels.label_axis('x', freq, ax=ax, tick_units=False)

    assert ax.get_xlabel().endswith('(MHz)')
    ticks = tick_texts(ax.xaxis)
    assert len(ticks) > 1
    assert not any('Hz' in t for t in ticks)


def test_units_in_the_ticks_leave_a_bare_label(ax) -> None:
    freq = frequency_axis(ax)

    labels.label_axis('x', freq, ax=ax, tick_units=True)

    assert '(' not in ax.get_xlabel()
    ticks = tick_texts(ax.xaxis)
    assert len(ticks) > 1
    assert all(t.endswith(' MHz') for t in ticks)


def test_ticks_and_label_use_one_frozen_prefix_after_a_zoom(ax) -> None:
    """the prefix is frozen from the data, so a view over a narrow span keeps the
    MHz label and the ticks stay mantissas in MHz"""
    freq = frequency_axis(ax)
    labels.label_axis('x', freq, ax=ax, tick_units=False)

    ax.set_xlim(1.0e9, 1.01e9)

    assert ax.get_xlabel().endswith('(MHz)')
    values = [float(t) for t in tick_texts(ax.xaxis) if t]
    assert values and all(1000 <= v <= 1010 for v in values)


def test_a_narrow_axis_far_from_zero_reads_in_the_prefix_of_its_span(ax) -> None:
    """3.695 to 3.71 GHz labels in MHz with ticks such as 3700, not 3.7 in GHz"""
    freq = xr.DataArray(
        np.linspace(3.695e9, 3.71e9, 4), dims='f', name='f', attrs=FREQUENCY_ATTRS
    )
    ax.plot(freq.values, np.zeros(freq.size))
    ax.set_xlim(3.695e9, 3.71e9)

    labels.label_axis('x', freq, ax=ax, tick_units=False)

    assert ax.get_xlabel().endswith('(MHz)')
    values = [float(t) for t in tick_texts(ax.xaxis) if t]
    assert 3700.0 in values


def test_all_nan_data_is_still_labeled(ax) -> None:
    da = xr.DataArray(np.full(4, np.nan), dims='f', name='psd', attrs=PSD_ATTRS)
    ax.plot(da.values)

    labels.label_axis('y', da, ax=ax, tick_units=False)

    assert ax.get_ylabel() == 'Power spectral density (dBm/360 kHz)'


def test_auto_tick_units_follow_the_unit_length(ax) -> None:
    """a 1 or 2 character unit goes in the ticks, a longer one in the label"""
    freq = frequency_axis(ax)
    labels.label_axis('x', freq, ax=ax)
    assert '(' not in ax.get_xlabel()

    da = xr.DataArray(np.linspace(-120, -60, 5), dims='f', name='psd', attrs=PSD_ATTRS)
    ax.plot(da.values)
    labels.label_axis('y', da, ax=ax)
    assert ax.get_ylabel().endswith('(dBm/360 kHz)')


def test_coordinate_supplies_the_attrs_and_the_values(ax) -> None:
    da = xr.DataArray(
        np.zeros(3),
        dims='f',
        coords={'f': ('f', np.array([1e3, 2e3, 3e3]), FREQUENCY_ATTRS)},
        name='power',
    )
    ax.plot(da.f.values, da.values)

    labels.label_axis('x', da, coord_name='f', ax=ax, tick_units=False)

    assert ax.get_xlabel() == 'Baseband frequency (kHz)'


def test_a_quantity_without_units_is_labeled_by_name(ax) -> None:
    da = xr.DataArray(np.arange(3.0), dims='f', name='fraction')
    ax.plot(da.values)

    labels.label_axis('y', da, ax=ax)

    assert ax.get_ylabel() == 'fraction'


def test_which_axis_is_validated(ax) -> None:
    da = xr.DataArray(np.arange(3.0), dims='f', name='fraction')
    with pytest.raises(ValueError, match='which_axis'):
        labels.label_axis('z', da, ax=ax)  # type: ignore[arg-type]


# %% label_by_coord and _coords_to_dicts


@pytest.mark.parametrize(
    ('fmt', 'expected'),
    [
        ('{center_frequency}', ['1000 MHz', '1100 MHz', '2000 MHz']),
        ('{center_frequency:G}', ['1 GHz', '1.1 GHz', '2 GHz']),
        ('{center_frequency:.0f}', ['1000000000', '1100000000', '2000000000']),
        ('{port}', ['0', '1', '2']),
        (
            '{start_time}',
            [
                '2026-01-01T00:00:00.000',
                '2026-01-01T00:00:01.000',
                '2026-01-01T00:00:02.000',
            ],
        ),
        ('{channel_name}', ['N77 A', 'N77 B', 'N78']),
        ('{flag}', ['True', 'False', 'True']),
        ('{band}', ['(1 MHz, 2 MHz)', '(1 MHz, 2 MHz)', 'None']),
    ],
    ids=[
        'float64_shared_prefix',
        'forced_prefix',
        'float_spec',
        'int',
        'datetime64',
        'str_title_case',
        'bool',
        'object_tuple_and_none',
    ],
)
def test_label_by_coord_renders_each_coordinate_kind(
    fmt: str, expected: list[str]
) -> None:
    """a float64 coordinate is formatted in its units with the prefix shared by the
    facet values (the 100 MHz step lowers 2 GHz to MHz); a format spec that is one
    prefix symbol forces the prefix, and any other spec formats the bare number"""
    assert labels.label_by_coord(facet_array(), fmt, coord_or_dim='capture') == expected


def test_label_by_coord_extra_fields_and_title_case() -> None:
    da = facet_array()

    out = labels.label_by_coord(
        da, '{name} {channel_name}', coord_or_dim='capture', title_case=False, name='x'
    )

    assert out == ['x n77 a', 'x n77 b', 'x n78']


def test_label_by_coord_coordinate_field_overrides_an_extra_field() -> None:
    da = facet_array()

    out = labels.label_by_coord(da, '{port}', coord_or_dim='capture', port='extra')

    assert out == ['0', '1', '2']


# %% label_legend


def test_legend_entries_share_one_prefix(ax) -> None:
    """the entries end with the units and the title has none"""
    da = facet_array()
    ax.plot(np.zeros((4, 3)))

    legend = labels.label_legend(da.center_frequency, ax=ax, tick_units=True)

    assert [t.get_text() for t in legend.get_texts()] == [
        '1000 MHz',
        '1100 MHz',
        '2000 MHz',
    ]
    assert legend.get_title().get_text() == 'Center frequency'


def test_legend_title_takes_the_units_from_bare_entries(ax) -> None:
    da = facet_array()
    ax.plot(np.zeros((4, 3)))

    legend = labels.label_legend(da.center_frequency, ax=ax, tick_units=False)

    assert [t.get_text() for t in legend.get_texts()] == ['1000', '1100', '2000']
    assert legend.get_title().get_text() == 'Center frequency (MHz)'


def test_legend_of_strings_writes_them_as_is(ax) -> None:
    da = facet_array()
    ax.plot(np.zeros((4, 3)))

    legend = labels.label_legend(da.channel_name, ax=ax)

    assert [t.get_text() for t in legend.get_texts()] == ['n77_a', 'n77_b', 'n78']
    assert legend.get_title().get_text() == 'channel_name'
