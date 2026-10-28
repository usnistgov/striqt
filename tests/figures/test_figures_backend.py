"""striqt.figures.backend: per-facet noise lines and limits"""

from __future__ import annotations

import types

import numpy as np
import pytest
import xarray as xr
from figures_datasets import make_dataset, predicted_levels_dBm

from striqt.figures import backend, specs, util


def fake_grid(name_dicts: list[dict[str, float] | None]) -> types.SimpleNamespace:
    axs = np.empty(len(name_dicts), dtype=object)
    return types.SimpleNamespace(name_dicts=np.array(name_dicts, dtype=object), axs=axs)


def noise_levels() -> xr.DataArray:
    return util.get_system_noise(make_dataset(), 'spectrogram_power_bin')


# %% _facet_values


def test_facet_values_by_dimension_position() -> None:
    """a facet dimension without coordinate values is matched by position"""
    values = noise_levels()
    grid = fake_grid([{'capture': 2}, {'capture': 0}, None])

    out = backend._facet_values(values, grid)

    expected = predicted_levels_dBm()
    assert out[0] == pytest.approx(expected[2], abs=1e-4)
    assert out[1] == pytest.approx(expected[0], abs=1e-4)
    assert out[2] is None


def test_facet_values_reduce_captures_that_share_a_facet() -> None:
    """2 captures at 1 GHz share one facet, which gets the minimum of the 2 levels"""
    values = noise_levels().set_xindex('center_frequency')
    grid = fake_grid([{'center_frequency': 1e9}, {'center_frequency': 2e9}])

    out = backend._facet_values(values, grid)

    expected = predicted_levels_dBm()
    assert out[0] == pytest.approx(expected[:2].min(), abs=1e-4)
    assert out[1] == pytest.approx(expected[2], abs=1e-4)


def test_facet_values_by_non_index_coordinate() -> None:
    values = noise_levels()
    grid = fake_grid([{'center_frequency': 2e9}, {'center_frequency': 1e9}])

    out = backend._facet_values(values, grid)

    expected = predicted_levels_dBm()
    assert out[0] == pytest.approx(expected[2], abs=1e-4)
    assert out[1] == pytest.approx(expected[:2].min(), abs=1e-4)


def test_facet_values_without_names_follow_the_capture_order() -> None:
    values = noise_levels()
    grid = types.SimpleNamespace(axs=np.empty(4, dtype=object))

    out = backend._facet_values(values, grid)

    np.testing.assert_allclose(out[:3], predicted_levels_dBm(), atol=1e-4)
    assert out[3] is None


# %% line, limit_facets, and mark_noise_level


def test_histogram_facet_limits_and_noise_lines_are_per_capture() -> None:
    """the facet of each capture is limited to the noise floor of that capture and
    marked with one vertical line at the system noise level of that capture"""
    import matplotlib.pyplot as plt

    opts = specs.SharedPlotOptions(col='capture', col_label_format='{capture}')
    plotter = backend.PlotBackend(opts, output_dir=None)
    ds = make_dataset()
    name, x = 'spectrogram_histogram', 'spectrogram_power_bin'
    sub = util.select_histogram_bins(ds, name, x)

    grid = plotter.line(sub, x=x, sharex=False)
    low, high = util.histogram_bin_limits(ds, name, x)
    plotter.limit_facets(grid, low, high)
    plotter.mark_noise_level(ds, x, grid, where='y')

    expected = predicted_levels_dBm()
    axs = [ax for ax in grid.axs.flat if ax.has_data()]
    assert len(axs) == 3
    for ax, level in zip(axs, expected):
        xmin, xmax = ax.get_xlim()
        assert xmin == pytest.approx(level - 12, abs=1e-3)
        assert xmax == pytest.approx(high, abs=1e-3)
        vertical = [
            line
            for line in ax.get_lines()
            if np.allclose(line.get_xdata(), level, atol=1e-3)
            and len(line.get_xdata()) == 2
        ]
        assert len(vertical) == 1, 'one noise line per facet'

    plt.close('all')
