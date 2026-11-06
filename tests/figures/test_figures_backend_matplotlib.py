"""striqt.figures.backend.matplotlib: per-facet noise lines, limits, and bands"""

from __future__ import annotations

import functools
import types

import numpy as np
import pytest
import xarray as xr
from figures_datasets import (
    CYCLIC_BOUND_DB,
    POWER_DETECTORS,
    cyclic_mean_dBm,
    make_dataset,
    predicted_levels_dBm,
)

from striqt.figures import specs, util
from striqt.figures.backend import matplotlib as mpl_backend


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

    out = mpl_backend._facet_values(values, grid)

    expected = predicted_levels_dBm()
    assert out[0] == pytest.approx(expected[2], abs=1e-4)
    assert out[1] == pytest.approx(expected[0], abs=1e-4)
    assert out[2] is None


def test_facet_values_reduce_captures_that_share_a_facet() -> None:
    """the facet shared by the 2 captures at 1 GHz takes the minimum of the 2 levels"""
    values = noise_levels().set_xindex('center_frequency')
    grid = fake_grid([{'center_frequency': 1e9}, {'center_frequency': 2e9}])

    out = mpl_backend._facet_values(values, grid)

    expected = predicted_levels_dBm()
    assert out[0] == pytest.approx(expected[:2].min(), abs=1e-4)
    assert out[1] == pytest.approx(expected[2], abs=1e-4)


def test_facet_values_by_non_index_coordinate() -> None:
    values = noise_levels()
    grid = fake_grid([{'center_frequency': 2e9}, {'center_frequency': 1e9}])

    out = mpl_backend._facet_values(values, grid)

    expected = predicted_levels_dBm()
    assert out[0] == pytest.approx(expected[2], abs=1e-4)
    assert out[1] == pytest.approx(expected[:2].min(), abs=1e-4)


def test_facet_values_without_names_follow_the_capture_order() -> None:
    values = noise_levels()
    grid = types.SimpleNamespace(axs=np.empty(4, dtype=object))

    out = mpl_backend._facet_values(values, grid)

    np.testing.assert_allclose(out[:3], predicted_levels_dBm(), atol=1e-4)
    assert out[3] is None


# %% line, limit_facets, and mark_noise_level


def test_histogram_facet_limits_and_noise_lines_are_per_capture() -> None:
    """the x limits of each facet start 12 dB below the noise level of that capture.
    One vertical line marks the system noise level of that capture."""
    import matplotlib.pyplot as plt

    opts = specs.SharedPlotOptions(col='capture', col_label_format='{capture}')
    plotter = mpl_backend.MatplotlibBackend(opts, output_dir=None)
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


# %% fill_between


def capture_backend(col_wrap: int | None = None) -> mpl_backend.MatplotlibBackend:
    """return a backend faceted on the `capture` dimension of the synthetic dataset.

    The dataset has no `start_time` coordinate. The suptitle and the facet titles
    format from `center_frequency`.
    """
    opts = specs.SharedPlotOptions(
        col='capture',
        col_wrap=col_wrap,
        col_label_format='{center_frequency}',
        suptitle_fmt='title',
    )
    return mpl_backend.MatplotlibBackend(opts, output_dir=None)


def cyclic_statistics() -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """return the (min, mean, max) cyclic channel power of the synthetic dataset."""
    da = make_dataset()['cyclic_channel_power']
    return tuple(da.sel(cyclic_statistic=s) for s in ('min', 'mean', 'max'))


def data_axes(grid) -> list:
    """return the visible facet axes of `grid`, in facet order."""
    return [ax for ax in grid.axs.flat if ax.get_visible()]


def test_fill_between_shades_one_band_per_hue_at_the_bound_levels() -> None:
    """each facet has one `PolyCollection` per detector in the color ``C{i}`` at the
    given alpha, spanning from the min to the max statistic of that capture."""
    import matplotlib.pyplot as plt
    from matplotlib import colors
    from matplotlib.collections import PolyCollection

    low, _, high = cyclic_statistics()
    alpha = 0.4

    grid = capture_backend().fill_between(
        low, high, x='cyclic_lag', hue='power_detector', alpha=alpha
    )

    axs = data_axes(grid)
    assert len(axs) == 3
    expected = cyclic_mean_dBm()
    for i_capture, ax in enumerate(axs):
        assert len(ax.collections) == len(POWER_DETECTORS)
        for i, band in enumerate(ax.collections):
            assert isinstance(band, PolyCollection)
            (facecolor,) = band.get_facecolor()
            assert tuple(facecolor[:3]) == pytest.approx(colors.to_rgb(f'C{i}'))
            assert facecolor[3] == pytest.approx(alpha)
            y = band.get_paths()[0].vertices[:, 1]
            level = expected[i_capture, i]
            assert y.min() == pytest.approx(level - CYCLIC_BOUND_DB, abs=1e-3)
            assert y.max() == pytest.approx(level + CYCLIC_BOUND_DB, abs=1e-3)

    plt.close('all')


def test_fill_between_on_a_line_grid_matches_the_line_colors() -> None:
    """the band of hue index `i` drawn onto the grid of `line` has the facecolor of
    the `i`th line on every facet."""
    import matplotlib.pyplot as plt
    from matplotlib import colors

    low, mean, high = cyclic_statistics()
    plotter = capture_backend()

    grid = plotter.line(mean, x='cyclic_lag', hue='power_detector')
    out = plotter.fill_between(
        low, high, x='cyclic_lag', hue='power_detector', target=grid
    )

    assert out is grid
    for ax in data_axes(grid):
        lines = ax.get_lines()
        assert len(lines) == len(POWER_DETECTORS)
        assert len(ax.collections) == len(POWER_DETECTORS)
        for line, band in zip(lines, ax.collections):
            (facecolor,) = band.get_facecolor()
            assert tuple(facecolor[:3]) == pytest.approx(
                colors.to_rgb(line.get_color())
            )

    plt.close('all')


def test_line_on_target_adds_lines_to_the_existing_figure() -> None:
    """a second `line` call with `target` doubles the lines on each facet and opens
    no new figure."""
    import matplotlib.pyplot as plt

    low, mean, _ = cyclic_statistics()
    plotter = capture_backend()

    grid = plotter.line(mean, x='cyclic_lag', hue='power_detector')
    fignums = plt.get_fignums()
    out = plotter.line(low, x='cyclic_lag', hue='power_detector', target=grid)

    assert out is grid
    assert plt.get_fignums() == fignums
    expected = cyclic_mean_dBm()
    for i_capture, ax in enumerate(data_axes(grid)):
        lines = ax.get_lines()
        assert len(lines) == 2 * len(POWER_DETECTORS)
        added = lines[len(POWER_DETECTORS) :]
        for i, line in enumerate(added):
            np.testing.assert_allclose(
                line.get_ydata(), expected[i_capture, i] - CYCLIC_BOUND_DB, atol=1e-3
            )

    plt.close('all')


@pytest.mark.parametrize('method', ['line', 'fill_between'])
def test_target_with_a_facet_the_data_lacks_raises(method: str) -> None:
    """a grid of 3 captures rejects data with 2 captures"""
    import matplotlib.pyplot as plt

    low, mean, high = cyclic_statistics()
    plotter = capture_backend()
    grid = plotter.line(mean, x='cyclic_lag', hue='power_detector')
    two = {'capture': [0, 1]}
    kws = {'x': 'cyclic_lag', 'hue': 'power_detector', 'target': grid}
    if method == 'line':
        call = functools.partial(plotter.line, mean.isel(two), **kws)
    else:
        call = functools.partial(
            plotter.fill_between, low.isel(two), high.isel(two), **kws
        )

    with pytest.raises(ValueError, match='capture'):
        call()

    plt.close('all')


def test_step_post_holds_each_value_until_the_next_sample() -> None:
    """every line has drawstyle ``steps-post``, and the lower edge of each band has
    the post-step vertices of the samples."""
    import matplotlib.pyplot as plt
    from matplotlib import cbook

    low, mean, high = cyclic_statistics()
    plotter = capture_backend()

    grid = plotter.line(mean, x='cyclic_lag', hue='power_detector', step='post')
    plotter.fill_between(
        low, high, x='cyclic_lag', hue='power_detector', step='post', target=grid
    )

    x = low['cyclic_lag'].values
    for i_capture, ax in enumerate(data_axes(grid)):
        for line in ax.get_lines():
            assert line.get_drawstyle() == 'steps-post'
        for i, band in enumerate(ax.collections):
            vertices = band.get_paths()[0].vertices
            edge = cbook.pts_to_poststep(x, low.values[i_capture, i])
            # vertex 0 is the first point of the upper edge; the lower edge follows
            np.testing.assert_allclose(vertices[1 : 1 + edge.shape[1]], edge.T)

    plt.close('all')


def test_fill_between_hides_the_padding_axis_of_a_wrapped_grid() -> None:
    """3 captures wrapped at 2 columns leave one of 4 axes hidden and empty"""
    import matplotlib.pyplot as plt

    low, _, high = cyclic_statistics()

    grid = capture_backend(col_wrap=2).fill_between(
        low, high, x='cyclic_lag', hue='power_detector'
    )

    assert grid.axs.shape == (2, 2)
    hidden = [ax for ax in grid.axs.flat if not ax.get_visible()]
    assert len(hidden) == 1
    assert len(hidden[0].collections) == 0
    assert all(len(ax.collections) == len(POWER_DETECTORS) for ax in data_axes(grid))

    plt.close('all')
