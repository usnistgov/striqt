"""striqt.figures.data_vars: the registered figure of each data variable"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from figures_datasets import (
    POWER_DETECTORS,
    capture_bandwidth,
    cyclic_mean_dBm,
    make_dataset,
    predicted_levels_dBm,
)

from striqt.figures import data_vars, specs
from striqt.figures.backend import MatplotlibBackend, PlotlyBackend

N_DETECTORS = len(POWER_DETECTORS)


def mpl_backend() -> MatplotlibBackend:
    """return a matplotlib backend faceted on `capture` that writes no file."""
    opts = specs.SharedPlotOptions(
        col='capture', col_label_format='{center_frequency}', suptitle_fmt='title'
    )
    return MatplotlibBackend(opts, output_dir=None)


def plotly_backend() -> PlotlyBackend:
    """return a plotly backend faceted on `capture`."""
    opts = specs.SharedPlotOptions(
        col='capture', col_label_format='{center_frequency:M}', suptitle_fmt='title'
    )
    return PlotlyBackend(opts)


def data_lines(ax) -> list:
    """return the lines of `ax` with one point per cyclic lag sample."""
    n = make_dataset().sizes['cyclic_lag']
    return [line for line in ax.get_lines() if len(line.get_xdata()) == n]


def noise_lines(ax, level: float) -> list:
    """return the 2-point horizontal lines of `ax` at `level` (dBm)."""
    return [
        line
        for line in ax.get_lines()
        if len(line.get_xdata()) == 2
        and np.allclose(line.get_ydata(), level, atol=1e-3)
    ]


# %% cyclic_channel_power


def test_cyclic_channel_power_on_matplotlib() -> None:
    """with no output directory the entry returns None. Each facet has one stepped
    line and one band per detector, with the line at the mean statistic of that
    capture, and one noise line at the system noise in the capture bandwidth."""
    import matplotlib.pyplot as plt

    ds = make_dataset()
    plotter = mpl_backend()

    out = data_vars.cyclic_channel_power(ds, plotter)

    assert out is None
    grid = plotter.last_grid
    assert grid is not None
    axs = [ax for ax in grid.axs.flat if ax.get_visible()]
    assert len(axs) == 3
    means = cyclic_mean_dBm()
    noise = predicted_levels_dBm(capture_bandwidth())
    for i_capture, ax in enumerate(axs):
        lines = data_lines(ax)
        assert len(lines) == N_DETECTORS
        for i, line in enumerate(lines):
            assert line.get_drawstyle() == 'steps-post'
            np.testing.assert_allclose(line.get_ydata(), means[i_capture, i], atol=1e-3)
        assert len(ax.collections) == N_DETECTORS
        assert len(noise_lines(ax, noise[i_capture])) == 1

    plt.close('all')


def test_cyclic_channel_power_on_plotly() -> None:
    """one pending figure with, per facet, one line trace per detector and 2 band
    traces per detector, and one noise shape per facet"""
    ds = make_dataset()
    plotter = plotly_backend()

    data_vars.cyclic_channel_power(ds, plotter)

    (fig,) = plotter.get_plot_data()
    for i in range(3):
        xaxis = 'x' if i == 0 else f'x{i + 1}'
        traces = [t for t in fig.data if (t.xaxis or 'x') == xaxis]
        lines = [t for t in traces if t.line.width != 0]
        bands = [t for t in traces if t.line.width == 0]
        assert len(lines) == N_DETECTORS
        assert len(bands) == 2 * N_DETECTORS
    assert len(fig.layout.shapes) == 3


@pytest.mark.parametrize(
    'kwargs',
    [{'center_statistic': 'median'}, {'bound_statistics': ('min', 'p99')}],
    ids=['center', 'bound'],
)
def test_cyclic_channel_power_rejects_a_statistic_the_data_lacks(
    kwargs: dict[str, Any],
) -> None:
    ds = make_dataset()

    with pytest.raises(ValueError, match='cyclic_statistic'):
        data_vars.cyclic_channel_power(ds, mpl_backend(), **kwargs)


def test_cyclic_channel_power_without_steps_connects_the_samples() -> None:
    import matplotlib.pyplot as plt

    plotter = mpl_backend()

    data_vars.cyclic_channel_power(make_dataset(), plotter, steps=False)

    grid = plotter.last_grid
    assert grid is not None
    for ax in grid.axs.flat:
        for line in data_lines(ax):
            assert line.get_drawstyle() == 'default'

    plt.close('all')


def test_cyclic_channel_power_linear_draws_the_power_in_mW() -> None:
    """with ``dB=False`` the line of each detector is ``10**(dBm/10)`` of the mean"""
    import matplotlib.pyplot as plt

    plotter = mpl_backend()

    data_vars.cyclic_channel_power(make_dataset(), plotter, dB=False)

    grid = plotter.last_grid
    assert grid is not None
    expected = 10 ** (cyclic_mean_dBm() / 10)
    axs = [ax for ax in grid.axs.flat if ax.get_visible()]
    for i_capture, ax in enumerate(axs):
        for i, line in enumerate(data_lines(ax)):
            np.testing.assert_allclose(
                line.get_ydata(), expected[i_capture, i], rtol=1e-5
            )

    plt.close('all')
