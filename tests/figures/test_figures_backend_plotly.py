"""striqt.figures.backend.plotly: trace pairs, facet ranges, and noise shapes"""

from __future__ import annotations

import functools

import numpy as np
import plotly.graph_objects as go
import pytest
import xarray as xr
from figures_datasets import (
    CYCLIC_BOUND_DB,
    POWER_DETECTORS,
    capture_bandwidth,
    cyclic_mean_dBm,
    make_dataset,
    predicted_levels_dBm,
)

from striqt.figures import specs
from striqt.figures.backend import PlotlyBackend

X = 'cyclic_lag'
HUE = 'power_detector'


def capture_backend() -> PlotlyBackend:
    """return a backend faceted on the `capture` dimension of the synthetic dataset."""
    opts = specs.SharedPlotOptions(
        col='capture', col_label_format='{center_frequency:M}', suptitle_fmt='title'
    )
    return PlotlyBackend(opts)


def cyclic_statistics(
    ds: xr.Dataset | None = None,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """return the (min, mean, max) cyclic channel power of `ds`, or of `make_dataset()`"""
    da = (make_dataset() if ds is None else ds)['cyclic_channel_power']
    return tuple(da.sel(cyclic_statistic=s) for s in ('min', 'mean', 'max'))


def axis_refs(i: int) -> tuple[str, str]:
    """return the ``(xaxis, yaxis)`` trace references of the 0-based facet `i`."""
    return ('x', 'y') if i == 0 else (f'x{i + 1}', f'y{i + 1}')


def facet_traces(fig: go.Figure, i: int) -> list[go.Scattergl]:
    """return the traces of `fig` on 0-based facet `i`, in draw order."""
    xaxis, _ = axis_refs(i)
    return [t for t in fig.data if (t.xaxis or 'x') == xaxis]


def rgb_of(color: str) -> tuple[int, int, int]:
    """return the RGB components of an ``'#rrggbb'`` or ``'rgba(r,g,b,a)'`` color."""
    if color.startswith('#'):
        return tuple(int(color[i : i + 2], 16) for i in (1, 3, 5))
    r, g, b, _ = color[len('rgba(') : -1].split(',')
    return int(r), int(g), int(b)


# %% fill_between


def test_fill_between_appends_a_lower_then_upper_trace_per_hue_per_facet() -> None:
    """each facet has 2 traces per detector: the lower edge, then the upper edge
    with ``fill == 'tonexty'``. Both reference the axes of the facet, have no line,
    and have no legend entry. The y values are the min and max statistics of the
    capture."""
    low, _, high = cyclic_statistics()
    plotter = capture_backend()

    fig = plotter.fill_between(low, high, x=X, hue=HUE)

    assert list(fig.layout.meta['facets']) == [0, 1, 2]
    expected = cyclic_mean_dBm()
    for i_capture in range(3):
        traces = facet_traces(fig, i_capture)
        assert len(traces) == 2 * len(POWER_DETECTORS)
        xaxis, yaxis = axis_refs(i_capture)
        for i, (lower, upper) in enumerate(zip(traces[::2], traces[1::2])):
            assert lower.fill is None
            assert upper.fill == 'tonexty'
            for t in (lower, upper):
                assert (t.xaxis, t.yaxis) == (xaxis, yaxis)
                assert t.line.width == 0
                assert t.showlegend is False
            level = expected[i_capture, i]
            np.testing.assert_allclose(lower.y, level - CYCLIC_BOUND_DB, atol=1e-3)
            np.testing.assert_allclose(upper.y, level + CYCLIC_BOUND_DB, atol=1e-3)


def test_fill_between_on_target_appends_traces_in_the_line_colors() -> None:
    """bands drawn onto the figure of `line` extend its traces, add no pending
    figure, and the band of hue index `i` has the RGB of the `i`th line"""
    low, mean, high = cyclic_statistics()
    plotter = capture_backend()

    fig = plotter.line(mean, x=X, hue=HUE)
    n_lines = len(fig.data)
    out = plotter.fill_between(low, high, x=X, hue=HUE, target=fig)

    assert out is fig
    assert len(fig.data) == n_lines + 2 * len(POWER_DETECTORS) * 3
    assert len(plotter.get_plot_data()) == 1
    for i_capture in range(3):
        traces = facet_traces(fig, i_capture)
        lines = traces[: len(POWER_DETECTORS)]
        uppers = traces[len(POWER_DETECTORS) + 1 :: 2]
        for line, upper in zip(lines, uppers):
            assert rgb_of(upper.fillcolor) == rgb_of(line.line.color)


# %% limit_facets


def test_limit_facets_sets_a_range_per_facet_from_the_capture_values() -> None:
    """each facet x range starts at the value of that capture and ends at None
    (autorange) when `high` is None"""
    _, mean, _ = cyclic_statistics()
    plotter = capture_backend()
    fig = plotter.line(mean, x=X, hue=HUE)
    lows = xr.DataArray([1e-3, 2e-3, 0.0], dims='capture')

    plotter.limit_facets(fig, lows, None)

    for i, key in enumerate(['xaxis', 'xaxis2', 'xaxis3']):
        lo, hi = fig.layout[key].range
        assert lo == pytest.approx(float(lows[i]))
        assert hi is None


# %% mark_noise_level


def test_mark_noise_level_edits_only_the_given_figure_by_facet_value() -> None:
    """the shapes of the given figure are at the system noise of the capture of
    each facet, in the facet order of that figure; another figure is unchanged"""
    plotter = capture_backend()
    _, mean_all, _ = cyclic_statistics()
    other = plotter.line(mean_all, x=X, hue=HUE)
    ds = make_dataset().isel(capture=[2, 0])
    _, mean, _ = cyclic_statistics(ds)
    fig = plotter.line(mean, x=X, hue=HUE)

    plotter.mark_noise_level(ds, 'cyclic_channel_power', fig, where='x')

    assert other.layout.shapes == ()
    shapes = fig.layout.shapes
    assert len(shapes) == 2
    expected = predicted_levels_dBm(capture_bandwidth())
    for shape, i_capture, (xaxis, yaxis) in zip(shapes, [2, 0], map(axis_refs, [0, 1])):
        assert shape.type == 'line'
        assert shape.y0 == shape.y1 == pytest.approx(expected[i_capture], abs=1e-3)
        assert shape.yref == yaxis
        assert shape.xref == f'{xaxis} domain'


# %% reset


@pytest.mark.parametrize('limits', [False, True], ids=['keep', 'forget'])
def test_reset_keeps_the_widened_limits_unless_limits_is_true(limits: bool) -> None:
    """after a draw of all 3 captures, a draw of the middle capture alone keeps the
    color and y range of the 3-capture data under ``reset()``, and takes the range
    of its own data under ``reset(limits=True)``"""
    wide = make_dataset()
    narrow = wide.isel(capture=[1])
    plotter = capture_backend()
    _, mean_wide, _ = cyclic_statistics(wide)
    _, mean_narrow, _ = cyclic_statistics(narrow)
    heat_kws = {'x': X, 'y': HUE}
    plotter.heatmap(mean_wide, **heat_kws)
    plotter.line(mean_wide, x=X, hue=HUE)
    wide_range = (float(np.nanmin(mean_wide)), float(np.nanmax(mean_wide)))
    narrow_range = (float(np.nanmin(mean_narrow)), float(np.nanmax(mean_narrow)))
    assert wide_range[0] < narrow_range[0] < narrow_range[1] < wide_range[1]

    plotter.reset(limits=limits)
    heat = plotter.heatmap(mean_narrow, **heat_kws)
    line = plotter.line(mean_narrow, x=X, hue=HUE)

    expected = narrow_range if limits else wide_range
    (trace,) = heat.data
    assert (trace.zmin, trace.zmax) == pytest.approx(expected)
    assert tuple(line.layout.yaxis.range) == pytest.approx(expected)


def test_reset_clears_the_pending_figures() -> None:
    _, mean, _ = cyclic_statistics()
    plotter = capture_backend()
    plotter.line(mean, x=X, hue=HUE)

    plotter.reset()

    assert plotter.get_plot_data() == []


# %% step and validation


@pytest.mark.parametrize(
    ('step', 'shape'), [('pre', 'vh'), ('post', 'hv'), ('mid', 'hvh')]
)
def test_step_selects_the_plotly_line_shape(step: str, shape: str) -> None:
    """``'post'`` is the plotly shape ``'hv'`` (horizontal, then vertical at the
    next sample), ``'pre'`` is ``'vh'``, and ``'mid'`` is ``'hvh'``, on lines and
    bands alike"""
    low, mean, high = cyclic_statistics()
    plotter = capture_backend()

    fig = plotter.line(mean, x=X, hue=HUE, step=step)
    plotter.fill_between(low, high, x=X, hue=HUE, step=step, target=fig)

    assert all(t.line.shape == shape for t in fig.data)


def test_line_rejects_a_yscale_other_than_linear_or_log() -> None:
    _, mean, _ = cyclic_statistics()

    with pytest.raises(ValueError, match='symlog'):
        capture_backend().line(mean, x=X, hue=HUE, yscale='symlog')


def test_figure_rejects_a_property_value_outside_the_plotly_schema() -> None:
    """the figure returned by `line` validates an axis type against the plotly
    enumeration on assignment"""
    _, mean, _ = cyclic_statistics()
    fig = capture_backend().line(mean, x=X, hue=HUE)

    with pytest.raises(ValueError, match='enumeration'):
        fig.update_yaxes(type='symlog')


@pytest.mark.parametrize('method', ['line', 'fill_between'])
def test_target_with_other_facets_raises(method: str) -> None:
    """a figure of 3 captures rejects data with 2 captures"""
    low, mean, high = cyclic_statistics()
    plotter = capture_backend()
    fig = plotter.line(mean, x=X, hue=HUE)
    two = {'capture': [0, 1]}
    kws = {'x': X, 'hue': HUE, 'target': fig}
    if method == 'line':
        call = functools.partial(plotter.line, mean.isel(two), **kws)
    else:
        call = functools.partial(
            plotter.fill_between, low.isel(two), high.isel(two), **kws
        )

    with pytest.raises(ValueError, match='facets'):
        call()
