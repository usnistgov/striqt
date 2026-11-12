"""plotly figures for the browser display of sweep results.

The grid of this backend is a `go.Figure` from `plotly.subplots.make_subplots`
with one subplot cell per facet. Facet `i` is the cell ``(1 + i // cols, 1 + i %
cols)``, with `cols` the smaller of the facet count and `col_wrap`, or 4 without
`col_wrap`. The traces of facet `i` reference the plotly axes ``x``/``y`` for
``i == 0`` and ``x{i+1}``/``y{i+1}`` after. ``layout.meta`` of each figure is set
to the ``variable`` name, the ``col_dim``, and the ``facets`` values in facet order.
"""

from __future__ import annotations

import functools
import math
from typing import TYPE_CHECKING, Any, Literal, Optional, Union, cast

import striqt.analysis as sa
import striqt.waveform as sw
from striqt.analysis.lib.util import np, xr

from .. import labels, specs, util
from .protocol import PlotBackend

if TYPE_CHECKING:
    from collections.abc import Iterator

    import plotly.graph_objects as go
    from plotly import subplots
    from xarray.core.types import ScaleOptions

    from .protocol import StepStyle
else:
    go = sw.util.lazy_import('plotly.graph_objects')
    subplots = sw.util.lazy_import('plotly.subplots')

_PALETTE = (
    '#636EFA',
    '#EF553B',
    '#00CC96',
    '#AB63FA',
    '#FFA15A',
    '#19D3F3',
    '#FF6692',
    '#B6E880',
    '#FF97FF',
    '#FECB52',
)
"""the plotly default qualitative colors, indexed by hue position"""

_STEP_SHAPES: dict[str, str] = {'post': 'hv', 'pre': 'vh', 'mid': 'hvh'}

_NOISE_COLOR = '#eec009'


def _to_numpy(arr: Any) -> np.ndarray:
    """return `arr` as a numpy array, with a cupy array transferred to the CPU"""
    if hasattr(arr, 'get'):
        return arr.get()
    return np.asarray(arr)


def _rgba(hex_color: str, alpha: float) -> str:
    """return `hex_color` (``'#rrggbb'``) as ``'rgba(r,g,b,a)'`` with opacity `alpha`"""
    r, g, b = (int(hex_color.lstrip('#')[i : i + 2], 16) for i in (0, 2, 4))
    return f'rgba({r},{g},{b},{alpha})'


def _axis_values(values: Any) -> np.ndarray:
    """return `values` flattened to 1-D, with a timedelta64 array in seconds"""
    arr = _to_numpy(values).flatten()
    if np.issubdtype(arr.dtype, np.timedelta64):
        # the plotly JSON encoder rejects datetime.timedelta
        return (arr / np.timedelta64(1, 's')).astype(np.float64)
    return arr


def _y_along(data: xr.DataArray, x: str) -> np.ndarray:
    """return the values of `data` as a 1-D array along the `x` dimension.

    With more than 1 entry along the dimensions other than `x`, the first entry
    along those dimensions is returned. Inf entries are returned as NaN.
    """
    y_data = _to_numpy(data.values).squeeze()
    x_size = data[x].size
    if y_data.size != x_size and x in data.dims:
        y_data = _to_numpy(data.values)
        x_dim_idx = list(data.dims).index(x)
        y_data = np.moveaxis(y_data, x_dim_idx, -1)
        y_data = y_data.reshape(-1, y_data.shape[-1])
        if y_data.shape[0] > 1:
            y_data = y_data[0]
    y_data = y_data.flatten()
    if np.issubdtype(y_data.dtype, np.floating):
        # Inf in a typed array breaks the autorange in the browser
        y_data = np.where(np.isinf(y_data), np.nan, y_data)
    return y_data


def _heatmap_z(data: xr.DataArray, x: str, y: str) -> np.ndarray:
    """return the values of `data` as a 2-D array of shape ``(len(y), len(x))``.

    Values on dimensions other than `x` and `y` are reduced to the first entry, and
    Inf entries are returned as NaN.
    """
    z_data = _to_numpy(data.values)
    while z_data.ndim > 2:
        z_data = z_data[0]
    if z_data.ndim == 1:
        z_data = z_data.reshape(1, -1)
    expected = (data[y].size, data[x].size)
    if z_data.shape != expected and z_data.shape == expected[::-1]:
        z_data = z_data.T
    if np.issubdtype(z_data.dtype, np.floating):
        z_data = np.where(np.isinf(z_data), np.nan, z_data)
    return z_data


def _finite_extent(values: Any) -> tuple[float, float]:
    """return the (min, max) of the finite entries of `values`, or (0, 1) without any.

    A non-numeric dtype gives (0, 1).
    """
    arr = _to_numpy(values)
    if arr.dtype.kind not in 'fiu':
        return 0.0, 1.0
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0, 1.0
    return float(np.nanmin(finite)), float(np.nanmax(finite))


def _finite_or_none(value: float | None) -> float | None:
    """return `value`, or `None` for `None` and for an infinite value"""
    if value is None or not np.isfinite(value):
        return None
    return value


def _grid_shape(n_facets: int, col_wrap: int | None) -> tuple[int, int]:
    """return the ``(rows, cols)`` of the subplot grid of `n_facets` facets.

    `cols` is the smaller of `n_facets` and `col_wrap`, or 4 without `col_wrap`.
    """
    cols = min(n_facets, col_wrap or 4)
    return math.ceil(n_facets / cols), cols


def _cell(i: int, cols: int) -> tuple[int, int]:
    """return the 1-based ``(row, col)`` subplot cell of facet `i` in a grid of `cols`"""
    return 1 + i // cols, 1 + i % cols


@functools.lru_cache(maxsize=16)
def _subplot_grid(rows: int, cols: int, titles: tuple[str, ...]) -> go.Figure:
    """return the empty subplot grid of `rows` by `cols` cells, one title per cell.

    Cells beyond the last title have no axes, and an empty title draws no text.
    The returned figure is shared across calls and must be copied with
    ``go.Figure(grid)`` before any change.
    """
    cell_specs = [
        [{} if r * cols + c < len(titles) else None for c in range(cols)]
        for r in range(rows)
    ]
    grid = subplots.make_subplots(
        rows=rows,
        cols=cols,
        specs=cell_specs,
        subplot_titles=list(titles),
        shared_yaxes=True,
    )
    # `go.Figure(grid)` re-validates the layout of the copy, which costs 7 ms with
    # the default template and 1 ms without
    grid.layout.template = None
    return grid


def _split_hue(data: xr.DataArray, hue: str | None) -> list[tuple[str, xr.DataArray]]:
    """return the ``(name, values)`` of each line of `data`, one per `hue` value.

    A `hue` that is a coordinate but not a dimension selects one line labeled by the
    first unique coordinate value, and `None` one line labeled by the name of `data`.
    """
    if hue is not None and hue in data.dims:
        return [
            (f'{hue}={v}', data.isel({hue: i})) for i, v in enumerate(data[hue].values)
        ]
    if hue is not None and hue in data.coords:
        hue_values = data[hue].values
        if np.ndim(hue_values) == 0:
            hue_val = hue_values.item()
        else:
            hue_val = np.unique(hue_values)[0]
        return [(f'{hue}={hue_val}', data)]
    return [(str(data.name), data)]


def _facet_arrays(
    data: xr.DataArray, col_dim: str | None, facets: list[Any]
) -> Iterator[tuple[Any, xr.DataArray]]:
    """yield the ``(facet value, values)`` of `data` on each facet, in facet order.

    `col_dim` None yields `data` once with the facet value None.
    """
    if col_dim is None:
        yield None, data
        return
    for i, facet in enumerate(facets):
        yield facet, data.isel({col_dim: i})


class PlotlyBackend(PlotBackend['go.Figure']):
    """the plotting backend that draws `go.Figure` grids for the browser.

    Each drawing call without `target` appends one figure to a pending list that
    `get_plot_data` returns and clears. The color limits of `heatmap` and the y
    limits of `line` and `fill_between` widen across calls and narrow only at
    `reset`. `sharex` has no effect.
    """

    def __init__(
        self,
        opts: specs.SharedPlotOptions,
    ):
        self.opts = opts
        self._pending: list[go.Figure] = []
        self._remembered_vmin: Optional[float] = None
        self._remembered_vmax: Optional[float] = None
        self._remembered_ymin: Optional[float] = None
        self._remembered_ymax: Optional[float] = None

    def reset(self, *, limits: bool = False) -> None:
        """clear the pending figures.

        Args:
            limits: True also clears the remembered color limits of `heatmap` and
                y limits of `line`; False keeps the remembered limits
        """
        self._pending.clear()
        if limits:
            self._remembered_vmin = None
            self._remembered_vmax = None
            self._remembered_ymin = None
            self._remembered_ymax = None

    def _get_col_dim(self, data: xr.DataArray) -> Optional[str]:
        """return the dimension for column faceting, or None.

        `opts.col` is returned when `opts.col` is a dimension of `data`. The
        dimension of a 1-D coordinate `opts.col` is returned when `data` has more
        than 1 value along that dimension. None is returned otherwise, and without
        `opts.col`.
        """
        col = self.opts.col
        if not col:
            return None

        if col in data.dims:
            return col

        if col in data.coords:
            coord = data.coords[col]
            if coord.dims and len(coord.dims) == 1:
                dim = coord.dims[0]
                if dim in data.dims and len(data[dim]) > 1:
                    return cast(str, dim)

        return None

    def _facets(self, data: xr.DataArray) -> tuple[str | None, list[Any]]:
        """return the facet dimension of `data` and the facet values in trace order.

        An unfaceted `data` gives ``(None, [None])``.
        """
        col_dim = self._get_col_dim(data)
        if col_dim is None:
            return None, [None]
        return col_dim, data[col_dim].values.tolist()

    def _new_figure(
        self, data: xr.DataArray, col_dim: str | None, facets: list[Any]
    ) -> go.Figure:
        """return the subplot grid of one drawing call on `data`.

        The grid is appended to the pending figures.
        """
        n_facets = len(facets)
        rows, cols = _grid_shape(n_facets, self.opts.col_wrap)
        if self.opts.col is None or self.opts.col_label_format is None:
            titles = ('',) * n_facets
        else:
            titles = tuple(_facet_titles(data, self.opts)[:n_facets])

        # make_subplots costs 4-6 ms per call, and `go.Figure(grid)` copies a cached
        # grid with its subplot references in 1 ms
        fig = go.Figure(_subplot_grid(rows, cols, titles))
        # the figure constructor applies the default template, which adds 6 kB to
        # every `to_json` message, and the browser page sets its own colors
        fig.layout.template = None
        fig.update_layout(
            uirevision='constant',
            meta={'variable': str(data.name), 'col_dim': col_dim, 'facets': facets},
        )
        self._pending.append(fig)
        return fig

    def _check_target(self, target: go.Figure, data: xr.DataArray) -> None:
        """raise `ValueError` unless the facets of `data` equal those of `target`.

        Args:
            target: a figure returned by a drawing method of this backend
            data: the array to draw onto `target`

        Raises:
            ValueError: if the facet dimension or the facet values of `data` differ
                from those in ``layout.meta`` of `target`
        """
        col_dim, facets = self._facets(data)
        meta = target.layout.meta or {}
        if col_dim != meta.get('col_dim') or facets != meta.get('facets'):
            raise ValueError(
                f'facets {col_dim}={facets} of {data.name!r} do not match the '
                f'target facets {meta.get("col_dim")}={meta.get("facets")}'
            )

    def _add_facet_traces(
        self, fig: go.Figure, traces: list[list[Any]], facets: list[Any]
    ) -> None:
        """add `traces[i]` to the cell of facet `i` of `fig`, in facet order"""
        _, cols = _grid_shape(len(facets), self.opts.col_wrap)
        flat: list[Any] = []
        rows_of: list[int] = []
        cols_of: list[int] = []
        for i, facet_traces in enumerate(traces):
            row, col = _cell(i, cols)
            flat.extend(facet_traces)
            rows_of.extend([row] * len(facet_traces))
            cols_of.extend([col] * len(facet_traces))
        fig.add_traces(flat, rows=rows_of, cols=cols_of)

    def heatmap(
        self,
        data: xr.DataArray,
        *,
        x: str,
        y: str,
        cmap: str = 'cubehelix',
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        vstep: Optional[float] = None,
    ) -> go.Figure:
        col_dim, facets = self._facets(data)

        # the remembered vmin/vmax span every port and every broadcast, so the
        # subplots share one colorbar scale
        data_np = _to_numpy(data.values)
        if vmin is None:
            current_vmin = float(np.nanmin(data_np))
            if self._remembered_vmin is None:
                self._remembered_vmin = current_vmin
            else:
                self._remembered_vmin = min(self._remembered_vmin, current_vmin)
            vmin = self._remembered_vmin
        if vmax is None:
            current_vmax = float(np.nanmax(data_np))
            if self._remembered_vmax is None:
                self._remembered_vmax = current_vmax
            else:
                self._remembered_vmax = max(self._remembered_vmax, current_vmax)
            vmax = self._remembered_vmax

        if vstep is not None:
            levels = util.quantized_value_range(data, vmin, vmax, vstep)
            colorscale = _make_quantized_colorscale(cmap, len(levels) - 1)
            # the color range is the rounded edges, so each band spans one `vstep`
            vmin, vmax = levels[0], levels[-1]
        else:
            colorscale = _mpl_to_plotly_colorscale(cmap)

        traces: list[list[Any]] = []
        for i, (facet, sub) in enumerate(_facet_arrays(data, col_dim, facets)):
            # the colorbar of the last facet is the shared scale
            last = i == len(facets) - 1
            trace = go.Heatmap(
                x=_axis_values(sub[x].values),
                y=_axis_values(sub[y].values),
                z=_heatmap_z(sub, x, y),
                colorscale=colorscale,
                zmin=vmin,
                zmax=vmax,
                showscale=last,
                colorbar=go.heatmap.ColorBar(title=_colorbar_title(data))
                if last
                else None,
                name=None if col_dim is None else f'{col_dim}={facet}',
            )
            traces.append([trace])

        fig = self._new_figure(data, col_dim, facets)
        self._add_facet_traces(fig, traces, facets)
        fig.update_xaxes(title_text=_axis_title(data[x]))
        fig.update_yaxes(title_text=_axis_title(data[y]), col=1)
        return fig

    def _widen_y(self, ymin: float, ymax: float) -> None:
        """widen the remembered y limits to include [`ymin`, `ymax`]"""
        if self._remembered_ymin is None:
            self._remembered_ymin = ymin
        else:
            self._remembered_ymin = min(self._remembered_ymin, ymin)
        if self._remembered_ymax is None:
            self._remembered_ymax = ymax
        else:
            self._remembered_ymax = max(self._remembered_ymax, ymax)

    def _y_range(self, axis_type: str | None) -> list[float | None]:
        """return the remembered y limits as a plotly axis range for `axis_type`.

        For a ``'log'`` `axis_type`, each entry is log10 of the limit. A `None` entry
        leaves that limit to autorange, and is returned for an unset limit, an
        infinite limit, or a limit at or below 0 on a log axis.
        """
        limits = [
            _finite_or_none(self._remembered_ymin),
            _finite_or_none(self._remembered_ymax),
        ]
        if axis_type != 'log':
            return limits
        return [None if v is None or v <= 0 else float(np.log10(v)) for v in limits]

    def line(
        self,
        data: xr.DataArray,
        *,
        x: str,
        hue: Optional[str] = None,
        yscale: ScaleOptions = 'linear',
        sharex: bool = True,
        ylim: Optional[tuple[Optional[float], Optional[float]]] = None,
        step: StepStyle | None = None,
        target: go.Figure | None = None,
    ) -> go.Figure:
        if yscale not in ('linear', 'log'):
            raise ValueError(f"yscale must be 'linear' or 'log', not {yscale!r}")
        if target is not None:
            self._check_target(target, data)

        col_dim, facets = self._facets(data)
        traces = [
            self._line_traces(
                sub,
                x,
                hue,
                step,
                legendgroup=None if col_dim is None else f'{col_dim}={facet}',
                showlegend=i == 0,
            )
            for i, (facet, sub) in enumerate(_facet_arrays(data, col_dim, facets))
        ]

        current_ymin, current_ymax = _finite_extent(data.values)
        if ylim is not None:
            current_ymin = current_ymin if ylim[0] is None else ylim[0]
            current_ymax = current_ymax if ylim[1] is None else ylim[1]
        self._widen_y(current_ymin, current_ymax)

        if target is not None:
            self._add_facet_traces(target, traces, facets)
            target.update_yaxes(range=self._y_range(target.layout.yaxis.type))
            return target

        fig = self._new_figure(data, col_dim, facets)
        self._add_facet_traces(fig, traces, facets)
        fig.update_xaxes(title_text=_axis_title(data[x]))
        fig.update_yaxes(type=yscale, range=self._y_range(yscale))
        fig.update_yaxes(title_text=_axis_title(data), col=1)
        fig.update_layout(legend=go.layout.Legend(orientation='h', y=-0.15))
        return fig

    def fill_between(
        self,
        low: xr.DataArray,
        high: xr.DataArray,
        *,
        x: str,
        hue: str | None = None,
        alpha: float = 0.25,
        step: StepStyle | None = None,
        target: go.Figure | None = None,
    ) -> go.Figure:
        if target is not None:
            self._check_target(target, low)

        col_dim, facets = self._facets(low)
        traces = [
            self._band_traces(
                sub_low,
                sub_high,
                x,
                hue,
                alpha,
                step,
                legendgroup=None if col_dim is None else f'{col_dim}={facet}',
            )
            for (facet, sub_low), (_, sub_high) in zip(
                _facet_arrays(low, col_dim, facets),
                _facet_arrays(high, col_dim, facets),
            )
        ]

        self._widen_y(_finite_extent(low.values)[0], _finite_extent(high.values)[1])

        if target is not None:
            self._add_facet_traces(target, traces, facets)
            target.update_yaxes(range=self._y_range(target.layout.yaxis.type))
            return target

        fig = self._new_figure(low, col_dim, facets)
        self._add_facet_traces(fig, traces, facets)
        fig.update_xaxes(title_text=_axis_title(low[x]))
        fig.update_yaxes(type='linear', range=self._y_range('linear'))
        fig.update_yaxes(title_text=_axis_title(low), col=1)
        fig.update_layout(legend=go.layout.Legend(orientation='h', y=-0.15))
        return fig

    def _band_traces(
        self,
        low: xr.DataArray,
        high: xr.DataArray,
        x: str,
        hue: str | None,
        alpha: float,
        step: StepStyle | None,
        legendgroup: str | None = None,
    ) -> list[go.Scattergl]:
        """return the lower- and upper-edge trace pair of each band on one facet.

        The 2 traces of each band are consecutive in the returned list.
        """
        x_data = _axis_values(low[x].values)
        shape = None if step is None else _STEP_SHAPES[step]

        traces: list[go.Scattergl] = []
        bands = zip(_split_hue(low, hue), _split_hue(high, hue))
        for i, ((name, lo), (_, hi)) in enumerate(bands):
            group = name if legendgroup is None else legendgroup
            lower = go.Scattergl(
                x=x_data,
                y=_y_along(lo, x),
                mode='lines',
                name=f'{name} low',
                legendgroup=group,
                showlegend=False,
                hoverinfo='skip',
                line=go.scattergl.Line(width=0, shape=shape),
            )
            upper = go.Scattergl(
                x=x_data,
                y=_y_along(hi, x),
                mode='lines',
                name=f'{name} high',
                legendgroup=group,
                showlegend=False,
                fill='tonexty',
                fillcolor=_rgba(_PALETTE[i % len(_PALETTE)], alpha),
                line=go.scattergl.Line(width=0, shape=shape),
            )
            traces.extend((lower, upper))
        return traces

    def _line_traces(
        self,
        data: xr.DataArray,
        x: str,
        hue: Optional[str],
        step: StepStyle | None,
        *,
        legendgroup: str | None = None,
        showlegend: bool = True,
    ) -> list[go.Scattergl]:
        """return one `go.Scattergl` per `hue` value of `data` on one facet"""
        x_data = _axis_values(data[x].values)
        # a single point draws nothing in 'lines' mode
        mode = 'markers' if x_data.size == 1 else 'lines'
        shape = None if step is None else _STEP_SHAPES[step]

        traces: list[go.Scattergl] = []
        for i, (name, sub) in enumerate(_split_hue(data, hue)):
            color = _PALETTE[i % len(_PALETTE)]
            traces.append(
                go.Scattergl(
                    x=x_data,
                    y=_y_along(sub, x),
                    mode=mode,
                    name=name,
                    legendgroup=name if legendgroup is None else legendgroup,
                    showlegend=showlegend,
                    line=go.scattergl.Line(color=color, shape=shape),
                    marker=go.scattergl.Marker(color=color),
                )
            )
        return traces

    def finish(
        self,
        target: go.Figure,
        xticklabelunits: Union[bool, Literal['auto']] = 'auto',
    ) -> None:
        """leave `target` unchanged."""

    def limit_facets(
        self,
        target: go.Figure,
        low: xr.DataArray | float | None,
        high: float | None,
    ) -> None:
        facets: list[Any] = target.layout.meta['facets']
        _, cols = _grid_shape(len(facets), self.opts.col_wrap)
        if isinstance(low, xr.DataArray):
            lows: list[float | None] = [float(v) for v in _to_numpy(low.values).flat]
        else:
            lows = [low] * len(facets)

        for i in range(len(facets)):
            lo = lows[i] if i < len(lows) else None
            row, col = _cell(i, cols)
            target.update_xaxes(range=[lo, high], row=row, col=col)

    def mark_noise_level(
        self,
        data: xr.Dataset,
        var_name: str,
        grid: go.Figure,
        where: Literal['x', 'y', 'colorbar'],
    ) -> None:
        if where not in ('x', 'y', 'colorbar'):
            raise TypeError('where must be "x", "y", or "colorbar"')

        noise = util.get_system_noise(data, var_name)
        if noise is None:
            return

        facets: list[Any] = grid.layout.meta['facets']
        _, cols = _grid_shape(len(facets), self.opts.col_wrap)
        line = {'color': _NOISE_COLOR, 'width': 1, 'dash': 'dot'}

        for i, noise_val in enumerate(_facet_noise(noise, grid)):
            if noise_val is None:
                continue
            row, col = _cell(i, cols)

            if where == 'x':
                grid.add_hline(
                    y=noise_val,
                    row=row,
                    col=col,
                    line=line,
                    exclude_empty_subplots=False,
                )
            elif where == 'y':
                grid.add_vline(
                    x=noise_val,
                    row=row,
                    col=col,
                    line=line,
                    exclude_empty_subplots=False,
                )
            else:
                colorbar_traces = [
                    t for t in grid.data if getattr(t, 'showscale', False)
                ]
                if not colorbar_traces:
                    raise TypeError('no colorbar in the figure')
                zmin = colorbar_traces[0].zmin
                zmax = colorbar_traces[0].zmax
                if zmin is None or zmax is None or zmax <= zmin:
                    continue

                # plotly annotations cannot reference colorbar values, so the marker
                # is placed in paper coordinates at the normalized (0-1) position
                # within the colorbar range
                colorbar_bottom = 0.1
                colorbar_top = 0.9
                clamped_noise = max(zmin, min(zmax, noise_val))
                normalized = (clamped_noise - zmin) / (zmax - zmin)
                y_paper = colorbar_bottom + normalized * (
                    colorbar_top - colorbar_bottom
                )

                grid.add_annotation(
                    x=1.02,
                    y=y_paper,
                    xref='paper',
                    yref='paper',
                    text='kTB',
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=1,
                    arrowcolor=_NOISE_COLOR,
                    font={'color': _NOISE_COLOR, 'size': 10},
                    ax=20,
                    ay=0,
                )

    def get_plot_data(self) -> list[go.Figure]:
        """return and clear the pending figures, in drawing order."""
        data = self._pending.copy()
        self._pending.clear()
        return data


def _facet_noise(noise: xr.DataArray, grid: go.Figure) -> list[float | None]:
    """return one value of `noise` per facet of `grid`, in facet order.

    The value of a facet is the minimum of `noise` at the facet value along the
    ``col_dim`` of `grid`, or None without a match. Without ``col_dim`` among the
    dimensions of `noise`, the values of `noise` are returned in order, one per
    facet, and None for each facet beyond the last value.
    """
    meta = grid.layout.meta or {}
    facets: list[Any] = meta.get('facets', [None])
    col_dim = meta.get('col_dim')
    values = _to_numpy(noise.values)

    if col_dim is None or col_dim not in noise.dims:
        flat = [float(v) for v in values.flat]
        return flat[: len(facets)] + [None] * (len(facets) - len(flat))

    coord = np.asarray(noise[col_dim].values)
    along = np.moveaxis(values, noise.dims.index(col_dim), 0)
    out: list[float | None] = []
    for facet in facets:
        matches = np.flatnonzero(coord == facet)
        if matches.size == 0:
            out.append(None)
        else:
            out.append(float(np.min(along[matches[0]])))
    return out


def _facet_titles(data: xr.DataArray, opts: specs.SharedPlotOptions) -> list[str]:
    """return the subplot title of each facet of `data` from `opts.col_label_format`"""
    return labels.label_by_coord(
        data,
        opts.col_label_format,
        coord_or_dim=opts.col or 'port',
        title_case=True,
        name=data.name,
        **data.attrs,
    )


def _measurand(da: xr.DataArray) -> sa.labels.Measurand:
    name = None if da.name is None else str(da.name)
    return sa.labels.measurand_from_attrs(da.attrs, name)


def _axis_title(da: xr.DataArray) -> str:
    """return the `standard_name` of `da` with its units in parentheses."""
    return sa.labels.quantity_label(_measurand(da), short=True)


def _colorbar_title(da: xr.DataArray) -> str:
    """return the units text of `da` from its attrs, or ``''`` without units."""
    return sa.labels.units_label(_measurand(da)) or ''


def _mpl_to_plotly_colorscale(cmap: str) -> Any:
    """return the Plotly colorscale for the matplotlib colormap name `cmap`.

    Returns:
        the Plotly name of a colormap that Plotly defines, otherwise a list of 256
        ``[position, 'rgb(r,g,b)']`` pairs sampled from matplotlib; ``'Viridis'``
        for a name matplotlib does not define
    """
    import matplotlib.pyplot as plt

    plotly_builtins = {
        'viridis': 'Viridis',
        'plasma': 'Plasma',
        'inferno': 'Inferno',
        'magma': 'Magma',
        'cividis': 'Cividis',
        'jet': 'Jet',
        'hot': 'Hot',
        'blues': 'Blues',
    }

    cmap_lower = cmap.lower()
    if cmap_lower in plotly_builtins:
        return plotly_builtins[cmap_lower]

    try:
        mpl_cmap = plt.get_cmap(cmap)
    except ValueError:
        return 'Viridis'

    n_colors = 256
    colorscale: list[list[Any]] = []
    for i in range(n_colors):
        pos = i / (n_colors - 1)
        rgba = mpl_cmap(pos)
        r, g, b = int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
        colorscale.append([pos, f'rgb({r},{g},{b})'])

    return colorscale


def _make_quantized_colorscale(cmap: str, n_bands: int) -> list[list[Any]]:
    """return a stepped Plotly colorscale of `n_bands` equal bands over ``[0, 1]``.

    Args:
        cmap: a matplotlib colormap name; an undefined name selects viridis
        n_bands: the number of color bands; a value below 1 selects 1

    Returns:
        a list of ``[position, color]`` pairs with 2 entries per band
    """
    import matplotlib.pyplot as plt

    n_levels = max(n_bands, 1)

    try:
        mpl_cmap = plt.get_cmap(cmap).resampled(n_levels)
    except ValueError:
        mpl_cmap = plt.get_cmap('viridis').resampled(n_levels)

    # each band has 2 entries of the same color, so the colorscale is flat
    # within a band
    colorscale: list[list[Any]] = []
    for i in range(n_levels):
        pos_start = i / n_levels
        pos_end = (i + 1) / n_levels

        rgba = mpl_cmap(i / (n_levels - 1) if n_levels > 1 else 0.5)
        r, g, b = int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
        color = f'rgb({r},{g},{b})'

        colorscale.append([pos_start, color])
        if i < n_levels - 1:
            colorscale.append([pos_end, color])
        else:
            colorscale.append([1.0, color])

    return colorscale
