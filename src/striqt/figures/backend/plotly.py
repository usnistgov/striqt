"""plotly trace and layout dicts for the browser display of sweep results.

The grid of this backend is a result dict with the keys ``type``, ``traces``,
``layout``, ``variable``, ``facets``, and ``col_dim``. The traces of facet `i`
reference the plotly axes ``x``/``y`` for ``i == 0`` and ``x{i+1}``/``y{i+1}``
after, in the order of the facet coordinate values.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Optional, Union, cast

import striqt.analysis as sa
from striqt.analysis.lib.util import np, xr

from .. import labels, specs, util

if TYPE_CHECKING:
    from xarray.core.types import ScaleOptions

    from .protocol import StepStyle

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


def _axis_refs(subplot_idx: int | None) -> tuple[str, str]:
    """return the ``(xaxis, yaxis)`` trace references of 1-based facet `subplot_idx`.

    `None` selects the first facet.
    """
    if subplot_idx is None or subplot_idx == 1:
        return 'x', 'y'
    return f'x{subplot_idx}', f'y{subplot_idx}'


def _layout_axis_key(axis: Literal['xaxis', 'yaxis'], facet_index: int) -> str:
    """return the layout key of `axis` for the 0-based `facet_index`"""
    return axis if facet_index == 0 else f'{axis}{facet_index + 1}'


def _y_along(data: xr.DataArray, x: str) -> np.ndarray:
    """return the values of `data` as a 1-D array along the `x` dimension.

    When `data` has values on dimensions other than `x`, only the first row along
    the remaining dimensions is returned.
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
    return y_data.flatten()


def _finite_extent(values: Any) -> tuple[float, float]:
    """return the (min, max) of the finite entries of `values`, or (0, 1) if none"""
    arr = _to_numpy(values)
    if arr.dtype.kind not in 'fiu':
        return 0.0, 1.0
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0, 1.0
    return float(np.nanmin(finite)), float(np.nanmax(finite))


class PlotlyBackend:
    """the plotting backend that emits plotly trace and layout dicts.

    Each drawing call appends one result dict to a pending list that `get_plot_data`
    returns and clears. The color limits of `heatmap` and the y limits of `line` only
    widen across calls.
    """

    opts: specs.SharedPlotOptions
    """the facet layout and label formats shared by every figure"""

    def __init__(
        self,
        opts: specs.SharedPlotOptions,
    ):
        self.opts = opts
        self._pending_traces: list[dict[str, Any]] = []
        self._layout: dict[str, Any] = {}
        self._remembered_vmin: Optional[float] = None
        self._remembered_vmax: Optional[float] = None
        self._remembered_ymin: Optional[float] = None
        self._remembered_ymax: Optional[float] = None

    def reset(self) -> None:
        """Clear per-broadcast state while preserving vmin/vmax memory."""
        self._pending_traces.clear()
        self._layout.clear()

    def _get_col_dim(self, data: xr.DataArray) -> Optional[str]:
        """return the dimension for column faceting, or None.

        `opts.col` is returned when `opts.col` is a dimension of `data`. When
        `opts.col` is a 1-D coordinate of `data`, the dimension of that coordinate is
        returned if `data` has more than 1 value along that dimension.
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

    def _new_result(
        self,
        kind: str,
        traces: list[dict[str, Any]],
        layout: dict[str, Any],
        data: xr.DataArray,
        col_dim: str | None,
        facets: list[Any],
    ) -> dict[str, Any]:
        """append and return the result dict of one drawing call"""
        result = {
            'type': kind,
            'traces': traces,
            'layout': layout,
            'variable': str(data.name),
            'facets': facets,
            'col_dim': col_dim,
        }
        self._pending_traces.append(result)
        return result

    def _check_target(self, target: dict[str, Any], data: xr.DataArray) -> None:
        """raise `ValueError` unless the facets of `data` equal those of `target`.

        Args:
            target: a result dict returned by a drawing method of this backend
            data: the array to draw onto `target`

        Raises:
            ValueError: if the facet dimension or the facet values of `data` differ
                from those recorded in `target`
        """
        col_dim, facets = self._facets(data)
        if col_dim != target.get('col_dim') or facets != target.get('facets'):
            raise ValueError(
                f'facets {col_dim}={facets} of {data.name!r} do not match the '
                f'target facets {target.get("col_dim")}={target.get("facets")}'
            )

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
    ) -> dict[str, Any]:
        """add a heatmap of `data` to the pending results.

        Every subplot (one per value of the column dimension) shares one colorbar
        scale. A `vmin` or `vmax` of None is the minimum or maximum of `data`,
        widened by the values of earlier calls.

        Args:
            data: the 2-D array to plot, with coordinates `x` and `y`
            x: the name of the x coordinate
            y: the name of the y coordinate
            cmap: a matplotlib colormap name
            vmin: the lower color limit, or None to infer
            vmax: the upper color limit, or None to infer
            vstep: the spacing of quantized color levels; None selects a continuous
                gradient

        Returns:
            a dict with keys ``type``, ``traces``, ``layout``, ``variable``,
            ``facets``, and ``col_dim``
        """
        traces = []
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
            colorscale = _make_quantized_colorscale(cmap, vmin, vmax, vstep)
        else:
            colorscale = _mpl_to_plotly_colorscale(cmap)

        if col_dim:
            n_subplots = len(facets)
            for i, col_val in enumerate(facets):
                sub = data.isel({col_dim: i})
                # Only show colorbar on the last subplot to avoid duplicates
                show_colorbar = i == n_subplots - 1
                trace = self._make_heatmap_trace(
                    sub,
                    x,
                    y,
                    colorscale,
                    vmin,
                    vmax,
                    subplot_idx=i + 1,
                    show_colorbar=show_colorbar,
                )
                trace['name'] = f'{col_dim}={col_val}'
                traces.append(trace)
        else:
            traces.append(
                self._make_heatmap_trace(
                    data, x, y, colorscale, vmin, vmax, show_colorbar=True
                )
            )

        layout = self._make_heatmap_layout(data, x, y, len(traces))
        return self._new_result('heatmap', traces, layout, data, col_dim, facets)

    def _make_heatmap_trace(
        self,
        data: xr.DataArray,
        x: str,
        y: str,
        colorscale: Any,
        vmin: Optional[float],
        vmax: Optional[float],
        subplot_idx: Optional[int] = None,
        show_colorbar: bool = True,
    ) -> dict[str, Any]:
        """Create a single heatmap trace.

        Args:
            data: The data array to plot
            x: Name of the x coordinate
            y: Name of the y coordinate
            colorscale: Plotly colorscale (string for built-in or list of [pos, color] pairs)
            vmin: Minimum value for color scale (shared across all subplots)
            vmax: Maximum value for color scale (shared across all subplots)
            subplot_idx: Index of the subplot (1-based)
            show_colorbar: Whether to show the colorbar for this trace
        """
        x_data = data[x].values
        y_data = data[y].values
        z_data = _to_numpy(data.values)

        # Ensure 2D by selecting first element along extra dimensions
        while z_data.ndim > 2:
            z_data = z_data[0]
        if z_data.ndim == 1:
            z_data = z_data.reshape(1, -1)

        # Plotly heatmap expects z with shape (len(y), len(x))
        if z_data.shape != (len(y_data), len(x_data)) and z_data.shape == (
            len(x_data),
            len(y_data),
        ):
            z_data = z_data.T

        # inf is replaced with NaN, which Plotly draws as a gap
        if np.issubdtype(z_data.dtype, np.floating):
            z_data = np.where(np.isinf(z_data), np.nan, z_data)

        # numpy arrays are passed as is. Figure.to_json() encodes a numpy array as a
        # base64 typed array
        trace: dict[str, Any] = {
            'type': 'heatmap',
            'x': x_data,
            'y': y_data,
            'z': z_data,
            'colorscale': colorscale,
            'showscale': show_colorbar,
        }

        if show_colorbar:
            trace['colorbar'] = {
                'title': {'text': _colorbar_title(data)},
            }

        if vmin is not None:
            trace['zmin'] = vmin
        if vmax is not None:
            trace['zmax'] = vmax

        if subplot_idx is not None:
            trace['xaxis'], trace['yaxis'] = _axis_refs(subplot_idx)

        return trace

    def _make_heatmap_layout(
        self, data: xr.DataArray, x: str, y: str, n_subplots: int
    ) -> dict[str, Any]:
        """Create layout for heatmap with optional subplots."""
        layout: dict[str, Any] = {
            'uirevision': 'constant',  # Preserve zoom/pan on updates
        }

        if self.opts.col is None or self.opts.col_label_format is None:
            pass
        else:
            layout['annotations'] = _format_subplots(data, self.opts)

        if n_subplots > 1:
            cols = min(n_subplots, self.opts.col_wrap or 4)
            rows = (n_subplots + cols - 1) // cols

            for i in range(n_subplots):
                row_idx = i // cols
                col_idx = i % cols
                suffix = str(i + 1) if i > 0 else ''

                x_domain = [col_idx / cols + 0.02, (col_idx + 1) / cols - 0.02]
                y_domain = [
                    1 - (row_idx + 1) / rows + 0.02,
                    1 - row_idx / rows - 0.08,
                ]

                layout[f'xaxis{suffix}'] = {
                    'title': {'text': _axis_title(data[x])},
                    'domain': x_domain,
                }
                layout[f'yaxis{suffix}'] = {
                    'title': {'text': _axis_title(data[y])},
                    'domain': y_domain,
                }
        else:
            layout['xaxis'] = {'title': {'text': _axis_title(data[x])}}
            layout['yaxis'] = {'title': {'text': _axis_title(data[y])}}

        return layout

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
        target: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """draw one ``scattergl`` line per `hue` value of `data` on each facet.

        The y limits are `ylim` where given and the finite extent of `data`
        otherwise, widened by the values of earlier calls.

        Args:
            data: the values on the y axis, with dimensions `x`, `hue` (when set), and
                the facet dimension of `opts`
            x: the dimension or coordinate of `data` on the x axis
            hue: the dimension of `data` that selects one line per value; `None`
                draws one line per facet
            yscale: the y axis scale, ``'linear'`` or ``'log'``
            sharex: accepted without effect on the layout
            ylim: the y limits; `None` in either position leaves that limit to the
                data extent
            step: when set, hold each value until the next sample as `StepStyle`
                selects; `None` connects the samples
            target: `None` builds a new result dict; a result dict returned by an
                earlier call receives the lines, and the facet values of `data` must
                equal those of the result

        Returns:
            the result dict drawn on

        Raises:
            ValueError: if `yscale` is neither ``'linear'`` nor ``'log'``, or if
                `target` has facets that `data` does not have
        """
        if yscale not in ('linear', 'log'):
            raise ValueError(f"yscale must be 'linear' or 'log', not {yscale!r}")
        if target is not None:
            self._check_target(target, data)

        traces = []
        col_dim, facets = self._facets(data)

        if col_dim:
            for i, col_val in enumerate(facets):
                sub = data.isel({col_dim: i})
                subplot_traces = self._make_line_traces(
                    sub, x, hue, subplot_idx=i + 1, step=step
                )
                for t in subplot_traces:
                    t['legendgroup'] = f'{col_dim}={col_val}'
                traces.extend(subplot_traces)
        else:
            traces.extend(self._make_line_traces(data, x, hue, step=step))

        current_ymin, current_ymax = _finite_extent(data.values)

        # the remembered limits only widen across broadcasts
        candidate_ymin = (
            ylim[0] if (ylim is not None and ylim[0] is not None) else current_ymin
        )
        if self._remembered_ymin is None:
            self._remembered_ymin = candidate_ymin
        else:
            self._remembered_ymin = min(self._remembered_ymin, candidate_ymin)

        candidate_ymax = (
            ylim[1] if (ylim is not None and ylim[1] is not None) else current_ymax
        )
        if self._remembered_ymax is None:
            self._remembered_ymax = candidate_ymax
        else:
            self._remembered_ymax = max(self._remembered_ymax, candidate_ymax)

        effective_ylim = (self._remembered_ymin, self._remembered_ymax)

        if target is not None:
            target['traces'].extend(traces)
            _widen_y_ranges(target['layout'], effective_ylim)
            return target

        layout = self._make_line_layout(
            data, x, yscale, len(facets), ylim=effective_ylim
        )
        return self._new_result('line', traces, layout, data, col_dim, facets)

    def fill_between(
        self,
        low: xr.DataArray,
        high: xr.DataArray,
        *,
        x: str,
        hue: str | None = None,
        alpha: float = 0.25,
        step: StepStyle | None = None,
        target: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """shade the band from `low` to `high` per `hue` value on each facet.

        The band of hue index `i` takes the color of the `i`th line of `line` with the
        same `hue`.

        Args:
            low: the lower edge of the band, with dimensions `x`, `hue` (when set),
                and the facet dimension of `opts`
            high: the upper edge of the band, with the dimensions of `low`
            x: the dimension or coordinate of `low` and `high` on the x axis
            hue: the dimension that selects one band per value; `None` shades one
                band per facet
            alpha: the opacity of the band, in [0, 1]
            step: when set, hold each edge until the next sample as `StepStyle`
                selects; `None` connects the samples
            target: `None` builds a new result dict; a result dict returned by an
                earlier call receives the bands, and the facet values of `low` must
                equal those of the result

        Returns:
            the result dict drawn on

        Raises:
            ValueError: if `target` has facets that `low` does not have
        """
        if target is not None:
            self._check_target(target, low)

        traces: list[dict[str, Any]] = []
        col_dim, facets = self._facets(low)

        if col_dim:
            for i, col_val in enumerate(facets):
                traces.extend(
                    self._make_band_traces(
                        low.isel({col_dim: i}),
                        high.isel({col_dim: i}),
                        x,
                        hue,
                        alpha,
                        step,
                        subplot_idx=i + 1,
                        legendgroup=f'{col_dim}={col_val}',
                    )
                )
        else:
            traces.extend(self._make_band_traces(low, high, x, hue, alpha, step))

        if target is not None:
            target['traces'].extend(traces)
            return target

        ylim = (_finite_extent(low.values)[0], _finite_extent(high.values)[1])
        layout = self._make_line_layout(low, x, 'linear', len(facets), ylim=ylim)
        return self._new_result('line', traces, layout, low, col_dim, facets)

    def _make_band_traces(
        self,
        low: xr.DataArray,
        high: xr.DataArray,
        x: str,
        hue: str | None,
        alpha: float,
        step: StepStyle | None,
        subplot_idx: int | None = None,
        legendgroup: str | None = None,
    ) -> list[dict[str, Any]]:
        """return the lower- and upper-edge trace pair of each band on one facet.

        Plotly fills ``tonexty`` to the preceding trace in ``data`` order. The 2
        traces of each band are consecutive in the returned list.
        """
        x_data = _to_json_serializable(np.asarray(low[x].values).flatten())
        xaxis, yaxis = _axis_refs(subplot_idx)
        line: dict[str, Any] = {'width': 0}
        if step is not None:
            line['shape'] = _STEP_SHAPES[step]

        if hue is not None and hue in low.dims:
            bands = [
                (i, low.isel({hue: i}), high.isel({hue: i}), f'{hue}={v}')
                for i, v in enumerate(low[hue].values)
            ]
        else:
            bands = [(0, low, high, str(low.name))]

        traces: list[dict[str, Any]] = []
        for i, lo, hi, name in bands:
            group = name if legendgroup is None else legendgroup
            base: dict[str, Any] = {
                'type': 'scattergl',
                'mode': 'lines',
                'x': x_data,
                'line': dict(line),
                'showlegend': False,
                'legendgroup': group,
                'xaxis': xaxis,
                'yaxis': yaxis,
            }
            lower = {
                **base,
                'y': _to_json_serializable(_y_along(lo, x)),
                'name': f'{name} low',
                'hoverinfo': 'skip',
            }
            upper = {
                **base,
                'y': _to_json_serializable(_y_along(hi, x)),
                'name': f'{name} high',
                'fill': 'tonexty',
                'fillcolor': _rgba(_PALETTE[i % len(_PALETTE)], alpha),
            }
            traces.extend((lower, upper))
        return traces

    def _make_line_traces(
        self,
        data: xr.DataArray,
        x: str,
        hue: Optional[str],
        subplot_idx: Optional[int] = None,
        step: StepStyle | None = None,
    ) -> list[dict[str, Any]]:
        """return one ``scattergl`` trace per `hue` value of `data` on one facet"""
        traces = []
        x_data = np.asarray(data[x].values).flatten()

        hue_is_dim = hue and hue in data.dims
        hue_is_coord = hue and hue in data.coords and hue not in data.dims

        def make_trace(y_data: np.ndarray, name: str, color: str) -> dict[str, Any]:
            # a single point draws nothing in 'lines' mode
            mode = 'markers' if len(x_data) == 1 else 'lines'
            line: dict[str, Any] = {'color': color}
            if step is not None:
                line['shape'] = _STEP_SHAPES[step]
            trace: dict[str, Any] = {
                'type': 'scattergl',
                'mode': mode,
                'x': _to_json_serializable(x_data),
                'y': _to_json_serializable(y_data),
                'name': name,
                'legendgroup': name,
                'showlegend': subplot_idx is None or subplot_idx == 1,
                'line': line,
                'marker': {'color': color},
            }
            if subplot_idx is not None:
                trace['xaxis'], trace['yaxis'] = _axis_refs(subplot_idx)
            return trace

        if hue_is_dim:
            for hue_idx, hue_val in enumerate(data[hue].values):
                sub = data.isel({hue: hue_idx})
                color = _PALETTE[hue_idx % len(_PALETTE)]
                traces.append(make_trace(_y_along(sub, x), f'{hue}={hue_val}', color))
        elif hue_is_coord:
            # `sel` does not accept a non-dimension coordinate, so the whole array
            # is drawn as one trace labeled by the first coordinate value
            hue_values = data[hue].values
            if np.ndim(hue_values) == 0:
                hue_val = hue_values.item()
            else:
                hue_val = np.unique(hue_values)[0]
            traces.append(
                make_trace(_y_along(data, x), f'{hue}={hue_val}', _PALETTE[0])
            )
        else:
            traces.append(make_trace(_y_along(data, x), str(data.name), _PALETTE[0]))

        return traces

    def _make_line_layout(
        self,
        data: xr.DataArray,
        x: str,
        yscale: ScaleOptions,
        n_subplots: int,
        *,
        ylim: Optional[tuple[Optional[float], Optional[float]]] = None,
    ) -> dict[str, Any]:
        """Create layout for line plot with optional subplots.

        Subplots in the same row share the same y-axis range.
        Only the leftmost subplot in each row shows y-axis title and tick labels.
        """
        layout: dict[str, Any] = {
            'uirevision': 'constant',
            'showlegend': True,
            'legend': {'orientation': 'h', 'y': -0.15},
        }

        if self.opts.col is None or self.opts.col_label_format is None:
            pass
        else:
            layout['annotations'] = _format_subplots(data, self.opts)

        if n_subplots > 1:
            cols = min(n_subplots, self.opts.col_wrap or 4)
            rows = (n_subplots + cols - 1) // cols

            if ylim is not None and ylim[0] is not None and ylim[1] is not None:
                y_range: list[Optional[float]] = [ylim[0], ylim[1]]
            else:
                # Compute shared y-axis range across all data, excluding inf values
                data_np = _to_numpy(data.values)
                finite_mask = np.isfinite(data_np)
                if np.any(finite_mask):
                    y_min = float(np.nanmin(data_np[finite_mask]))
                    y_max = float(np.nanmax(data_np[finite_mask]))
                else:
                    # All values are inf/nan - use default range
                    y_min, y_max = 0.0, 1.0
                # Add 5% padding
                y_range_pad = (y_max - y_min) * 0.05
                y_range = [y_min - y_range_pad, y_max + y_range_pad]

                if ylim is not None:
                    if ylim[0] is not None:
                        y_range[0] = ylim[0]
                    if ylim[1] is not None:
                        y_range[1] = ylim[1]

            for i in range(n_subplots):
                row_idx = i // cols
                col_idx = i % cols
                suffix = str(i + 1) if i > 0 else ''
                is_leftmost = col_idx == 0

                x_domain = [col_idx / cols + 0.05, (col_idx + 1) / cols - 0.02]
                y_domain = [
                    1 - (row_idx + 1) / rows + 0.1,
                    1 - row_idx / rows - 0.05,
                ]

                layout[f'xaxis{suffix}'] = {
                    'title': {'text': _axis_title(data[x])},
                    'domain': x_domain,
                }

                # inf is not valid JSON
                sanitized_y_range = [
                    None if (v is not None and np.isinf(v)) else v for v in y_range
                ]
                y_axis_config: dict[str, Any] = {
                    'domain': y_domain,
                    'type': yscale,
                    'range': sanitized_y_range,
                }

                # Only show y-axis title and tick labels on leftmost column
                if is_leftmost:
                    y_axis_config['title'] = {'text': _axis_title(data)}
                else:
                    y_axis_config['title'] = {'text': ''}
                    y_axis_config['showticklabels'] = False

                layout[f'yaxis{suffix}'] = y_axis_config
        else:
            layout['xaxis'] = {'title': {'text': _axis_title(data[x])}}
            y_axis_config = {
                'title': {'text': _axis_title(data)},
                'type': yscale,
            }
            if ylim is not None and (ylim[0] is not None or ylim[1] is not None):
                y_range_single: list[Optional[float]] = [
                    ylim[0]
                    if (ylim[0] is not None and not np.isinf(ylim[0]))
                    else None,
                    ylim[1]
                    if (ylim[1] is not None and not np.isinf(ylim[1]))
                    else None,
                ]
                y_axis_config['range'] = y_range_single
            layout['yaxis'] = y_axis_config

        return layout

    def finish(
        self,
        target: dict[str, Any],
        xticklabelunits: Union[bool, Literal['auto']] = 'auto',
    ) -> None:
        """leave `target` unchanged."""

    def limit_facets(
        self,
        target: dict[str, Any],
        low: xr.DataArray | float | None,
        high: float | None,
    ) -> None:
        """set the x range of each facet of `target` to (`low`, `high`).

        Args:
            target: the result dict to limit
            low: the lower limit, or a DataArray with one value per facet in facet
                order; `None` selects the finite minimum of the x data on the facet
            high: the upper limit; `None` selects the finite maximum of the x data
                on the facet
        """
        n_facets = len(target.get('facets', [None]))
        if isinstance(low, xr.DataArray):
            lows: list[float | None] = [float(v) for v in _to_numpy(low.values).flat]
        else:
            lows = [low] * n_facets

        layout = target['layout']
        for i in range(n_facets):
            xref = _axis_refs(i + 1)[0]
            lo = lows[i] if i < len(lows) else None
            hi = high
            if lo is None or hi is None:
                x_values = [
                    _to_numpy(t['x'])
                    for t in target['traces']
                    if t.get('xaxis', 'x') == xref and 'x' in t
                ]
                if x_values:
                    x_min, x_max = _finite_extent(np.concatenate(x_values))
                else:
                    x_min, x_max = 0.0, 1.0
                lo = x_min if lo is None else lo
                hi = x_max if hi is None else hi
            axis = layout.setdefault(_layout_axis_key('xaxis', i), {})
            axis['range'] = [lo, hi]

    def mark_noise_level(
        self,
        data: xr.Dataset,
        var_name: str,
        grid: dict[str, Any],
        where: Literal['x', 'y', 'colorbar'],
    ) -> None:
        """draw the system noise level (in dBm) of `var_name` on each facet of `grid`.

        Nothing is drawn when `data` has no `system_noise` variable or `var_name` has
        no noise bandwidth. The level of a facet is the `system_noise` entry at the
        facet value recorded in `grid`.

        Args:
            data: the dataset with the `system_noise` variable (dBm/Hz) and `var_name`
            var_name: the data variable or power-bin coordinate that sets the noise
                bandwidth
            grid: the result dict to draw on
            where: ``'x'`` draws a horizontal line for power on the y axis, ``'y'`` a
                vertical line for power on the x axis, and ``'colorbar'`` an
                annotation beside the colorbar of a heatmap

        Raises:
            TypeError: if `where` is ``'colorbar'`` and no trace of `grid` draws a
                colorbar, or if `where` is none of the 3 values
        """
        noise = util.get_system_noise(data, var_name)
        if noise is None:
            return

        layout = grid['layout']
        shapes = layout.setdefault('shapes', [])

        for i, noise_val in enumerate(_facet_noise(noise, grid)):
            if noise_val is None:
                continue
            xref, yref = _axis_refs(i + 1)

            if where == 'x':
                shapes.append({
                    'type': 'line',
                    'x0': 0,
                    'x1': 1,
                    'xref': f'{xref} domain',
                    'y0': noise_val,
                    'y1': noise_val,
                    'yref': yref,
                    'line': {'color': _NOISE_COLOR, 'width': 1, 'dash': 'dot'},
                })
            elif where == 'y':
                shapes.append({
                    'type': 'line',
                    'x0': noise_val,
                    'x1': noise_val,
                    'xref': xref,
                    'y0': 0,
                    'y1': 1,
                    'yref': f'{yref} domain',
                    'line': {'color': _NOISE_COLOR, 'width': 1, 'dash': 'dot'},
                })
            elif where == 'colorbar':
                colorbar_traces = [t for t in grid['traces'] if t.get('showscale')]
                if not colorbar_traces:
                    raise TypeError('no colorbar in the result dict')
                zmin = colorbar_traces[0].get('zmin')
                zmax = colorbar_traces[0].get('zmax')
                if zmin is None or zmax is None or zmax <= zmin:
                    continue

                # Plotly annotations cannot reference colorbar values, so the marker
                # is placed in paper coordinates at the normalized (0-1) position
                # within the colorbar range
                colorbar_bottom = 0.1
                colorbar_top = 0.9
                clamped_noise = max(zmin, min(zmax, noise_val))
                normalized = (clamped_noise - zmin) / (zmax - zmin)
                y_paper = colorbar_bottom + normalized * (
                    colorbar_top - colorbar_bottom
                )

                layout.setdefault('annotations', []).append({
                    'x': 1.02,
                    'y': y_paper,
                    'xref': 'paper',
                    'yref': 'paper',
                    'text': 'kTB',
                    'showarrow': True,
                    'arrowhead': 2,
                    'arrowsize': 1,
                    'arrowwidth': 1,
                    'arrowcolor': _NOISE_COLOR,
                    'font': {'color': _NOISE_COLOR, 'size': 10},
                    'ax': 20,
                    'ay': 0,
                })
            else:
                raise TypeError('where must be "x", "y", or "colorbar"')

    def get_plot_data(self) -> list[dict[str, Any]]:
        """Return all pending plot data and clear the buffer."""
        data = self._pending_traces.copy()
        self._pending_traces.clear()
        return data


def _widen_y_ranges(
    layout: dict[str, Any], ylim: tuple[float | None, float | None]
) -> None:
    """widen every finite y axis ``range`` of `layout` to include `ylim`"""
    for key, axis in layout.items():
        if not key.startswith('yaxis') or not isinstance(axis, dict):
            continue
        y_range = axis.get('range')
        if y_range is None:
            continue
        if ylim[0] is not None and np.isfinite(ylim[0]):
            y_range[0] = ylim[0] if y_range[0] is None else min(y_range[0], ylim[0])
        if ylim[1] is not None and np.isfinite(ylim[1]):
            y_range[1] = ylim[1] if y_range[1] is None else max(y_range[1], ylim[1])


def _facet_noise(noise: xr.DataArray, grid: dict[str, Any]) -> list[float | None]:
    """return one value of `noise` per facet of `grid`, in facet order.

    When the ``col_dim`` of `grid` is a dimension of `noise`, the value of a facet is
    the entry of `noise` at the matching coordinate value along that dimension, or
    None without a match. Otherwise the values of `noise` are returned in order, one
    per facet, with None for each facet beyond the last value.
    """
    facets: list[Any] = grid.get('facets', [None])
    col_dim = grid.get('col_dim')
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


def _to_json_serializable(
    arr: Any, preserve_inf: bool = False
) -> Union[np.ndarray, list, float, int, str, None]:
    """convert an array or scalar to a Plotly graph object input.

    Numeric arrays are returned as contiguous numpy arrays, with a cupy array
    transferred to the CPU. String, object, and datetime64 arrays are returned as
    lists, and a timedelta64 array as seconds. In arrays, NaN is kept. In lists and
    scalars, NaN is returned as None.

    Args:
        arr: the array or scalar
        preserve_inf: if True, Inf is kept in arrays and returned as ±1e38 in lists
            and scalars; if False, Inf is NaN in arrays and None in lists and scalars
    """
    # Use large but finite values for Inf when preserving
    _INF_REPLACEMENT = 1e38
    _NEG_INF_REPLACEMENT = -1e38

    def _sanitize_value(v):
        """return `v` with NaN as None and Inf as `preserve_inf` selects"""
        if isinstance(v, float):
            if np.isnan(v):
                return None
            if np.isinf(v):
                if preserve_inf:
                    return _INF_REPLACEMENT if v > 0 else _NEG_INF_REPLACEMENT
                else:
                    return None  # Skip infinite values in line plots
        return v

    def _sanitize_list(lst):
        """Recursively sanitize a list."""
        result: list[Any] = []
        for item in lst:
            if isinstance(item, list):
                result.append(_sanitize_list(item))
            elif isinstance(item, float):
                result.append(_sanitize_value(item))
            else:
                result.append(item)
        return result

    # cupy arrays transfer to the CPU
    if hasattr(arr, 'get'):
        arr = arr.get()

    if isinstance(arr, np.ndarray):
        # String types - must use list format
        if arr.dtype.kind in ('U', 'S', 'O'):
            return arr.tolist()
        # datetime64 must use list format
        if np.issubdtype(arr.dtype, np.datetime64):
            return [str(x) for x in arr]
        if np.issubdtype(arr.dtype, np.timedelta64):
            arr = (arr / np.timedelta64(1, 's')).astype(np.float64)

        if arr.dtype.kind in ('f', 'i', 'u'):
            arr = np.ascontiguousarray(arr)
            if not preserve_inf and np.issubdtype(arr.dtype, np.floating):
                arr = np.where(np.isinf(arr), np.nan, arr)
            return arr

        # Fallback to list format for other types
        return _sanitize_list(arr.tolist())

    if isinstance(arr, (np.floating, np.integer)):
        val = float(arr)
        return _sanitize_value(val)
    if isinstance(arr, float):
        return _sanitize_value(arr)
    return arr


def _format_subplots(data, opts: specs.SharedPlotOptions):
    col_titles = labels.label_by_coord(
        data,
        opts.col_label_format,
        coord_or_dim=opts.col or 'port',
        title_case=True,
        name=data.name,
        **data.attrs,
    )

    return [
        {
            'text': t,
            'showarrow': False,
            'xref': 'paper',
            'yref': 'paper',
            'x': ((i + 0.5) / len(col_titles)),
            'y': 1.0,  # X/Y placement on canvas
            'xanchor': 'center',
            'yanchor': 'bottom',
        }
        for i, t in enumerate(col_titles)
    ]


def _measurand(da: xr.DataArray) -> sa.labels.Measurand:
    name = None if da.name is None else str(da.name)
    return sa.labels.measurand_from_attrs(da.attrs, name)


def _axis_title(da: xr.DataArray) -> str:
    """return the name of `da` with its units in parentheses, from its attrs."""
    return sa.labels.quantity_label(_measurand(da))


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

    # Built-in Plotly colormaps that can be used directly
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
        # Fallback to viridis if colormap not found
        return 'Viridis'

    n_colors = 256
    colorscale: list[list[Any]] = []
    for i in range(n_colors):
        pos = i / (n_colors - 1)
        rgba = mpl_cmap(pos)
        r, g, b = int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
        colorscale.append([pos, f'rgb({r},{g},{b})'])

    return colorscale


def _make_quantized_colorscale(
    cmap: str,
    vmin: float,
    vmax: float,
    vstep: float,
) -> list[list[Any]]:
    """return a stepped Plotly colorscale with bands of `vstep` from `vmin` to `vmax`.

    The bands follow the matplotlib `BoundaryNorm` over a resampled colormap.

    Args:
        cmap: a matplotlib colormap name; an undefined name selects viridis
        vmin: the lower color limit
        vmax: the upper color limit
        vstep: the spacing of the color levels, in the units of the data

    Returns:
        a list of ``[position, color]`` pairs with 2 entries per band
    """
    import matplotlib.pyplot as plt

    # data=None is accepted when vmin and vmax are both given
    levels = util.quantized_value_range(cast('xr.DataArray', None), vmin, vmax, vstep)
    n_levels = len(levels) - 1

    n_levels = max(n_levels, 1)

    try:
        mpl_cmap = plt.get_cmap(cmap).resampled(n_levels)
    except ValueError:
        mpl_cmap = plt.get_cmap('viridis').resampled(n_levels)

    # each band takes 2 entries of the same color, so the colorscale is flat
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
