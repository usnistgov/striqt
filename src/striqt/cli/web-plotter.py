#!/usr/bin/env python3
"""
Web-based real-time display for radio diagnostic tooling.

This is a variant of live-plotter that uses FastAPI/uvicorn to serve
real-time visualizations via a web browser, taking advantage of GPU
acceleration on clients through WebGL (via Plotly.js).

Usage:
    python _experiments/web-plotter.py path/to/spec.yaml
    # Then open http://localhost:8000 in a browser
"""

from __future__ import annotations

import asyncio
import json
import sys
import typing
from contextlib import asynccontextmanager
import functools
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Literal, Optional, Set, Tuple, TypedDict, Union, cast

import click
import numpy as np
import xarray as xr
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

def _to_numpy(arr: Any) -> np.ndarray:
    """Convert array to numpy, handling cupy arrays transparently.

    This enables web-plotter to work on systems where xarray data
    is backed by cupy arrays (e.g., Jetson TX2i with GPU acceleration).

    Note: Only data variables (DataArray.values) may be cupy arrays.
    Coordinates are always numpy arrays and don't need this conversion.
    """
    if hasattr(arr, 'get'):
        # cupy array - transfer to CPU
        return arr.get()  # ty: ignore[return-value]
    return np.asarray(arr)

# Note: StaticFiles not currently used but available if needed
# from fastapi.staticfiles import StaticFiles

if typing.TYPE_CHECKING:
    import striqt.sensor as ss
    import striqt.figures as sf
    import striqt.analysis as sa


# ---------------------------------------------------------------------------
# Type definitions for app state
# ---------------------------------------------------------------------------


class CoordinateDict(TypedDict):
    """A single coordinate entry with name, values, and unit."""

    name: str
    values: List[str]
    unit: str


class ExtraCoordDict(TypedDict):
    """A single extra coordinate entry (from AcquisitionInfo) with name and value."""

    name: str
    value: str
    unit: str


class CoordinatesData(TypedDict, total=False):
    """Coordinates data structure sent to the frontend.

    Keys:
        num_ports: Number of receiver ports
        groups: Dict mapping class name to list of coordinate dicts (capture fields)
        extra_coords: Dict mapping class name to list of extra coordinate dicts (AcquisitionInfo)
    """

    num_ports: int
    groups: Dict[str, List[CoordinateDict]]
    extra_coords: Dict[str, List[ExtraCoordDict]]


class AppState(TypedDict, total=False):
    """Type definition for the global application state.

    Keys:
        delayed_dataset: Current DelayedDataset from acquisition
        plotter: Retained WebPlotBackend instance (preserves vmin/vmax across captures)
        selected_variable: Currently selected data variable name
        available_variables: List of available data variable names
        variable_labels: Mapping from variable name to display label (standard_name)
        websockets: Set of connected WebSocket clients
        broadcast_pending: Flag to prevent broadcast queue buildup
        data_updated: Flag indicating new data is available
        plot_opts: Plot options from spec file
        data_select: Selection dict from DataOptions.select for xarray.sel()
        spec_filename: Name of the YAML/JSON spec file being used
    """

    delayed_dataset: Optional['ss.lib.compute.DelayedDataset']
    plotter: Optional['WebPlotBackend']
    selected_variable: str
    available_variables: List[str]
    variable_labels: Dict[str, str]
    websockets: Set[WebSocket]
    broadcast_pending: bool
    data_updated: bool
    plot_opts: Optional['sf.specs.PlotOptions']
    data_select: Dict[str, Any]
    spec_filename: str


# ---------------------------------------------------------------------------
# Plot configuration (mirrors live-plotter)
# ---------------------------------------------------------------------------

plot_opts_dict = {
    'data': {
        'select': {
            'channel_power_bin': 'slice(-100, -15)',
            'spectrogram_power_bin': 'slice(-130, -50)',
            'spectrogram_time': 'slice(0, 20e-3)',
        },
        'sweep_index': -1,
    },
    'plotter': {
        'col': 'port',
        'col_label_format': 'Port {port} {channel_name}',
        'style': None,
        'filename_fmt': '{name}.svg',
        'suptitle_fmt': '',
    },
    'variables': {
        'spectrogram': {},
        'power_spectral_density': {},
    },
}


# ---------------------------------------------------------------------------
# WebPlotBackend - FastAPI/Plotly-based backend mirroring PlotBackend
# ---------------------------------------------------------------------------


class WebPlotBackend:
    """
    A plot backend that generates Plotly.js-compatible JSON for WebGL rendering.

    This mirrors the interface of striqt.figures.backend.PlotBackend but outputs
    data structures suitable for real-time web visualization instead of matplotlib
    figures.
    """

    def __init__(
        self,
        opts: 'sf.specs.SharedPlotOptions',
    ):
        # Store opts, using defaults if not provided
        self.opts = opts
        self._pending_traces: List[Dict[str, Any]] = []
        self._layout: Dict[str, Any] = {}
        # Track vmin/vmax across captures for consistent colorbar scaling (heatmaps)
        self._remembered_vmin: Optional[float] = None
        self._remembered_vmax: Optional[float] = None
        # Track ymin/ymax across captures for consistent y-axis scaling (line plots)
        self._remembered_ymin: Optional[float] = None
        self._remembered_ymax: Optional[float] = None

    def reset(self) -> None:
        """Clear per-broadcast state while preserving vmin/vmax memory."""
        self._pending_traces.clear()
        self._layout.clear()

    def _get_subplot_key(self, data: xr.DataArray) -> str:
        """Generate a subplot identifier from column/row coordinates."""
        parts = []
        col = self.opts.col
        row = self.opts.row
        if col and col in data.coords:
            parts.append(f'{col}={data[col].values}')
        if row and row in data.coords:
            parts.append(f'{row}={data[row].values}')
        return '_'.join(parts) if parts else 'main'

    def _get_col_dim(self, data: xr.DataArray) -> Optional[str]:
        """Find the dimension to use for column faceting.

        If opts.col is a dimension, use it directly.
        If opts.col is a coordinate, find the dimension it belongs to.
        Returns None if no suitable dimension is found.
        """
        col = self.opts.col
        if not col:
            return None

        # If col is directly a dimension, use it
        if col in data.dims:
            return col

        # If col is a coordinate, find which dimension it belongs to
        if col in data.coords:
            coord = data.coords[col]
            # A coordinate's dims tell us which dimension(s) it indexes
            if coord.dims and len(coord.dims) == 1:
                dim = coord.dims[0]
                # Only use if the dimension exists in data and has multiple values
                if dim in data.dims and len(data[dim]) > 1:
                    return dim

        return None

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
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Create a heatmap trace for Plotly.

        All subplots (e.g., multiple RX ports) share the same colorbar scale.

        Args:
            data: The data array to plot
            x: Name of the x coordinate
            y: Name of the y coordinate
            cmap: Matplotlib colormap name
            vmin: Minimum value for color scale
            vmax: Maximum value for color scale
            vstep: If provided, quantize the colormap into discrete levels
                   spaced by vstep. This creates a stepped/banded colorbar
                   instead of a continuous gradient.
        """
        # Handle faceting by iterating over col dimension
        traces = []
        col_dim = self._get_col_dim(data)

        # Compute shared vmin/vmax across ALL data (all ports) for consistent colorbar
        data_np = _to_numpy(data.values)
        if vmin is None:
            # Remember the min vmin across captures for consistent colorbar scaling
            current_vmin = float(np.nanmin(data_np))
            if self._remembered_vmin is None:
                self._remembered_vmin = current_vmin
            else:
                self._remembered_vmin = min(self._remembered_vmin, current_vmin)
            vmin = self._remembered_vmin
        if vmax is None:
            # Remember the max vmax across captures for consistent colorbar scaling
            current_vmax = float(np.nanmax(data_np))
            if self._remembered_vmax is None:
                self._remembered_vmax = current_vmax
            else:
                self._remembered_vmax = max(self._remembered_vmax, current_vmax)
            vmax = self._remembered_vmax

        # Build colorscale - quantized if vstep is provided
        if vstep is not None:
            colorscale = _make_quantized_colorscale(cmap, vmin, vmax, vstep)
        else:
            colorscale = _mpl_to_plotly_colorscale(cmap)

        if col_dim:
            n_subplots = len(data[col_dim].values)
            for i, col_val in enumerate(data[col_dim].values):
                sub = data.sel({col_dim: col_val})
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

        result = {
            'type': 'heatmap',
            'traces': traces,
            'layout': self._make_heatmap_layout(data, x, y, len(traces)),
            'variable': str(data.name),
        }
        self._pending_traces.append(result)
        return result

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
    ) -> Dict[str, Any]:
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
        # Final shape validation - z should be (len(y), len(x))
        if z_data.shape != (len(y_data), len(x_data)):
            # Try transpose if dimensions are swapped
            if z_data.shape == (len(x_data), len(y_data)):
                z_data = z_data.T

        # Convert to lists for JSON serialization
        # Use preserve_inf=True for heatmaps to show -Inf as very low values
        trace: Dict[str, Any] = {
            'type': 'heatmap',  # Use 'heatmapgl' for WebGL acceleration
            'x': _to_json_serializable(x_data),
            'y': _to_json_serializable(y_data),
            'z': _to_json_serializable(z_data, preserve_inf=True),
            'colorscale': colorscale,
            'showscale': show_colorbar,  # Only show colorbar when requested
        }

        # Add colorbar configuration only if showing
        if show_colorbar:
            trace['colorbar'] = {
                'title': {'text': _format_units_only(data)},
            }

        if vmin is not None:
            trace['zmin'] = vmin
        if vmax is not None:
            trace['zmax'] = vmax

        if subplot_idx is not None:
            trace['xaxis'] = f'x{subplot_idx}' if subplot_idx > 1 else 'x'
            trace['yaxis'] = f'y{subplot_idx}' if subplot_idx > 1 else 'y'

        return trace

    def _make_heatmap_layout(
        self, data: xr.DataArray, x: str, y: str, n_subplots: int
    ) -> Dict[str, Any]:
        """Create layout for heatmap with optional subplots."""
        layout: Dict[str, Any] = {
            'uirevision': 'constant',  # Preserve zoom/pan on updates
        }

        if self.opts.col is None or self.opts.col_label_format is None:
            pass
        else:
            layout['annotations'] = _format_subplots(data, self.opts)

        if n_subplots > 1:
            # Create subplot grid
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
                    'title': {'text': _format_coord_label(data[x])},
                    'domain': x_domain,
                }
                layout[f'yaxis{suffix}'] = {
                    'title': {'text': _format_coord_label(data[y])},
                    'domain': y_domain,
                }
        else:
            layout['xaxis'] = {'title': {'text': _format_coord_label(data[x])}}
            layout['yaxis'] = {'title': {'text': _format_coord_label(data[y])}}

        return layout

    def line(
        self,
        data: xr.DataArray,
        *,
        x: str,
        hue: Optional[str] = None,
        yscale: Literal['linear', 'log'] = 'linear',
        ylim: Optional[tuple[Optional[float], Optional[float]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Create line plot traces for Plotly (uses WebGL scattergl for performance).
        """
        traces = []
        col_dim = self._get_col_dim(data)

        if col_dim:
            for i, col_val in enumerate(data[col_dim].values):
                sub = data.sel({col_dim: col_val})
                subplot_traces = self._make_line_traces(sub, x, hue, subplot_idx=i + 1)
                for t in subplot_traces:
                    t['legendgroup'] = f'{col_dim}={col_val}'
                traces.extend(subplot_traces)
        else:
            traces.extend(self._make_line_traces(data, x, hue))

        n_subplots = len(data[col_dim].values) if col_dim else 1

        # Track and expand y-axis limits persistently across captures
        # Compute current data range (excluding inf values)
        data_np = _to_numpy(data.values)
        finite_mask = np.isfinite(data_np)
        if np.any(finite_mask):
            current_ymin = float(np.nanmin(data_np[finite_mask]))
            current_ymax = float(np.nanmax(data_np[finite_mask]))
        else:
            current_ymin, current_ymax = 0.0, 1.0

        # Build effective ylim by combining caller-specified limits with remembered limits
        # For ymin: track the minimum (expand downward)
        # For ymax: track the maximum (expand upward)

        # Handle ymin - use caller-specified value if provided, otherwise use data min
        candidate_ymin = ylim[0] if (ylim is not None and ylim[0] is not None) else current_ymin
        if self._remembered_ymin is None:
            self._remembered_ymin = candidate_ymin
        else:
            self._remembered_ymin = min(self._remembered_ymin, candidate_ymin)

        # Handle ymax - use caller-specified value if provided, otherwise use data max
        candidate_ymax = ylim[1] if (ylim is not None and ylim[1] is not None) else current_ymax
        if self._remembered_ymax is None:
            self._remembered_ymax = candidate_ymax
        else:
            self._remembered_ymax = max(self._remembered_ymax, candidate_ymax)

        effective_ylim = (self._remembered_ymin, self._remembered_ymax)

        result = {
            'type': 'line',
            'traces': traces,
            'layout': self._make_line_layout(data, x, yscale, n_subplots, ylim=effective_ylim),
            'variable': str(data.name),
        }
        self._pending_traces.append(result)
        return result

    def _make_line_traces(
        self,
        data: xr.DataArray,
        x: str,
        hue: Optional[str],
        subplot_idx: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Create line traces, optionally split by hue dimension."""
        traces = []
        x_data = np.asarray(data[x].values).flatten()

        # Plotly default color sequence for consistent colors across subplots
        plotly_colors = [
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
        ]

        # Check if hue is a dimension or coordinate we can iterate over
        hue_is_dim = hue and hue in data.dims
        hue_is_coord = hue and hue in data.coords and hue not in data.dims
        
        if hue_is_dim:
            # hue is a dimension - iterate over its values
            hue_values = data[hue].values
            for hue_idx, hue_val in enumerate(hue_values):
                sub = data.sel({hue: hue_val})
                # Squeeze out the hue dimension, then flatten remaining dims
                y_data = _to_numpy(sub.values).squeeze()

                # Ensure x and y have matching lengths
                if y_data.size != x_data.size:
                    # If y has extra dimensions, try to align with x
                    if x in sub.dims:
                        # Select along x dimension to get 1D array
                        y_data = _to_numpy(sub.values)
                        # Flatten all dims except x, keeping x as the last dim
                        x_dim_idx = list(sub.dims).index(x)
                        # Move x dim to last position and flatten others
                        y_data = np.moveaxis(y_data, x_dim_idx, -1)
                        y_data = y_data.reshape(-1, y_data.shape[-1])
                        # If still multi-row, take first row (or could average)
                        if y_data.shape[0] > 1:
                            y_data = y_data[0]
                        y_data = y_data.flatten()
                    else:
                        y_data = y_data.flatten()

                # Use markers only when there's a single point (lines won't show)
                mode = 'markers' if len(x_data) == 1 else 'lines'

                # Use consistent color for same hue value across subplots
                color = plotly_colors[hue_idx % len(plotly_colors)]
                hue_name = f'{hue}={hue_val}'

                trace: Dict[str, Any] = {
                    'type': 'scattergl',  # WebGL-accelerated scatter
                    'mode': mode,
                    'x': _to_json_serializable(x_data),
                    'y': _to_json_serializable(y_data),
                    'name': hue_name,
                    'legendgroup': hue_name,  # Group same hue across subplots
                    'showlegend': subplot_idx is None
                    or subplot_idx == 1,  # Only show legend for first subplot
                    'line': {'color': color},
                    'marker': {'color': color},
                }

                if subplot_idx is not None:
                    trace['xaxis'] = f'x{subplot_idx}' if subplot_idx > 1 else 'x'
                    trace['yaxis'] = f'y{subplot_idx}' if subplot_idx > 1 else 'y'

                traces.append(trace)
        elif hue_is_coord:
            # hue is a coordinate but not a dimension - use it for labeling
            # but iterate over the coordinate values
            hue_values = data[hue].values
            # If hue_values is 0-d or scalar, make it iterable
            if np.ndim(hue_values) == 0:
                hue_values = [hue_values.item()]
            else:
                hue_values = np.unique(hue_values)
            
            for hue_idx, hue_val in enumerate(hue_values):
                # For non-dimension coordinates, we can't select - just use the data
                y_data = _to_numpy(data.values).squeeze()

                # Ensure x and y have matching lengths
                if y_data.size != x_data.size:
                    if x in data.dims:
                        x_dim_idx = list(data.dims).index(x)
                        y_data = np.moveaxis(y_data, x_dim_idx, -1)
                        y_data = y_data.reshape(-1, y_data.shape[-1])
                        if y_data.shape[0] > 1:
                            y_data = y_data[0]
                        y_data = y_data.flatten()
                    else:
                        y_data = y_data.flatten()

                mode = 'markers' if len(x_data) == 1 else 'lines'
                color = plotly_colors[hue_idx % len(plotly_colors)]
                hue_name = f'{hue}={hue_val}'

                trace: Dict[str, Any] = {
                    'type': 'scattergl',
                    'mode': mode,
                    'x': _to_json_serializable(x_data),
                    'y': _to_json_serializable(y_data),
                    'name': hue_name,
                    'legendgroup': hue_name,
                    'showlegend': subplot_idx is None or subplot_idx == 1,
                    'line': {'color': color},
                    'marker': {'color': color},
                }

                if subplot_idx is not None:
                    trace['xaxis'] = f'x{subplot_idx}' if subplot_idx > 1 else 'x'
                    trace['yaxis'] = f'y{subplot_idx}' if subplot_idx > 1 else 'y'

                traces.append(trace)
                # Only add one trace for non-dimension hue (can't actually split)
                break
        else:
            y_data = _to_numpy(data.values).squeeze()

            # Ensure x and y have matching lengths
            if y_data.size != x_data.size:
                if x in data.dims:
                    x_dim_idx = list(data.dims).index(x)
                    y_data = np.moveaxis(y_data, x_dim_idx, -1)
                    y_data = y_data.reshape(-1, y_data.shape[-1])
                    if y_data.shape[0] > 1:
                        y_data = y_data[0]
                    y_data = y_data.flatten()
                else:
                    y_data = y_data.flatten()

            # Use markers only when there's a single point (lines won't show)
            mode = 'markers' if len(x_data) == 1 else 'lines'

            trace_name = str(data.name)
            trace = {
                'type': 'scattergl',
                'mode': mode,
                'x': _to_json_serializable(x_data),
                'y': _to_json_serializable(y_data),
                'name': trace_name,
                'legendgroup': trace_name,
                'showlegend': subplot_idx is None or subplot_idx == 1,
                'line': {'color': plotly_colors[0]},
                'marker': {'color': plotly_colors[0]},
            }

            if subplot_idx is not None:
                trace['xaxis'] = f'x{subplot_idx}' if subplot_idx > 1 else 'x'
                trace['yaxis'] = f'y{subplot_idx}' if subplot_idx > 1 else 'y'

            traces.append(trace)

        return traces

    def _make_line_layout(
        self,
        data: xr.DataArray,
        x: str,
        yscale: str,
        n_subplots: int,
        *,
        ylim: Optional[tuple[Optional[float], Optional[float]]] = None,
    ) -> Dict[str, Any]:
        """Create layout for line plot with optional subplots.

        Subplots in the same row share the same y-axis range.
        Only the leftmost subplot in each row shows y-axis title and tick labels.
        """
        import striqt.figures as sf

        layout: Dict[str, Any] = {
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

            # Use ylim if fully specified, otherwise compute from data
            if ylim is not None and ylim[0] is not None and ylim[1] is not None:
                # Both limits provided - use them directly
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

                # Apply ylim overrides if provided
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
                    'title': {'text': _format_coord_label(data[x])},
                    'domain': x_domain,
                }

                # Sanitize y_range to ensure no inf values (not valid JSON)
                sanitized_y_range = [
                    None if (v is not None and np.isinf(v)) else v for v in y_range
                ]
                y_axis_config: Dict[str, Any] = {
                    'domain': y_domain,
                    'type': yscale,
                    'range': sanitized_y_range,
                }

                # Only show y-axis title and tick labels on leftmost column
                if is_leftmost:
                    y_axis_config['title'] = {'text': _format_label_with_units(data)}
                else:
                    y_axis_config['title'] = {'text': ''}
                    y_axis_config['showticklabels'] = False

                layout[f'yaxis{suffix}'] = y_axis_config
        else:
            layout['xaxis'] = {'title': {'text': _format_coord_label(data[x])}}
            y_axis_config = {
                'title': {'text': _format_label_with_units(data)},
                'type': yscale,
            }
            # Apply ylim if provided for single subplot
            if ylim is not None and (ylim[0] is not None or ylim[1] is not None):
                y_range_single: list[Optional[float]] = [
                    ylim[0] if (ylim[0] is not None and not np.isinf(ylim[0])) else None,
                    ylim[1] if (ylim[1] is not None and not np.isinf(ylim[1])) else None,
                ]
                y_axis_config['range'] = y_range_single
            layout['yaxis'] = y_axis_config

        return layout

    def _coord_kws(self, **kwargs) -> Dict[str, Any]:
        """
        Return coordinate/layout kwargs for plotting.

        This mirrors PlotBackend._coord_kws() to provide compatible interface
        for striqt.figures.data_vars functions.
        """
        col = self.opts.col
        if col is None and self.opts.row is None:
            col = '_view'

        return {
            **kwargs,
            'col_wrap': self.opts.col_wrap,
            'row': self.opts.row,
            'col': col,
        }

    def finish(
        self,
        grid: Dict[str, Any],
        xticklabelunits: Union[bool, Literal['auto']] = 'auto',
    ) -> None:
        """
        Finalize the plot (no-op for web backend, plot data already collected).

        This mirrors PlotBackend.finish() to provide compatible interface
        for striqt.figures.data_vars functions.
        """
        # For web backend, the plot data is already collected in _pending_traces
        # by heatmap() and line() methods. Nothing more to do here.
        pass

    def mark_noise_level(
        self,
        data: xr.Dataset,
        var_name: str,
        grid: Dict[str, Any],
        where: Literal['x', 'y', 'colorbar'],
    ) -> None:
        """
        Add noise level markers to the plot.

        This mirrors PlotBackend.mark_noise_level() to provide compatible interface
        for striqt.figures.data_vars functions.

        For web backend, we add shape annotations to the Plotly layout.
        """
        from striqt.figures import util

        noise = util.get_system_noise(data, var_name)
        if noise is None:
            return

        # Find the most recent plot data to add noise markers to
        if not self._pending_traces:
            return

        plot_data = self._pending_traces[-1]
        layout = plot_data.get('layout', {})

        # Initialize shapes list if not present
        if 'shapes' not in layout:
            layout['shapes'] = []

        # Add noise level lines for each port
        noise_values = _to_numpy(noise.values).flat if hasattr(noise, 'values') else [noise]

        for i, noise_val in enumerate(noise_values):
            noise_float = float(noise_val)

            if where == 'x':
                # Horizontal line at noise level (for line plots with y=power)
                shape = {
                    'type': 'line',
                    'x0': 0,
                    'x1': 1,
                    'xref': f'x{i + 1} domain' if i > 0 else 'x domain',
                    'y0': noise_float,
                    'y1': noise_float,
                    'yref': f'y{i + 1}' if i > 0 else 'y',
                    'line': {'color': '#eec009', 'width': 1, 'dash': 'dot'},
                }
            elif where == 'y':
                # Vertical line at noise level (for histograms with x=power)
                shape = {
                    'type': 'line',
                    'x0': noise_float,
                    'x1': noise_float,
                    'xref': f'x{i + 1}' if i > 0 else 'x',
                    'y0': 0,
                    'y1': 1,
                    'yref': f'y{i + 1} domain' if i > 0 else 'y domain',
                    'line': {'color': '#eec009', 'width': 1, 'dash': 'dot'},
                }
            elif where == 'colorbar':
                # For heatmaps, add a marker on the colorbar
                # Plotly annotations can't directly reference colorbar values, so we need
                # to compute the normalized position (0-1) within the colorbar range
                # and use paper coordinates for the y position
                traces = plot_data.get('traces', [])
                if traces:
                    # Get the z range from the first trace with a colorbar
                    zmin = None
                    zmax = None
                    for trace in traces:
                        if trace.get('showscale', False):
                            zmin = trace.get('zmin')
                            zmax = trace.get('zmax')
                            break

                    if zmin is not None and zmax is not None and zmax > zmin:
                        # Compute normalized position (0-1) within the colorbar range
                        # Colorbar typically spans from y=0.1 to y=0.9 in paper coords
                        colorbar_bottom = 0.1
                        colorbar_top = 0.9
                        colorbar_height = colorbar_top - colorbar_bottom

                        # Clamp noise value to the colorbar range
                        clamped_noise = max(zmin, min(zmax, noise_float))
                        normalized = (clamped_noise - zmin) / (zmax - zmin)
                        y_paper = colorbar_bottom + normalized * colorbar_height

                        if 'annotations' not in layout:
                            layout['annotations'] = []
                        layout['annotations'].append({
                            'x': 1.02,
                            'y': y_paper,
                            'xref': 'paper',
                            'yref': 'paper',
                            'text': 'kTB',
                            'showarrow': True,
                            'arrowhead': 2,
                            'arrowsize': 1,
                            'arrowwidth': 1,
                            'arrowcolor': '#eec009',
                            'font': {'color': '#eec009', 'size': 10},
                            'ax': 20,
                            'ay': 0,
                        })
                continue  # Skip adding to shapes for colorbar

            layout['shapes'].append(shape)

        plot_data['layout'] = layout

    def get_plot_data(self) -> List[Dict[str, Any]]:
        """Return all pending plot data and clear the buffer."""
        data = self._pending_traces.copy()
        self._pending_traces.clear()
        return data


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _to_json_serializable(
    arr: Union[np.ndarray, Any], preserve_inf: bool = False
) -> Union[list, float, int, str, None]:
    """Convert numpy arrays to JSON-serializable format.

    Handles NaN and Inf values:
    - NaN -> None (JSON null)
    - Inf -> None by default (for line plots where gaps are preferred)
    - Inf -> large finite value if preserve_inf=True (for heatmaps)

    Args:
        arr: The array or value to convert
        preserve_inf: If True, convert Inf to large finite values (for heatmaps).
                     If False, convert Inf to None (for line plots).
    """
    # Use large but finite values for Inf when preserving
    _INF_REPLACEMENT = 1e38
    _NEG_INF_REPLACEMENT = -1e38

    def _sanitize_value(v):
        """Convert NaN to None, Inf based on preserve_inf setting."""
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
        result = []
        for item in lst:
            if isinstance(item, list):
                result.append(_sanitize_list(item))
            elif isinstance(item, float):
                result.append(_sanitize_value(item))
            else:
                result.append(item)
        return result

    # Handle cupy arrays by converting to numpy first
    if hasattr(arr, 'get'):
        arr = arr.get()  # ty: ignore[call-non-callable]

    if isinstance(arr, np.ndarray):
        if arr.dtype.kind in ('U', 'S', 'O'):  # String types
            return arr.tolist()
        # Handle datetime64
        if np.issubdtype(arr.dtype, np.datetime64):
            return [str(x) for x in arr]
        # Handle timedelta64
        if np.issubdtype(arr.dtype, np.timedelta64):
            return (arr / np.timedelta64(1, 's')).tolist()  # Convert to seconds
        # Convert to list and sanitize NaN/Inf values
        return _sanitize_list(arr.tolist())
    if isinstance(arr, (np.floating, np.integer)):
        val = float(arr)
        return _sanitize_value(val)
    if isinstance(arr, float):
        return _sanitize_value(arr)
    return arr


def _format_label(data: xr.DataArray) -> str:
    """Format a label from DataArray name (without units) - for plot titles."""
    return data.attrs.get('long_name', data.attrs['standard_name'])


def _format_subplots(data, opts: 'sf.specs.SharedPlotOptions'):
    import striqt.figures as sf

    col_titles = sf.labels.label_by_coord(
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


def _format_label_with_units(data: xr.DataArray) -> str:
    """Format a label from DataArray name with units - for axis labels."""
    name = data.attrs['standard_name']
    units = data.attrs.get('units', '')
    if units:
        return f'{name} ({units})'
    return name


def _format_units_only(data: xr.DataArray) -> str:
    """Format only the units from DataArray - for colorbar labels."""
    units = data.attrs.get('units', '')
    return units


def _format_coord_label(coord: xr.DataArray) -> str:
    """Format a coordinate label."""
    name = coord.attrs['standard_name']
    units = coord.attrs.get('units', '')
    if units:
        return f'{name} ({units})'
    return name


def _mpl_to_plotly_colorscale(cmap: str) -> Any:
    """Convert matplotlib colormap name to Plotly colorscale.

    Returns a Plotly-compatible colorscale - either a string for built-in
    colormaps or a list of [position, color] pairs for custom ones.
    For colormaps not built into Plotly, extracts colors from matplotlib.
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

    # For other colormaps (like cubehelix), extract from matplotlib
    try:
        mpl_cmap = plt.get_cmap(cmap)
    except ValueError:
        # Fallback to viridis if colormap not found
        return 'Viridis'

    # Sample the colormap at 256 points and convert to Plotly format
    n_colors = 256
    colorscale: List[List[Any]] = []
    for i in range(n_colors):
        pos = i / (n_colors - 1)
        rgba = mpl_cmap(pos)
        # Convert to rgb string format
        r, g, b = int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
        colorscale.append([pos, f'rgb({r},{g},{b})'])

    return colorscale


def _make_quantized_colorscale(
    cmap: str,
    vmin: float,
    vmax: float,
    vstep: float,
) -> List[List[Any]]:
    """Create a quantized (stepped/banded) colorscale for Plotly.

    This creates discrete color bands instead of a continuous gradient,
    matching the behavior of matplotlib's BoundaryNorm with a resampled colormap.

    Args:
        cmap: Matplotlib colormap name
        vmin: Minimum value for color scale
        vmax: Maximum value for color scale
        vstep: Step size between color levels

    Returns:
        Plotly colorscale as list of [position, color] pairs with discrete bands
    """
    import matplotlib.pyplot as plt

    import striqt.figures as sf

    # Compute quantized bin edges using striqt's utility function
    # Pass None for data since vmin/vmax are already provided
    levels = sf.util.quantized_value_range(None, vmin, vmax, vstep)  # type: ignore[arg-type]
    n_levels = len(levels) - 1

    if n_levels < 1:
        n_levels = 1

    # Get the matplotlib colormap and resample to n_levels colors
    try:
        mpl_cmap = plt.get_cmap(cmap).resampled(n_levels)
    except ValueError:
        mpl_cmap = plt.get_cmap('viridis').resampled(n_levels)

    # Build a stepped colorscale where each color band is constant
    # For n_levels, we need 2*n_levels entries to create flat bands
    colorscale: List[List[Any]] = []
    for i in range(n_levels):
        # Normalized position for this band
        pos_start = i / n_levels
        pos_end = (i + 1) / n_levels

        # Get color for this level (sample at center of band)
        rgba = mpl_cmap(i / (n_levels - 1) if n_levels > 1 else 0.5)
        r, g, b = int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
        color = f'rgb({r},{g},{b})'

        # Add two entries with same color to create a flat band
        colorscale.append([pos_start, color])
        if i < n_levels - 1:
            # For all but the last band, add the end position with same color
            colorscale.append([pos_end, color])
        else:
            # Last band ends at 1.0
            colorscale.append([1.0, color])

    return colorscale


# ---------------------------------------------------------------------------
# Use striqt.figures.data_vars registry directly
# ---------------------------------------------------------------------------


def get_data_plots_registry():
    """Get the data variable plotting functions from striqt.figures.data_vars."""
    from striqt.figures.data_vars import _data_plots

    return _data_plots


# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------

# Global state for the web server (typed for improved IDE support and type checking)
_app_state: AppState = {
    'delayed_dataset': None,
    'plotter': None,
    'selected_variable': 'spectrogram',
    'available_variables': [],
    'variable_labels': {},
    'websockets': set(),
    'broadcast_pending': False,
    'data_updated': False,
    'plot_opts': None,
    'data_select': {},
    'spec_filename': '',
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI app startup/shutdown."""
    yield
    # Cleanup on shutdown
    _app_state['websockets'].clear()


app = FastAPI(
    title='Striqt WebView',
    description='Real-time radio diagnostic visualization',
    lifespan=lifespan,
)


# Path to HTML template file
_HTML_TEMPLATE_PATH = Path(__file__).parent / 'web-plotter.html'


@app.get('/', response_class=HTMLResponse)
async def get_index():
    """Serve the main HTML page."""
    return _HTML_TEMPLATE_PATH.read_text()


@app.websocket('/ws')
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time data streaming."""
    await websocket.accept()
    _app_state['websockets'].add(websocket)

    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)

            if msg['type'] == 'get_state':
                # Extract coordinates on demand (deferred from update_dataset)
                coordinates: CoordinatesData = {'num_ports': 1, 'groups': {}, 'extra_coords': {}}
                if _app_state['delayed_dataset'] is not None:
                    coordinates = _extract_capture_coordinates(_app_state['delayed_dataset'])
                await websocket.send_json({
                    'type': 'state',
                    'available_variables': _app_state['available_variables'],
                    'variable_labels': _app_state['variable_labels'],
                    'selected_variable': _app_state['selected_variable'],
                    'coordinates': coordinates,
                    'spec_filename': _app_state['spec_filename'],
                })
                # If we have data, also trigger a plot update for this client
                if _app_state['delayed_dataset'] is not None:
                    await broadcast_plot_update()
            elif msg['type'] == 'select_variable':
                _app_state['selected_variable'] = msg['variable']
                # Reset remembered vmin/vmax/ymin/ymax when switching variables
                plotter = _app_state['plotter']
                if plotter is not None:
                    plotter._remembered_vmin = None
                    plotter._remembered_vmax = None
                    plotter._remembered_ymin = None
                    plotter._remembered_ymax = None
                # Trigger immediate update with new variable
                await broadcast_plot_update()

    except WebSocketDisconnect:
        _app_state['websockets'].discard(websocket)
    except Exception as e:
        print(f'WebSocket error: {e}')
        _app_state['websockets'].discard(websocket)


async def broadcast_plot_update():
    """Broadcast plot data to all connected clients.

    Uses a flag-based approach to prevent queue buildup when the display
    falls behind the data acquisition rate. If a broadcast is already in
    progress, new requests are skipped but the data_updated flag ensures
    the latest data will be sent when the current broadcast completes.
    """
    # Skip if a broadcast is already in progress
    if _app_state['broadcast_pending']:
        _app_state['data_updated'] = True  # Mark that we have newer data
        return

    _app_state['broadcast_pending'] = True

    try:
        await _do_broadcast()
    finally:
        _app_state['broadcast_pending'] = False

        # If new data arrived while we were broadcasting, schedule another update
        if _app_state['data_updated']:
            _app_state['data_updated'] = False
            # Use create_task to avoid recursion depth issues
            asyncio.create_task(_do_broadcast_if_needed())


async def _do_broadcast_if_needed():
    """Helper to trigger another broadcast if data was updated."""
    if (
        _app_state['delayed_dataset'] is not None
        and not _app_state['broadcast_pending']
    ):
        await broadcast_plot_update()


def _select_delayed_variable(result: 'ss.lib.compute.DelayedDataset', name: str):
    import dataclasses

    delayed_var = result.delayed[name]
    return dataclasses.replace(result, delayed={name: delayed_var})


def _prepare_plot_data_sync(
    result: 'ss.lib.compute.DelayedDataset',
    plotter: WebPlotBackend,
    variable: str,
    data_select: Dict[str, Any],
) -> tuple[list[Dict[str, Any]], CoordinatesData]:
    """Synchronous helper to prepare plot data - runs in thread pool.

    This function performs CPU/GPU-bound work that would block the event loop:
    - Coordinate extraction from the delayed dataset
    - Materializing delayed computations (GPU transfer on TX2i)
    - Running the plot function to generate traces
    - JSON-serializable data preparation

    Args:
        result: Delayed dataset from acquisition
        plotter: WebPlotBackend instance for generating plot traces
        variable: Name of the data variable to plot
        data_select: Selection dict from PlotOptions.data.select, to be filtered
            and passed to xarray.Dataset.sel()

    Returns:
        Tuple of (plot_data, coordinates) ready for JSON serialization
    """
    from striqt.sensor.lib import compute

    plotter.reset()  # Clear traces/layout but keep vmin/vmax memory

    # Extract capture coordinates (deferred from update_dataset to reduce acquisition thread work)
    coordinates = _extract_capture_coordinates(result)

    data_plots = get_data_plots_registry()
    plot_func = data_plots[variable]

    # Pull the relevant variable and materialize it into an xarray dataset
    # This is the main blocking operation - triggers GPU computation on TX2i
    delayed = _select_delayed_variable(result, variable)
    ds = compute.from_delayed(delayed).set_xindex('port')

    # Apply data selection from PlotOptions.data.select
    # Filter to only include keys that are valid indexes in this dataset
    if data_select:
        valid_indexes = set(ds.indexes.keys())
        filtered_select = {k: v for k, v in data_select.items() if k in valid_indexes}
        if filtered_select:
            ds = ds.sel(filtered_select)

    # Call the striqt.figures.data_vars function with our WebPlotBackend
    # The function will call plotter.heatmap/line and plotter.finish
    plot_func(ds, plotter)

    plot_data = plotter.get_plot_data()

    return plot_data, coordinates


async def _do_broadcast():
    """Internal broadcast implementation.

    Uses asyncio.to_thread() to run blocking operations (GPU computation,
    coordinate extraction, plotting) in a thread pool. This prevents the
    event loop from being blocked, which would cause:
    - New WebSocket connections to hang
    - Signal handlers (Ctrl+C) to be unresponsive
    - Existing connections to time out
    """
    result = _app_state['delayed_dataset']
    if result is None:
        return

    # Use retained plotter from app state (preserves vmin/vmax across captures)
    plotter = _app_state['plotter']
    if plotter is None:
        return

    variable = _app_state['selected_variable']
    data_plots = get_data_plots_registry()

    if variable in data_plots:
        try:
            # Run blocking operations in thread pool to avoid blocking event loop
            # This is critical for TX2i where GPU operations are slow
            data_select = _app_state['data_select']
            plot_data, coordinates = await asyncio.to_thread(
                _prepare_plot_data_sync, result, plotter, variable, data_select
            )

            if not plot_data:
                # No plot data was generated - likely the function uses
                # matplotlib-specific code (FacetGrid, ax=) that doesn't work
                # with WebPlotBackend
                error_message = json.dumps({
                    'type': 'error',
                    'message': f"'{variable}' uses matplotlib-specific plotting that isn't supported in web view yet",
                })
                for ws in _app_state['websockets']:
                    try:
                        await ws.send_text(error_message)
                    except Exception:
                        pass
                return

            # JSON serialization can also be slow for large datasets - run in thread
            message = await asyncio.to_thread(
                json.dumps,
                {
                    'type': 'plot_data',
                    'data': plot_data,
                    'coordinates': coordinates,
                    'available_variables': _app_state['available_variables'],
                    'variable_labels': _app_state['variable_labels'],
                    'selected_variable': _app_state['selected_variable'],
                    'spec_filename': _app_state['spec_filename'],
                },
            )

            # Broadcast to all connected clients
            disconnected = set()
            for ws in _app_state['websockets']:
                try:
                    await ws.send_text(message)
                except Exception:
                    disconnected.add(ws)

            _app_state['websockets'] -= disconnected

        except Exception as e:
            import traceback

            traceback.print_exc()

            # Send error to clients
            error_message = json.dumps({
                'type': 'error',
                'message': f"Error plotting '{variable}': {str(e)}",
            })
            for ws in _app_state['websockets']:
                try:
                    await ws.send_text(error_message)
                except Exception:
                    pass


@functools.cache
def _get_field_origins(cls: type) -> Dict[str, type]:
    """Find which class in the MRO first defines each msgspec field.

    Returns a dict mapping field name to the highest-level class that defines it.
    """
    import msgspec

    field_origins: Dict[str, type] = {}

    # Walk MRO from base to derived (reversed)
    mro = [
        c
        for c in cls.__mro__
        if hasattr(c, '__struct_fields__') and isinstance(c.__struct_fields__, tuple)
    ]

    for klass in reversed(mro):
        struct_fields: Tuple[str, ...] = klass.__struct_fields__  # ty: ignore
        for field_name in struct_fields:
            # First class to define it wins (since we're going base->derived)
            if field_name not in field_origins:
                field_origins[field_name] = klass

    return field_origins


def _extract_extra_coords(
    result: 'ss.lib.compute.DelayedDataset',
) -> Dict[str, List[Dict[str, Any]]]:
    """Extract extra_coords (AcquisitionInfo) fields from dataset, grouped by defining class.

    Skips the 'signal_trigger' field if present.

    Args:
        result: DelayedDataset containing extra_coords (AcquisitionInfo)

    Returns:
        Dict mapping class name to list of coordinate dicts. Each coordinate dict
        has 'name', 'value', and 'unit' keys.
    """
    from collections import defaultdict

    extra_coords = result.extra_coords
    if extra_coords is None:
        return {}

    # Get field origins from the extra_coords type
    field_origins = _get_field_origins(type(extra_coords))

    # Group fields by their defining class
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for field_name, defining_class in field_origins.items():
        # Skip signal_trigger
        if field_name == 'signal_trigger':
            continue

        value = getattr(extra_coords, field_name, None)
        if value is None:
            continue

        # Format the value
        formatted_value = _format_coord_value(value)

        # Try to get unit from type hints or attrs (AcquisitionInfo fields typically don't have units)
        unit = ''

        coord_dict = {
            'name': field_name,
            'value': formatted_value,
            'unit': unit,
        }

        qualname = f'{defining_class.__module__}.{defining_class.__name__}'
        grouped[qualname].append(coord_dict)

    return grouped


def _extract_capture_coordinates(
    result: 'ss.lib.compute.DelayedDataset',
) -> CoordinatesData:
    """Extract capture coordinates from dataset, grouped by defining class.

    Only includes coordinates that correspond to fields in the capture spec.

    Args:
        result: DelayedDataset containing capture information and coordinates

    Returns:
        CoordinatesData with 'num_ports' (number of ports), 'groups' (dict mapping
        class name to list of coordinate dicts), and 'extra_coords' (dict
        mapping class name to list of extra coordinate dicts from AcquisitionInfo).
        Each coordinate dict has 'name', 'values' (list), and 'unit' keys.
    """
    from collections import defaultdict
    import striqt.sensor as ss

    capture_coords = ss.lib.compute.datasets.build_capture_coords(
        result.capture, result.extra_coords, result.config.sweep_spec.loops
    )

    if capture_coords is None:
        raise TypeError('did not receive capture information')

    splits = ss.specs.helpers.split_capture_ports(result.capture)
    port_count = len(splits)

    # Get field origins from the capture spec
    field_origins = _get_field_origins(type(result.capture))

    # Group coordinates by their defining class (only include fields from capture spec)
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for field_name, defining_class in field_origins.items():
        # Only include if this field exists as a coordinate in the dataset
        if field_name not in capture_coords:
            continue

        coord = capture_coords[field_name]
        value = coord.values

        # Get unit from attrs
        unit = coord.attrs.get('units', '')

        # Format values as a list (one per port if multi-valued)
        # Apply SI prefix for numeric values with units
        if value.ndim == 0:
            # Scalar value - same for all ports
            raw_values = [value.item()] * port_count
        elif value.size == port_count:
            # One value per port
            raw_values = [v for v in value.flat]
        elif value.size <= 5:
            # Small array but not matching ports - show as single combined value
            combined = ', '.join(_format_coord_value(v) for v in value.flat)
            raw_values = combined  # Signal to use combined string
        else:
            # Large array - show summary
            raw_values = None  # Signal to use summary string

        # Apply SI prefix formatting for numeric values with units
        if raw_values is not None and unit:
            # Check if all values are numeric (float/int)
            numeric_vals = [
                float(v)
                for v in raw_values
                if isinstance(v, (int, float, np.integer, np.floating))
                and np.isfinite(v)
            ]
            if len(numeric_vals) == len(raw_values):
                # All values are numeric - apply SI prefix based on smallest absolute value
                pow10, prefix = _get_si_prefix_for_values(numeric_vals, unit)
                values = [
                    _format_value_with_prefix(float(v), pow10) for v in raw_values
                ]
                # Update unit with prefix
                unit = f'{prefix}{unit}'
            else:
                # Mixed or non-numeric - format without SI prefix
                values = [_format_coord_value(v) for v in raw_values]
        elif raw_values is not None:
            # No unit - format without SI prefix
            values = [_format_coord_value(v) for v in raw_values]
        elif value.size <= 5:
            # Small array combined string
            combined = ', '.join(_format_coord_value(v) for v in value.flat)
            values = [combined] * port_count
        else:
            # Large array summary
            values = [f'[{value.size} values]'] * port_count

        coord_dict = {
            'name': field_name,
            'values': values,  # List of values, one per port
            'unit': unit,
        }

        class_name = f'{defining_class.__module__}.{defining_class.__name__}'
        grouped[class_name].append(coord_dict)

    # Extract extra_coords (AcquisitionInfo fields)
    extra_coords_groups = _extract_extra_coords(result)

    return cast(CoordinatesData, {
        'num_ports': port_count,
        'groups': grouped,
        'extra_coords': extra_coords_groups,
    })


def _get_si_prefix_for_values(values: List[float], unit: str) -> tuple[int, str]:
    """Determine the SI prefix based on the smallest absolute non-zero value.

    Returns (power_of_10, prefix_string).
    Uses _ENG_PREFIXES from striqt.analysis.lib.dataarrays.
    """
    from striqt.analysis.lib.dataarrays import _ENG_PREFIXES
    import math

    # dB units don't get SI prefixes
    if unit.lower().startswith('db'):
        return 0, ''

    # Filter to finite non-zero values
    finite_vals = [abs(v) for v in values if np.isfinite(v) and v != 0]
    if not finite_vals:
        return 0, ''

    # Use smallest absolute value to determine prefix
    min_val = min(finite_vals)

    # Calculate power of 1000 (engineering notation)
    pow10 = int(math.floor(math.log10(min_val) / 3) * 3)

    # Clamp to available prefixes
    pow10 = int(np.clip(pow10, min(_ENG_PREFIXES), max(_ENG_PREFIXES)))

    return pow10, _ENG_PREFIXES.get(pow10, '')


def _format_value_with_prefix(value: float, pow10: int) -> str:
    """Format a value using the given power of 10."""
    from striqt.analysis.lib.dataarrays import _ENG_PREFIXES

    if not np.isfinite(value):
        return str(value)

    scaled = value / (10.0**pow10)

    # Handle rounding to 1000 case
    if abs(scaled) >= 1000 and pow10 < max(_ENG_PREFIXES.keys()):
        scaled /= 1000
        # Note: we don't adjust pow10 here since we want consistent prefix

    return f'{scaled:.4g}'


def _format_coord_value(value: Any) -> str:
    """Format a single coordinate value for display (without SI prefix)."""
    if isinstance(value, (np.floating, float)):
        # Format floats nicely
        if abs(value) >= 1e6 or (abs(value) < 1e-3 and value != 0):
            return f'{value:.3e}'
        return f'{value:.4g}'
    if isinstance(value, np.datetime64):
        # Format datetime
        return str(value)[:19]  # Trim to seconds
    if isinstance(value, np.timedelta64):
        # Convert to seconds
        seconds = value / np.timedelta64(1, 's')
        return f'{seconds:.3g} s'
    if isinstance(value, bytes):
        return value.decode('utf-8', errors='replace')
    return str(value)
import re


def _build_variable_labels_from_registry(
    analysis_spec: 'sa.specs.AnalysisGroup',
    registry: 'sa.lib.register.AnalysisRegistry',
) -> Dict[str, str]:
    """Build a mapping from variable names to human-readable labels.

    This is called once at startup to avoid repeated introspection on every capture.
    It iterates over the analysis spec fields and looks up the corresponding
    spec class in the registry to derive labels.

    Args:
        analysis_spec: The analysis specification from the sweep spec (spec.analysis)
        registry: The striqt.analysis.registry containing AnalysisInfo entries

    Returns:
        Dict mapping variable names (e.g., 'spectrogram_histogram') to labels
        (e.g., 'Spectrogram Histogram')
    """
    labels: Dict[str, str] = {}

    # Build a reverse lookup: variable name -> spec class name
    name_to_class: Dict[str, str] = {}
    for spec_type, info in registry.items():
        name_to_class[info.name] = spec_type.__name__

    # Iterate over the analysis spec fields
    for field_name in analysis_spec.__struct_fields__:
        value = getattr(analysis_spec, field_name)
        if value is not None:
            # This field is enabled in the spec
            if field_name in name_to_class:
                labels[field_name] = _spec_class_to_label(name_to_class[field_name])
            else:
                # Fallback: convert snake_case to Title Case
                labels[field_name] = field_name.replace('_', ' ').title()

    return labels


def _spec_class_to_label(class_name: str) -> str:
    """Convert a spec class name to a human-readable label.

    Examples:
        SpectrogramHistogram -> "Spectrogram Histogram"
        Cellular5GNRSSBSpectrogram -> "Cellular 5G-NR SSB Spectrogram"
        ChannelPowerTimeSeries -> "Channel Power Time Series"
        CellularCyclicAutocorrelator -> "Cellular Cyclic Autocorrelator"
    """
    # Strip leading underscores (private class names)
    result = class_name.lstrip('_')

    # Replace known abbreviations with spaced versions (order matters - longer first)
    abbreviations = [
        ('5GNR', ' 5G-NR '),
        ('SSB', ' SSB '),
        ('PSS', ' PSS '),
        ('SSS', ' SSS '),
        ('PSD', ' PSD '),
        ('IQ', ' IQ '),
    ]

    for abbrev, replacement in abbreviations:
        result = result.replace(abbrev, replacement)

    # Insert spaces before uppercase letters (handles CamelCase)
    # But preserve consecutive uppercase (acronyms)
    result = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', result)
    result = re.sub(r'(?<=[A-Z])(?=[A-Z][a-z])', ' ', result)

    # Clean up multiple spaces
    result = re.sub(r'\s+', ' ', result).strip()

    return result


def update_dataset(result: 'ss.lib.compute.DelayedDataset'):
    """Update the dataset and trigger broadcast to clients.

    Note: Variable labels are built once at startup in run_server() using
    _build_variable_labels_from_registry(), not on every capture.
    """
    _app_state['delayed_dataset'] = result

    # Update available variables based on what's in the dataset
    data_plots = get_data_plots_registry()
    available = [n for n in result.delayed.keys() if n in data_plots]

    _app_state['available_variables'] = available

    # If current selection is not available, switch to first available
    if _app_state['selected_variable'] not in available and available:
        _app_state['selected_variable'] = available[0]


# ---------------------------------------------------------------------------
# Main runner (mirrors live-plotter structure)
# ---------------------------------------------------------------------------


def run_server(
    path: str,
    host: str = '0.0.0.0',
    port: int = 8000,
):
    """Run the web server with data acquisition loop."""
    import striqt.analysis as sa
    import striqt.sensor as ss
    import striqt.figures as sf
    import uvicorn
    import threading

    # Store the spec filename for display in the web UI header
    _app_state['spec_filename'] = Path(path).name

    if path.endswith('.yaml') or path.endswith('.yml'):
        spec = ss.read_yaml_spec(path)
    elif path.endswith('.json'):
        spec = ss.read_json_spec(path)
    else:
        raise click.ClickException('expected file to have .json or .yaml suffix')

    # Disable sink to handle data ourselves
    spec = spec.replace(
        extensions=spec.extensions.replace(sink='striqt.sensor.sinks.NoSink')
    )

    if spec.plot_hint is None:
        plot_opts = sf.specs.PlotOptions(
            data=sf.specs.DataOptions(
                sweep_index=-1,
                select= {
                    # 'channel_power_bin': slice(-100, -15),
                    # 'spectrogram_power_bin': slice(-130, -50),
                    'spectrogram_time': slice(0, 5e-3),
                }
            ),
            plotter=sf.specs.SharedPlotOptions(
                col='port',
                col_label_format='Port {port}',
                style=None,
                filename_fmt='',
                suptitle_fmt='{name}',
            ),
        )
    else:
        plot_opts = sf.specs.PlotOptions.from_spec(spec.plot_hint)

    _app_state['plot_opts'] = plot_opts
    _app_state['plotter'] = WebPlotBackend(plot_opts.plotter)
    # Store data selection dict for xarray.sel() - will be filtered to valid indexes per variable
    _app_state['data_select'] = dict(plot_opts.data.select)

    # Build variable labels once at startup from the analysis spec and registry
    _app_state['variable_labels'] = _build_variable_labels_from_registry(
        spec.analysis, sa.registry
    )

    # Event loop for async operations
    loop = asyncio.new_event_loop()

    def acquisition_loop():
        """Run data acquisition in a separate thread."""

        def on_data(result: 'ss.lib.compute.DelayedDataset'):
            """Callback when new data arrives."""
            # Store capture spec for field origin introspection
            update_dataset(result)

            # Schedule broadcast on the event loop
            asyncio.run_coroutine_threadsafe(broadcast_plot_update(), loop)

        ctx = ss.open_resources(spec, path)

        with ctx as resources:
            resources['sink'].append = on_data
            while True:
                sweep = ss.iterate_sweep(
                    resources, yield_values=True, always_yield=True
                )
                for _ in sweep:
                    pass

    # Start acquisition in background thread
    acq_thread = threading.Thread(target=acquisition_loop, daemon=True)
    acq_thread.start()

    # Run uvicorn with the event loop
    config = uvicorn.Config(app, host=host, port=port, loop='asyncio')
    server = uvicorn.Server(config)

    # Run the server
    loop.run_until_complete(server.serve())


@click.command('Web-based real-time radio diagnostic display')
@click.argument('path', type=click.Path(exists=True, dir_okay=False), required=True)
@click.option(
    '--host',
    '-h',
    default='0.0.0.0',
    show_default=True,
    help='Host to bind the server to',
)
@click.option(
    '--port',
    '-p',
    default=8000,
    show_default=True,
    help='Port to bind the server to',
)
def cli(path: str, host: str, port: int):
    """
    Run a web-based real-time display for radio diagnostics.

    PATH is the path to a JSON or YAML acquisition specification file.
    """
    run_server(path, host=host, port=port)


if __name__ == '__main__':
    cli()

