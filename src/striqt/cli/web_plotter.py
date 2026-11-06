#!/usr/bin/env python3
"""serve a live browser display of sweep results.

FastAPI and uvicorn serve the page and stream updates over a websocket. Plotly.js
renders the updates in the browser through WebGL. The page is at
http://localhost:8000 for ``web-plotter path/to/spec.yaml``.
"""

from __future__ import annotations

import asyncio
import functools
from contextlib import asynccontextmanager
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
    TypedDict,
    cast,
)

import click
import msgspec
import numpy as np
import plotly.graph_objects as go
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

if TYPE_CHECKING:
    import striqt.analysis as sa
    import striqt.figures as sf
    import striqt.sensor as ss
    from striqt.figures.backend import PlotlyBackend


_json_encoder = msgspec.json.Encoder()
_json_decoder = msgspec.json.Decoder()


# ---------------------------------------------------------------------------
# Type definitions for app state
# ---------------------------------------------------------------------------


class CoordinateDict(TypedDict):
    """A single coordinate entry with name, values, and unit."""

    name: str
    values: list[str]
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
    groups: dict[str, list[CoordinateDict]]
    extra_coords: dict[str, list[ExtraCoordDict]]


class AppState(TypedDict):
    """Type definition for the global application state."""

    delayed_dataset: Optional[ss.lib.compute.DelayedDataset]
    plotter: Optional[PlotlyBackend]
    selected_variable: str
    available_variables: list[str]
    variable_labels: dict[str, str]
    websockets: set[WebSocket]
    broadcast_pending: bool
    data_updated: bool
    plot_opts: Optional[sf.specs.PlotOptions]
    data_select: dict[str, Any]
    spec_filename: str


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
    yield  # noqa: RUF075
    # Cleanup on shutdown
    _app_state['websockets'].clear()


app = FastAPI(
    title='Striqt WebView',
    description='Real-time radio diagnostic visualization',
    lifespan=lifespan,
)


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
            msg = _json_decoder.decode(data)

            if msg['type'] == 'get_state':
                # Extract coordinates on demand (deferred from update_dataset)
                coordinates: CoordinatesData = {
                    'num_ports': 1,
                    'groups': {},
                    'extra_coords': {},
                }
                if _app_state['delayed_dataset'] is not None:
                    coordinates = _extract_capture_coordinates(
                        _app_state['delayed_dataset']
                    )
                # msgspec encodes to bytes; decode to str so the frame is sent as
                # text and the browser's JSON.parse(event.data) works unchanged.
                await websocket.send_text(
                    _json_encoder.encode({
                        'type': 'state',
                        'available_variables': _app_state['available_variables'],
                        'variable_labels': _app_state['variable_labels'],
                        'selected_variable': _app_state['selected_variable'],
                        'coordinates': coordinates,
                        'spec_filename': _app_state['spec_filename'],
                    }).decode()
                )
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
    """send the current plot to every connected websocket.

    A call during a broadcast in progress sets `data_updated` instead of queuing.
    One more broadcast follows the current one.
    """
    if _app_state['broadcast_pending']:
        _app_state['data_updated'] = True
        return

    _app_state['broadcast_pending'] = True

    try:
        await _do_broadcast()
    finally:
        _app_state['broadcast_pending'] = False

        if _app_state['data_updated']:
            _app_state['data_updated'] = False
            # Use create_task to avoid recursion depth issues
            asyncio.create_task(_do_broadcast_if_needed())


async def _do_broadcast_if_needed():
    """broadcast again when data arrived during the previous broadcast"""
    if (
        _app_state['delayed_dataset'] is not None
        and not _app_state['broadcast_pending']
    ):
        await broadcast_plot_update()


def _select_delayed_variable(result: ss.lib.compute.DelayedDataset, name: str):
    import dataclasses

    delayed_var = result.delayed[name]
    return dataclasses.replace(result, delayed={name: delayed_var})


def _prepare_plot_data_sync(
    result: ss.lib.compute.DelayedDataset,
    plotter: PlotlyBackend,
    variable: str,
    data_select: dict[str, Any],
) -> tuple[list[dict[str, Any]], CoordinatesData]:
    """return the plot results and capture coordinates of `variable` in `result`.

    The call blocks on the materialization of the delayed dataset (a GPU transfer
    on the TX2i).

    Args:
        result: the delayed dataset of the analyzed capture
        plotter: the backend that produces the results; `reset` is called first
        variable: the name of the data variable to plot
        data_select: the `PlotOptions.data.select` selection; keys that are not
            indexes of the dataset are dropped

    Returns:
        the list of result dicts from `plotter`, and the `CoordinatesData`
    """
    from striqt.sensor.lib import compute

    plotter.reset()

    # Extract capture coordinates (deferred from update_dataset to reduce acquisition thread work)
    coordinates = _extract_capture_coordinates(result)

    data_plots = get_data_plots_registry()
    plot_func = data_plots[variable]

    delayed = _select_delayed_variable(result, variable)
    ds = compute.from_delayed(delayed).set_xindex('port')

    # `select` may name indexes of other variables
    if data_select:
        valid_indexes = set(ds.indexes.keys())
        filtered_select = {k: v for k, v in data_select.items() if k in valid_indexes}
        if filtered_select:
            ds = ds.sel(filtered_select)

    plot_func(ds, plotter)

    plot_data = plotter.get_plot_data()

    return plot_data, coordinates


def _serialize_plot_message(
    plot_data: list[dict[str, Any]],
    coordinates: CoordinatesData,
    available_variables: list[str],
    variable_labels: dict[str, str],
    selected_variable: str,
    spec_filename: str,
) -> str:
    """return the ``plot_data`` message as JSON text.

    Numpy arrays in the traces are encoded as base64 typed arrays, as by
    `Figure.to_json`.
    """
    serialized_plots = []
    for plot in plot_data:
        traces = plot.get('traces', [])
        layout = plot.get('layout', {})

        go_traces: list[Any] = []
        for trace in traces:
            trace_type = trace.get('type', 'scatter')
            if trace_type == 'heatmap':
                go_traces.append(go.Heatmap(trace))
            elif trace_type == 'scattergl':
                go_traces.append(go.Scattergl(trace))
            elif trace_type == 'scatter':
                go_traces.append(go.Scatter(trace))
            else:
                go_traces.append(trace)

        fig = go.Figure(data=go_traces, layout=layout)
        serialized_plots.append(_json_decoder.decode(cast(str, fig.to_json())))

    # msgspec encodes to bytes; decode to str so callers can send it as a text
    # WebSocket frame (keeps the browser's JSON.parse(event.data) path unchanged).
    return _json_encoder.encode({
        'type': 'plot_data',
        'data': serialized_plots,
        'coordinates': coordinates,
        'available_variables': available_variables,
        'variable_labels': variable_labels,
        'selected_variable': selected_variable,
        'spec_filename': spec_filename,
    }).decode()


async def _do_broadcast():
    """Internal broadcast implementation."""
    result = _app_state['delayed_dataset']
    if result is None:
        return

    plotter = _app_state['plotter']
    if plotter is None:
        return

    variable = _app_state['selected_variable']
    data_plots = get_data_plots_registry()

    if variable in data_plots:
        try:
            data_select = _app_state['data_select']
            plot_data, coordinates = await asyncio.to_thread(
                _prepare_plot_data_sync, result, plotter, variable, data_select
            )

            if not plot_data:
                # the plot function of `variable` returned without a drawing call
                error_message = _json_encoder.encode({
                    'type': 'error',
                    'message': f"'{variable}' produced no plot data for the web view",
                }).decode()
                for ws in _app_state['websockets']:
                    try:
                        await ws.send_text(error_message)
                    except Exception:  # noqa: S110
                        pass
                return

            message = await asyncio.to_thread(
                _serialize_plot_message,
                plot_data,
                coordinates,
                _app_state['available_variables'],
                _app_state['variable_labels'],
                _app_state['selected_variable'],
                _app_state['spec_filename'],
            )

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

            error_message = _json_encoder.encode({
                'type': 'error',
                'message': f"Error plotting '{variable}': {e!s}",
            }).decode()
            for ws in _app_state['websockets']:
                try:
                    await ws.send_text(error_message)
                except Exception:  # noqa: S110
                    pass


@functools.cache
def _get_field_origins(cls: type) -> dict[str, type]:
    """return the class in the MRO of `cls` that first defines each msgspec field.

    Returns:
        a dict from field name to the base-most class with the field in
        ``__struct_fields__``
    """
    field_origins: dict[str, type] = {}

    # Walk MRO from base to derived (reversed)
    mro = [
        c
        for c in cls.__mro__
        if hasattr(c, '__struct_fields__') and isinstance(c.__struct_fields__, tuple)
    ]

    for klass in reversed(mro):
        struct_fields: tuple[str, ...] = klass.__struct_fields__  # ty: ignore
        for field_name in struct_fields:
            if field_name not in field_origins:
                field_origins[field_name] = klass

    return field_origins


def _extract_extra_coords(
    result: ss.lib.compute.DelayedDataset,
) -> dict[str, list[dict[str, Any]]]:
    """Extract extra_coords (AcquisitionInfo) fields from dataset, grouped by defining class.

    Skips the 'signal_trigger' field if present.

    Args:
        result: DelayedDataset containing extra_coords (AcquisitionInfo)

    Returns:
        Dict mapping class name to list of coordinate dicts. Each coordinate dict
        has 'name', 'value', and 'unit' keys.
    """
    from collections import defaultdict

    import striqt.analysis as sa

    extra_coords = result.extra_coords
    if extra_coords is None:
        return {}

    field_origins = _get_field_origins(type(extra_coords))
    field_attrs = sa.specs.helpers.get_capture_type_attrs(type(extra_coords))

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for field_name, defining_class in field_origins.items():
        if field_name == 'signal_trigger':
            continue

        value = getattr(extra_coords, field_name, None)
        if value is None:
            continue

        formatted_value = _format_coord_value(value)
        unit = field_attrs.get(field_name, {}).get('units', '')

        coord_dict = {
            'name': field_name,
            'value': formatted_value,
            'unit': unit,
        }

        qualname = f'{defining_class.__module__}.{defining_class.__name__}'
        grouped[qualname].append(coord_dict)

    return grouped


def _extract_capture_coordinates(
    result: ss.lib.compute.DelayedDataset,
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
    import striqt.waveform as sw

    capture_coords = ss.lib.compute.datasets.build_capture_coords(
        result.capture, result.extra_coords, result.config.sweep_spec.loops
    )

    if capture_coords is None:
        raise TypeError('did not receive capture information')

    splits = ss.specs.captures.split_capture_ports(result.capture)
    port_count = len(splits)

    field_origins = _get_field_origins(type(result.capture))

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for field_name, defining_class in field_origins.items():
        if field_name not in capture_coords:
            continue

        coord = capture_coords[field_name]
        value = coord.values

        unit = coord.attrs.get('units', '')

        # Format values as a list (one per port if multi-valued)
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
            numeric_vals = [
                float(v)
                for v in raw_values
                if isinstance(v, (int, float, np.integer, np.floating))
                and np.isfinite(v)
            ]
            if len(numeric_vals) == len(raw_values):
                p = sw.units.si_prefix(numeric_vals, unit)
                values = [
                    sw.units.format_quantity(v, None, prefix=p) for v in numeric_vals
                ]
                unit = f'{p.symbol}{unit}'
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

    extra_coords_groups = _extract_extra_coords(result)

    return cast(
        CoordinatesData,
        {
            'num_ports': port_count,
            'groups': grouped,
            'extra_coords': extra_coords_groups,
        },
    )


def _format_coord_value(value: Any) -> str:
    """Format a single coordinate value for display (without SI prefix)."""
    if isinstance(value, (np.floating, float)):
        if abs(value) >= 1e6 or (abs(value) < 1e-3 and value != 0):
            return f'{value:.3e}'
        return f'{value:.4g}'
    if isinstance(value, np.datetime64):
        return str(value)[:19]  # Trim to seconds
    if isinstance(value, np.timedelta64):
        # Convert to seconds
        seconds = value / np.timedelta64(1, 's')
        return f'{seconds:.3g} s'
    if isinstance(value, bytes):
        return value.decode('utf-8', errors='replace')
    return str(value)


def _build_variable_labels_from_registry(
    analysis_spec: sa.specs.AnalysisGroup,
    registry: sa.lib.register.AnalysisRegistry,
) -> dict[str, str]:
    """return a label for each enabled field of `analysis_spec`, keyed by field name.

    The label of a field with a `registry` entry is the `standard_name` of the
    registration. A field without an entry is labeled by the field name in title
    case with underscores replaced by spaces.
    """
    standard_names = {
        info.name: str(info.attrs['standard_name']) for info in registry.values()
    }
    labels: dict[str, str] = {}
    for field_name in analysis_spec.__struct_fields__:
        if getattr(analysis_spec, field_name) is None:
            continue
        fallback = field_name.replace('_', ' ').title()
        labels[field_name] = standard_names.get(field_name, fallback)
    return labels


def update_dataset(result: ss.lib.compute.DelayedDataset):
    """store the dataset and refresh the list of plottable variables.

    The selected variable falls back to the first plottable variable when the
    current selection is not plottable.
    """
    _app_state['delayed_dataset'] = result

    data_plots = get_data_plots_registry()
    available = [n for n in result.delayed if n in data_plots]

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
    import threading

    import uvicorn

    import striqt.analysis as sa
    import striqt.figures as sf
    import striqt.sensor as ss
    from striqt.figures.backend import PlotlyBackend

    # Store the spec filename for display in the web UI header
    _app_state['spec_filename'] = Path(path).name

    if path.endswith(('.yaml', '.yml')):
        spec = ss.read_yaml_spec(path)
    elif path.endswith('.json'):
        spec = ss.read_json_spec(path)
    else:
        raise click.ClickException('expected file to have .json or .yaml suffix')

    # a null sink path writes nothing, and on_data receives the results instead
    spec = spec.replace(sink=spec.sink.replace(path=None))

    # the plotter shows the sweep the action runs, which may differ from the spec
    sweep = ss.lib.actions.registry.lookup(spec)(spec).sweep

    if sweep.plot_hint is None:
        plot_opts = sf.specs.PlotOptions(
            data=sf.specs.DataOptions(
                sweep_index=-1,
                select={
                    # 'channel_power_bin': slice(-100, -15),
                    # 'spectrogram_power_bin': slice(-130, -50),
                    'spectrogram_time': slice(0, 5e-3),
                },
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
        plot_opts = sf.specs.PlotOptions.from_spec(sweep.plot_hint)

    _app_state['plot_opts'] = plot_opts
    _app_state['plotter'] = PlotlyBackend(plot_opts.plotter)
    # the selection is filtered to valid indexes per variable in _prepare_plot_data_sync
    _app_state['data_select'] = dict(plot_opts.data.select)

    _app_state['variable_labels'] = _build_variable_labels_from_registry(
        sweep.analysis, sa.registry
    )

    loop = asyncio.new_event_loop()

    def acquisition_loop():
        """Run data acquisition in a separate thread."""

        def on_data(result: ss.lib.compute.DelayedDataset) -> None:
            """Callback when new data arrives."""
            update_dataset(result)

            # Schedule broadcast on the event loop
            asyncio.run_coroutine_threadsafe(broadcast_plot_update(), loop)

        ctx = ss.open_resources(spec, path, on_analyzed=on_data)

        with ctx as resources:
            while True:
                sweep = ss.iterate_sweep(
                    resources, yield_values=True, always_yield=True
                )
                for _ in sweep:
                    pass

    acq_thread = threading.Thread(target=acquisition_loop, daemon=True)
    acq_thread.start()

    config = uvicorn.Config(app, host=host, port=port, loop='asyncio')
    server = uvicorn.Server(config)

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
