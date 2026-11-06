#!/usr/bin/env python3
"""serve a live browser display of sweep results.

FastAPI and uvicorn serve the page and stream updates over a websocket. Plotly.js
renders the updates in the browser through WebGL. The page is at
http://localhost:8000 for ``web-plotter path/to/spec.yaml``.

The sweep runs with a null sink. Each acquisition evaluates only the measurement on
display, through the `select_analysis` hook, and the acquisition sequence follows the
capture and loop values selected in the page, through the `captures` keyword of
`iterate_sweep`.
"""

from __future__ import annotations

import asyncio
import dataclasses
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, cast

import click
import msgspec
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Iterator

    import plotly.graph_objects as go

    import striqt.figures as sf
    import striqt.sensor as ss
    from striqt.figures.backend import PlotlyBackend
    from striqt.sensor.lib.resources import Resources
    from striqt.sensor.specs.sequencing import PlanChoices


_json_encoder = msgspec.json.Encoder()
_json_decoder = msgspec.json.Decoder()

_HTML_TEMPLATE_PATH = Path(__file__).parent / 'web-plotter.html'


@dataclasses.dataclass
class PreviewState:
    """the state shared by the acquisition thread and the websocket server.

    The acquisition thread sets `sweep`, `source_id`, `choices`, `plotter`, and
    `dataset`. The websocket handlers set the selections. A selection is replaced
    as a whole, never mutated in place.
    """

    spec_filename: str = ''
    """the file name of the spec, shown in the page header"""
    sweep: Optional[ss.specs.Sweep] = None
    """the sweep that the open resources run"""
    source_id: Optional[str] = None
    """the id of the open source"""
    plot_opts: Optional[sf.specs.PlotOptions] = None
    """the plot options from ``sweep.plot_hint`` or the preview default"""
    plotter: Optional[PlotlyBackend] = None
    """the backend that draws the selected variable"""
    data_select: dict[str, Any] = dataclasses.field(default_factory=dict)
    """the ``plot_opts.data.select`` indexers, filtered per variable at draw time"""
    available_variables: list[str] = dataclasses.field(default_factory=list)
    """the enabled measurements of the sweep that have a registered plot"""
    variable_labels: dict[str, str] = dataclasses.field(default_factory=dict)
    """the display label of each available variable"""
    selected_variable: str = 'spectrogram'
    """the variable on display; the only measurement evaluated per acquisition"""
    choices: Optional[PlanChoices] = None
    """the capture entries and loop values the page offers"""
    selected_capture: Optional[int] = None
    """the index into ``sweep.captures`` to acquire, or None for every entry"""
    selected_loops: dict[tuple[str, str], Any] = dataclasses.field(default_factory=dict)
    """the loop values to acquire, keyed by ``(isin, field)``; a missing loop
    acquires every value"""
    dataset: Optional[ss.lib.compute.DelayedDataset] = None
    """the latest analyzed acquisition"""
    websockets: set[WebSocket] = dataclasses.field(default_factory=set)
    """the connected clients"""
    broadcast_pending: bool = False
    """True while a broadcast is in progress"""
    data_updated: bool = False
    """True when a dataset arrived during a broadcast"""
    stop: threading.Event = dataclasses.field(default_factory=threading.Event)
    """set to end the acquisition thread"""
    thread: Optional[threading.Thread] = None
    """the acquisition thread, once started"""


STATE = PreviewState()


def _logger():
    import striqt.analysis as sa

    return sa.util.get_logger('sweep')


# %% analysis and capture selection


def select_displayed(
    capture: ss.specs.SensorCapture, analysis: ss.specs.AnalysisGroup
) -> ss.specs.AnalysisGroup:
    """return `analysis` narrowed to the variable on display.

    The whole group is returned when the selected variable is not an enabled field
    of `analysis`.
    """
    import striqt.analysis as sa

    name = STATE.selected_variable
    if name not in analysis.to_dict():
        return analysis
    return sa.specs.helpers.narrow_analysis_group(analysis, [name])


def current_plan(state: PreviewState) -> tuple[ss.specs.SensorCapture, ...]:
    """return the captures that match the selections of `state`, in sweep order."""
    from striqt.sensor.specs import sequencing

    assert state.sweep is not None
    return sequencing.select_plan(
        state.sweep,
        state.source_id,
        capture=state.selected_capture,
        loops=state.selected_loops,
    )


def selected_captures(state: PreviewState) -> Iterator[ss.specs.SensorCapture]:
    """yield the captures of `current_plan`, wrapping, until ``state.stop`` is set.

    The plan is re-read before each capture. A selection change takes effect at the
    next acquisition. An empty plan waits, polling ``state.stop``.
    """
    i = 0
    while not state.stop.is_set():
        plan = current_plan(state)
        if len(plan) == 0:
            state.stop.wait(0.1)
            continue
        yield plan[i % len(plan)]
        i += 1


# %% messages to the page


def capture_options(choices: PlanChoices | None) -> dict[str, Any]:
    """return the capture and loop choices as the ``capture_options`` message field."""
    if choices is None:
        return {'captures': [], 'loops': []}
    return {
        'captures': list(choices.captures),
        'loops': [
            {'field': loop.field, 'isin': loop.isin, 'labels': list(loop.labels)}
            for loop in choices.loops
        ],
    }


def capture_selection(state: PreviewState) -> dict[str, Any]:
    """return the selections of `state` as indexes into `capture_options`."""
    loops: list[int | None] = []
    for loop in () if state.choices is None else state.choices.loops:
        key = (loop.isin, loop.field)
        if key in state.selected_loops:
            loops.append(loop.values.index(state.selected_loops[key]))
        else:
            loops.append(None)
    return {'capture': state.selected_capture, 'loops': loops}


def apply_capture_selection(state: PreviewState, msg: dict[str, Any]) -> None:
    """set the selections of `state` from a ``select_captures`` message.

    Args:
        state: the state to update
        msg: ``{'capture': int | None, 'loops': [int | None, ...]}`` with indexes
            into the `capture_options` lists, in the order of ``choices.loops``
    """
    if state.choices is None:
        return
    capture = msg.get('capture')
    state.selected_capture = None if capture is None else int(capture)
    selected: dict[tuple[str, str], Any] = {}
    indexes = msg.get('loops') or []
    for loop, index in zip(state.choices.loops, indexes):
        if index is not None:
            selected[loop.isin, loop.field] = loop.values[int(index)]
    state.selected_loops = selected


def coordinates_message(dd: ss.lib.compute.DelayedDataset) -> dict[str, Any]:
    """return the capture and acquisition fields of `dd` as the ``coordinates`` field.

    Returns:
        ``num_ports``, ``groups`` (capture fields per defining class, one value per
        port), and ``extra_coords`` (acquisition info per defining class)
    """
    import striqt.sensor as ss
    from striqt.sensor.lib.compute import summary

    groups = {
        group.owner: [
            {'name': f.name, 'values': list(f.values), 'unit': f.units}
            for f in group.fields
        ]
        for group in summary.capture_fields(dd)
    }
    extra = {
        group.owner: [
            {'name': f.name, 'value': f.values[0], 'unit': f.units}
            for f in group.fields
        ]
        for group in summary.acquisition_fields(dd)
    }
    return {
        'num_ports': len(ss.specs.captures.split_capture_ports(dd.capture)),
        'groups': groups,
        'extra_coords': extra,
    }


def state_message(state: PreviewState) -> dict[str, Any]:
    """return the ``state`` message: the variables, selections, and coordinates."""
    coordinates: dict[str, Any] = {'num_ports': 1, 'groups': {}, 'extra_coords': {}}
    if state.dataset is not None:
        coordinates = coordinates_message(state.dataset)
    return {
        'type': 'state',
        'available_variables': state.available_variables,
        'variable_labels': state.variable_labels,
        'selected_variable': state.selected_variable,
        'coordinates': coordinates,
        'spec_filename': state.spec_filename,
        'capture_options': capture_options(state.choices),
        'capture_selection': capture_selection(state),
    }


def _encode(message: dict[str, Any]) -> str:
    # msgspec encodes to bytes; a text frame keeps the browser's JSON.parse path
    return _json_encoder.encode(message).decode()


# %% FastAPI application


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """end the acquisition thread when the server shuts down."""
    yield  # noqa: RUF075
    STATE.stop.set()
    STATE.websockets.clear()
    if STATE.thread is not None:
        STATE.thread.join()


app = FastAPI(
    title='Striqt WebView',
    description='Real-time radio diagnostic visualization',
    lifespan=lifespan,
)


@app.get('/', response_class=HTMLResponse)
async def get_index() -> str:
    return _HTML_TEMPLATE_PATH.read_text()


async def _send_all(message: str) -> None:
    """send `message` to every client, dropping the clients that fail."""
    disconnected = set()
    for ws in STATE.websockets:
        try:
            await ws.send_text(message)
        except Exception:
            disconnected.add(ws)
    STATE.websockets -= disconnected


async def broadcast_state() -> None:
    await _send_all(_encode(state_message(STATE)))


@app.websocket('/ws')
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    STATE.websockets.add(websocket)

    try:
        while True:
            msg = _json_decoder.decode(await websocket.receive_text())

            if msg['type'] == 'get_state':
                await websocket.send_text(_encode(state_message(STATE)))
                if STATE.dataset is not None:
                    await broadcast_plot_update()
            elif msg['type'] == 'select_variable':
                STATE.selected_variable = msg['variable']
                if STATE.plotter is not None:
                    STATE.plotter.reset(limits=True)
                await broadcast_plot_update()
            elif msg['type'] == 'select_captures':
                apply_capture_selection(STATE, msg)
                await broadcast_state()

    except WebSocketDisconnect:
        STATE.websockets.discard(websocket)
    except Exception:
        _logger().exception('websocket handler failed')
        STATE.websockets.discard(websocket)


async def broadcast_plot_update() -> None:
    """send the current plot to every connected websocket.

    A call during a broadcast in progress sets `data_updated` instead of queuing.
    One more broadcast follows the current one.
    """
    if STATE.broadcast_pending:
        STATE.data_updated = True
        return

    STATE.broadcast_pending = True

    try:
        await _do_broadcast()
    finally:
        STATE.broadcast_pending = False

        if STATE.data_updated:
            STATE.data_updated = False
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                # the server stopped during the broadcast
                return
            # a task rather than recursion, which would grow with the backlog
            loop.create_task(_do_broadcast_if_needed())


async def _do_broadcast_if_needed() -> None:
    if STATE.dataset is not None and not STATE.broadcast_pending:
        await broadcast_plot_update()


def _select_delayed_variable(
    result: ss.lib.compute.DelayedDataset, name: str
) -> ss.lib.compute.DelayedDataset:
    return dataclasses.replace(result, delayed={name: result.delayed[name]})


def _prepare_plot_data_sync(
    result: ss.lib.compute.DelayedDataset,
    plotter: PlotlyBackend,
    variable: str,
    data_select: dict[str, Any],
) -> list[go.Figure]:
    """return the figures of `variable` in `result`.

    The call blocks on the materialization of the delayed dataset (a GPU transfer
    on the TX2i).

    Args:
        result: the delayed dataset of the analyzed capture
        plotter: the backend that produces the figures; `reset` is called first
        variable: the name of the data variable to plot
        data_select: the `PlotOptions.data.select` selection; keys that are not
            indexes of the dataset are dropped

    Returns:
        the figures from `plotter` in drawing order
    """
    import striqt.figures as sf
    from striqt.sensor.lib import compute

    plotter.reset()

    plot_func = sf.data_var_plotters()[variable]

    delayed = _select_delayed_variable(result, variable)
    ds = compute.from_delayed(delayed).set_xindex('port')

    # `select` may name indexes of other variables
    if data_select:
        valid_indexes = set(ds.indexes.keys())
        filtered_select = {k: v for k, v in data_select.items() if k in valid_indexes}
        if filtered_select:
            ds = ds.sel(filtered_select)

    plot_func(ds, plotter)

    return plotter.get_plot_data()


def _serialize_plot_message(
    plot_data: list[go.Figure],
    coordinates: dict[str, Any],
    state: PreviewState,
) -> str:
    """return the ``plot_data`` message as JSON text.

    Each figure is encoded by `Figure.to_json`, which writes numpy arrays as base64
    typed arrays.
    """
    serialized_plots = [
        _json_decoder.decode(cast(str, fig.to_json())) for fig in plot_data
    ]

    return _encode({
        'type': 'plot_data',
        'data': serialized_plots,
        'coordinates': coordinates,
        'available_variables': state.available_variables,
        'variable_labels': state.variable_labels,
        'selected_variable': state.selected_variable,
        'spec_filename': state.spec_filename,
    })


async def _do_broadcast() -> None:
    result = STATE.dataset
    plotter = STATE.plotter
    if result is None or plotter is None or not STATE.websockets:
        return

    variable = STATE.selected_variable
    if variable not in result.delayed:
        # the acquisition in flight was analyzed under the previous selection
        return

    try:
        plot_data = await asyncio.to_thread(
            _prepare_plot_data_sync, result, plotter, variable, STATE.data_select
        )

        if not plot_data:
            await _send_all(
                _encode({
                    'type': 'error',
                    'message': f"'{variable}' produced no plot data for the web view",
                })
            )
            return

        coordinates = await asyncio.to_thread(coordinates_message, result)
        message = await asyncio.to_thread(
            _serialize_plot_message, plot_data, coordinates, STATE
        )
        await _send_all(message)

    except Exception as e:
        _logger().exception(f"plotting '{variable}' failed")
        await _send_all(
            _encode({'type': 'error', 'message': f"Error plotting '{variable}': {e!s}"})
        )


# %% the acquisition thread


def _default_plot_options() -> sf.specs.PlotOptions:
    import striqt.figures as sf

    return sf.specs.PlotOptions(
        data=sf.specs.DataOptions(
            sweep_index=-1,
            select={'spectrogram_time': slice(0, 5e-3)},
        ),
        plotter=sf.specs.SharedPlotOptions(
            col='port',
            col_label_format='Port {port}',
            style=None,
            filename_fmt='',
            suptitle_fmt='{name}',
        ),
    )


def configure(state: PreviewState, resources: Resources) -> None:
    """set the sweep-derived fields of `state` from the open `resources`.

    The variables offered are the enabled measurements of the sweep with a plot in
    `striqt.figures.data_var_plotters`. The selected variable falls back to the
    first offered one when the current selection is not offered.
    """
    import striqt.figures as sf
    from striqt.figures.backend import PlotlyBackend
    from striqt.sensor.lib.compute import summary
    from striqt.sensor.specs import sequencing

    sweep = resources['sweep_spec']
    state.sweep = sweep
    state.source_id = resources['source'].source_id

    if sweep.plot_hint is None:
        plot_opts = _default_plot_options()
    else:
        plot_opts = sf.specs.PlotOptions.from_spec(sweep.plot_hint)
    state.plot_opts = plot_opts
    state.plotter = PlotlyBackend(plot_opts.plotter)
    state.data_select = dict(plot_opts.data.select)

    plotters = sf.data_var_plotters()
    labels = summary.analysis_labels(sweep.analysis)
    state.available_variables = [name for name in labels if name in plotters]
    state.variable_labels = {name: labels[name] for name in state.available_variables}
    if (
        state.selected_variable not in state.available_variables
        and state.available_variables
    ):
        state.selected_variable = state.available_variables[0]

    state.choices = sequencing.plan_choices(sweep, state.source_id)


def run_server(path: str, host: str = '0.0.0.0', port: int = 8000) -> None:
    """run the web server and the acquisition thread for the spec at `path`.

    The sink of the spec is replaced by a null sink, so nothing is written. The
    server exits when the acquisition thread ends, and the acquisition thread ends
    after the capture in progress when the server exits.

    Raises:
        click.ClickException: if `path` has a suffix other than ``.yaml``, ``.yml``,
            or ``.json``
    """
    import uvicorn

    import striqt.sensor as ss
    from striqt.cli.sensor_sweep import log_overloads

    STATE.spec_filename = Path(path).name

    if path.endswith(('.yaml', '.yml')):
        spec = ss.read_yaml_spec(path)
    elif path.endswith('.json'):
        spec = ss.read_json_spec(path)
    else:
        raise click.ClickException('expected file to have .json or .yaml suffix')

    # a null sink path writes nothing, and on_analyzed receives the results instead
    spec = spec.replace(sink=spec.sink.replace(path=None))

    loop = asyncio.new_event_loop()
    config = uvicorn.Config(app, host=host, port=port, loop='asyncio')
    server = uvicorn.Server(config)

    def on_data(result: ss.lib.compute.DelayedDataset) -> None:
        log_overloads(result)
        STATE.dataset = result
        if loop.is_running():
            asyncio.run_coroutine_threadsafe(broadcast_plot_update(), loop)

    def acquisition_loop() -> None:
        try:
            ctx = ss.open_resources(
                spec, path, on_analyzed=on_data, select_analysis=select_displayed
            )
            with ctx as resources:
                configure(STATE, resources)
                asyncio.run_coroutine_threadsafe(broadcast_state(), loop)
                sweep = ss.iterate_sweep(
                    resources,
                    yield_values=False,
                    always_yield=True,
                    captures=selected_captures(STATE),
                )
                try:
                    for _ in sweep:
                        if STATE.stop.is_set():
                            break
                finally:
                    sweep.close()
        except Exception:
            _logger().exception('the acquisition thread failed')
        finally:
            server.should_exit = True

    STATE.thread = threading.Thread(
        target=acquisition_loop, name='striqt-preview', daemon=True
    )
    STATE.thread.start()

    try:
        loop.run_until_complete(server.serve())
    finally:
        STATE.stop.set()
        STATE.thread.join()


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
def cli(path: str, host: str, port: int) -> None:
    """
    Run a web-based real-time display for radio diagnostics.

    PATH is the path to a JSON or YAML acquisition specification file.
    """
    run_server(path, host=host, port=port)


if __name__ == '__main__':
    cli()
