"""serve the studio page and stream its updates over a websocket.

Every message to the page is built from the `StudioSession` that the worker thread
of `session` updates; the static files of the page are served from the ``static``
directory beside this module, and ``plotly.min.js`` from the `plotly` package.
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import logging
import os
import socket
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, cast

import msgspec
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

import striqt.analysis as sa

from . import logs, session as session_
from .session import StudioSession

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    import plotly.graph_objects as go

    import striqt.sensor as ss
    from striqt.figures.backend import PlotlyBackend


_json_encoder = msgspec.json.Encoder(enc_hook=sa.specs.helpers._enc_hook_no_tuple_keys)
_json_decoder = msgspec.json.Decoder()

_STATIC_DIR = Path(__file__).parent / 'static'

SESSION = StudioSession()
"""the session of the running server"""


@dataclasses.dataclass
class Clients:
    """the connected websockets and the state of the plot broadcast"""

    websockets: set[WebSocket] = dataclasses.field(default_factory=set)
    broadcast_pending: bool = False
    """True while a plot broadcast is in progress"""
    data_updated: bool = False
    """True when a dataset arrived during a broadcast"""
    log_handler: Optional[logs.WebSocketLogHandler] = None
    """the handler forwarding the records of the run in progress, or None"""


CLIENTS = Clients()


# %% messages to the page


def capture_options(choices: ss.specs.sequencing.PlanChoices | None) -> dict[str, Any]:
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


def capture_selection(session: StudioSession) -> dict[str, Any]:
    """return the selections of `session` as indexes into `capture_options`."""
    loops: list[int | None] = []
    for loop in () if session.open is None else session.open.choices.loops:
        key = (loop.isin, loop.field)
        if key in session.selected_loops:
            loops.append(loop.values.index(session.selected_loops[key]))
        else:
            loops.append(None)
    return {'capture': session.selected_capture, 'loops': loops}


def coordinates_message(
    dd: ss.lib.compute.DelayedDataset, varying: frozenset[str] | None = None
) -> dict[str, Any]:
    """return the capture fields of `dd` as the ``coordinates`` field.

    Args:
        dd: the analyzed acquisition
        varying: the field names to keep, or None to keep every field

    Returns:
        a dict with ``num_ports`` and ``fields`` (the kept capture fields in
        declaration order, one value per port)
    """
    import striqt.sensor as ss
    from striqt.sensor.lib.compute import summary

    fields = [
        {'name': f.name, 'values': list(f.values), 'unit': f.units}
        for group in summary.capture_fields(dd)
        for f in group.fields
        if varying is None or f.name in varying
    ]
    return {
        'num_ports': len(ss.specs.captures.split_capture_ports(dd.capture)),
        'fields': fields,
    }


def run_message(session: StudioSession) -> dict[str, Any]:
    """return the ``run`` field of the state message: the output path and progress."""
    run = session.run
    opened = session.open
    path = None if opened is None else opened.sink_path
    adjustments = {} if opened is None else opened.adjustments
    message: dict[str, Any] = {
        'sink_path': path,
        'exists': path is not None and Path(path).exists(),
        'adjustments': {k: list(v) for k, v in adjustments.items()},
    }
    if run is not None:
        message.update({
            'count': run.count,
            'captures_per_sweep': run.per_sweep,
            'acquired': run.acquired,
            'capture_index': run.capture_index,
            'sweep_index': run.sweep_index,
            'started': run.started,
            'finished': run.finished,
            'error': run.error,
        })
    return message


def state_message(session: StudioSession) -> dict[str, Any]:
    """return the ``state`` message: the mode, variables, selections, and progress.

    ``coordinates`` is None until the first acquisition is analyzed.
    """
    opened = session.open
    coordinates: dict[str, Any] | None = None
    if session.dataset is not None and opened is not None:
        coordinates = coordinates_message(session.dataset, opened.varying_fields)
    spec = session_.current_spec(session)
    prompt = session.prompt
    return {
        'type': 'state',
        'mode': session.mode,
        'spec_filename': Path(session.spec_path).name,
        'action_class': None if spec is None else type(spec).__name__,
        'available_variables': []
        if opened is None
        else list(opened.available_variables),
        'variable_labels': {} if opened is None else opened.variable_labels,
        'selected_variable': session.selected_variable,
        'coordinates': coordinates,
        'capture_options': capture_options(None if opened is None else opened.choices),
        'capture_selection': capture_selection(session),
        'running': session.preview_enabled,
        'run': run_message(session),
        'prompt': None if prompt is None else {'id': prompt.id, 'text': prompt.text},
        'apply_error': session.apply_error,
        'validation': {
            'ok': not session.validation_errors,
            'errors': session.validation_errors,
        },
        'editable': session.provenance is not None,
        'changed_sections': session_.changed_sections(session),
        'applied_differs': (
            spec is not None
            and session.applied_spec is not None
            and spec != session.applied_spec
        ),
    }


def spec_message(session: StudioSession) -> dict[str, Any]:
    """return the ``spec`` message: the merged tree and the file of each value.

    Returns:
        a dict with ``tree`` (mapping keys as strings), ``files`` (paths relative to
        the directory of the spec, in first-read order), ``origin`` (JSON pointer to ``[file index, pointer in
        that file]``), and ``regions`` (pointer of each include to its file indexes)
    """
    prov = session.provenance
    if prov is None:
        return {'type': 'spec', 'tree': None, 'files': [], 'origin': {}, 'regions': {}}
    paths = list(prov.files)
    index = {path: i for i, path in enumerate(paths)}
    root = session_.spec_dir(session)

    return {
        'type': 'spec',
        'tree': json_tree(prov.tree),
        'files': [os.path.relpath(p, root) for p in paths],
        'origin': {
            pointer: [index[path], local]
            for pointer, (path, local) in prov.origin.items()
        },
        'regions': {
            pointer: [index[path] for path in files]
            for pointer, files in prov.regions.items()
        },
    }


def json_tree(node: Any) -> Any:
    """return `node` with every mapping key that is not a string replaced by `str(key)`.

    A sequence-valued YAML key (``[0, 1]: 0`` in an `adjust_captures` lookup) is a
    tuple in the merged tree and ``'(0, 1)'`` in the provenance pointers.
    `_enc_hook_no_tuple_keys`, which encodes the spec defaults and the zarr attrs,
    writes the same key as ``'[0, 1]'``.
    """
    if isinstance(node, dict):
        return {
            key if isinstance(key, str) else str(key): json_tree(value)
            for key, value in node.items()
        }
    if isinstance(node, (list, tuple)):
        return [json_tree(item) for item in node]
    return node


def schema_message(session: StudioSession) -> dict[str, Any]:
    """return the ``schema`` message: the editor schema of the spec class."""
    return {'type': 'schema', 'schema': session_.spec_schema_json(session)}


def _encode(message: dict[str, Any]) -> str:
    # msgspec encodes to bytes. The page parses text frames with JSON.parse.
    return _json_encoder.encode(message).decode()


# %% FastAPI application


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """end the worker thread when the server shuts down."""
    yield  # noqa: RUF075
    SESSION.stop.set()
    CLIENTS.websockets.clear()
    if SESSION.thread is not None:
        await asyncio.to_thread(SESSION.thread.join)


app = FastAPI(
    title='striqt studio',
    description='edit, preview, and run a sweep specification',
    lifespan=lifespan,
)

app.mount('/static', StaticFiles(directory=str(_STATIC_DIR)), name='static')


@app.get('/', response_class=HTMLResponse)
async def get_index() -> str:
    return (_STATIC_DIR / 'index.html').read_text()


@app.get('/vendor/plotly.min.js')
async def get_plotly_js() -> FileResponse:
    """serve the plotly.js bundle of the installed `plotly` package."""
    import plotly

    path = Path(plotly.__file__).parent / 'package_data' / 'plotly.min.js'
    return FileResponse(path, media_type='application/javascript')


async def _send_all(message: str) -> None:
    """send `message` to every client, dropping each client on a send error."""
    disconnected = set()
    for ws in CLIENTS.websockets:
        try:
            await ws.send_text(message)
        except Exception:
            disconnected.add(ws)
    CLIENTS.websockets -= disconnected


async def send_message(message: dict[str, Any]) -> None:
    """encode `message` and send it to every client."""
    await _send_all(_encode(message))


async def broadcast_state() -> None:
    await send_message(state_message(SESSION))


async def broadcast_spec() -> None:
    await send_message(spec_message(SESSION))


def log_level(name: Any) -> int:
    """return the logging level named by `name`, or INFO for an unknown name."""
    from striqt.sensor.lib.util import _LOG_LEVEL_NAMES

    return _LOG_LEVEL_NAMES.get(str(name).lower(), logging.INFO)


async def _error(websocket: WebSocket, text: str, command: str | None = None) -> None:
    """send an ``error`` message naming the refused `command`, then the state."""
    message: dict[str, Any] = {'type': 'error', 'message': text}
    if command is not None:
        message['command'] = command
    await websocket.send_text(_encode(message))
    if command is not None:
        await websocket.send_text(_encode(state_message(SESSION)))


async def handle_edit(websocket: WebSocket, msg: dict[str, Any]) -> None:
    """write the edits of an ``edit`` message, then send the spec and the validation.

    Args:
        websocket: the client that sent the message
        msg: ``{'edits': [{'pointer', 'op', 'value_text'}, ...]}``
    """
    from striqt.analysis.lib import yaml as spec_yaml

    prov = SESSION.provenance
    if prov is None:
        await _error(websocket, 'a JSON spec is not editable in the page', 'edit')
        return
    edits = [
        spec_yaml.YAMLEdit(e['pointer'], e['op'], e.get('value_text'))
        for e in msg['edits']
    ]
    try:
        await asyncio.to_thread(spec_yaml.apply_yaml_edits, prov, edits)
    except (KeyError, ValueError) as ex:
        await _error(websocket, f'{type(ex).__name__}: {ex}', 'edit')
        return
    await asyncio.to_thread(session_.revalidate, SESSION)
    await broadcast_spec()
    await broadcast_state()


async def handle_revert(websocket: WebSocket, msg: dict[str, Any]) -> None:
    """restore one top-level section of the files to the applied values.

    Args:
        websocket: the client that sent the message
        msg: ``{'section': <top-level key>}``
    """
    from striqt.analysis.lib import yaml as spec_yaml

    prov = SESSION.provenance
    if prov is None or SESSION.applied_tree is None:
        await _error(websocket, 'a JSON spec is not editable in the page', 'revert')
        return
    try:
        edits = session_.revert_edits(SESSION, str(msg['section']))
        if edits:
            await asyncio.to_thread(spec_yaml.apply_yaml_edits, prov, edits)
    except (KeyError, ValueError, TypeError) as ex:
        await _error(websocket, f'{type(ex).__name__}: {ex}', 'revert')
        return
    await asyncio.to_thread(session_.revalidate, SESSION)
    await broadcast_spec()
    await broadcast_state()


async def handle_message(websocket: WebSocket, msg: dict[str, Any]) -> None:
    """act on one message from a client."""
    kind = msg['type']
    if kind == 'get_state':
        await websocket.send_text(_encode(state_message(SESSION)))
        await websocket.send_text(_encode(schema_message(SESSION)))
        await websocket.send_text(_encode(spec_message(SESSION)))
        if SESSION.dataset is not None:
            await broadcast_plot_update()
    elif kind == 'select_variable':
        SESSION.selected_variable = msg['variable']
        await asyncio.to_thread(_reset_plotter)
        await broadcast_plot_update()
    elif kind == 'select_captures':
        session_.apply_capture_selection(SESSION, msg)
        await broadcast_state()
    elif kind == 'set_running':
        SESSION.preview_enabled = bool(msg['running'])
        await broadcast_state()
    elif kind == 'edit':
        await handle_edit(websocket, msg)
    elif kind == 'revert':
        await handle_revert(websocket, msg)
    elif kind == 'apply':
        if SESSION.mode in ('running', 'prompting'):
            await _error(websocket, 'stop the run before applying the spec', kind)
        elif SESSION.edited_spec is None:
            await _error(websocket, 'the spec files do not validate', kind)
        else:
            tree = None
            if SESSION.provenance is not None:
                tree = copy.deepcopy(SESSION.provenance.tree)
            SESSION.commands.put(session_.Apply(SESSION.edited_spec, tree))
    elif kind == 'run_start':
        if SESSION.mode != 'preview':
            await _error(
                websocket, f'a run cannot start in mode {SESSION.mode!r}', kind
            )
        else:
            SESSION.commands.put(
                session_.StartRun(
                    overwrite=bool(msg.get('overwrite')),
                    log_level=log_level(msg.get('log_level')),
                )
            )
    elif kind == 'run_stop':
        if SESSION.run is not None:
            SESSION.run.stop.set()
    elif kind == 'prompt_answer':
        if not session_.answer_prompt(SESSION, int(msg['id']), str(msg['text'])):
            await _error(websocket, 'no prompt is waiting for that answer', kind)
    elif kind == 'set_log_level':
        if CLIENTS.log_handler is not None:
            CLIENTS.log_handler.setLevel(log_level(msg.get('level')))
    else:
        await _error(websocket, f'unknown message type {kind!r}', kind)


@app.websocket('/ws')
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    CLIENTS.websockets.add(websocket)

    try:
        while True:
            msg = _json_decoder.decode(await websocket.receive_text())
            await handle_message(websocket, msg)
    except WebSocketDisconnect:
        CLIENTS.websockets.discard(websocket)
    except Exception:
        session_._logger().exception('websocket handler failed')
        CLIENTS.websockets.discard(websocket)


# %% the plot broadcast


async def broadcast_plot_update() -> None:
    """send the current plot to every connected websocket.

    During a broadcast in progress, the function sets `data_updated` and returns,
    and one more broadcast follows the current one.
    """
    if CLIENTS.broadcast_pending:
        CLIENTS.data_updated = True
        return

    CLIENTS.broadcast_pending = True

    try:
        await _do_broadcast()
    finally:
        CLIENTS.broadcast_pending = False
        if CLIENTS.data_updated:
            CLIENTS.data_updated = False
            _schedule_follow_up_broadcast()


def _schedule_follow_up_broadcast() -> None:
    """schedule one more broadcast as a task, unless the server has stopped."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # the server stopped during the broadcast
        return
    # a task rather than recursion, which would grow with the backlog
    loop.create_task(_do_broadcast_if_needed())


async def _do_broadcast_if_needed() -> None:
    if SESSION.dataset is not None and not CLIENTS.broadcast_pending:
        await broadcast_plot_update()


def _select_delayed_variable(
    result: ss.lib.compute.DelayedDataset, name: str
) -> ss.lib.compute.DelayedDataset:
    return dataclasses.replace(result, delayed={name: result.delayed[name]})


_PLOT_LOCK = threading.Lock()
"""serializes the draws and resets of the plotter between the server threads"""


def _reset_plotter() -> None:
    with _PLOT_LOCK:
        if SESSION.open is not None:
            SESSION.open.plotter.reset(limits=True)


def _prepare_plot_data_sync(
    result: ss.lib.compute.DelayedDataset,
    plotter: PlotlyBackend,
    variable: str,
    data_select: dict[str, Any],
) -> list[go.Figure]:
    """return the figures of `variable` in `result`.

    The function blocks until the delayed dataset is materialized, and holds the
    plotter lock while drawing.

    Args:
        result: the delayed dataset of the analyzed capture
        plotter: the backend that produces the figures, reset before drawing
        variable: the name of the data variable to plot
        data_select: the `PlotOptions.data.select` selection; keys that are not
            indexes of the dataset are dropped

    Returns:
        the figures from `plotter` in drawing order
    """
    import striqt.figures as sf
    from striqt.sensor.lib import compute

    plot_func = sf.data_var_plotters()[variable]

    delayed = _select_delayed_variable(result, variable)
    ds = compute.from_delayed(delayed).set_xindex('port')

    # `select` may name indexes of other variables
    if data_select:
        valid_indexes = set(ds.indexes.keys())
        filtered_select = {k: v for k, v in data_select.items() if k in valid_indexes}
        if filtered_select:
            ds = ds.sel(filtered_select)

    with _PLOT_LOCK:
        plotter.reset()
        plot_func(ds, plotter)
        return plotter.get_plot_data()


def _serialize_plot_message(
    plot_data: list[go.Figure],
    coordinates: dict[str, Any],
    session: StudioSession,
) -> str:
    """return the ``plot_data`` message as JSON text.

    Numpy arrays in each figure are encoded as base64 typed arrays by
    `Figure.to_json`. ``selected_variable`` names the variable the figures show.
    """
    serialized_plots = [
        _json_decoder.decode(cast(str, fig.to_json())) for fig in plot_data
    ]

    return _encode({
        'type': 'plot_data',
        'data': serialized_plots,
        'coordinates': coordinates,
        'selected_variable': session.selected_variable,
    })


async def _do_broadcast() -> None:
    result = SESSION.dataset
    opened = SESSION.open
    if result is None or opened is None or not CLIENTS.websockets:
        return
    plotter = opened.plotter

    variable = SESSION.selected_variable
    if variable not in result.delayed:
        # the acquisition in flight was analyzed under the previous selection
        return

    try:
        plot_data = await asyncio.to_thread(
            _prepare_plot_data_sync, result, plotter, variable, opened.data_select
        )

        if not plot_data:
            await send_message({
                'type': 'error',
                'message': f"'{variable}' produced no plot data for the web view",
            })
            return

        coordinates = await asyncio.to_thread(
            coordinates_message, result, opened.varying_fields
        )
        message = await asyncio.to_thread(
            _serialize_plot_message, plot_data, coordinates, SESSION
        )
        await _send_all(message)

    except Exception as e:
        session_._logger().exception(f"plotting '{variable}' failed")
        await send_message({
            'type': 'error',
            'message': f"Error plotting '{variable}': {e!s}",
        })


# %% the server


WILDCARD_HOSTS = ('', '0.0.0.0', '::')


def serving_urls(host: str, port: int) -> list[str]:
    """return the URLs at which a server bound to `host` answers.

    A wildcard `host` (``''``, ``'0.0.0.0'``, or ``'::'``) expands to the loopback
    address, the host name, the fully qualified host name when it differs, and the
    IPv4 address of every network interface other than loopback. For any other
    `host`, the list holds the one URL at that host.
    """
    if host not in WILDCARD_HOSTS:
        return [f'http://{host}:{port}']

    import psutil

    names = ['127.0.0.1', socket.gethostname()]
    fqdn = socket.getfqdn()
    if fqdn not in names:
        names.append(fqdn)
    for addresses in psutil.net_if_addrs().values():
        for address in addresses:
            if address.family != socket.AF_INET:
                continue
            if address.address.startswith('127.') or address.address in names:
                continue
            names.append(address.address)
    return [f'http://{name}:{port}' for name in names]


def session_events(
    loop: asyncio.AbstractEventLoop, server: Any
) -> session_.SessionEvents:
    """return the `SessionEvents` that schedule the broadcasts on `loop`.

    `server` is the uvicorn server, which exits when the worker thread ends.
    """

    def schedule(coro: Any) -> None:
        if loop.is_running() and not SESSION.stop.is_set():
            asyncio.run_coroutine_threadsafe(coro, loop)
        else:
            coro.close()

    def attach_logs(level: int) -> logs.WebSocketLogHandler:
        handler = logs.WebSocketLogHandler(send_message, loop, level)
        logging.getLogger(logs.PARENT_LOGGER).addHandler(handler)
        CLIENTS.log_handler = handler
        return handler

    def detach_logs(handler: logs.WebSocketLogHandler) -> None:
        logging.getLogger(logs.PARENT_LOGGER).removeHandler(handler)
        if CLIENTS.log_handler is handler:
            CLIENTS.log_handler = None

    return session_.SessionEvents(
        state_changed=lambda: schedule(broadcast_state()),
        data_ready=lambda: schedule(broadcast_plot_update()),
        attach_logs=attach_logs,
        detach_logs=detach_logs,
        closed=lambda: setattr(server, 'should_exit', True),
    )


def run_server(path: str, host: str = '0.0.0.0', port: int = 8000) -> None:
    """run the web server and the worker thread for the spec at `path`.

    The server exits when the worker thread ends, and the worker thread ends after
    the capture in progress when the server exits. The URLs
    of `serving_urls` are logged at INFO on the ``sweep`` logger once the server
    starts, and the ``uvicorn.error`` logger level is set to WARNING.

    Raises:
        click.ClickException: if `path` has a suffix other than ``.yaml``, ``.yml``,
            or ``.json``
    """
    import click
    import uvicorn

    import striqt.sensor as ss

    SESSION.spec_path = str(Path(path).resolve())
    try:
        spec = ss.read_spec(SESSION.spec_path)
    except ValueError as ex:
        raise click.ClickException(str(ex)) from ex
    SESSION.applied_spec = spec
    session_.revalidate(SESSION)
    if SESSION.provenance is not None:
        SESSION.applied_tree = copy.deepcopy(SESSION.provenance.tree)

    loop = asyncio.new_event_loop()
    config = uvicorn.Config(app, host=host, port=port, loop='asyncio')
    server = uvicorn.Server(config)
    # the uvicorn startup line names only the bind address, 0.0.0.0 by default.
    # serve() logs the reachable URLs instead, and WARNING still passes uvicorn errors
    logging.getLogger('uvicorn.error').setLevel(logging.WARNING)

    async def serve() -> None:
        task = asyncio.ensure_future(server.serve())
        while not server.started and not task.done():
            await asyncio.sleep(0.05)
        if server.started:
            urls = ', '.join(serving_urls(host, port))
            session_._logger().info(f'serving the studio at {urls}')
        await task

    session_.start(SESSION, session_events(loop, server))

    try:
        loop.run_until_complete(serve())
    finally:
        SESSION.stop.set()
