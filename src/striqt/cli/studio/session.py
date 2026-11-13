"""the acquisition session behind the page: one open source, preview and run in turn.

One worker thread holds the resources open for the applied spec and executes the
commands queued by the websocket handlers in order. A preview acquires the selected
captures into a `NoSink` with the analysis narrowed to the displayed variable; a run
acquires the plan of the full spec into the sink of its runner; an apply closes the
resources and reopens the resources for the edited spec. No two of these overlap.
"""

from __future__ import annotations

import contextlib
import dataclasses
import itertools
import logging
import queue
import threading
import time
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

import striqt.analysis as sa
import striqt.sensor as ss

if TYPE_CHECKING:
    from collections.abc import Iterator

    import striqt.figures as sf
    from striqt.analysis.lib.yaml import ProvenanceTree
    from striqt.figures.backend import PlotlyBackend
    from striqt.sensor.lib.resources import Resources
    from striqt.sensor.specs.sequencing import PlanChoices

    from .logs import WebSocketLogHandler


# %% commands and run state


@dataclasses.dataclass(frozen=True)
class Apply:
    """reopen the resources for `spec`"""

    spec: ss.specs.ActionSpec
    tree: Any = None
    """the merged tree of the files as decoded for `spec`, kept as `applied_tree`"""


class OpenFailed(Exception):
    """the resources for an applied spec did not open"""


@dataclasses.dataclass(frozen=True)
class StartRun:
    """run the applied spec into its sink"""

    overwrite: bool = False
    """True removes a store already at the sink path"""
    log_level: int = logging.INFO
    """the level of the log records forwarded to the page during the run"""


@dataclasses.dataclass(frozen=True)
class Close:
    """end the worker thread"""


Command = Union[Apply, StartRun, Close]


@dataclasses.dataclass
class RunState:
    """the progress of one run"""

    count: int
    """the acquisitions the plan produces"""
    per_sweep: int = 0
    """the captures in one pass of the plan, the period of `capture_index`"""
    overwrite: bool = False
    """True when the run removed a store already at the sink path"""
    acquired: int = 0
    """the acquisitions analyzed so far"""
    capture_index: int = -1
    """the `capture_index` of the latest analyzed acquisition"""
    sweep_index: int = 0
    """the `sweep_index` of the latest analyzed acquisition"""
    started: float = dataclasses.field(default_factory=time.time)
    """the unix time (s) at which the run began"""
    finished: Optional[float] = None
    """the unix time (s) at which the run ended, or None while it runs"""
    error: Optional[str] = None
    """the traceback text of the exception that ended the run, or None"""
    stop: threading.Event = dataclasses.field(default_factory=threading.Event)
    """the event that ends the run after the acquisition in flight"""


@dataclasses.dataclass
class PendingPrompt:
    """an operator prompt waiting for an answer from the page"""

    id: int
    text: str
    answer: Optional[str] = None
    event: threading.Event = dataclasses.field(default_factory=threading.Event)


_prompt_ids = itertools.count(1)


@dataclasses.dataclass
class StudioSession:
    """the state shared by the worker thread and the websocket server.

    The worker thread sets the fields derived from the open resources, `mode`, `run`,
    and `prompt`. The websocket handlers set the selections and queue commands. A
    selection is replaced as a whole, never mutated in place.
    """

    spec_path: str = ''
    """the absolute path of the spec file"""
    mode: str = 'closed'
    """``'closed'``, ``'idle'``, ``'preview'``, ``'running'``, or ``'prompting'``"""
    applied_spec: Optional[ss.specs.ActionSpec] = None
    """the spec the resources are open for, with the sink path as written"""
    edited_spec: Optional[ss.specs.ActionSpec] = None
    """the latest valid spec decoded from the files, or None when they fail to decode"""
    validation_errors: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    """``{'path': <JSON pointer>, 'locations': [<pointer>, ...], 'message': ...}`` for
    each decode error of the files"""
    provenance: Optional[ProvenanceTree] = None
    """the merged tree of the YAML files with the file of each value, or None for a
    JSON spec"""
    applied_tree: Any = None
    """the merged tree of the files as decoded for `applied_spec`"""
    apply_error: Optional[str] = None
    """the traceback text of the last `Apply` that failed to open, or None"""
    sweep: Optional[ss.specs.Sweep] = None
    """the sweep spec of the open resources"""
    source_id: Optional[str] = None
    """the id of the open source"""
    plot_opts: Optional[sf.specs.PlotOptions] = None
    """the plot options from ``sweep.plot_hint`` or the preview default"""
    plotter: Optional[PlotlyBackend] = None
    """the backend that draws the selected variable"""
    data_select: dict[str, Any] = dataclasses.field(default_factory=dict)
    """the ``plot_opts.data.select`` indexers"""
    available_variables: list[str] = dataclasses.field(default_factory=list)
    """the enabled measurements of the sweep that have a registered plot"""
    variable_labels: dict[str, str] = dataclasses.field(default_factory=dict)
    """the display label of each available variable"""
    selected_variable: str = 'spectrogram'
    """the variable on display"""
    choices: Optional[PlanChoices] = None
    """the capture entries and loop values available for selection"""
    selected_capture: Optional[int] = None
    """the index into ``sweep.captures`` to acquire, or None for every entry"""
    selected_loops: dict[tuple[str, str], Any] = dataclasses.field(default_factory=dict)
    """the loop values to acquire, keyed by ``(isin, field)``, with every value kept
    for a loop absent from the mapping"""
    varying_fields: frozenset[str] = frozenset()
    """the capture fields that differ between the ``captures:`` entries of the applied
    spec, or between the ports of one entry; a field varied by a loop or an
    adjustment alone is not included"""
    adjustments: dict[str, tuple[Any, ...]] = dataclasses.field(default_factory=dict)
    """the unique values of each adjusted capture field, from `list_capture_adjustments`"""
    sink_path: Optional[str] = None
    """the resolved output path of a run, or None when the spec writes nothing"""
    dataset: Optional[ss.lib.compute.DelayedDataset] = None
    """the latest analyzed acquisition"""
    run: Optional[RunState] = None
    """the latest run, or None before the first"""
    prompt: Optional[PendingPrompt] = None
    """the operator prompt awaiting an answer, or None"""
    running: bool = True
    """True while the preview acquires; False pauses before the next capture"""
    commands: queue.Queue = dataclasses.field(default_factory=queue.Queue)
    """the commands for the worker thread, executed in order"""
    stop: threading.Event = dataclasses.field(default_factory=threading.Event)
    """the event that ends the worker thread when set"""
    thread: Optional[threading.Thread] = None
    """the worker thread, or None before start"""


@dataclasses.dataclass(frozen=True)
class SessionEvents:
    """the callables through which the worker thread reaches the server.

    Each is called from the worker thread and returns at once.
    """

    state_changed: Callable[[], None]
    """called after a change to `mode`, `run`, `prompt`, or a field set by `configure`"""
    data_ready: Callable[[], None]
    """called after `StudioSession.dataset` is replaced"""
    attach_logs: Callable[[int], WebSocketLogHandler]
    """returns a handler that forwards the ``striqt`` records at the level to the page"""
    detach_logs: Callable[[WebSocketLogHandler], None]
    """removes a handler returned by `attach_logs`"""
    closed: Callable[[], None]
    """called once when the worker thread ends"""


def _logger() -> logging.LoggerAdapter:
    return sa.util.get_logger('sweep')


# %% analysis and capture selection


def select_displayed(
    session: StudioSession,
    capture: ss.specs.SensorCapture,
    analysis: ss.specs.AnalysisGroup,
) -> ss.specs.AnalysisGroup:
    """return `analysis` narrowed to the variable on display.

    The whole group is returned when the selected variable is not an enabled field
    of `analysis`.
    """
    name = session.selected_variable
    if name not in analysis.to_dict():
        return analysis
    return sa.specs.helpers.narrow_analysis_group(analysis, [name])


def current_plan(session: StudioSession) -> tuple[ss.specs.SensorCapture, ...]:
    """return the captures that match the selections of `session`, in sweep order."""
    from striqt.sensor.specs import sequencing

    assert session.sweep is not None
    return sequencing.select_plan(
        session.sweep,
        session.source_id,
        capture=session.selected_capture,
        loops=session.selected_loops,
    )


def selected_captures(session: StudioSession) -> Iterator[ss.specs.SensorCapture]:
    """yield the captures of `current_plan` in a cycle until a command is queued.

    The plan is re-read before each capture, and a selection change takes effect at
    the next acquisition. The generator polls every 0.1 s while the plan is empty or
    ``session.running`` is False, and ends when ``session.stop`` is set or
    ``session.commands`` holds a command.

    Yields:
        the next capture of the plan, cycling to the first after the last
    """
    i = 0
    while not session.stop.is_set() and session.commands.empty():
        plan = current_plan(session)
        if len(plan) == 0 or not session.running:
            session.stop.wait(0.1)
            continue
        yield plan[i % len(plan)]
        i += 1


def select_first(session: StudioSession) -> None:
    """select the first ``captures:`` entry and the first value of each loop.

    No entry is selected when the sweep lists no captures, and a loop without
    values is left unselected. `session` is unchanged when ``session.choices`` is
    None.
    """
    choices = session.choices
    if choices is None:
        return
    session.selected_capture = 0 if choices.captures else None
    session.selected_loops = {
        (loop.isin, loop.field): loop.values[0] for loop in choices.loops if loop.values
    }


def apply_capture_selection(session: StudioSession, msg: dict[str, Any]) -> None:
    """set the selections of `session` from a ``select_captures`` message.

    A capture index is clamped to the ``captures:`` list and a loop index to the
    values of the loop. None keeps every entry or every value. `session` is
    unchanged when ``session.choices`` is None.

    Args:
        session: the session to update
        msg: ``{'capture': int | None, 'loops': [int | None, ...]}`` with indexes
            into the option lists, in the order of ``choices.loops``
    """
    if session.choices is None:
        return
    capture = msg.get('capture')
    if capture is None or not session.choices.captures:
        session.selected_capture = None
    else:
        session.selected_capture = min(
            max(int(capture), 0), len(session.choices.captures) - 1
        )
    selected: dict[tuple[str, str], Any] = {}
    indexes = msg.get('loops') or []
    for loop, index in zip(session.choices.loops, indexes):
        if index is not None and loop.values:
            selected[loop.isin, loop.field] = loop.values[
                min(max(int(index), 0), len(loop.values) - 1)
            ]
    session.selected_loops = selected


# %% the open resources


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


def output_path(
    spec: ss.specs.ActionSpec, format_path: ss.specs.paths.PathFormatter | None
) -> Optional[str]:
    """return the resolved path that a run of `spec` writes, or None for no output.

    A `ManualYFactorCalibration` writes the corrections file of `corrections_path`;
    a spec with a `ZarrSink` writes the store of `resolve_store_path`. A path
    template that `format_path` cannot fill is returned as written.
    """
    from striqt.sensor.lib import io as sensor_io
    from striqt.sensor.lib.actions import yfactor

    if spec.sink.path is None:
        return None
    try:
        if isinstance(spec, ss.specs.ManualYFactorCalibration):
            runner = yfactor.ManualYFactorRunner(spec)
            return str(yfactor.corrections_path(runner.sweep, format_path))
        if isinstance(spec.sink, ss.specs.ZarrSink):
            return str(sensor_io.resolve_store_path(spec.sink, format_path=format_path))
    except (KeyError, ValueError):
        return spec.sink.path
    return spec.sink.path


def configure(
    session: StudioSession,
    spec: ss.specs.ActionSpec,
    tree: Any,
    resources: Resources,
) -> None:
    """set the fields of `session` derived from `spec` and the open `resources`.

    `spec` becomes `applied_spec` and `tree` becomes `applied_tree` together.

    The variables offered are the enabled measurements of the sweep with a plot in
    `striqt.figures.data_var_plotters`. `selected_variable` is replaced by the first
    offered variable when the current selection is not offered. The capture
    selection starts at the first entry and the first value of each loop.
    """
    import striqt.figures as sf
    from striqt.figures.backend import PlotlyBackend
    from striqt.sensor.lib.compute import summary
    from striqt.sensor.specs import sequencing

    sweep = resources['sweep_spec']
    session.applied_spec = spec
    session.applied_tree = tree
    session.sweep = sweep
    session.source_id = resources['source'].source_id

    if sweep.plot_hint is None:
        plot_opts = _default_plot_options()
    else:
        plot_opts = sf.specs.PlotOptions.from_spec(sweep.plot_hint)
    session.plot_opts = plot_opts
    session.plotter = PlotlyBackend(plot_opts.plotter)
    session.data_select = dict(plot_opts.data.select)

    plotters = sf.data_var_plotters()
    labels = summary.analysis_labels(sweep.analysis)
    session.available_variables = [name for name in labels if name in plotters]
    session.variable_labels = {
        name: labels[name] for name in session.available_variables
    }
    if (
        session.selected_variable not in session.available_variables
        and session.available_variables
    ):
        session.selected_variable = session.available_variables[0]

    session.choices = sequencing.plan_choices(sweep, session.source_id)
    select_first(session)
    # the loop selectors show the looped values, so the table lists the fields that
    # differ between the entries themselves
    entries = sweep.replace(loops=(), adjust_captures={})
    session.varying_fields = summary.varying_fields(entries, session.source_id)
    session.adjustments = sequencing.list_capture_adjustments(sweep, session.source_id)
    session.sink_path = output_path(spec, resources['format_path'])


def null_sink(spec: ss.specs.ActionSpec) -> ss.specs.ActionSpec:
    """return `spec` with the sink path set to None, so that no store is written."""
    return spec.replace(sink=spec.sink.replace(path=None))


# %% operator prompts


def ask_operator(session: StudioSession, events: SessionEvents, text: str) -> str:
    """post `text` as a prompt to the page and block until the answer arrives.

    Raises:
        EOFError: if the session or the run is stopped before an answer arrives
    """
    pending = PendingPrompt(id=next(_prompt_ids), text=text)
    previous = session.mode
    session.prompt = pending
    session.mode = 'prompting'
    events.state_changed()
    try:
        while not pending.event.wait(0.1):
            run_stopped = session.run is not None and session.run.stop.is_set()
            if session.stop.is_set() or run_stopped:
                raise EOFError('the run was stopped before the prompt was answered')
        return pending.answer or ''
    finally:
        session.prompt = None
        session.mode = previous
        events.state_changed()


def answer_prompt(session: StudioSession, prompt_id: int, text: str) -> bool:
    """deliver `text` as the answer to the pending prompt with id `prompt_id`.

    Returns:
        True when the prompt was pending, False when no prompt has that id
    """
    pending = session.prompt
    if pending is None or pending.id != prompt_id:
        return False
    pending.answer = text
    pending.event.set()
    return True


# %% the worker thread


def run_preview(
    session: StudioSession, resources: Resources, events: SessionEvents
) -> None:
    """acquire the selected captures into a `NoSink` until a command is queued.

    The analysis of each capture is narrowed to the displayed variable, and each
    analyzed dataset replaces ``session.dataset``.
    """
    from striqt.sensor.lib import actions
    from striqt.sensor.lib.resources import SweepCallbacks

    def on_analyzed(result: ss.lib.compute.DelayedDataset) -> None:
        session.dataset = result
        events.data_ready()

    session.mode = 'preview'
    events.state_changed()

    callbacks = SweepCallbacks(
        on_analyzed=on_analyzed,
        select_analysis=lambda capture, analysis: select_displayed(
            session, capture, analysis
        ),
    )
    with ss.sinks.NoSink(resources['sweep_spec']) as sink:
        generator = actions.iterate_captures(
            resources,
            captures=selected_captures(session),
            yield_values=False,
            always_yield=True,
            sink=sink,
            callbacks=callbacks,
        )
        try:
            for _ in generator:
                pass
        finally:
            generator.close()


def run_sweep(
    session: StudioSession,
    resources: Resources,
    events: SessionEvents,
    command: StartRun,
) -> None:
    """run the applied spec into the sink of its runner.

    The run ends after the plan, after the acquisition in flight when
    ``session.run.stop`` is set, or on an exception, which is kept as
    ``session.run.error``. Overloads are logged on the ``analysis`` logger and the
    spectrogram SNR on each cached spectrogram, as in `sensor-sweep`.
    """
    from striqt.cli.sensor_sweep import log_overloads, log_spectrogram_snr
    from striqt.sensor.lib import actions
    from striqt.sensor.lib.resources import SweepCallbacks
    from striqt.sensor.specs import sequencing

    spec = session.applied_spec
    assert spec is not None
    runner = actions.registry.lookup(spec)(spec)
    sweep = runner.sweep
    source_id = resources['source'].source_id

    run = RunState(
        count=sequencing.count_acquisitions(sweep, source_id),
        per_sweep=len(sequencing.loop_captures(sweep, source_id)),
        overwrite=command.overwrite,
    )
    session.run = run
    session.mode = 'running'
    events.state_changed()

    def on_analyzed(result: ss.lib.compute.DelayedDataset) -> None:
        log_overloads(result)
        run.acquired += 1
        run.capture_index = result.extra_coords.capture_index
        run.sweep_index = result.extra_coords.sweep_index
        session.dataset = result
        events.data_ready()
        events.state_changed()

    callbacks = SweepCallbacks(
        on_analyzed=on_analyzed,
        on_sink_cached=log_spectrogram_snr,
        prompt=lambda text: ask_operator(session, events, text),
    )

    handler = events.attach_logs(command.log_level)
    try:
        sink = runner.open_sink(resources['format_path'], force=command.overwrite)
        with sink:
            generator = actions.iterate_sweep(
                resources,
                yield_values=False,
                always_yield=True,
                sink=sink,
                runner=runner,
                action_spec=spec,
                sweep_spec=sweep,
                callbacks=callbacks,
            )
            try:
                for _ in generator:
                    if run.stop.is_set() or session.stop.is_set():
                        break
            finally:
                generator.close()
    except Exception:
        run.error = traceback.format_exc()
        _logger().exception('the run failed')
    finally:
        events.detach_logs(handler)
        run.finished = time.time()
        session.mode = 'idle'
        events.state_changed()


def serve_spec(
    session: StudioSession,
    spec: ss.specs.ActionSpec,
    tree: Any,
    events: SessionEvents,
) -> Command:
    """open the resources for `spec` and execute the queued commands until an `Apply` or `Close`.

    The resources are opened with a None sink path, so nothing is written until a
    `StartRun`. A preview runs whenever no command is queued.

    Returns:
        the `Apply` or `Close` command that ended the block

    Raises:
        OpenFailed: if the resources do not open, from the exception raised
    """
    with contextlib.ExitStack() as stack:
        try:
            ctx = ss.open_resources(null_sink(spec), session.spec_path)
            resources = stack.enter_context(ctx)
        except Exception as ex:
            raise OpenFailed(str(ex)) from ex
        configure(session, spec, tree, resources)
        session.mode = 'idle'
        events.state_changed()
        while not session.stop.is_set():
            run_preview(session, resources, events)
            if session.stop.is_set():
                break
            command: Command = session.commands.get()
            if isinstance(command, StartRun):
                run_sweep(session, resources, events, command)
            else:
                return command
    return Close()


def worker(session: StudioSession, events: SessionEvents) -> None:
    """the body of the worker thread: serve each applied spec in turn.

    An applied spec that fails to open is logged on the ``sweep`` logger and kept as
    ``session.apply_error``, and the previous spec is reopened. The thread ends on a
    `Close` command, when ``session.stop`` is set, or on an exception outside a run,
    which is logged on the ``sweep`` logger.
    """
    spec = session.applied_spec
    tree = session.applied_tree
    assert spec is not None
    fallback: Optional[tuple[ss.specs.ActionSpec, Any]] = None
    try:
        while not session.stop.is_set():
            try:
                command = serve_spec(session, spec, tree, events)
            except OpenFailed:
                if fallback is None:
                    raise
                session.apply_error = traceback.format_exc()
                _logger().exception(
                    'the applied spec did not open; reopening the previous spec'
                )
                spec, tree = fallback
                fallback = None
                continue
            if isinstance(command, Apply):
                fallback = (spec, tree)
                spec, tree = command.spec, command.tree
                session.apply_error = None
            else:
                break
    except Exception:
        _logger().exception('the acquisition thread failed')
    finally:
        session.mode = 'closed'
        events.state_changed()
        events.closed()


def start(session: StudioSession, events: SessionEvents) -> threading.Thread:
    """start the worker thread for ``session.applied_spec`` and return it."""
    session.thread = threading.Thread(
        target=worker, args=(session, events), name='striqt-studio', daemon=True
    )
    session.thread.start()
    return session.thread


# %% the spec files


def read_spec(path: str) -> ss.specs.ActionSpec:
    """decode the spec at `path` by its suffix.

    Raises:
        ValueError: if `path` has a suffix other than ``.yaml``, ``.yml``, or ``.json``
    """
    if path.endswith(('.yaml', '.yml')):
        return ss.read_yaml_spec(path)
    if path.endswith('.json'):
        return ss.read_json_spec(path)
    raise ValueError('expected file to have .json or .yaml suffix')


def validation_pointers(text: str) -> list[str]:
    """return the JSON pointer of each msgspec path (``$.a[0].b``) in `text`, in order.

    A `SpecValidationError` names the failing field first and the further
    locations of the failure after ``at``; a plain msgspec error names one path.
    """
    import re

    pointers = []
    for match in re.finditer(r'\$((?:\.[A-Za-z_]\w*|\[\d+\])*)', text):
        parts = re.findall(r'\.([A-Za-z_]\w*)|\[(\d+)\]', match.group(1))
        pointers.append(''.join(f'/{name or index}' for name, index in parts))
    return pointers


def revalidate(session: StudioSession) -> None:
    """decode the spec files again and set `edited_spec` and `validation_errors`."""
    import msgspec

    try:
        session.edited_spec = read_spec(session.spec_path)
    except msgspec.ValidationError as ex:
        session.edited_spec = None
        text = str(ex)
        pointers = validation_pointers(text) or ['']
        session.validation_errors = [
            {'path': pointers[0], 'locations': pointers[1:], 'message': text}
        ]
    except Exception as ex:  # a YAML syntax error or a missing include
        session.edited_spec = None
        session.validation_errors = [
            {'path': '', 'locations': [], 'message': f'{type(ex).__name__}: {ex}'}
        ]
    else:
        session.validation_errors = []


def spec_schema_json(session: StudioSession) -> dict[str, Any]:
    """return the editor schema of the edited spec class, or of the applied one."""
    from striqt.sensor.specs import schema

    spec = session.edited_spec or session.applied_spec
    assert spec is not None
    return schema.schema_json(type(spec))


def spec_dir(session: StudioSession) -> Path:
    """return the directory of the spec file."""
    return Path(session.spec_path).parent


# %% reverting a section to the applied tree


def yaml_text(value: Any) -> str:
    """return `value` as one line of YAML flow text.

    A string that YAML 1.1 would read as another type is quoted, so the text reads
    back as the same value.
    """
    import yaml

    text = yaml.safe_dump(value, default_flow_style=True, width=float('inf'))
    return text.strip().removesuffix('\n...').strip()


def tree_edits(current: Any, target: Any, pointer: str = '') -> list[Any]:
    """return the edits that turn `current` into `target` at `pointer`.

    Mappings are compared key by key and lists element by element when the lengths
    agree, so an unchanged value is left in place; a list with a changed length and
    every other difference is replaced as a whole.

    Returns:
        `YAMLEdit` instances in application order: deletions of keys absent from
        `target`, then sets
    """
    from striqt.analysis.lib.yaml import YAMLEdit

    if isinstance(current, dict) and isinstance(target, dict):
        edits: list[Any] = [
            YAMLEdit(f'{pointer}/{_escape(key)}', 'delete', None)
            for key in current
            if key not in target
        ]
        for key, value in target.items():
            child = f'{pointer}/{_escape(key)}'
            if key not in current:
                edits.append(YAMLEdit(child, 'set', yaml_text(value)))
            else:
                edits.extend(tree_edits(current[key], value, child))
        return edits
    if (
        isinstance(current, list)
        and isinstance(target, list)
        and len(current) == len(target)
    ):
        edits = []
        for i, (a, b) in enumerate(zip(current, target)):
            edits.extend(tree_edits(a, b, f'{pointer}/{i}'))
        return edits
    if current == target and type(current) is type(target):
        return []
    return [YAMLEdit(pointer, 'set', yaml_text(target))]


def _escape(key: Any) -> str:
    return str(key).replace('~', '~0').replace('/', '~1')


def changed_sections(session: StudioSession) -> list[str]:
    """return the top-level keys with a file value that differs from `applied_tree`."""
    prov = session.provenance
    if prov is None or session.applied_tree is None:
        return []
    current = prov.tree if isinstance(prov.tree, dict) else {}
    applied = session.applied_tree if isinstance(session.applied_tree, dict) else {}
    names = list(current) + [k for k in applied if k not in current]
    return [str(k) for k in names if current.get(k) != applied.get(k)]


def revert_edits(session: StudioSession, section: str) -> list[Any]:
    """return the edits that restore `section` of the files to `applied_tree`.

    Raises:
        KeyError: if `section` is a key of neither tree
    """
    from striqt.analysis.lib.yaml import YAMLEdit

    prov = session.provenance
    assert prov is not None and session.applied_tree is not None
    current = prov.tree
    applied = session.applied_tree
    if section not in current and section not in applied:
        raise KeyError(section)
    pointer = f'/{_escape(section)}'
    if section not in applied:
        return [YAMLEdit(pointer, 'delete', None)]
    if section not in current:
        return [YAMLEdit(pointer, 'set', yaml_text(applied[section]))]
    return tree_edits(current[section], applied[section], pointer)
