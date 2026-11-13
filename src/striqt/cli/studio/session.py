"""the acquisition session behind the page: one open source, preview and run in turn.

One worker thread holds the resources open for the applied spec and executes the
commands of the page in order, so a preview, a run, and a reopen never overlap.
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


@dataclasses.dataclass(frozen=True)
class OpenSpec:
    """the facts derived from the applied spec and the open resources.

    An instance is built by `configure` once the resources open and replaced as a
    whole on the next apply.
    """

    sweep: ss.specs.Sweep
    """the sweep spec of the open resources"""
    source_id: str
    """the id of the open source"""
    choices: PlanChoices
    """the capture entries and loop values available for selection"""
    varying_fields: frozenset[str]
    """the capture fields that differ between the ``captures:`` entries of the spec,
    or between the ports of one entry; a field varied by a loop or an adjustment
    alone is not included"""
    adjustments: dict[str, tuple[Any, ...]]
    """the unique values of each adjusted capture field, from `list_capture_adjustments`"""
    sink_path: Optional[str]
    """the resolved output path of a run, or None when the spec writes nothing"""
    available_variables: tuple[str, ...]
    """the enabled measurements of the sweep that have a registered plot"""
    variable_labels: dict[str, str]
    """the display label of each available variable"""
    plotter: PlotlyBackend
    """the backend that draws the selected variable"""
    data_select: dict[str, Any]
    """the ``plot_opts.data.select`` indexers"""


@dataclasses.dataclass
class StudioSession:
    """the state shared by the worker thread and the websocket server.

    The worker thread writes `open`, the applied spec, the mode, the run, and the
    prompt; the websocket handlers write the selections and the edited files. A
    selection is replaced as a whole, never mutated in place.
    """

    spec_path: str = ''
    """the absolute path of the spec file"""
    mode: str = 'closed'
    """``'closed'``, ``'idle'``, ``'preview'``, ``'running'``, or ``'prompting'``"""
    applied_spec: Optional[ss.specs.ActionSpec] = None
    """the spec the resources are open for, with the sink path as written"""
    applied_tree: Any = None
    """the merged tree of the files as decoded for `applied_spec`"""
    apply_error: Optional[str] = None
    """the traceback text of the last `Apply` that failed to open, or None"""
    edited_spec: Optional[ss.specs.ActionSpec] = None
    """the latest valid spec decoded from the files, or None when they fail to decode"""
    validation_errors: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    """``{'path': <JSON pointer>, 'locations': [<pointer>, ...], 'message': ...}`` for
    each decode error of the files"""
    provenance: Optional[ProvenanceTree] = None
    """the merged tree of the YAML files with the file of each value, or None for a
    JSON spec"""
    open: Optional[OpenSpec] = None
    """the facts derived from the open resources, or None while closed"""
    selected_variable: str = 'spectrogram'
    """the variable on display"""
    selected_capture: Optional[int] = None
    """the index into ``sweep.captures`` to acquire, or None for every entry"""
    selected_loops: dict[tuple[str, str], Any] = dataclasses.field(default_factory=dict)
    """the loop values to acquire, keyed by ``(isin, field)``, with every value kept
    for a loop absent from the mapping"""
    preview_enabled: bool = True
    """True while the preview acquires; False pauses before the next capture"""
    dataset: Optional[ss.lib.compute.DelayedDataset] = None
    """the latest analyzed acquisition"""
    run: Optional[RunState] = None
    """the latest run, or None before the first"""
    prompt: Optional[PendingPrompt] = None
    """the operator prompt awaiting an answer, or None"""
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
    """called after a change to the mode, the run, the prompt, or the open resources"""
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

    assert session.open is not None
    return sequencing.select_plan(
        session.open.sweep,
        session.open.source_id,
        capture=session.selected_capture,
        loops=session.selected_loops,
    )


def selected_captures(session: StudioSession) -> Iterator[ss.specs.SensorCapture]:
    """yield the captures of `current_plan` in a cycle until a command is queued.

    The plan is re-read before each capture, and a selection change takes effect at
    the next acquisition. The generator polls every 0.1 s while the plan is empty or
    ``session.preview_enabled`` is False, and ends when ``session.stop`` is set or
    ``session.commands`` holds a command.

    Yields:
        the next capture of the plan, cycling to the first after the last
    """
    i = 0
    while not session.stop.is_set() and session.commands.empty():
        plan = current_plan(session)
        if len(plan) == 0 or not session.preview_enabled:
            session.stop.wait(0.1)
            continue
        yield plan[i % len(plan)]
        i += 1


def select_first(session: StudioSession) -> None:
    """select the first ``captures:`` entry and the first value of each loop.

    No entry is selected when the sweep lists no captures, and a loop without
    values is left unselected. `session` is unchanged while the resources are closed.
    """
    if session.open is None:
        return
    choices = session.open.choices
    session.selected_capture = 0 if choices.captures else None
    session.selected_loops = {
        (loop.isin, loop.field): loop.values[0] for loop in choices.loops if loop.values
    }


def apply_capture_selection(session: StudioSession, msg: dict[str, Any]) -> None:
    """set the selections of `session` from a ``select_captures`` message.

    A capture index is clamped to the ``captures:`` list and a loop index to the
    values of the loop. None keeps every entry or every value. `session` is
    unchanged while the resources are closed.

    Args:
        session: the session to update
        msg: ``{'capture': int | None, 'loops': [int | None, ...]}`` with indexes
            into the option lists, in the order of ``choices.loops``
    """
    if session.open is None:
        return
    choices = session.open.choices
    capture = msg.get('capture')
    if capture is None or not choices.captures:
        session.selected_capture = None
    else:
        session.selected_capture = min(max(int(capture), 0), len(choices.captures) - 1)
    selected: dict[tuple[str, str], Any] = {}
    indexes = msg.get('loops') or []
    for loop, index in zip(choices.loops, indexes):
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


def configure(
    session: StudioSession,
    spec: ss.specs.ActionSpec,
    tree: Any,
    resources: Resources,
) -> None:
    """set `session.open` from `spec` and the open `resources`.

    `spec` becomes `applied_spec` and `tree` becomes `applied_tree` together. The
    variables offered are the enabled measurements of the sweep with a plot in
    `striqt.figures.data_var_plotters`. `selected_variable` is replaced by the first
    offered variable when the current selection is not offered. The capture
    selection starts at the first entry and the first value of each loop.
    """
    import striqt.figures as sf
    from striqt.figures.backend import PlotlyBackend
    from striqt.sensor.lib import actions
    from striqt.sensor.lib.compute import summary
    from striqt.sensor.specs import sequencing

    sweep = resources['sweep_spec']
    source_id = resources['source'].source_id
    session.applied_spec = spec
    session.applied_tree = tree

    if sweep.plot_hint is None:
        plot_opts = _default_plot_options()
    else:
        plot_opts = sf.specs.PlotOptions.from_spec(sweep.plot_hint)

    plotters = sf.data_var_plotters()
    labels = summary.analysis_labels(sweep.analysis)
    available = tuple(name for name in labels if name in plotters)
    if session.selected_variable not in available and available:
        session.selected_variable = available[0]

    # the loop selectors show the looped values, so the table lists the fields that
    # differ between the entries themselves
    entries = sweep.replace(loops=(), adjust_captures={})
    session.open = OpenSpec(
        sweep=sweep,
        source_id=source_id,
        choices=sequencing.plan_choices(sweep, source_id),
        varying_fields=summary.varying_fields(entries, source_id),
        adjustments=sequencing.list_capture_adjustments(sweep, source_id),
        sink_path=actions.core.output_path(spec, resources['format_path']),
        available_variables=available,
        variable_labels={name: labels[name] for name in available},
        plotter=PlotlyBackend(plot_opts.plotter),
        data_select=dict(plot_opts.data.select),
    )
    select_first(session)


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
    ``session.run.error``. The hooks are those of `logging_callbacks`.
    """
    from striqt.cli.sensor_sweep import logging_callbacks
    from striqt.sensor.lib import actions
    from striqt.sensor.specs import sequencing

    spec = session.applied_spec
    assert spec is not None
    runner = actions.registry.lookup(spec)(spec)
    sweep = runner.sweep
    source_id = resources['source'].source_id

    run = RunState(
        count=sequencing.count_acquisitions(sweep, source_id),
        per_sweep=len(sequencing.loop_captures(sweep, source_id)),
    )
    session.run = run
    session.mode = 'running'
    events.state_changed()

    def on_analyzed(result: ss.lib.compute.DelayedDataset) -> None:
        run.acquired += 1
        run.capture_index = result.extra_coords.capture_index
        run.sweep_index = result.extra_coords.sweep_index
        session.dataset = result
        events.data_ready()
        events.state_changed()

    callbacks = logging_callbacks(
        on_analyzed=on_analyzed,
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
            ctx = ss.open_resources(ss.specs.action.null_sink(spec), session.spec_path)
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
    """serve each applied spec in turn, as the body of the worker thread.

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


def validation_pointers(text: str) -> list[str]:
    """return the JSON pointer of each msgspec path (``$.a[0].b``) in `text`, in order.

    A plain `msgspec.ValidationError` names its path in the message only; a
    `SpecValidationError` carries the path and the locations as fields, read by
    `pointers_of`.
    """
    import re

    pointers = []
    for match in re.finditer(r'\$((?:\.[A-Za-z_]\w*|\[\d+\])*)', text):
        parts = re.findall(r'\.([A-Za-z_]\w*)|\[(\d+)\]', match.group(1))
        pointers.append(''.join(f'/{name or index}' for name, index in parts))
    return pointers


def revalidate(session: StudioSession) -> None:
    """load the spec files again and set `provenance`, `edited_spec`, and `validation_errors`.

    A YAML spec is read twice: once for the provenance of each value and once as
    the bound spec. A file that fails to load or decode leaves `edited_spec` None and
    one error; a `SpecValidationError` adds the JSON pointers of the failing field
    and of the further locations the error names.
    """
    import msgspec

    from striqt.analysis.lib import yaml as spec_yaml

    if session.spec_path.endswith(('.yaml', '.yml')):
        try:
            session.provenance = spec_yaml.load_yaml_provenance(session.spec_path)
        except Exception as ex:
            session.edited_spec = None
            session.validation_errors = [
                {'path': '', 'locations': [], 'message': f'{type(ex).__name__}: {ex}'}
            ]
            return
    try:
        session.edited_spec = ss.read_spec(session.spec_path)
    except msgspec.ValidationError as ex:
        session.edited_spec = None
        text = str(ex)
        if isinstance(ex, sa.specs.helpers.SpecValidationError):
            path, locations = sa.specs.helpers.pointers_of(ex)
        else:
            pointers = validation_pointers(text) or ['']
            path, locations = pointers[0], pointers[1:]
        session.validation_errors = [
            {'path': path, 'locations': locations, 'message': text}
        ]
    except Exception as ex:
        session.edited_spec = None
        session.validation_errors = [
            {'path': '', 'locations': [], 'message': f'{type(ex).__name__}: {ex}'}
        ]
    else:
        session.validation_errors = []


def spec_schema_json(session: StudioSession) -> dict[str, Any]:
    """return the editor schema of the edited spec class, or of the applied one."""
    from striqt.sensor.specs import frontend

    spec = current_spec(session)
    assert spec is not None
    return frontend.schema_json(type(spec))


def current_spec(session: StudioSession) -> Optional[ss.specs.ActionSpec]:
    """return the latest valid spec of the files, or the applied spec when none decodes."""
    return session.edited_spec or session.applied_spec


def spec_dir(session: StudioSession) -> Path:
    """return the directory of the spec file."""
    return Path(session.spec_path).parent


# %% reverting a section to the applied tree


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
    from striqt.analysis.lib import yaml as spec_yaml
    from striqt.analysis.lib.yaml import YAMLEdit

    prov = session.provenance
    assert prov is not None and session.applied_tree is not None
    current = prov.tree
    applied = session.applied_tree
    if section not in current and section not in applied:
        raise KeyError(section)
    pointer = spec_yaml.json_pointer([section])
    if section not in applied:
        return [YAMLEdit(pointer, 'delete', None)]
    if section not in current:
        return [YAMLEdit(pointer, 'set', spec_yaml.yaml_flow_text(applied[section]))]
    return spec_yaml.tree_edits(current[section], applied[section], pointer)
