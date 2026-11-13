"""striqt.cli.studio.session: the analysis and capture selections of the preview, the
operator prompts, and the validation of the spec files
"""

from __future__ import annotations

import threading

import pytest
from studio_helpers import SNR_LOOP, HookedStop, looped_sweep, preview_session
from synthetic_sources import ANALYSIS, TONE_OFFSETS, tone_captures

from striqt.cli.studio import session
from striqt.sensor.specs import sequencing

# %% select_displayed


def test_select_displayed_narrows_to_the_selected_variable():
    state = session.StudioSession(selected_variable='iq_waveform')
    (capture,) = tone_captures(TONE_OFFSETS[:1])
    narrowed = session.select_displayed(state, capture, ANALYSIS)
    assert type(narrowed) is type(ANALYSIS)
    assert list(narrowed.to_dict()) == ['iq_waveform']
    assert narrowed.iq_waveform == ANALYSIS.iq_waveform


def test_select_displayed_passes_an_unknown_selection_through():
    state = session.StudioSession(selected_variable='spectrogram')
    (capture,) = tone_captures(TONE_OFFSETS[:1])
    assert session.select_displayed(state, capture, ANALYSIS) is ANALYSIS


# %% selected_captures


def test_select_first_picks_the_first_entry_and_loop_value():
    state = preview_session(looped_sweep())
    assert state.selected_capture == 0
    assert state.selected_loops == {('capture', 'snr'): SNR_LOOP.values[0]}


def test_selected_captures_follows_the_selection_at_the_next_item():
    sweep = looped_sweep()
    state = preview_session(sweep)
    (first,) = sequencing.select_plan(
        sweep, None, capture=0, loops={('capture', 'snr'): SNR_LOOP.values[0]}
    )
    items = session.selected_captures(state)

    assert next(items) == first
    assert next(items) == first

    state.selected_capture = None
    state.selected_loops = {}
    plan = sequencing.loop_captures(sweep)
    assert next(items) in plan
    assert len({next(items) for _ in range(len(plan))}) > 1

    state.selected_capture = 2
    only_third = sequencing.select_plan(sweep, None, capture=2)
    assert len(only_third) == len(SNR_LOOP.values)
    assert next(items) in only_third

    state.stop.set()
    with pytest.raises(StopIteration):
        next(items)


def test_selected_captures_ends_when_a_command_is_queued():
    state = preview_session(looped_sweep())
    items = session.selected_captures(state)
    next(items)
    state.commands.put(session.StartRun())
    with pytest.raises(StopIteration):
        next(items)


def test_selected_captures_waits_while_not_running():
    """a paused generator polls `stop` and yields nothing until `running` is set"""
    sweep = looped_sweep()
    state = preview_session(sweep)
    state.stop = HookedStop(lambda stop: setattr(state, 'preview_enabled', True))
    items = session.selected_captures(state)
    first = next(items)
    assert state.stop.waits == 0

    state.preview_enabled = False
    assert next(items) == first
    assert state.stop.waits == 1
    state.stop.set()


def test_selected_captures_waits_through_an_empty_selection():
    sweep = looped_sweep()
    state = preview_session(sweep)
    state.selected_loops = {('capture', 'snr'): 5}
    state.stop = HookedStop(threading.Event.set)
    assert list(session.selected_captures(state)) == []
    assert state.stop.waits == 1


# %% apply_capture_selection


def test_apply_capture_selection_round_trips():
    state = preview_session(looped_sweep())
    session.apply_capture_selection(state, {'capture': 1, 'loops': [1]})
    assert state.selected_capture == 1
    assert state.selected_loops == {('capture', 'snr'): SNR_LOOP.values[1]}

    session.apply_capture_selection(state, {'capture': None, 'loops': [None]})
    assert state.selected_capture is None
    assert state.selected_loops == {}


def test_apply_capture_selection_clamps_indexes_to_the_choices():
    state = preview_session(looped_sweep())
    session.apply_capture_selection(state, {'capture': 99, 'loops': [-3]})
    assert state.selected_capture == len(TONE_OFFSETS) - 1
    assert state.selected_loops == {('capture', 'snr'): SNR_LOOP.values[0]}


# %% operator prompts


def _events(state):
    """events that record each state change"""
    changes = []
    events = session.SessionEvents(
        state_changed=lambda: changes.append(state.mode),
        data_ready=lambda: None,
        attach_logs=lambda level: None,
        detach_logs=lambda handler: None,
        closed=lambda: None,
    )
    return events, changes


def test_ask_operator_blocks_until_the_answer_arrives():
    state = session.StudioSession(mode='running')
    events, changes = _events(state)
    answers = []

    def ask():
        answers.append(session.ask_operator(state, events, 'enable the diode'))

    thread = threading.Thread(target=ask)
    thread.start()
    for _ in range(100):
        if state.prompt is not None:
            break
        threading.Event().wait(0.01)
    assert state.prompt is not None
    assert state.mode == 'prompting'
    assert state.prompt.text == 'enable the diode'

    assert not session.answer_prompt(state, state.prompt.id + 1, 'wrong id')
    assert session.answer_prompt(state, state.prompt.id, 'y')
    thread.join(5)
    assert answers == ['y']
    assert state.prompt is None
    assert state.mode == 'running'
    assert changes == ['prompting', 'running']


def test_ask_operator_raises_when_the_run_stops():
    state = session.StudioSession(mode='running', run=session.RunState(count=1))
    events, _ = _events(state)
    state.run.stop.set()
    with pytest.raises(EOFError):
        session.ask_operator(state, events, 'confirm')
    assert state.prompt is None


# %% validation


@pytest.mark.parametrize(
    ('text', 'pointers'),
    [
        ('Expected `int`, got `str` - at `$.captures[0].gain`', ['/captures/0/gain']),
        ('$.loops[2]: a loop names no field', ['/loops/2']),
        (
            '$.analysis.spectrogram: too long - at $.captures[0] on $.captures[1]',
            ['/analysis/spectrogram', '/captures/0', '/captures/1'],
        ),
        ('no path here', []),
    ],
)
def test_validation_pointers(text, pointers):
    assert session.validation_pointers(text) == pointers


def test_revalidate_reports_the_path_of_a_bad_value(tmp_path):
    import shutil

    from conftest import SWEEP_DIR

    shutil.copytree(SWEEP_DIR, tmp_path / 'sweeps')
    path = tmp_path / 'sweeps' / 'synthetic.yaml'
    state = session.StudioSession(spec_path=str(path))

    session.revalidate(state)
    assert state.validation_errors == []
    assert state.edited_spec is not None

    text = path.read_text().replace('duration: 2e-3 # s', 'duration: -1 # s', 1)
    path.write_text(text)
    session.revalidate(state)
    assert state.edited_spec is None
    (error,) = state.validation_errors
    assert '/captures/0' in [error['path'], *error['locations']]


# %% the worker thread


def wait_for(condition, timeout=60.0, what='the condition'):
    """block until `condition()` is true, failing after `timeout` s"""
    import time

    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            pytest.fail(f'{what} did not hold within {timeout} s')
        time.sleep(0.02)


def test_worker_previews_runs_and_applies_in_turn(tmp_path, monkeypatch):
    """the worker opens the synthetic sweep, previews, writes one store for a run,
    reopens for an applied spec with one capture, and ends on Close"""
    import logging
    import shutil

    from conftest import SWEEP_DIR

    import striqt.sensor as ss

    monkeypatch.chdir(tmp_path)
    shutil.copytree(SWEEP_DIR, tmp_path / 'sweeps')
    path = tmp_path / 'sweeps' / 'synthetic.yaml'
    spec = ss.read_yaml_spec(str(path))
    state = session.StudioSession(
        spec_path=str(path), applied_spec=spec, edited_spec=spec
    )
    modes = []
    datasets = []
    closed = threading.Event()
    events = session.SessionEvents(
        state_changed=lambda: modes.append(state.mode),
        data_ready=lambda: datasets.append((state.mode, state.dataset)),
        attach_logs=logging.NullHandler,
        detach_logs=lambda handler: None,
        closed=closed.set,
    )

    outputs = tmp_path / 'sweeps' / 'outputs'
    before = set(outputs.glob('*.zarr.zip')) if outputs.exists() else set()
    thread = session.start(state, events)
    try:
        wait_for(lambda: state.mode == 'preview', what='the preview')
        wait_for(lambda: len(datasets) >= 1, what='the first preview dataset')
        opened = state.open
        assert opened is not None
        assert opened.sink_path is not None
        assert opened.sink_path.endswith('.zarr.zip')
        # snr is looped and equal in both entries; the LO frequency differs between them
        assert 'external_lo_frequency' in opened.varying_fields
        assert 'snr' not in opened.varying_fields
        assert len(opened.choices.captures) == len(spec.captures)
        # the preview analyzes the displayed variable only
        assert datasets[0][0] == 'preview'
        assert list(datasets[0][1].delayed) == [state.selected_variable]

        state.commands.put(session.StartRun())
        wait_for(lambda: state.run is not None and state.run.finished is not None)
        run = state.run
        assert run.error is None, run.error
        assert run.count == ss.specs.sequencing.count_acquisitions(
            spec, opened.source_id
        )
        assert run.acquired == run.count
        stores = set(outputs.glob('*.zarr.zip')) - before
        assert len(stores) == 1
        # the run analyzes the whole group
        run_sets = [set(dd.delayed) for mode, dd in datasets if mode == 'running']
        assert len(run_sets) == run.count
        assert run_sets[-1] == set(spec.analysis.to_dict())
        wait_for(lambda: state.mode == 'preview', what='the preview after the run')

        one_capture = spec.replace(captures=spec.captures[:1])
        state.commands.put(session.Apply(one_capture))
        wait_for(
            lambda: state.applied_spec is one_capture and state.mode == 'preview',
            what='the applied spec',
        )
        assert len(state.open.choices.captures) == 1

        state.commands.put(session.Close())
        wait_for(closed.is_set, what='the close')
    finally:
        state.stop.set()
        thread.join(60)
    assert not thread.is_alive()
    assert state.mode == 'closed'
    assert modes[:2] == ['idle', 'preview']
    assert 'running' in modes


def test_run_stop_ends_the_run_early(tmp_path, monkeypatch):
    import logging
    import shutil

    from conftest import SWEEP_DIR

    import striqt.sensor as ss

    monkeypatch.chdir(tmp_path)
    shutil.copytree(SWEEP_DIR, tmp_path / 'sweeps')
    path = tmp_path / 'sweeps' / 'synthetic.yaml'
    spec = ss.read_yaml_spec(str(path))
    state = session.StudioSession(spec_path=str(path), applied_spec=spec)
    events = session.SessionEvents(
        state_changed=lambda: None,
        data_ready=lambda: (
            state.run is not None and state.run.acquired >= 2 and state.run.stop.set()
        ),
        attach_logs=logging.NullHandler,
        detach_logs=lambda handler: None,
        closed=lambda: None,
    )
    thread = session.start(state, events)
    try:
        wait_for(lambda: state.mode == 'preview')
        state.commands.put(session.StartRun())
        wait_for(lambda: state.run is not None and state.run.finished is not None)
        assert state.run.error is None
        assert 2 <= state.run.acquired < state.run.count
        wait_for(lambda: state.mode == 'preview')
    finally:
        state.stop.set()
        thread.join(60)


# %% reverting a section


@pytest.mark.parametrize(
    ('value', 'text'),
    [
        (70, '70'),
        ('3750e6', '3750e6'),
        ('yes', "'yes'"),
        (None, 'null'),
        ([0, 1], '[0, 1]'),
        ({'a': 1, 'b': [True, 'x']}, '{a: 1, b: [true, x]}'),
    ],
)
def test_yaml_text_reads_back_as_the_value(value, text):
    import yaml

    assert session.yaml_text(value) == text
    assert yaml.safe_load(text) == value


def test_tree_edits_touch_only_the_differences():
    current = {'a': 1, 'b': {'c': 2, 'd': 3}, 'e': [1, 2], 'f': [1, 2, 3]}
    target = {'a': 1, 'b': {'c': 2, 'd': 4, 'g': 5}, 'f': [1, 2], 'h': 'x'}
    edits = session.tree_edits(current, target, '/s')
    assert [(e.pointer, e.op, e.value_text) for e in edits] == [
        ('/s/e', 'delete', None),
        ('/s/b/d', 'set', '4'),
        ('/s/b/g', 'set', '5'),
        ('/s/f', 'set', '[1, 2]'),
        ('/s/h', 'set', 'x'),
    ]


def test_tree_edits_of_equal_trees_are_empty():
    tree = {'a': [1, {'b': 'c'}]}
    assert session.tree_edits(tree, dict(tree)) == []


def test_changed_sections_and_revert_edits(tmp_path):
    import shutil

    from conftest import SWEEP_DIR

    from striqt.analysis.lib import yaml as spec_yaml

    shutil.copytree(SWEEP_DIR, tmp_path / 'sweeps')
    path = tmp_path / 'sweeps' / 'synthetic.yaml'
    state = session.StudioSession(spec_path=str(path))
    state.provenance = spec_yaml.load_yaml_provenance(path)
    state.applied_tree = spec_yaml.load_yaml_provenance(path).tree
    assert session.changed_sections(state) == []

    spec_yaml.apply_yaml_edits(
        state.provenance,
        [
            spec_yaml.YAMLEdit('/captures/0/snr', 'set', '70'),
            spec_yaml.YAMLEdit('/loops/0/count', 'set', '3'),
        ],
    )
    state.provenance = spec_yaml.load_yaml_provenance(path)
    assert session.changed_sections(state) == ['captures', 'loops']

    edits = session.revert_edits(state, 'captures')
    assert [(e.pointer, e.op, e.value_text) for e in edits] == [
        ('/captures/0/snr', 'set', '80')
    ]
    spec_yaml.apply_yaml_edits(state.provenance, edits)
    state.provenance = spec_yaml.load_yaml_provenance(path)
    assert session.changed_sections(state) == ['loops']
    with pytest.raises(KeyError):
        session.revert_edits(state, 'nonexistent')


def test_failed_apply_keeps_the_previous_spec(tmp_path, monkeypatch):
    """an applied spec that does not open is recorded as `apply_error` and the worker
    reopens the previous spec"""
    import logging
    import shutil

    from conftest import SWEEP_DIR

    import striqt.sensor as ss

    monkeypatch.chdir(tmp_path)
    shutil.copytree(SWEEP_DIR, tmp_path / 'sweeps')
    path = tmp_path / 'sweeps' / 'synthetic.yaml'
    spec = ss.read_yaml_spec(str(path))
    bad = spec.replace(captures=spec.captures[:1])
    state = session.StudioSession(
        spec_path=str(path), applied_spec=spec, applied_tree={'spec': 'old'}
    )
    opened = []
    real_open = ss.open_resources

    def open_resources(target, *args, **kwargs):
        opened.append(target)
        if target.captures == bad.captures:
            raise RuntimeError('the radio is unplugged')
        return real_open(target, *args, **kwargs)

    monkeypatch.setattr(session.ss, 'open_resources', open_resources)
    events = session.SessionEvents(
        state_changed=lambda: None,
        data_ready=lambda: None,
        attach_logs=logging.NullHandler,
        detach_logs=lambda handler: None,
        closed=lambda: None,
    )
    thread = session.start(state, events)
    try:
        wait_for(lambda: state.mode == 'preview')
        state.commands.put(session.Apply(bad, {'spec': 'new'}))
        wait_for(lambda: state.apply_error is not None, what='the apply error')
        wait_for(lambda: state.mode == 'preview', what='the preview after the failure')
        assert 'the radio is unplugged' in state.apply_error
        assert state.applied_spec is spec
        assert state.applied_tree == {'spec': 'old'}
        assert len(opened) == 3
        assert thread.is_alive()
    finally:
        state.stop.set()
        thread.join(60)
