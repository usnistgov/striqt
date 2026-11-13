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
    state.stop = HookedStop(lambda stop: setattr(state, 'running', True))
    items = session.selected_captures(state)
    first = next(items)
    assert state.stop.waits == 0

    state.running = False
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
