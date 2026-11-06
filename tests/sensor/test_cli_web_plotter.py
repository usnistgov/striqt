"""striqt.cli.web_plotter: the analysis and capture selections of the preview, and the
messages built from an analyzed capture
"""

from __future__ import annotations

import json
import threading

import plotly.graph_objects as go
import pytest
from synthetic_sources import (
    ANALYSIS,
    SPECTROGRAM,
    TONE_OFFSETS,
    preset_sweep,
    tone_captures,
)

import striqt.sensor as ss
from striqt.cli import web_plotter
from striqt.sensor.specs import sequencing

SNR_LOOP = ss.specs.List(field='snr', values=(10.0, 20.0))


def looped_sweep(analysis=SPECTROGRAM):
    return preset_sweep(
        'single_tone',
        tone_captures(TONE_OFFSETS),
        analysis=analysis,
        loops=(ss.specs.Repeat(count=2), SNR_LOOP),
    )


def preview_state(sweep) -> web_plotter.PreviewState:
    """return a state over `sweep` with the first entry of every choice selected"""
    state = web_plotter.PreviewState(sweep=sweep)
    state.choices = sequencing.plan_choices(sweep, None)
    web_plotter.select_first(state)
    return state


class HookedStop(threading.Event):
    """an Event with a `wait` that calls ``on_wait(self)`` and returns without
    blocking"""

    def __init__(self, on_wait):
        super().__init__()
        self.on_wait = on_wait
        self.waits = 0

    def wait(self, timeout=None):
        self.waits += 1
        self.on_wait(self)
        return self.is_set()


@pytest.fixture(scope='module')
def analyzed():
    """the `DelayedDataset` of one spectrogram capture"""
    sweep = preset_sweep(
        'single_tone', tone_captures(TONE_OFFSETS[:1]), analysis=SPECTROGRAM
    )
    with ss.open_resources(sweep, None) as res:
        results = ss.iterate_sweep(res, sink=ss.sinks.NoSink(sweep))
        return next(dd for dd in results if dd is not None)


# %% select_displayed


def test_select_displayed_narrows_to_the_selected_variable(monkeypatch):
    monkeypatch.setattr(web_plotter.STATE, 'selected_variable', 'iq_waveform')
    (capture,) = tone_captures(TONE_OFFSETS[:1])
    narrowed = web_plotter.select_displayed(capture, ANALYSIS)
    assert type(narrowed) is type(ANALYSIS)
    assert list(narrowed.to_dict()) == ['iq_waveform']
    assert narrowed.iq_waveform == ANALYSIS.iq_waveform


def test_select_displayed_passes_an_unknown_selection_through(monkeypatch):
    monkeypatch.setattr(web_plotter.STATE, 'selected_variable', 'spectrogram')
    (capture,) = tone_captures(TONE_OFFSETS[:1])
    assert web_plotter.select_displayed(capture, ANALYSIS) is ANALYSIS


# %% selected_captures


def test_select_first_picks_the_first_entry_and_loop_value():
    state = preview_state(looped_sweep())
    assert state.selected_capture == 0
    assert state.selected_loops == {('capture', 'snr'): SNR_LOOP.values[0]}


def test_selected_captures_follows_the_selection_at_the_next_item():
    sweep = looped_sweep()
    state = preview_state(sweep)
    (first,) = sequencing.select_plan(
        sweep, None, capture=0, loops={('capture', 'snr'): SNR_LOOP.values[0]}
    )
    items = web_plotter.selected_captures(state)

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


def test_selected_captures_waits_while_not_running():
    """a paused generator polls `stop` and yields nothing until `running` is set"""
    sweep = looped_sweep()
    state = preview_state(sweep)
    state.stop = HookedStop(lambda stop: setattr(state, 'running', True))
    items = web_plotter.selected_captures(state)
    first = next(items)
    assert state.stop.waits == 0

    state.running = False
    assert next(items) == first
    assert state.stop.waits == 1
    state.stop.set()


def test_selected_captures_waits_through_an_empty_selection():
    sweep = looped_sweep()
    state = preview_state(sweep)
    state.selected_loops = {('capture', 'snr'): 5}
    state.stop = HookedStop(threading.Event.set)
    assert list(web_plotter.selected_captures(state)) == []
    assert state.stop.waits == 1


# %% capture selection messages


def test_capture_options_list_one_entry_per_loop_and_capture():
    state = preview_state(looped_sweep())
    options = web_plotter.capture_options(state.choices)
    assert len(options['captures']) == len(TONE_OFFSETS)
    assert [loop['field'] for loop in options['loops']] == ['snr']
    assert options['loops'][0]['labels'] == list(state.choices.loops[0].labels)


def test_apply_capture_selection_round_trips():
    state = preview_state(looped_sweep())
    assert web_plotter.capture_selection(state) == {'capture': 0, 'loops': [0]}

    web_plotter.apply_capture_selection(state, {'capture': 1, 'loops': [1]})
    assert state.selected_capture == 1
    assert state.selected_loops == {('capture', 'snr'): SNR_LOOP.values[1]}
    assert web_plotter.capture_selection(state) == {'capture': 1, 'loops': [1]}

    web_plotter.apply_capture_selection(state, {'capture': None, 'loops': [None]})
    assert state.selected_capture is None
    assert state.selected_loops == {}
    assert web_plotter.capture_selection(state) == {'capture': None, 'loops': [None]}


def test_apply_capture_selection_clamps_indexes_to_the_choices():
    state = preview_state(looped_sweep())
    web_plotter.apply_capture_selection(state, {'capture': 99, 'loops': [-3]})
    assert state.selected_capture == len(TONE_OFFSETS) - 1
    assert state.selected_loops == {('capture', 'snr'): SNR_LOOP.values[0]}


# %% plot and coordinate messages


def test_prepare_plot_data_returns_figures(analyzed):
    from striqt.figures.backend import PlotlyBackend

    plotter = PlotlyBackend(web_plotter._default_plot_options().plotter)
    figures = web_plotter._prepare_plot_data_sync(analyzed, plotter, 'spectrogram', {})
    assert len(figures) >= 1
    assert all(isinstance(fig, go.Figure) for fig in figures)
    assert all(len(fig.data) >= 1 for fig in figures)


def test_coordinates_message_has_one_value_per_port(analyzed):
    message = web_plotter.coordinates_message(analyzed)
    ports = len(ss.specs.captures.split_capture_ports(analyzed.capture))
    assert message['num_ports'] == ports
    names = {f['name'] for fields in message['groups'].values() for f in fields}
    assert {'frequency_offset', 'sample_rate', 'duration'} <= names
    for fields in message['groups'].values():
        for field in fields:
            assert len(field['values']) == ports
    extra = {f['name'] for fields in message['extra_coords'].values() for f in fields}
    assert 'capture_index' in extra
    assert 'signal_trigger' not in extra


def test_messages_are_json_text(analyzed):
    from striqt.figures.backend import PlotlyBackend

    state = preview_state(looped_sweep())
    state.dataset = analyzed
    state.available_variables = ['spectrogram']
    plotter = PlotlyBackend(web_plotter._default_plot_options().plotter)
    figures = web_plotter._prepare_plot_data_sync(analyzed, plotter, 'spectrogram', {})

    plot = json.loads(
        web_plotter._serialize_plot_message(
            figures, web_plotter.coordinates_message(analyzed), state
        )
    )
    assert plot['type'] == 'plot_data'
    assert len(plot['data']) == len(figures)
    assert plot['available_variables'] == ['spectrogram']

    status = json.loads(web_plotter._encode(web_plotter.state_message(state)))
    assert status['type'] == 'state'
    assert status['capture_options']['loops'][0]['field'] == 'snr'
    assert status['capture_selection'] == {'capture': 0, 'loops': [0]}
    assert status['coordinates']['num_ports'] == 1


# %% websocket endpoint


def test_websocket_serves_the_state_the_plot_and_a_selection(analyzed, monkeypatch):
    """the handler answers ``get_state`` with the state and then the plot of the
    stored dataset, and answers ``select_captures`` with the updated state"""
    from fastapi.testclient import TestClient

    from striqt.figures.backend import PlotlyBackend

    state = web_plotter.STATE
    configured = preview_state(looped_sweep())
    for name in ('sweep', 'choices'):
        monkeypatch.setattr(state, name, getattr(configured, name))
    monkeypatch.setattr(state, 'dataset', analyzed)
    monkeypatch.setattr(
        state, 'plotter', PlotlyBackend(web_plotter._default_plot_options().plotter)
    )
    monkeypatch.setattr(state, 'available_variables', ['spectrogram'])
    monkeypatch.setattr(state, 'selected_variable', 'spectrogram')
    monkeypatch.setattr(state, 'data_select', {})
    monkeypatch.setattr(state, 'selected_capture', configured.selected_capture)
    monkeypatch.setattr(state, 'selected_loops', configured.selected_loops)
    monkeypatch.setattr(state, 'websockets', set())
    monkeypatch.setattr(state, 'running', True)
    monkeypatch.setattr(state, 'stop', threading.Event())

    with TestClient(web_plotter.app) as client, client.websocket_connect('/ws') as ws:
        ws.send_json({'type': 'get_state'})
        status = ws.receive_json()
        plot = ws.receive_json()
        ws.send_json({'type': 'select_captures', 'capture': 1, 'loops': [0]})
        updated = ws.receive_json()
        ws.send_json({'type': 'set_running', 'running': False})
        paused = ws.receive_json()

    assert status['type'] == 'state'
    assert status['capture_options']['loops'][0]['field'] == 'snr'
    assert plot['type'] == 'plot_data'
    assert plot['selected_variable'] == 'spectrogram'
    assert len(plot['data']) >= 1
    assert updated['type'] == 'state'
    assert updated['capture_selection'] == {'capture': 1, 'loops': [0]}
    assert updated['running'] is True
    assert state.selected_loops == {('capture', 'snr'): SNR_LOOP.values[0]}
    assert paused['type'] == 'state'
    assert paused['running'] is False
    assert state.running is False


# %% serving_urls


def test_serving_urls_of_a_wildcard_bind_start_with_loopback_and_the_host_name():
    import socket

    urls = web_plotter.serving_urls('0.0.0.0', 8000)
    assert urls[:2] == ['http://127.0.0.1:8000', f'http://{socket.gethostname()}:8000']
    assert len(urls) == len(set(urls))
    assert all(url.endswith(':8000') for url in urls)


def test_serving_urls_of_an_explicit_host_is_that_host():
    assert web_plotter.serving_urls('192.0.2.7', 8001) == ['http://192.0.2.7:8001']
