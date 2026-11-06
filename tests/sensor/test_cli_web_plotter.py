"""striqt.cli.web_plotter: the analysis and capture selections of the preview, and the
messages built from an analyzed capture
"""

from __future__ import annotations

import json
import threading

import plotly.graph_objects as go
import pytest
from synthetic_sources import ANALYSIS, SPECTROGRAM, preset_sweep, tone_captures

import striqt.sensor as ss
from striqt.cli import web_plotter
from striqt.sensor.specs import sequencing

OFFSETS = (1e6, 2e6, 3e6)
SNR_LOOP = ss.specs.List(field='snr', values=(10.0, 20.0))


def looped_sweep(analysis=SPECTROGRAM):
    return preset_sweep(
        'single_tone',
        tone_captures(OFFSETS),
        analysis=analysis,
        loops=(ss.specs.Repeat(count=2), SNR_LOOP),
    )


def preview_state(sweep) -> web_plotter.PreviewState:
    state = web_plotter.PreviewState(sweep=sweep)
    state.choices = sequencing.plan_choices(sweep, None)
    return state


@pytest.fixture(scope='module')
def analyzed():
    """the `DelayedDataset` of one spectrogram capture"""
    sweep = preset_sweep(
        'single_tone', tone_captures(OFFSETS[:1]), analysis=SPECTROGRAM
    )
    with ss.open_resources(sweep, None) as res:
        results = ss.iterate_sweep(res, sink=ss.sinks.NoSink(sweep))
        return next(dd for dd in results if dd is not None)


# %% select_displayed


def test_select_displayed_narrows_to_the_selected_variable(monkeypatch):
    monkeypatch.setattr(web_plotter.STATE, 'selected_variable', 'iq_waveform')
    (capture,) = tone_captures(OFFSETS[:1])
    narrowed = web_plotter.select_displayed(capture, ANALYSIS)
    assert type(narrowed) is type(ANALYSIS)
    assert list(narrowed.to_dict()) == ['iq_waveform']
    assert narrowed.iq_waveform == ANALYSIS.iq_waveform


def test_select_displayed_passes_an_unknown_selection_through(monkeypatch):
    monkeypatch.setattr(web_plotter.STATE, 'selected_variable', 'spectrogram')
    (capture,) = tone_captures(OFFSETS[:1])
    assert web_plotter.select_displayed(capture, ANALYSIS) is ANALYSIS


# %% selected_captures


def test_selected_captures_follows_the_selection_at_the_next_item():
    sweep = looped_sweep()
    state = preview_state(sweep)
    plan = sequencing.loop_captures(sweep)
    items = web_plotter.selected_captures(state)

    assert next(items) == plan[0]
    assert next(items) == plan[1]

    state.selected_capture = 2
    only_third = sequencing.select_plan(sweep, None, capture=2)
    assert len(only_third) == len(SNR_LOOP.values)
    assert next(items) in only_third
    assert next(items) in only_third

    state.stop.set()
    with pytest.raises(StopIteration):
        next(items)


def test_selected_captures_waits_through_an_empty_selection():
    sweep = looped_sweep()
    state = preview_state(sweep)
    state.selected_loops = {('capture', 'snr'): 5}
    items = web_plotter.selected_captures(state)
    threading.Timer(0.3, state.stop.set).start()
    assert list(items) == []


# %% capture selection messages


def test_capture_options_list_one_entry_per_loop_and_capture():
    state = preview_state(looped_sweep())
    options = web_plotter.capture_options(state.choices)
    assert len(options['captures']) == len(OFFSETS)
    assert [loop['field'] for loop in options['loops']] == ['snr']
    assert options['loops'][0]['labels'] == list(state.choices.loops[0].labels)


def test_apply_capture_selection_round_trips():
    state = preview_state(looped_sweep())
    web_plotter.apply_capture_selection(state, {'capture': 1, 'loops': [1]})
    assert state.selected_capture == 1
    assert state.selected_loops == {('capture', 'snr'): SNR_LOOP.values[1]}
    assert web_plotter.capture_selection(state) == {'capture': 1, 'loops': [1]}

    web_plotter.apply_capture_selection(state, {'capture': None, 'loops': [None]})
    assert state.selected_capture is None
    assert state.selected_loops == {}
    assert web_plotter.capture_selection(state) == {'capture': None, 'loops': [None]}


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
    assert status['capture_selection'] == {'capture': None, 'loops': [None]}
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
    monkeypatch.setattr(state, 'selected_capture', None)
    monkeypatch.setattr(state, 'selected_loops', {})
    monkeypatch.setattr(state, 'websockets', set())
    monkeypatch.setattr(state, 'stop', threading.Event())

    with TestClient(web_plotter.app) as client, client.websocket_connect('/ws') as ws:
        ws.send_json({'type': 'get_state'})
        status = ws.receive_json()
        plot = ws.receive_json()
        ws.send_json({'type': 'select_captures', 'capture': 1, 'loops': [0]})
        updated = ws.receive_json()

    assert status['type'] == 'state'
    assert status['capture_options']['loops'][0]['field'] == 'snr'
    assert plot['type'] == 'plot_data'
    assert plot['selected_variable'] == 'spectrogram'
    assert len(plot['data']) >= 1
    assert updated['type'] == 'state'
    assert updated['capture_selection'] == {'capture': 1, 'loops': [0]}
    assert state.selected_loops == {('capture', 'snr'): SNR_LOOP.values[0]}
