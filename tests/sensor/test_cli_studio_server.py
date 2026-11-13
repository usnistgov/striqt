"""striqt.cli.studio.server: the messages built from an analyzed capture and the
websocket endpoint
"""

from __future__ import annotations

import json
import threading

import plotly.graph_objects as go
import pytest
from studio_helpers import SNR_LOOP, analyzed_dataset, looped_sweep, preview_session
from synthetic_sources import TONE_OFFSETS

import striqt.sensor as ss
from striqt.cli.studio import server, session


@pytest.fixture(scope='module')
def analyzed():
    """the `DelayedDataset` of one spectrogram capture"""
    return analyzed_dataset()


# %% capture selection messages


def test_capture_options_list_one_entry_per_loop_and_capture():
    state = preview_session(looped_sweep())
    options = server.capture_options(state.choices)
    assert len(options['captures']) == len(TONE_OFFSETS)
    assert [loop['field'] for loop in options['loops']] == ['snr']
    assert options['loops'][0]['labels'] == list(state.choices.loops[0].labels)


def test_capture_selection_round_trips():
    state = preview_session(looped_sweep())
    assert server.capture_selection(state) == {'capture': 0, 'loops': [0]}
    session.apply_capture_selection(state, {'capture': 1, 'loops': [1]})
    assert server.capture_selection(state) == {'capture': 1, 'loops': [1]}
    session.apply_capture_selection(state, {'capture': None, 'loops': [None]})
    assert server.capture_selection(state) == {'capture': None, 'loops': [None]}


# %% plot and coordinate messages


def test_prepare_plot_data_returns_figures(analyzed):
    from striqt.figures.backend import PlotlyBackend

    plotter = PlotlyBackend(session._default_plot_options().plotter)
    figures = server._prepare_plot_data_sync(analyzed, plotter, 'spectrogram', {})
    assert len(figures) >= 1
    assert all(isinstance(fig, go.Figure) for fig in figures)
    assert all(len(fig.data) >= 1 for fig in figures)


def test_coordinates_message_has_one_value_per_port(analyzed):
    message = server.coordinates_message(analyzed)
    ports = len(ss.specs.captures.split_capture_ports(analyzed.capture))
    assert message['num_ports'] == ports
    names = {f['name'] for fields in message['groups'].values() for f in fields}
    assert {'frequency_offset', 'sample_rate', 'duration'} <= names
    for fields in message['groups'].values():
        for field in fields:
            assert len(field['values']) == ports
    assert 'extra_coords' not in message


def test_coordinates_message_keeps_the_varying_fields_only(analyzed):
    message = server.coordinates_message(analyzed, frozenset({'frequency_offset'}))
    names = [f['name'] for fields in message['groups'].values() for f in fields]
    assert names == ['frequency_offset']


def test_messages_are_json_text(analyzed):
    from striqt.figures.backend import PlotlyBackend

    state = preview_session(looped_sweep())
    state.dataset = analyzed
    state.available_variables = ['spectrogram']
    plotter = PlotlyBackend(session._default_plot_options().plotter)
    figures = server._prepare_plot_data_sync(analyzed, plotter, 'spectrogram', {})

    plot = json.loads(
        server._serialize_plot_message(
            figures, server.coordinates_message(analyzed), state
        )
    )
    assert plot['type'] == 'plot_data'
    assert len(plot['data']) == len(figures)
    assert plot['available_variables'] == ['spectrogram']

    status = json.loads(server._encode(server.state_message(state)))
    assert status['type'] == 'state'
    assert status['capture_options']['loops'][0]['field'] == 'snr'
    assert status['capture_selection'] == {'capture': 0, 'loops': [0]}
    assert status['coordinates']['num_ports'] == 1
    assert status['mode'] == 'closed'
    assert status['run']['sink_path'] is None
    assert status['validation'] == {'ok': True, 'errors': []}


# %% websocket endpoint


def test_websocket_serves_the_state_the_plot_and_a_selection(analyzed, monkeypatch):
    """the handler answers ``get_state`` with the state and then the plot of the
    stored dataset, and answers ``select_captures`` with the updated state"""
    from fastapi.testclient import TestClient

    from striqt.figures.backend import PlotlyBackend

    state = server.SESSION
    configured = preview_session(looped_sweep())
    for name in ('sweep', 'choices'):
        monkeypatch.setattr(state, name, getattr(configured, name))
    monkeypatch.setattr(state, 'applied_spec', configured.sweep)
    monkeypatch.setattr(state, 'dataset', analyzed)
    monkeypatch.setattr(
        state, 'plotter', PlotlyBackend(session._default_plot_options().plotter)
    )
    monkeypatch.setattr(state, 'available_variables', ['spectrogram'])
    monkeypatch.setattr(state, 'selected_variable', 'spectrogram')
    monkeypatch.setattr(state, 'data_select', {})
    monkeypatch.setattr(state, 'selected_capture', configured.selected_capture)
    monkeypatch.setattr(state, 'selected_loops', configured.selected_loops)
    monkeypatch.setattr(server.CLIENTS, 'websockets', set())
    monkeypatch.setattr(state, 'running', True)
    monkeypatch.setattr(state, 'stop', threading.Event())

    with TestClient(server.app) as client, client.websocket_connect('/ws') as ws:
        ws.send_json({'type': 'get_state'})
        status = ws.receive_json()
        schema = ws.receive_json()
        spec = ws.receive_json()
        plot = ws.receive_json()
        ws.send_json({'type': 'select_captures', 'capture': 1, 'loops': [0]})
        updated = ws.receive_json()
        ws.send_json({'type': 'set_running', 'running': False})
        paused = ws.receive_json()

    assert status['type'] == 'state'
    assert status['capture_options']['loops'][0]['field'] == 'snr'
    assert schema['type'] == 'schema'
    assert [s['name'] for s in schema['schema']['sections']][:2] == [
        'description',
        'source',
    ]
    assert spec == {
        'type': 'spec',
        'tree': None,
        'files': [],
        'origin': {},
        'regions': {},
    }
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

    urls = server.serving_urls('0.0.0.0', 8000)
    assert urls[:2] == ['http://127.0.0.1:8000', f'http://{socket.gethostname()}:8000']
    assert len(urls) == len(set(urls))
    assert all(url.endswith(':8000') for url in urls)


def test_serving_urls_of_an_explicit_host_is_that_host():
    assert server.serving_urls('192.0.2.7', 8001) == ['http://192.0.2.7:8001']
