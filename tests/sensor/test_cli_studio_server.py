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
    options = server.capture_options(state.open.choices)
    assert len(options['captures']) == len(TONE_OFFSETS)
    assert [loop['field'] for loop in options['loops']] == ['snr']
    assert options['loops'][0]['labels'] == list(state.open.choices.loops[0].labels)


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
    names = {f['name'] for f in message['fields']}
    assert {'frequency_offset', 'sample_rate', 'duration'} <= names
    for field in message['fields']:
        assert len(field['values']) == ports
    assert 'extra_coords' not in message


def test_coordinates_message_keeps_the_varying_fields_only(analyzed):
    message = server.coordinates_message(analyzed, frozenset({'frequency_offset'}))
    assert [f['name'] for f in message['fields']] == ['frequency_offset']


def test_state_message_encodes_to_json_text(analyzed):
    state = preview_session(looped_sweep(), available_variables=('spectrogram',))
    state.dataset = analyzed
    status = json.loads(server._encode(server.state_message(state)))
    assert status['type'] == 'state'
    assert status['coordinates']['num_ports'] == 1
    assert status['run']['sink_path'] is None
    assert status['validation'] == {'ok': True, 'errors': []}


# %% websocket endpoint


def test_websocket_serves_the_state_the_plot_and_a_selection(analyzed, monkeypatch):
    """the handler answers ``get_state`` with the state and then the plot of the
    stored dataset, and answers ``select_captures`` with the updated state"""
    from fastapi.testclient import TestClient

    state = server.SESSION
    configured = preview_session(looped_sweep(), available_variables=('spectrogram',))
    monkeypatch.setattr(state, 'open', configured.open)
    monkeypatch.setattr(state, 'applied_spec', configured.open.sweep)
    monkeypatch.setattr(state, 'dataset', analyzed)
    monkeypatch.setattr(state, 'selected_variable', 'spectrogram')
    monkeypatch.setattr(state, 'selected_capture', configured.selected_capture)
    monkeypatch.setattr(state, 'selected_loops', configured.selected_loops)
    monkeypatch.setattr(server.CLIENTS, 'websockets', set())
    monkeypatch.setattr(state, 'preview_enabled', True)
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
    assert state.preview_enabled is False


# %% serving_urls


def test_serving_urls_of_a_wildcard_bind_start_with_loopback_and_the_host_name():
    import socket

    urls = server.serving_urls('0.0.0.0', 8000)
    assert urls[:2] == ['http://127.0.0.1:8000', f'http://{socket.gethostname()}:8000']
    assert len(urls) == len(set(urls))
    assert all(url.endswith(':8000') for url in urls)


def test_serving_urls_of_an_explicit_host_is_that_host():
    assert server.serving_urls('192.0.2.7', 8001) == ['http://192.0.2.7:8001']


# %% edits over the websocket


def test_edit_writes_the_file_and_reports_the_validation(sweeps_copy, monkeypatch):
    from fastapi.testclient import TestClient

    from striqt.analysis.lib import yaml as spec_yaml

    path = sweeps_copy / 'synthetic.yaml'
    spec = ss.read_yaml_spec(str(path))
    state = server.SESSION
    monkeypatch.setattr(state, 'spec_path', str(path))
    monkeypatch.setattr(state, 'applied_spec', spec)
    monkeypatch.setattr(state, 'edited_spec', spec)
    monkeypatch.setattr(state, 'provenance', spec_yaml.load_yaml_provenance(path))
    monkeypatch.setattr(state, 'validation_errors', [])
    monkeypatch.setattr(server.CLIENTS, 'websockets', set())

    with TestClient(server.app) as client, client.websocket_connect('/ws') as ws:
        ws.send_json({
            'type': 'edit',
            'edits': [
                {'pointer': '/captures/0/duration', 'op': 'set', 'value_text': '3e-3'}
            ],
        })
        spec_msg = ws.receive_json()
        status = ws.receive_json()
        reread = ss.read_yaml_spec(str(path))
        ws.send_json({
            'type': 'edit',
            'edits': [
                {'pointer': '/captures/0/duration', 'op': 'set', 'value_text': '-1'}
            ],
        })
        bad_spec = ws.receive_json()
        bad_status = ws.receive_json()
        ws.send_json({'type': 'apply'})
        refused = ws.receive_json()
        after_refusal = ws.receive_json()

    assert spec_msg['type'] == 'spec'
    # YAML 1.1 reads `3e-3` as text; msgspec converts the text when decoding the spec
    assert spec_msg['tree']['captures'][0]['duration'] == '3e-3'
    assert spec_msg['files'][0] == 'synthetic.yaml'
    assert spec_msg['origin']['/captures/0/duration'][0] == 0
    assert status['validation'] == {'ok': True, 'errors': []}
    assert status['applied_differs'] is True
    assert reread.captures[0].duration == 3e-3

    assert bad_spec['tree']['captures'][0]['duration'] == -1
    assert bad_status['validation']['ok'] is False
    (error,) = bad_status['validation']['errors']
    assert '/captures/0' in [error['path'], *error['locations']]
    assert refused == {
        'type': 'error',
        'message': 'the spec files do not validate',
        'command': 'apply',
    }
    assert after_refusal['type'] == 'state'


# %% spec message


def test_spec_message_of_a_site_sweep_encodes_with_relative_paths():
    from conftest import SITE_DIR

    from striqt.analysis.lib import yaml as spec_yaml

    path = SITE_DIR / 'site-cpu.yaml'
    state = session.StudioSession(spec_path=str(path))
    state.provenance = spec_yaml.load_yaml_provenance(path)

    message = json.loads(server._encode(server.spec_message(state)))

    assert message['files'][0] == 'site-cpu.yaml'
    assert '../fragments/source/tone.yaml' in message['files']
    assert message['regions']['/source'] == [
        message['files'].index('../fragments/source/tone.yaml'),
        message['files'].index('../fragments/source/tone_unsync.yaml'),
    ]
    # a sequence-valued lookup key arrives as text
    lookups = message['tree']['adjust_captures']
    keys = {
        key
        for block in lookups.values()
        for remap in block.values()
        if isinstance(remap, dict)
        for key in remap.get('lookup', {})
    }
    assert all(isinstance(key, str) for key in keys)


# %% revert over the websocket


def test_revert_restores_the_applied_section(sweeps_copy, monkeypatch):
    import copy

    from fastapi.testclient import TestClient

    from striqt.analysis.lib import yaml as spec_yaml

    path = sweeps_copy / 'synthetic.yaml'
    original = path.read_text()
    spec = ss.read_yaml_spec(str(path))
    state = server.SESSION
    monkeypatch.setattr(state, 'spec_path', str(path))
    monkeypatch.setattr(state, 'applied_spec', spec)
    monkeypatch.setattr(state, 'edited_spec', spec)
    monkeypatch.setattr(state, 'provenance', spec_yaml.load_yaml_provenance(path))
    monkeypatch.setattr(state, 'applied_tree', copy.deepcopy(state.provenance.tree))
    monkeypatch.setattr(state, 'validation_errors', [])
    monkeypatch.setattr(server.CLIENTS, 'websockets', set())

    with TestClient(server.app) as client, client.websocket_connect('/ws') as ws:
        ws.send_json({
            'type': 'edit',
            'edits': [
                {'pointer': '/captures/0/duration', 'op': 'set', 'value_text': '3e-3'}
            ],
        })
        ws.receive_json()
        edited = ws.receive_json()
        ws.send_json({'type': 'revert', 'section': 'captures'})
        ws.receive_json()
        reverted = ws.receive_json()

    assert edited['changed_sections'] == ['captures']
    assert reverted['changed_sections'] == []
    assert reverted['applied_differs'] is False
    assert path.read_text() == original
