"""sink path formatting: get_format_fields, get_path_fields, PathFormatter"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from sweep_strategies import make_sweep

import striqt.sensor as ss

H = ss.specs.helpers
PROPERTY = settings(
    suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None
)


def test_format_fields_are_extracted_in_order():
    assert H.get_format_fields('a{x}b{y!r}c{z:04d}{{}}') == ['x', 'y', 'z']
    assert H.get_format_fields('plain') == []


@given(names=st.lists(st.from_regex(r'\A[a-z][a-z0-9_]{0,8}\Z'), max_size=5))
@PROPERTY
def test_format_fields_roundtrip(names):
    template = ''.join(f'{{{name}}}' for name in names)
    assert H.get_format_fields(template) == names


def test_path_fields_with_spec_path(cw_sweep, cw_spec_path):
    fields = H.get_path_fields(cw_sweep, source_id='abcd', spec_path=cw_spec_path)
    assert set(fields) == {
        'start_time',
        'sensor_binding',
        'spec_name',
        'parent_name',
        'source_id',
    }
    assert fields['sensor_binding'] == 'single_tone'
    assert fields['spec_name'] == 'cw-cpu'
    assert fields['parent_name'] == 'sweeps'
    assert fields['source_id'] == 'abcd'
    assert re.fullmatch(r'\d{8}-\d{2}h\d{2}m\d{2}', fields['start_time'])


def test_path_fields_without_spec_path(cw_sweep):
    fields = H.get_path_fields(cw_sweep, source_id='abcd')
    assert set(fields) == {'start_time', 'sensor_binding', 'source_id'}


def test_path_fields_accept_a_callable_source_id(cw_sweep):
    fields = H.get_path_fields(cw_sweep, source_id=lambda: 'abcd')
    assert fields['source_id'] == 'abcd'


def test_path_fields_include_fixed_adjustments_only():
    remap = ss.specs.CaptureRemap(key='frequency_offset', lookup={100: 1.0})
    sweep = make_sweep(adjust_captures={'defaults': {'lo_shift': 'left', 'snr': remap}})
    fields = H.get_path_fields(sweep, source_id='abcd')
    assert fields['lo_shift'] == 'left'
    assert 'snr' not in fields


def test_path_fields_are_cached_per_argument_set(cw_sweep):
    # the lru_cache also freezes start_time for repeated identical arguments
    first = H.get_path_fields(cw_sweep, source_id='cafe')
    assert H.get_path_fields(cw_sweep, source_id='cafe') is first


def test_formatter_passes_through_paths_without_fields(cw_sweep, monkeypatch):
    def fail(*args, **kws):
        raise AssertionError('lookup.id must not be called')

    monkeypatch.setattr(ss.lib.controller.lookup, 'id', fail)
    assert H.PathFormatter(cw_sweep)('outputs/noformat.zarr') == 'outputs/noformat.zarr'


def test_formatter_substitutes_fields(cw_sweep, cw_spec_path, fake_source_id):
    formatter = H.PathFormatter(cw_sweep, spec_path=cw_spec_path)
    result = formatter('outputs/{spec_name}-{source_id}.zarr')
    assert result == f'outputs/cw-cpu-{fake_source_id}.zarr'


def test_formatter_expands_user(cw_sweep, cw_spec_path, fake_source_id):
    result = H.PathFormatter(cw_sweep, spec_path=cw_spec_path)('~/{parent_name}')
    assert result == str(Path.home() / 'sweeps')


def test_formatter_names_the_unknown_field_and_the_allowed_ones(
    cw_sweep, fake_source_id
):
    with pytest.raises(KeyError, match="'nope'") as info:
        H.PathFormatter(cw_sweep)('{nope}')
    assert 'source_id' in str(info.value)


def test_formatter_resolves_spec_path(cw_sweep):
    relative = 'tests/sensor/sweeps/cw-cpu.yaml'
    assert H.PathFormatter(cw_sweep, spec_path=relative).spec_path.is_absolute()
    assert H.PathFormatter(cw_sweep).spec_path is None


@pytest.mark.xfail(
    strict=True,
    reason="Sink.path defaults to '{yaml_name}-{start_time}' but get_path_fields "
    'provides spec_name, not yaml_name',
)
def test_formatter_accepts_the_default_sink_path(
    cw_sweep, cw_spec_path, fake_source_id
):
    result = H.PathFormatter(cw_sweep, spec_path=cw_spec_path)(ss.specs.Sink().path)
    assert 'cw-cpu-' in result
