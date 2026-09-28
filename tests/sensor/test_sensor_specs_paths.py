"""striqt.sensor.specs.paths: sink path formatting (get_format_fields,
get_path_fields, PathFormatter)

The site-shaped cases use the extension binding in sweeps/src/extensions.py and the
override files in sweeps/sites*, mirroring the downstream sensor configuration.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from conftest import SITE_SPEC
from hypothesis import given
from hypothesis import strategies as st
from site_strategies import OTHER_ID, RADIO_ID
from sweep_strategies import make_sweep

import striqt.sensor as ss

P = ss.specs.paths
Remap = ss.specs.CaptureRemap


# %% get_format_fields, get_path_fields and PathFormatter


def test_format_fields_are_extracted_in_order():
    assert P.get_format_fields('a{x}b{y!r}c{z:04d}{{}}') == ['x', 'y', 'z']
    assert P.get_format_fields('plain') == []


@given(names=st.lists(st.from_regex(r'\A[a-z][a-z0-9_]{0,8}\Z'), max_size=5))
def test_format_fields_roundtrip(names):
    template = ''.join(f'{{{name}}}' for name in names)
    assert P.get_format_fields(template) == names


def test_path_fields_with_spec_path(synthetic_sweep, synthetic_spec_path):
    fields = P.get_path_fields(
        synthetic_sweep, source_id='abcd', spec_path=synthetic_spec_path
    )
    assert set(fields) == {
        'start_time',
        'sensor_binding',
        'spec_name',
        'parent_name',
        'source_id',
    }
    assert fields['sensor_binding'] == 'single_tone'
    assert fields['spec_name'] == 'synthetic'
    assert fields['parent_name'] == 'sweeps'
    assert fields['source_id'] == 'abcd'
    assert re.fullmatch(r'\d{8}-\d{2}h\d{2}m\d{2}', fields['start_time'])


def test_path_fields_without_spec_path(synthetic_sweep):
    fields = P.get_path_fields(synthetic_sweep, source_id='abcd')
    assert set(fields) == {'start_time', 'sensor_binding', 'source_id'}


def test_path_fields_accept_a_callable_source_id(synthetic_sweep):
    fields = P.get_path_fields(synthetic_sweep, source_id=lambda: 'abcd')
    assert fields['source_id'] == 'abcd'


def test_path_fields_include_fixed_adjustments_only():
    remap = Remap(key='frequency_offset', lookup={100: 1.0})
    sweep = make_sweep(adjust_captures={'defaults': {'lo_shift': 'left', 'snr': remap}})
    fields = P.get_path_fields(sweep, source_id='abcd')
    assert fields['lo_shift'] == 'left'
    assert 'snr' not in fields


def test_path_fields_only_string_fixed_values_from_site_files(site_sweep):
    # documented as current: numeric and null fixed values are not formattable
    fields = P.get_path_fields(site_sweep, source_id=RADIO_ID)
    assert fields['radio_name'] == 'radio02'
    assert fields['site_name'] == 'WAPA-north'
    assert not {'gain', 'mast_height', 'azimuth_offset'} & set(fields)
    assert 'mast_height' not in P.get_path_fields(site_sweep, source_id=OTHER_ID)


def test_formatter_passes_through_paths_without_fields(synthetic_sweep, monkeypatch):
    def fail(*args, **kws):
        raise AssertionError('lookup.id must not be called')

    monkeypatch.setattr(ss.lib.controller.lookup, 'id', fail)
    formatter = P.PathFormatter(synthetic_sweep)
    assert formatter('outputs/noformat.zarr') == 'outputs/noformat.zarr'


def test_formatter_substitutes_fields(
    synthetic_sweep, synthetic_spec_path, fake_source_id
):
    formatter = P.PathFormatter(synthetic_sweep, spec_path=synthetic_spec_path)
    result = formatter('outputs/{spec_name}-{source_id}.zarr')
    assert result == f'outputs/synthetic-{fake_source_id}.zarr'


def test_formatter_expands_user(synthetic_sweep, synthetic_spec_path, fake_source_id):
    result = P.PathFormatter(synthetic_sweep, spec_path=synthetic_spec_path)(
        '~/{parent_name}'
    )
    assert result == str(Path.home() / 'sweeps')


def test_formatter_names_the_unknown_field_and_the_allowed_ones(
    synthetic_sweep, fake_source_id
):
    with pytest.raises(KeyError, match="'nope'") as info:
        P.PathFormatter(synthetic_sweep)('{nope}')
    assert 'source_id' in str(info.value)


def test_formatter_accepts_the_default_sink_path(
    synthetic_sweep, synthetic_spec_path, fake_source_id
):
    result = P.PathFormatter(synthetic_sweep, spec_path=synthetic_spec_path)(
        ss.specs.Sink().path
    )
    assert 'synthetic-' in result


def test_formatter_fills_the_site_sink_template(site_sweep, fake_radio_id):
    path = P.PathFormatter(site_sweep, spec_path=SITE_SPEC)(site_sweep.sink.path)
    assert path.startswith('../outputs/WAPA-north/site-cpu_site_')
    assert path.endswith('.zarr.zip')
