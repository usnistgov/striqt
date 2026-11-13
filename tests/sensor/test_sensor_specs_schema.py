"""striqt.sensor.specs.schema: the editor schema of action spec classes"""

from __future__ import annotations

from typing import Optional

import msgspec

import striqt.analysis as sa
import striqt.sensor as ss
from striqt.sensor.specs import schema
from striqt.sensor.specs.types import PerPort


def qualname(cls: type) -> str:
    return f'{cls.__module__}.{cls.__qualname__}'


def by_name(fields) -> dict:
    return {f.name: f for f in fields}


def sections(spec_cls: type) -> dict:
    return {s.name: s for s in schema.spec_schema(spec_cls).sections}


class SiteCapture(ss.specs.SingleToneCapture, frozen=True, kw_only=True):
    antenna_model: Optional[PerPort[str]] = None


# %% field_owners


def test_field_owners_of_soapy_capture():
    owners = schema.field_owners(ss.specs.SoapyCapture)
    assert owners['duration'] is sa.specs.structs.Capture
    assert owners['port'] is ss.specs.SensorCapture
    assert owners['gain'] is ss.specs.SoapyCapture
    assert list(owners) == list(ss.specs.SoapyCapture.__struct_fields__)


def test_field_owners_of_a_downstream_capture():
    owners = schema.field_owners(SiteCapture)
    assert owners['antenna_model'] is SiteCapture
    assert len(set(owners.values())) == 4


# %% struct_schema


def test_soapy_capture_per_port_fields():
    fields = by_name(schema.struct_schema(ss.specs.SoapyCapture).fields)
    per_port = {name for name, f in fields.items() if f.per_port}
    assert per_port == {'port', 'center_frequency', 'gain', 'external_lo_frequency'}


def test_soapy_capture_owners():
    fields = by_name(schema.struct_schema(ss.specs.SoapyCapture).fields)
    assert fields['duration'].owner == qualname(sa.specs.structs.Capture)
    assert fields['gain'].owner == qualname(ss.specs.SoapyCapture)


def test_soapy_capture_field_kinds():
    fields = by_name(schema.struct_schema(ss.specs.SoapyCapture).fields)

    assert fields['lo_shift'].kind == 'literal'
    assert fields['lo_shift'].choices == ('left', 'right', 'none')
    assert fields['lo_shift'].default == 'none'

    assert fields['backend_sample_rate'].kind == 'number'
    assert fields['backend_sample_rate'].optional
    assert not fields['backend_sample_rate'].required
    assert fields['backend_sample_rate'].gt == 0

    assert fields['center_frequency'].kind == 'number'
    assert fields['center_frequency'].units == 'Hz'
    assert fields['center_frequency'].required
    assert fields['center_frequency'].default is None
    assert fields['center_frequency'].gt == 0

    assert fields['external_lo_frequency'].per_port
    assert fields['external_lo_frequency'].optional
    assert fields['external_lo_frequency'].kind == 'number'

    assert fields['adjust_analysis'].kind == 'opaque'
    assert fields['adjust_analysis'].read_only

    assert fields['port'].kind == 'integer'
    assert fields['port'].ge == 0
    assert fields['host_resample'].kind == 'boolean'
    assert fields['analysis_bandwidth'].default == 'inf'


def test_description_is_set_only_when_it_differs_from_standard_name():
    capture = by_name(schema.struct_schema(ss.specs.SoapyCapture).fields)
    assert capture['center_frequency'].standard_name == 'Center frequency for each port'
    assert capture['center_frequency'].description is None
    assert capture['host_resample'].standard_name is None
    assert capture['host_resample'].description is None

    cal = by_name(schema.struct_schema(ss.specs.ManualYFactorSpec).fields)
    assert cal['enr'].standard_name == 'Excess noise ratio'
    assert cal['enr'].description.startswith('Excess noise ratio, referred to')


def test_downstream_optional_per_port_string():
    fields = by_name(schema.struct_schema(SiteCapture).fields)
    f = fields['antenna_model']
    assert f.owner == qualname(SiteCapture)
    assert f.per_port
    assert f.optional
    assert f.kind == 'string'
    assert not f.required


def test_struct_schema_of_a_tagged_loop():
    s = schema.struct_schema(ss.specs.Range)
    assert s.cls == qualname(ss.specs.Range)
    assert s.tag_value == 'range'
    assert [f.name for f in s.fields] == ['field', 'isin', 'start', 'stop', 'step']


def test_struct_schema_of_a_fraction_and_fixed_tuple():
    cal = by_name(schema.struct_schema(ss.specs.ManualYFactorSpec).fields)
    assert cal['detector_period'].kind == 'fraction'
    assert cal['detector_period'].optional
    assert cal['implied_loops'].kind == 'list'
    assert cal['implied_loops'].item.kind == 'string'
    assert cal['implied_loops'].length is None

    cell = by_name(schema.struct_schema(ss.specs.CellChannel).fields)
    guard = cell['channel_guard_bandwidths']
    assert guard.kind == 'list'
    assert guard.length == 2
    assert guard.item.kind == 'number'
    assert guard.default == [0.0, 0.0]


def test_struct_schema_of_a_source():
    src = by_name(schema.struct_schema(ss.specs.SoapySource).fields)
    assert src['signal_trigger'].kind == 'opaque'
    assert src['signal_trigger'].optional
    assert src['bursts'].kind == 'struct'
    assert [f.name for f in src['bursts'].fields] == ['on_gap', 'buffers', 'trigger_on']
    assert src['bursts'].default == {
        'on_gap': 'skip',
        'buffers': 1,
        'trigger_on': 'first',
    }
    assert src['adc_overload_limit'].kind == 'number'
    assert src['adc_overload_limit'].le == 0
    assert src['adc_overload_limit'].units == 'dBFS'
    assert src['adc_overload_limit'].optional


# %% field_schema


def test_field_schema_of_a_bool_literal_union():
    f = schema.field_schema(
        'average_rbs',
        sa.specs.types.CellularAverageRBs,
        owner=ss.specs.SoapyCapture,
        default=False,
        required=False,
    )
    assert f.kind == 'literal'
    assert f.choices == (True, False, 'half')
    assert f.default is False


def test_field_schema_of_a_window_union_is_opaque():
    f = schema.field_schema(
        'window',
        sa.specs.types.WindowType,
        owner=ss.specs.SoapyCapture,
        default='hamming',
        required=False,
    )
    assert f.kind == 'opaque'
    assert f.read_only


# %% spec_schema


def test_sweep_sections():
    secs = sections(ss.specs.Sweep)
    assert list(secs) == list(ss.specs.Sweep.__struct_fields__)

    assert secs['captures'].kind == 'struct_list'
    assert secs['captures'].schema.cls == qualname(ss.specs.SensorCapture)

    loops = secs['loops']
    assert loops.kind == 'tagged_list'
    assert loops.schema.tag == 'kind'
    assert [v.tag_value for v in loops.schema.variants] == [
        'repeat',
        'list',
        'range',
        'frequencybinrange',
    ]

    analysis = secs['analysis']
    assert analysis.kind == 'optional_struct_dict'
    assert analysis.schema.cls == qualname(sa.specs.BuiltinAnalyses)
    names = [f.name for f in analysis.schema.fields]
    assert names == list(sa.specs.BuiltinAnalyses.__struct_fields__)
    for f in analysis.schema.fields:
        assert f.kind == 'struct'
        assert f.optional
        assert f.fields

    sink = secs['sink']
    assert sink.kind == 'struct'
    store = by_name(sink.schema.fields)['store']
    assert store.kind == 'literal'
    assert store.choices == ('zip', 'directory')

    for name in ('adjust_captures', 'plot_hint'):
        assert secs[name].kind == 'opaque'
        assert secs[name].read_only

    for name in ('source', 'peripherals', 'description', 'extensions', 'options'):
        assert secs[name].kind == 'struct'


def test_unbound_sweep_has_no_binding():
    s = schema.spec_schema(ss.specs.Sweep)
    assert s.sensor_binding is None
    assert s.action_cls == qualname(ss.specs.Sweep)


def test_bound_calibration_sections():
    spec_cls = ss.bindings.air7101b_calibration.sensor.sweep_spec_cls
    s = schema.spec_schema(spec_cls)
    assert s.sensor_binding == 'air7101b_calibration'
    secs = {sec.name: sec for sec in s.sections}
    assert 'plot_hint' not in secs
    assert secs['calibration'].kind == 'struct'
    assert secs['calibration'].schema.cls == qualname(ss.specs.ManualYFactorSpec)
    assert secs['captures'].schema.cls == qualname(
        ss.specs.captures.get_capture_type(spec_cls)
    )
    assert secs['analysis'].kind == 'opaque'
    assert secs['mock_source'].kind == 'opaque'


def test_bound_cellular_sections():
    spec_cls = ss.bindings.air7101b_cellular.sensor.sweep_spec_cls
    secs = sections(spec_cls)
    channels = secs['cell_channels']
    assert channels.kind == 'dict'
    assert channels.schema.kind == 'struct'
    assert by_name(channels.schema.fields)['subcarrier_spacing'].units == 'Hz'
    assert secs['analysis'].kind == 'optional_struct_dict'
    assert secs['analysis'].schema.cls == qualname(ss.specs.Cellular5GNRAnalysis)


# %% schema_json


def test_schema_json_round_trips():
    j = schema.schema_json(ss.specs.Sweep)
    assert msgspec.json.decode(msgspec.json.encode(j)) == j
    assert j['action_cls'] == qualname(ss.specs.Sweep)
    assert [s['name'] for s in j['sections']] == list(ss.specs.Sweep.__struct_fields__)
