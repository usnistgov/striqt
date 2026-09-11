"""adjust_captures shaped like downstream site override files, on the site binding"""

from __future__ import annotations

import msgspec
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from site_strategies import (
    ANTENNA_MODELS,
    CHANNEL_NAMES,
    OTHER_ID,
    RADIO_ID,
    SITE_ADJUST,
    Remap,
    SiteSweepCls,
    capture_dict,
    center_frequency_keys,
    make_site_capture,
    make_site_sweep,
)
from sweep_strategies import make_sweep

import striqt.sensor as ss

H = ss.specs.helpers
PROPERTY = settings(
    suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None
)


def first_capture(sweep, source_id, **capture_kws):
    if capture_kws:
        sweep = sweep.replace(captures=(make_site_capture(**capture_kws),))
    return H.loop_captures(sweep, source_id=source_id)[0]


def test_site_fields_are_accepted_by_the_bound_capture_class():
    sweep = make_site_sweep(adjust_captures=SITE_ADJUST)
    lookup = sweep.adjust_captures['defaults']['channel_name'].lookup
    assert set(map(type, lookup)) == {float}
    assert SiteSweepCls.from_dict(sweep.to_dict()) == sweep


def test_yaml_and_direct_adjustments_agree(site_sweep):
    direct = make_site_sweep(adjust_captures=SITE_ADJUST)
    assert site_sweep.adjust_captures == direct.adjust_captures


def test_builtin_binding_rejects_site_fields():
    match = (
        "adjust_captures field 'channel_name' was not defined in capture class "
        "'striqt.sensor.specs.SingleToneCapture'"
    )
    with pytest.raises(msgspec.ValidationError, match=match):
        make_sweep(adjust_captures={'defaults': {'channel_name': 'x'}})


def test_source_block_chains_through_defaults(site_sweep):
    c = first_capture(site_sweep, RADIO_ID, port=(0, 1))
    assert c.radio_name == 'radio02'
    assert c.site_name == 'WAPA-north'
    assert c.antenna_name == ('Omni', '1x32')
    assert c.antenna_polarization == ('Vertical', 'Linear +45')
    assert c.antenna_model == ('OmniModel', 'PanelModel')
    assert c.antenna_index == (0, 1)
    assert c.channel_name == '3750 MHz'
    assert c.gain == 0
    assert c.mast_height == pytest.approx(1.7)
    assert c.azimuth_offset == 150


def test_source_override_of_a_defaults_alias_keeps_its_evaluation_position():
    # the source block lists antenna_model before antenna_name, but antenna_name
    # was declared first in defaults, so the merged field order still resolves it first
    adjust = {
        'defaults': {
            'antenna_name': 'Unspecified',
            'antenna_model': Remap(key='antenna_name', lookup=ANTENNA_MODELS),
        },
        RADIO_ID: {
            'antenna_model': Remap(key='antenna_name', lookup=ANTENNA_MODELS),
            'antenna_name': Remap(key='port', lookup={0: 'Omni'}),
        },
    }
    sweep = make_site_sweep(adjust_captures=adjust)
    assert first_capture(sweep, RADIO_ID).antenna_model == 'OmniModel'


def test_adjustments_override_explicit_capture_values(site_sweep):
    c = first_capture(
        site_sweep, RADIO_ID, port=(0, 1), antenna_polarization=('V', 'H')
    )
    assert c.antenna_polarization == ('Vertical', 'Linear +45')


def test_unknown_source_uses_defaults_only(site_sweep):
    c = first_capture(site_sweep, 'deadbeef0000')
    assert c.antenna_name == 'Unspecified'
    assert c.antenna_polarization == 'Unspecified'
    assert c.channel_name == '3750 MHz'
    assert c.gain == -10
    assert c.radio_name == ''


def test_scalar_miss_returns_the_null_default(site_sweep):
    c = first_capture(site_sweep, RADIO_ID, center_frequency=3960e6)
    assert c.channel_name is None


@given(keys=st.lists(center_frequency_keys, min_size=2, max_size=2))
@PROPERTY
def test_per_port_hits_map_elementwise(site_sweep, keys):
    capture = capture_dict(port=tuple(range(len(keys))), center_frequency=tuple(keys))
    result = H.adjust_captures(capture, site_sweep.adjust_captures, RADIO_ID)
    assert result['channel_name'] == tuple(CHANNEL_NAMES[k] for k in keys)


@pytest.mark.xfail(
    strict=True,
    raises=KeyError,
    reason='the per-port branch of do_lookup indexes the lookup directly, bypassing '
    '`default`; this is how per-port center_frequency captures fail downstream',
)
def test_per_port_miss_uses_the_default(site_sweep):
    capture = capture_dict(port=(0, 1), center_frequency=(7350e6, 7350e6))
    result = H.adjust_captures(capture, site_sweep.adjust_captures, RADIO_ID)
    assert result['channel_name'] == (None, None)


def test_per_port_miss_raises_key_error(site_sweep):
    # documented as current; see the xfail above
    capture = capture_dict(port=(0, 1), center_frequency=(7350e6, 7350e6))
    with pytest.raises(KeyError):
        H.adjust_captures(capture, site_sweep.adjust_captures, RADIO_ID)


@pytest.mark.parametrize(
    'center_frequency, gain', [(3750e6, 0), (3900e6, -10)], ids=['hit', 'miss']
)
def test_single_element_tuple_key_remap_is_optional(site_sweep, center_frequency, gain):
    c = first_capture(site_sweep, OTHER_ID, center_frequency=center_frequency)
    assert c.gain == gain


def test_multi_field_key_with_per_port_values(site_sweep):
    c = first_capture(site_sweep, OTHER_ID, port=(0, 1), switch_input=1)
    assert c.antenna_index == (2, 3)


def test_remap_keyed_on_an_unknown_field_is_dropped():
    # documented as current; see the xfail below
    remap = Remap(key='centre_frequency', lookup={1.0: 'x'})
    sweep = make_site_sweep(adjust_captures={'defaults': {'channel_name': remap}})
    assert dict(sweep.adjust_captures['defaults']) == {}


@pytest.mark.xfail(
    strict=True,
    reason='_convert_label_lookup_keys prunes remaps keyed on unknown fields with a '
    '`continue` that precedes the intended raise',
)
def test_remap_keyed_on_an_unknown_field_is_rejected():
    remap = Remap(key='centre_frequency', lookup={1.0: 'x'})
    with pytest.raises(msgspec.ValidationError, match='centre_frequency'):
        make_site_sweep(adjust_captures={'defaults': {'channel_name': remap}})


@pytest.mark.xfail(
    strict=True,
    raises=msgspec.ValidationError,
    reason='after a fixed value or remap is processed its lookup type is forced to '
    'str, so remaps keyed on numeric aliases reject numeric lookup keys',
)
def test_remap_keyed_on_a_fixed_numeric_alias_accepts_numeric_keys():
    adjust = {
        'defaults': {
            'switch_input': 1,
            'antenna_index': Remap(key='switch_input', lookup={1: 7}),
        }
    }
    sweep = make_site_sweep(adjust_captures=adjust)
    assert first_capture(sweep, None).antenna_index == 7


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason='remaps are evaluated in declaration order, so a remap keyed on an alias '
    'declared after it sees the unadjusted capture value and is silently omitted',
)
def test_chained_remap_declared_before_its_key_resolves():
    adjust = {
        'defaults': {
            'antenna_model': Remap(key='antenna_name', lookup=ANTENNA_MODELS),
            'antenna_name': Remap(key='port', lookup={0: 'Omni'}),
        }
    }
    sweep = make_site_sweep(adjust_captures=adjust)
    result = H.adjust_captures(capture_dict(), sweep.adjust_captures, None)
    assert result.get('antenna_model') == 'OmniModel'


def test_remaps_see_loop_values_given_as_numbers(site_sweep):
    loops = (ss.specs.List(field='center_frequency', values=(3750e6, 3900e6)),)
    sweep = site_sweep.replace(loops=loops)
    captures = H.loop_captures(sweep, source_id=RADIO_ID)
    assert [c.channel_name for c in captures] == ['3750 MHz', '3900 MHz']


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason='loop points are coerced to the field type only after adjust_captures has '
    'run, so remaps keyed on a looped field see the raw YAML strings and miss',
)
def test_remaps_see_loop_values_given_as_yaml_strings(site_sweep):
    loops = [
        {'kind': 'list', 'field': 'center_frequency', 'values': ['3750e6', '3900e6']}
    ]
    sweep = SiteSweepCls.from_dict({**site_sweep.to_dict(), 'loops': loops})
    captures = H.loop_captures(sweep, source_id=RADIO_ID)
    assert [c.channel_name for c in captures] == ['3750 MHz', '3900 MHz']


def test_empty_string_source_key_is_valid_hex():
    # documented as current: the built-in function sources report '' as their id
    sweep = make_site_sweep(adjust_captures={'': {'radio_name': 'anon'}})
    result = H.adjust_captures(capture_dict(), sweep.adjust_captures, '')
    assert result == {'radio_name': 'anon'}


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason='the error text says "global" but the accepted key is "defaults"',
)
def test_source_key_error_names_defaults():
    with pytest.raises(msgspec.ValidationError, match='defaults'):
        make_site_sweep(adjust_captures={'zz': {'radio_name': 'x'}})


def test_only_string_fixed_values_are_path_fields(site_sweep):
    # documented as current: numeric and null fixed values are not formattable
    fields = H.get_path_fields(site_sweep, source_id=RADIO_ID)
    assert fields['radio_name'] == 'radio02'
    assert fields['site_name'] == 'WAPA-north'
    assert not {'gain', 'mast_height', 'azimuth_offset'} & set(fields)
    assert 'mast_height' not in H.get_path_fields(site_sweep, source_id=OTHER_ID)


def test_path_formatter_fills_the_site_sink_template(
    site_sweep, site_spec_path, fake_radio_id
):
    path = H.PathFormatter(site_sweep, spec_path=site_spec_path)(site_sweep.sink.path)
    assert path.startswith('../outputs/WAPA-north/site-cpu_site_')
    assert path.endswith('.zarr.zip')


def test_list_capture_adjustments_for_a_radio(site_sweep):
    listed = H.list_capture_adjustments(site_sweep, RADIO_ID)
    assert listed['radio_name'] == ('radio02',)
    assert listed['antenna_name'] == (('Omni', '1x32'),)
    assert listed['channel_name'] == ('3750 MHz',)
