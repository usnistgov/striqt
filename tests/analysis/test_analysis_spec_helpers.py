"""Meta, the msgspec conversion hooks, json_schema and coordinate inference"""

from __future__ import annotations

import fractions
from typing import Any, Optional

import msgspec
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from msgspec import inspect as mi

import striqt.analysis as sa
import striqt.sensor as ss
from striqt.analysis.specs.helpers import (
    Meta,
    convert_dict,
    convert_spec,
    frozendict,
    get_capture_type_attrs,
    infer_coord_info,
    json_schema,
)

PROPERTY = settings(
    suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None
)

ATTRS_XFAIL = pytest.mark.xfail(
    strict=True,
    reason='get_capture_type_attrs reads raw Annotated aliases, which carry no .extra; '
    'the metadata is on msgspec.inspect.type_info(cls).fields',
)


def test_meta_without_units():
    meta = Meta('Duration')
    assert meta.description == 'Duration'
    assert meta.extra == {'standard_name': 'Duration'}


def test_meta_with_units_and_constraints():
    meta = Meta('Center frequency', 'Hz', gt=0)
    assert meta.extra == {'standard_name': 'Center frequency', 'units': 'Hz'}
    assert meta.gt == 0


def test_meta_rejects_duplicate_description():
    with pytest.raises(TypeError):
        Meta('x', description='y')


@given(name=st.text(min_size=1), units=st.one_of(st.none(), st.text(min_size=1)))
@PROPERTY
def test_meta_units_key_only_when_given(name, units):
    extra = Meta(name, units).extra
    assert extra['standard_name'] == name
    assert ('units' in extra) == (units is not None)


def test_capture_type_attrs_covers_every_field():
    attrs = get_capture_type_attrs(ss.specs.SingleToneCapture)
    assert set(attrs) == set(ss.specs.SingleToneCapture.__struct_fields__)


@ATTRS_XFAIL
def test_capture_type_attrs_units():
    attrs = get_capture_type_attrs(ss.specs.SingleToneCapture)
    assert attrs['duration'] == {
        'standard_name': 'Duration of the analysis waveform',
        'units': 's',
    }
    assert attrs['port'] == {'standard_name': 'Input port indices'}
    assert attrs['host_resample'] == {}


@ATTRS_XFAIL
def test_capture_type_attrs_optional_union():
    attrs = get_capture_type_attrs(ss.specs.SingleToneCapture)
    assert attrs['backend_sample_rate']['units'] == 'S/s'
    assert 'standard_name' in attrs['snr']


class SchemaProbe(msgspec.Struct):
    f: fractions.Fraction
    d: frozendict


class SchemaUnknown(msgspec.Struct):
    z: complex


def _properties(schema: dict) -> dict:
    name = schema['$ref'].rsplit('/', 1)[-1]
    return schema['$defs'][name]['properties']


def test_json_schema_custom_types():
    props = _properties(json_schema(SchemaProbe))
    assert props['f'] == {'type': ['string', 'number']}
    assert props['d'] == {'type': 'object'}


def test_json_schema_unknown_type_raises():
    with pytest.raises(TypeError, match='no schema handler'):
        json_schema(SchemaUnknown)


def test_json_schema_of_bound_sweep(cw_sweep):
    assert '$defs' in json_schema(type(cw_sweep))


@pytest.mark.parametrize(
    'value, expected',
    [
        ('13/28', fractions.Fraction(13, 28)),
        (0.5, fractions.Fraction(1, 2)),
        (3, fractions.Fraction(3)),
    ],
)
def test_convert_dict_fraction(value, expected):
    assert convert_dict(value, fractions.Fraction) == expected


@pytest.mark.parametrize(
    'value, expected', [('1e6', 1e6), ('inf', float('inf')), (2, 2.0)]
)
def test_convert_dict_lax_floats(value, expected):
    # loop points arrive as strings from YAML; strict=False is what coerces them
    assert convert_dict(value, float) == expected


@given(x=st.fractions())
@PROPERTY
def test_convert_dict_fraction_roundtrip(x):
    assert convert_dict(str(x), fractions.Fraction) == x


class DictField(sa.specs.SpecBase, frozen=True):
    d: dict[str, Any] = msgspec.field(default_factory=dict)


def test_convert_dict_freezes_spec_fields_to_their_depth():
    spec = convert_dict({'d': {'a': [1, {'b': 2}]}}, DictField)
    assert isinstance(spec.d, frozendict)
    assert type(spec.d['a']) is tuple
    # the field's freeze depth is 1, so the dict inside the tuple stays a dict
    assert spec.d == {'a': (1, {'b': 2})}


def test_convert_spec_downcasts_to_base_capture():
    tone = ss.specs.SingleToneCapture(port=0, frequency_offset=5.0)
    base = convert_spec(tone, ss.specs.SensorCapture)
    assert type(base) is ss.specs.SensorCapture
    assert base.port == 0
    assert not hasattr(base, 'frequency_offset')
    assert base == ss.specs.SensorCapture.from_spec(tone)


@pytest.mark.xfail(
    strict=True,
    reason='_enc_hook_no_tuple_keys rewrites the keys of the frozendict it was given, '
    'and the hook is lru-cached',
)
def test_to_dict_without_tuple_keys_does_not_mutate():
    remap = ss.specs.CaptureRemap(key=('a', 'b'), lookup={(1, 2): 'x'})
    encoded = remap.to_dict(allow_tuple_keys=False)
    assert encoded['lookup'] == {'[1, 2]': 'x'}
    assert remap.lookup == {(1, 2): 'x'}


@pytest.mark.parametrize(
    'type_, default',
    [(float, 0.0), (bool, False), (int, 0), (str, ''), (dict[str, Any], {})],
)
def test_infer_coord_info_builtins(type_, default):
    attrs, value = infer_coord_info(mi.type_info(type_))
    assert attrs == {}
    assert value == default
    assert type(value) is type(default)


def test_infer_coord_info_surfaces_meta_and_skips_tuple_union_members():
    attrs, value = infer_coord_info(ss.specs.types.Port)
    assert attrs == {'standard_name': 'Input port indices'}
    assert value == 0
    attrs, value = infer_coord_info(ss.specs.types.LOFrequency)
    assert 'standard_name' in attrs
    assert value == 0 and isinstance(value, float)


def test_infer_coord_info_unwraps_optional():
    attrs, value = infer_coord_info(Optional[ss.specs.types.BackendSampleRate])
    assert attrs['units'] == 'S/s'
    assert value == 0 and isinstance(value, float)


def test_infer_coord_info_literal_returns_a_type_object():
    _, value = infer_coord_info(ss.specs.types.LOShift)
    assert value is str


def test_infer_coord_info_custom_types():
    import pandas as pd

    assert infer_coord_info(mi.type_info(pd.Timestamp))[1] == pd.Timestamp(0)
    # every custom type is presumed to be a timestamp when timestamps are allowed
    fraction = mi.type_info(fractions.Fraction)
    assert infer_coord_info(fraction)[1] == pd.Timestamp(0)
    assert infer_coord_info(fraction, allow_timestamps=False)[1] == fractions.Fraction(
        0
    )


def test_infer_coord_info_rejects_ambiguous_union():
    with pytest.raises(TypeError, match='cannot determine xarray type'):
        infer_coord_info(sa.specs.types.WindowType)


def test_infer_coord_info_rejects_unsupported_types():
    with pytest.raises(TypeError, match='unsupported msgspec field type'):
        infer_coord_info(mi.type_info(list[int]))


def test_infer_coord_info_accepts_raw_annotation():
    alias = ss.specs.types.BackendSampleRate
    assert infer_coord_info(alias) == infer_coord_info(mi.type_info(alias))
