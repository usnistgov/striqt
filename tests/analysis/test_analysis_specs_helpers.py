"""striqt.analysis.specs.helpers: frozendict, freezing, Meta, conversion hooks,
json_schema and coordinate inference"""

from __future__ import annotations

import fractions
import pickle
from typing import Annotated, Any, Optional, Union

import msgspec
import pytest
from analysis_strategies import (
    container_depth,
    frozendict_dicts,
    has_frozen_below,
    has_mutable_below,
    json_trees,
    listify,
    scalars,
    tuplify,
    unhashable_frozendicts,
)
from hypothesis import given
from hypothesis import strategies as st
from msgspec import inspect as mi

import striqt.analysis as sa
import striqt.sensor as ss
from striqt.analysis.specs.helpers import (
    Meta,
    convert_dict,
    convert_spec,
    freeze,
    frozendict,
    get_capture_type_attrs,
    infer_coord_info,
    inspect_freeze_depths,
    json_schema,
    unfreeze,
)

ATTRS_XFAIL = pytest.mark.xfail(
    strict=True,
    reason='get_capture_type_attrs reads raw Annotated aliases, which carry no .extra; '
    'the metadata is on msgspec.inspect.type_info(cls).fields',
)
VAR_TUPLE_DEPTH_XFAIL = pytest.mark.xfail(
    strict=True,
    reason='_inspect_container_depth (analysis/specs/helpers.py:341) handles '
    'TupleType but not VarTupleType, so tuple[T, ...] fields get depth 0',
)


# %% frozendict


class TestFrozendict:
    @given(d=frozendict_dicts)
    def test_mapping_laws(self, d):
        fd = frozendict(d)
        assert len(fd) == len(d)
        assert list(fd) == list(d)
        assert list(fd.keys()) == list(d.keys())
        assert list(fd.values()) == list(d.values())
        assert list(fd.items()) == list(d.items())
        for key in d:
            assert key in fd
            assert fd[key] == d[key]
        assert object() not in fd
        assert fd == d
        assert d == fd

    @given(d=frozendict_dicts, data=st.data())
    def test_hash_is_order_independent(self, d, data):
        items = list(d.items())
        permuted = frozendict(data.draw(st.permutations(items)))
        assert hash(frozendict(items)) == hash(permuted)
        assert len({frozendict(items), permuted}) == 1

    @given(fd=unhashable_frozendicts)
    def test_unhashable_value_raises(self, fd):
        with pytest.raises(TypeError, match='unhashable frozendict entry'):
            hash(fd)

    @given(a=frozendict_dicts, b=frozendict_dicts)
    def test_or_returns_frozendict(self, a, b):
        fa = frozendict(a)
        merged = {**a, **b}
        for other in (b, frozendict(b)):
            result = fa | other
            assert isinstance(result, frozendict)
            assert result == merged
        with pytest.raises(TypeError):
            fa | 5

    @given(a=frozendict_dicts, b=frozendict_dicts)
    def test_ror_returns_plain_dict(self, a, b):
        fb = frozendict(b)
        merged = {**a, **b}
        result = a | fb
        assert type(result) is dict
        assert result == merged
        assert list(a.items()) | fb == merged
        with pytest.raises(TypeError, match='unsupported mapping'):
            5 | fb

    def test_inplace_update_is_frozen(self):
        fd = frozendict({'a': 1})
        with pytest.raises(TypeError, match='frozen'):
            fd |= {'b': 2}
        with pytest.raises(TypeError, match='frozen'):
            fd.update({'b': 2})
        assert fd == {'a': 1}

    @given(d=frozendict_dicts)
    def test_pickle_roundtrip(self, d):
        fd = frozendict(d)
        restored = pickle.loads(pickle.dumps(fd))
        assert isinstance(restored, frozendict)
        assert restored == fd

    def test_constructors_copy(self):
        fd = frozendict(a=1, b=2)
        assert fd == {'a': 1, 'b': 2}
        assert frozendict.fromkeys('ab', 0) == {'a': 0, 'b': 0}
        copied = fd.copy()
        assert isinstance(copied, frozendict)
        assert copied == fd
        assert copied is not fd

    def test_nested_frozendict_is_hashable(self):
        nested = frozendict({'a': frozendict({'b': (1, 2)})})
        assert hash(nested) == hash(frozendict({'a': frozendict({'b': (1, 2)})}))


# %% freeze and unfreeze


class TestFreeze:
    @given(tree=json_trees())
    def test_is_idempotent(self, tree):
        frozen = freeze(tree)
        refrozen = freeze(frozen)
        assert refrozen == frozen
        assert type(refrozen) is type(frozen)
        assert not has_mutable_below(refrozen)

    @given(tree=json_trees())
    def test_matches_oracle(self, tree):
        assert freeze(tree) == tuplify(tree)

    @given(x=scalars)
    def test_scalars_pass_through_by_identity(self, x):
        assert freeze(x) is x
        assert unfreeze(x) is x

    @given(tree=json_trees(), max_depth=st.integers(min_value=-1, max_value=4))
    def test_depth_converts_only_shallower_levels(self, tree, max_depth):
        result = freeze(tree, max_depth)
        assert listify(result) == listify(tree)
        # any max_depth <= 1 converts just the root; the rest of the tree is untouched
        assert has_mutable_below(result) == has_mutable_below(tree, max(max_depth, 1))
        if max_depth <= 1:
            assert result == freeze(tree, 1)
        if max_depth >= container_depth(tree):
            assert result == freeze(tree)

    @pytest.mark.xfail(
        strict=True,
        reason='freeze recurses into dict but not frozendict, so mutable values '
        'inside an existing frozendict survive; unfreeze handles both',
    )
    def test_recurses_into_frozendict(self):
        frozen = freeze({'x': frozendict({'y': [1]})})
        assert frozen['x']['y'] == (1,)
        assert not has_mutable_below(frozen)


class TestUnfreeze:
    @given(tree=json_trees())
    def test_roundtrip(self, tree):
        thawed = unfreeze(freeze(tree))
        assert thawed == listify(tree)
        assert not has_frozen_below(thawed)

    @given(tree=json_trees(), max_depth=st.integers(min_value=-1, max_value=4))
    def test_depth_converts_only_shallower_levels(self, tree, max_depth):
        frozen = freeze(tree)
        result = unfreeze(frozen, max_depth)
        assert listify(result) == listify(tree)
        assert has_frozen_below(result) == has_frozen_below(frozen, max(max_depth, 1))
        if max_depth <= 1:
            assert result == unfreeze(frozen, 1)
        if max_depth >= container_depth(tree):
            assert result == unfreeze(frozen)


# %% inspect_freeze_depths


class DepthProbe(msgspec.Struct):
    a: dict[str, list[dict[str, int]]]
    b: Annotated[list[int], Meta('b')]
    c: tuple[int, ...]
    d: set[int]
    e: Optional[dict[str, Any]]  # noqa: UP045
    f: tuple[str, float]
    g: Union[int, list[list[int]]]  # noqa: UP007
    hh: int
    i: fractions.Fraction


@VAR_TUPLE_DEPTH_XFAIL
def test_freeze_depths_count_dict_list_and_tuple_nesting():
    # sets and scalars contribute no depth; freeze() does not convert sets
    assert inspect_freeze_depths(DepthProbe) == {
        'a': 3,
        'b': 1,
        'c': 1,
        'e': 1,
        'f': 1,
        'g': 2,
    }


def test_freeze_depths_are_cached():
    assert inspect_freeze_depths(DepthProbe) is inspect_freeze_depths(DepthProbe)


@VAR_TUPLE_DEPTH_XFAIL
def test_freeze_depths_of_real_specs(synthetic_sweep):
    assert inspect_freeze_depths(ss.specs.SingleToneCapture) == {
        'port': 1,
        'adjust_analysis': 1,
        'external_lo_frequency': 1,
    }
    assert inspect_freeze_depths(ss.specs.CaptureRemap) == {'key': 1, 'lookup': 1}
    assert inspect_freeze_depths(type(synthetic_sweep)) == {
        'captures': 1,
        'loops': 1,
        'adjust_captures': 2,
    }


# %% Meta and get_capture_type_attrs


@pytest.mark.parametrize('units', [None, 'Hz'])
def test_meta_units_key_only_when_given(units):
    meta = Meta('Center frequency', units, gt=0)
    assert meta.description == 'Center frequency'
    assert meta.gt == 0
    expected = {'standard_name': 'Center frequency'}
    if units is not None:
        expected['units'] = units
    assert meta.extra == expected


@ATTRS_XFAIL
def test_capture_type_attrs():
    attrs = get_capture_type_attrs(ss.specs.SingleToneCapture)
    assert set(attrs) == set(ss.specs.SingleToneCapture.__struct_fields__)
    assert attrs['duration'] == {
        'standard_name': 'Duration of the analysis waveform',
        'units': 's',
    }
    assert attrs['port'] == {'standard_name': 'Input port indices'}
    assert attrs['host_resample'] == {}
    # the Optional[...] alias of a field still has to reach the attrs
    assert attrs['backend_sample_rate']['units'] == 'S/s'
    assert 'standard_name' in attrs['snr']


# %% json_schema


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


def test_json_schema_of_bound_sweep(synthetic_sweep):
    schema = json_schema(type(synthetic_sweep))
    name = schema['$ref'].rsplit('/', 1)[-1]
    sweep_def = schema['$defs'][name]
    assert sweep_def['properties']['sensor_binding'] == {'enum': [name]}
    assert 'sensor_binding' in sweep_def['required']

    capture_ref = sweep_def['properties']['captures']['items']['$ref']
    capture_props = schema['$defs'][capture_ref.rsplit('/', 1)[-1]]['properties']
    assert (
        capture_props['duration']['description'] == 'Duration of the analysis waveform'
    )
    assert capture_props['port']['description'] == 'Input port indices'


# %% convert_dict, convert_spec and the encode/decode hooks


def test_convert_dict_lax_floats():
    # loop points arrive as strings from YAML; strict=False is what coerces them
    assert convert_dict('1e6', float) == pytest.approx(1e6)


@given(x=st.fractions())
def test_convert_dict_fraction_roundtrip(x):
    assert convert_dict(str(x), fractions.Fraction) == x


@pytest.mark.parametrize(
    'value, expected',
    [
        (0.001, fractions.Fraction(1, 1000)),
        (1 / 28000, fractions.Fraction(1, 28000)),
        (13 / 28, fractions.Fraction(13, 28)),
        (0.5, fractions.Fraction(1, 2)),
    ],
    ids=['1e-3', '1/28000', '13/28', '1/2'],
)
def test_convert_dict_fraction_snaps_floats(value, expected):
    assert convert_dict(value, fractions.Fraction) == expected


def test_convert_dict_fraction_keeps_strings_and_ints_exact():
    big = fractions.Fraction(1152921504606847, 1152921504606846976)
    assert convert_dict(str(big), fractions.Fraction) == big
    assert convert_dict(1, fractions.Fraction) == fractions.Fraction(1)


class DictField(sa.specs.SpecBase, frozen=True):
    d: dict[str, Any] = msgspec.field(default_factory=dict)


def test_spec_dict_fields_are_frozen_to_their_depth(construct):
    spec = construct(DictField, d={'a': [1, {'b': 2}]})
    assert isinstance(spec.d, frozendict)
    assert type(spec.d['a']) is tuple
    # the field's freeze depth is 1, so the dict inside the tuple stays a dict
    assert spec.d == {'a': (1, {'b': 2})}
    assert type(spec.d['a'][1]) is dict
    hash(construct(DictField, d={'a': [1]}))


@pytest.mark.xfail(
    strict=True,
    reason='SpecBase.__post_init__ (analysis/specs/structs.py:82-86) freezes a '
    'dict[str, Any] field to depth 2, so a list nested 3 deep stays a list',
)
def test_three_deep_adjust_analysis_is_hashable():
    spec = ss.specs.SingleToneCapture(
        port=0, adjust_analysis={'spectrogram': {'window': ['kaiser', 8]}}
    )
    hash(spec)
    assert spec.adjust_analysis['spectrogram']['window'] == ('kaiser', 8)


def test_convert_spec_downcasts_to_base_capture():
    tone = ss.specs.SingleToneCapture(port=0, frequency_offset=5.0)
    base = convert_spec(tone, ss.specs.SensorCapture)
    assert type(base) is ss.specs.SensorCapture
    assert base.port == 0
    assert not hasattr(base, 'frequency_offset')


def test_to_dict_without_tuple_keys_does_not_mutate():
    remap = ss.specs.CaptureRemap(key=('a', 'b'), lookup={(1, 2): 'x'})
    encoded = remap.to_dict(allow_tuple_keys=False)
    assert encoded['lookup'] == {'[1,2]': 'x'}
    assert remap.lookup == {(1, 2): 'x'}


def test_to_dict_without_tuple_keys_roundtrips_string_components():
    remap = ss.specs.CaptureRemap(key=('lo_shift', 'port'), lookup={('none', 0): 1.0})
    encoded = remap.to_dict(allow_tuple_keys=False)
    assert encoded['lookup'] == {'["none",0]': 1.0}
    assert ss.specs.CaptureRemap.from_dict(encoded) == remap
    assert remap.lookup == {('none', 0): 1.0}


# %% infer_coord_info


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
