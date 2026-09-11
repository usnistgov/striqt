"""CaptureRemap.__post_init__: decoding of multi-field lookup keys"""

from __future__ import annotations

import json

import msgspec
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import striqt.sensor as ss
from striqt.analysis.specs.helpers import frozendict

Remap = ss.specs.CaptureRemap
PROPERTY = settings(
    suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None
)
key_pairs = st.tuples(
    st.integers(min_value=-100, max_value=100),
    st.integers(min_value=-100, max_value=100),
)


def test_multi_key_json_string_keys_become_tuples(construct):
    remap = construct(Remap, key=('a', 'b'), lookup={'[1, 2]': 5, '["x", 3]': 6})
    assert isinstance(remap.lookup, frozendict)
    assert remap.lookup == {(1, 2): 5, ('x', 3): 6}
    hash(remap)


def test_multi_key_tuple_keys_pass_through():
    remap = Remap(key=('a', 'b'), lookup={(1, 2): 5})
    assert remap.lookup == {(1, 2): 5}


@given(keys=st.lists(key_pairs, unique=True, max_size=6))
@PROPERTY
def test_multi_key_roundtrip(construct, keys):
    lookup = {json.dumps(list(k)): i for i, k in enumerate(keys)}
    remap = construct(Remap, key=('a', 'b'), lookup=lookup)
    assert remap.lookup == {k: i for i, k in enumerate(keys)}


@pytest.mark.parametrize('key', ['a', ('a',)])
def test_single_field_key_leaves_string_keys(construct, key):
    remap = construct(Remap, key=key, lookup={'[1]': 5})
    assert remap.lookup == {'[1]': 5}


def test_multi_key_non_json_key_raises_decode_error(construct):
    # not translated into a spec-level message
    with pytest.raises(msgspec.DecodeError):
        construct(Remap, key=('a', 'b'), lookup={'foo': 5})


def test_multi_key_non_string_key_raises_type_error():
    with pytest.raises(TypeError, match='bytes-like object'):
        Remap(key=('a', 'b'), lookup={1: 5})


def test_nested_lookup_values_are_frozen(construct):
    remap = construct(Remap, key='a', lookup={'x': [1, 2]})
    assert remap.lookup == frozendict({'x': (1, 2)})


def test_replace_reconverts_keys():
    remap = Remap(key=('a', 'b'), lookup={(1, 2): 5})
    assert remap.replace(lookup={'[3, 4]': 1}).lookup == {(3, 4): 1}
