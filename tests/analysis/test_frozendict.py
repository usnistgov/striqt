"""frozendict: the hashable mapping behind every frozen spec field"""

from __future__ import annotations

import pickle

import pytest
from frozen_strategies import frozendict_dicts, unhashable_frozendicts
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from striqt.analysis.specs.helpers import frozendict

PROPERTY = settings(
    suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None
)


@given(d=frozendict_dicts)
@PROPERTY
def test_mapping_laws(d):
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
@PROPERTY
def test_hash_is_order_independent(d, data):
    items = list(d.items())
    permuted = frozendict(data.draw(st.permutations(items)))
    assert hash(frozendict(items)) == hash(permuted)
    assert len({frozendict(items), permuted}) == 1


def test_hash_is_cached():
    fd = frozendict({'a': 1})
    assert fd._hash is None
    h = hash(fd)
    assert fd._hash == h
    assert hash(frozendict()) == 0


@given(fd=unhashable_frozendicts)
@PROPERTY
def test_unhashable_value_raises(fd):
    with pytest.raises(TypeError, match='unhashable frozendict entry'):
        hash(fd)


@given(a=frozendict_dicts, b=frozendict_dicts)
@PROPERTY
def test_or_returns_frozendict(a, b):
    fa = frozendict(a)
    merged = {**a, **b}
    for other in (b, frozendict(b)):
        result = fa | other
        assert isinstance(result, frozendict)
        assert result == merged
    with pytest.raises(TypeError):
        fa | 5


@given(a=frozendict_dicts, b=frozendict_dicts)
@PROPERTY
def test_ror_returns_plain_dict(a, b):
    fb = frozendict(b)
    merged = {**a, **b}
    result = a | fb
    assert type(result) is dict
    assert result == merged
    assert list(a.items()) | fb == merged
    with pytest.raises(TypeError, match='unsupported mapping'):
        5 | fb


def test_inplace_update_is_frozen():
    fd = frozendict({'a': 1})
    with pytest.raises(TypeError, match='frozen'):
        fd |= {'b': 2}
    with pytest.raises(TypeError, match='frozen'):
        fd.update({'b': 2})
    assert fd == {'a': 1}


@given(d=frozendict_dicts)
@PROPERTY
def test_pickle_roundtrip(d):
    fd = frozendict(d)
    restored = pickle.loads(pickle.dumps(fd))
    assert isinstance(restored, frozendict)
    assert restored == fd


def test_constructors_copy_and_repr():
    fd = frozendict(a=1, b=2)
    assert fd == {'a': 1, 'b': 2}
    assert frozendict.fromkeys('ab', 0) == {'a': 0, 'b': 0}
    copied = fd.copy()
    assert isinstance(copied, frozendict)
    assert copied == fd
    assert copied is not fd
    assert repr(fd) == "frozendict({'a': 1, 'b': 2})"


def test_nested_frozendict_is_hashable():
    nested = frozendict({'a': frozendict({'b': (1, 2)})})
    assert hash(nested) == hash(frozendict({'a': frozendict({'b': (1, 2)})}))
