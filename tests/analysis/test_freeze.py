"""freeze, unfreeze and inspect_freeze_depths: how spec field values become hashable"""

from __future__ import annotations

import fractions
from typing import Annotated, Any, Optional, Union

import msgspec
import pytest
from frozen_strategies import (
    container_depth,
    has_frozen_below,
    has_mutable_below,
    json_trees,
    listify,
    scalars,
    tuplify,
)
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import striqt.sensor as ss
from striqt.analysis.specs.helpers import (
    Meta,
    freeze,
    frozendict,
    inspect_freeze_depths,
    unfreeze,
)

PROPERTY = settings(
    suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None
)


@given(tree=json_trees())
@PROPERTY
def test_freeze_removes_mutable_containers(tree):
    frozen = freeze(tree)
    assert not has_mutable_below(frozen)
    hash(frozen)


@given(tree=json_trees())
@PROPERTY
def test_freeze_is_idempotent(tree):
    frozen = freeze(tree)
    assert freeze(frozen) == frozen


@given(tree=json_trees())
@PROPERTY
def test_freeze_matches_oracle(tree):
    assert freeze(tree) == tuplify(tree)


@given(x=scalars)
@PROPERTY
def test_scalars_pass_through_by_identity(x):
    assert freeze(x) is x
    assert unfreeze(x) is x


@given(tree=json_trees(), max_depth=st.integers(min_value=-1, max_value=4))
@PROPERTY
def test_freeze_depth_converts_only_shallower_levels(tree, max_depth):
    result = freeze(tree, max_depth)
    assert listify(result) == listify(tree)
    # any max_depth <= 1 converts just the root; the rest of the tree is untouched
    assert has_mutable_below(result) == has_mutable_below(tree, max(max_depth, 1))
    if max_depth <= 1:
        assert result == freeze(tree, 1)
    if max_depth >= container_depth(tree):
        assert result == freeze(tree)


@given(tree=json_trees())
@PROPERTY
def test_unfreeze_roundtrip(tree):
    assert unfreeze(freeze(tree)) == listify(tree)


def test_unfreeze_converts_frozendict_and_tuple_at_every_level():
    frozen = frozendict({'x': (1, frozendict({'z': (2,)}))})
    thawed = unfreeze(frozen)
    assert thawed == {'x': [1, {'z': [2]}]}
    assert type(thawed) is dict
    assert type(thawed['x']) is list
    assert type(thawed['x'][1]) is dict
    assert type(thawed['x'][1]['z']) is list


@given(tree=json_trees(), max_depth=st.integers(min_value=-1, max_value=4))
@PROPERTY
def test_unfreeze_depth_converts_only_shallower_levels(tree, max_depth):
    frozen = freeze(tree)
    result = unfreeze(frozen, max_depth)
    assert listify(result) == listify(tree)
    assert has_frozen_below(result) == has_frozen_below(frozen, max(max_depth, 1))
    if max_depth <= 1:
        assert result == unfreeze(frozen, 1)
    if max_depth >= container_depth(tree):
        assert result == unfreeze(frozen)


@pytest.mark.xfail(
    strict=True,
    reason='freeze recurses into dict but not frozendict, so mutable values inside an '
    'existing frozendict survive; unfreeze handles both',
)
def test_freeze_recurses_into_frozendict():
    frozen = freeze({'x': frozendict({'y': [1]})})
    assert frozen['x']['y'] == (1,)
    assert not has_mutable_below(frozen)


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


def test_freeze_depths_count_dict_list_and_fixed_tuple_nesting():
    # variable-length tuples, sets and scalars contribute no depth
    assert inspect_freeze_depths(DepthProbe) == {
        'a': 3,
        'b': 1,
        'e': 1,
        'f': 1,
        'g': 2,
    }
    assert inspect_freeze_depths(DepthProbe) is inspect_freeze_depths(DepthProbe)


def test_freeze_depths_of_real_specs(cw_sweep):
    assert inspect_freeze_depths(ss.specs.SingleToneCapture) == {'adjust_analysis': 1}
    assert inspect_freeze_depths(ss.specs.CaptureRemap) == {'lookup': 1}
    assert inspect_freeze_depths(type(cw_sweep)) == {'adjust_captures': 2}
