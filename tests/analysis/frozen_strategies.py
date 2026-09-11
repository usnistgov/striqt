"""strategies and oracles for the freeze/unfreeze and frozendict tests"""

from __future__ import annotations

from hypothesis import strategies as st

from striqt.analysis.specs.helpers import frozendict


def _children(obj):
    if isinstance(obj, (dict, frozendict)):
        return obj.values()
    if isinstance(obj, (list, tuple)):
        return obj
    return ()


def container_depth(obj) -> int:
    if isinstance(obj, (list, tuple, dict, frozendict)):
        return 1 + max((container_depth(v) for v in _children(obj)), default=0)
    return 0


def has_mutable_below(obj, level: int = 0) -> bool:
    """True if a list or dict sits at nesting level >= level (the root is level 0)"""
    if level <= 0 and isinstance(obj, (list, dict)):
        return True
    return any(has_mutable_below(v, level - 1) for v in _children(obj))


def has_frozen_below(obj, level: int = 0) -> bool:
    """True if a tuple or frozendict sits at nesting level >= level"""
    if level <= 0 and isinstance(obj, (tuple, frozendict)):
        return True
    return any(has_frozen_below(v, level - 1) for v in _children(obj))


def tuplify(obj):
    if isinstance(obj, (list, tuple)):
        return tuple(tuplify(v) for v in obj)
    if isinstance(obj, (dict, frozendict)):
        return frozendict({k: tuplify(v) for k, v in obj.items()})
    return obj


def listify(obj):
    if isinstance(obj, (list, tuple)):
        return [listify(v) for v in obj]
    if isinstance(obj, (dict, frozendict)):
        return {k: listify(v) for k, v in obj.items()}
    return obj


scalars = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-(10**6), max_value=10**6),
    st.floats(allow_nan=False, allow_infinity=False),
    st.text(max_size=8),
)
dict_keys = st.one_of(st.text(max_size=6), st.integers(min_value=-5, max_value=5))


def json_trees(max_leaves: int = 25):
    return st.recursive(
        scalars,
        lambda children: st.one_of(
            st.lists(children, max_size=4),
            st.lists(children, max_size=4).map(tuple),
            st.dictionaries(dict_keys, children, max_size=4),
        ),
        max_leaves=max_leaves,
    )


hashable_values = st.one_of(scalars, st.tuples(scalars, scalars))
frozendict_dicts = st.dictionaries(dict_keys, hashable_values, max_size=6)
unhashable_values = st.one_of(
    st.lists(scalars, max_size=3), st.dictionaries(dict_keys, scalars, max_size=3)
)
unhashable_frozendicts = st.dictionaries(
    dict_keys, unhashable_values, min_size=1, max_size=4
).map(frozendict)
