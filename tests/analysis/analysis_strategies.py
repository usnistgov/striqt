"""strategies and oracles for the analysis tests: the capture/delay/range domains
that __post_init__ validates, the freeze/unfreeze and frozendict trees, and the
registry walk that fetches a measurement's tolerance.

Not a conftest: a second rootless conftest would shadow the root one that other test
directories import by name.
"""

from __future__ import annotations

from conftest import scalars
from hypothesis import strategies as st

import striqt.analysis as sa
from striqt.analysis.specs.helpers import frozendict

# %% registered tolerances


def registered_tolerance(capture, spec, **kwargs) -> sa.specs.Tolerance:
    """the tolerance registered for the measurement that `spec` selects, fetched
    through the registry walk a consumer would use"""
    name = sa.registry[type(spec)].name
    group = sa.registry.tospec()(**{name: spec})
    return sa.registry.tolerances(capture, group, **kwargs)[name]


# %% capture, delay and range domains

_SAMPLE_RATES = (1e6, 2e6, 7.68e6, 15.36e6)
_SSB_SAMPLE_RATE = 7.68e6

# a fractional part of at least 1e-3 samples stays clear of the 1e-6 sample
# tolerance in Capture.__post_init__
fractional_parts = st.floats(min_value=1e-3, max_value=1 - 1e-3)


@st.composite
def integer_sample_captures(draw):
    sample_rate = draw(st.sampled_from(_SAMPLE_RATES))
    count = draw(st.integers(min_value=1, max_value=10**6))
    return {'duration': count / sample_rate, 'sample_rate': sample_rate}


@st.composite
def fractional_sample_captures(draw):
    sample_rate = draw(st.sampled_from(_SAMPLE_RATES))
    count = draw(st.integers(min_value=0, max_value=10**6))
    frac = draw(fractional_parts)
    return {'duration': (count + frac) / sample_rate, 'sample_rate': sample_rate}


@st.composite
def integer_delay_kwargs(draw):
    n = draw(st.integers(min_value=0, max_value=10**5))
    return {'sample_rate': _SSB_SAMPLE_RATE, 'delay': n / _SSB_SAMPLE_RATE}


@st.composite
def fractional_delay_kwargs(draw):
    n = draw(st.integers(min_value=0, max_value=10**5))
    frac = draw(fractional_parts)
    return {'sample_rate': _SSB_SAMPLE_RATE, 'delay': (n + frac) / _SSB_SAMPLE_RATE}


range_starts = st.integers(min_value=0, max_value=50)


@st.composite
def valid_frame_ranges(draw):
    start = draw(range_starts)
    if start > 0:
        stop = draw(st.integers(min_value=start, max_value=60))
    else:
        stop = draw(st.integers(min_value=-5, max_value=60))
    return (start, stop)


@st.composite
def valid_symbol_ranges(draw):
    start = draw(range_starts)
    if start == 0:
        stop = draw(st.one_of(st.none(), st.integers(min_value=-5, max_value=60)))
    else:
        stop = draw(st.integers(min_value=start, max_value=60))
    return (start, stop)


@st.composite
def descending_ranges(draw):
    start = draw(st.integers(min_value=1, max_value=50))
    stop = draw(st.integers(min_value=-5, max_value=start - 1))
    return (start, stop)


# %% freeze/unfreeze and frozendict


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


dict_keys = st.one_of(st.text(max_size=6), st.integers(min_value=-5, max_value=5))


def json_trees(max_leaves: int = 25):
    def containers(children):
        lists = st.lists(children, max_size=4)
        dicts = st.dictionaries(dict_keys, children, max_size=4)
        return st.one_of(lists, lists.map(tuple), dicts)

    return st.recursive(scalars, containers, max_leaves=max_leaves)


hashable_values = st.one_of(scalars, st.tuples(scalars, scalars))
frozendict_dicts = st.dictionaries(dict_keys, hashable_values, max_size=6)
unhashable_values = st.one_of(
    st.lists(scalars, max_size=3), st.dictionaries(dict_keys, scalars, max_size=3)
)
unhashable_frozendicts = st.dictionaries(
    dict_keys, unhashable_values, min_size=1, max_size=4
).map(frozendict)
