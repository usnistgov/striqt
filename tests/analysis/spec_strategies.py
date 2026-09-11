"""strategies for the spec __post_init__ tests.

Not a conftest: a second rootless conftest would shadow the root one that other test
directories import by name.
"""

from __future__ import annotations

from hypothesis import strategies as st

SAMPLE_RATES = (1e6, 2e6, 7.68e6, 15.36e6)
SSB_SAMPLE_RATE = 7.68e6

# a fractional part of at least 1e-3 samples stays clear of the 1e-6 sample
# tolerance in Capture.__post_init__
fractional_parts = st.floats(min_value=1e-3, max_value=1 - 1e-3)


@st.composite
def integer_sample_captures(draw):
    sample_rate = draw(st.sampled_from(SAMPLE_RATES))
    count = draw(st.integers(min_value=1, max_value=10**6))
    return {'duration': count / sample_rate, 'sample_rate': sample_rate}


@st.composite
def fractional_sample_captures(draw):
    sample_rate = draw(st.sampled_from(SAMPLE_RATES))
    count = draw(st.integers(min_value=0, max_value=10**6))
    frac = draw(fractional_parts)
    return {'duration': (count + frac) / sample_rate, 'sample_rate': sample_rate}


@st.composite
def integer_delay_kwargs(draw):
    n = draw(st.integers(min_value=0, max_value=10**5))
    return {'sample_rate': SSB_SAMPLE_RATE, 'delay': n / SSB_SAMPLE_RATE}


@st.composite
def fractional_delay_kwargs(draw):
    n = draw(st.integers(min_value=0, max_value=10**5))
    frac = draw(fractional_parts)
    return {'sample_rate': SSB_SAMPLE_RATE, 'delay': (n + frac) / SSB_SAMPLE_RATE}


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
