"""strategies and factories for the sweep and capture helper tests.

Not a conftest: a second rootless conftest would shadow the root one that other test
directories import by name.
"""

from __future__ import annotations

import math

from hypothesis import strategies as st

import striqt.sensor as ss

SOURCE = ss.specs.FunctionSource(master_clock_rate=125e6, num_rx_ports=2)
CaptureCls = ss.bindings.single_tone.schema.capture
SweepCls = ss.bindings.single_tone.sensor.sweep_spec_cls

LO_SHIFTS = ('left', 'right', 'none')
LIST_LOOP_FIELDS = ('frequency_offset', 'snr', 'lo_shift')
SAMPLE_RATES = (1e6, 2e6, 4e6)


def make_capture(**kws):
    kws = {'port': 0, 'sample_rate': 1e6, 'duration': 1e-3, **kws}
    return CaptureCls(**kws)


def make_sweep(captures=(), loops=(), adjust_captures=None, **kws):
    if adjust_captures is None:
        adjust_captures = {}
    return SweepCls(
        source=SOURCE,
        captures=tuple(captures),
        loops=tuple(loops),
        adjust_captures=adjust_captures,
        **kws,
    )


def loop_point_count(loops) -> int:
    return math.prod(len(l.get_points()) for l in loops if l.field is not None)


scalars = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-(10**6), max_value=10**6),
    st.floats(allow_nan=False, allow_infinity=False),
    st.text(max_size=8),
)

port_scalars = st.integers(min_value=0, max_value=3)
port_tuples = st.lists(port_scalars, min_size=1, max_size=3, unique=True).map(tuple)
port_values = st.one_of(port_scalars, port_tuples)
lo_frequencies = st.floats(min_value=1e6, max_value=1e10)


@st.composite
def ports_and_lo(draw):
    """a port tuple with a matching external_lo_frequency tuple, scalar, or None"""
    port = draw(port_tuples)
    per_port = st.tuples(*[lo_frequencies] * len(port))
    lo = draw(st.one_of(st.none(), lo_frequencies, per_port))
    return port, lo


@st.composite
def capture_kwargs(draw, ports=port_values):
    sample_rate = draw(st.sampled_from(SAMPLE_RATES))
    count = draw(st.integers(min_value=10, max_value=1000))
    return {
        'port': draw(ports),
        'sample_rate': sample_rate,
        # an integer sample count keeps Capture.__post_init__ satisfied
        'duration': count / sample_rate,
        'frequency_offset': draw(st.floats(min_value=0, max_value=1e5)),
        'snr': draw(st.one_of(st.none(), st.floats(min_value=0, max_value=60))),
    }


def capture_tuples(min_size: int = 1, max_size: int = 3):
    return st.lists(capture_kwargs(), min_size=min_size, max_size=max_size).map(
        lambda kws: tuple(make_capture(**k) for k in kws)
    )


def _unique_floats(min_size: int = 0, max_size: int = 4):
    return st.lists(
        st.floats(min_value=0, max_value=1e5),
        min_size=min_size,
        max_size=max_size,
        unique=True,
    )


@st.composite
def list_loop(draw, field: str, min_size: int = 0):
    if field == 'lo_shift':
        values = draw(
            st.lists(
                st.sampled_from(LO_SHIFTS), min_size=min_size, max_size=3, unique=True
            )
        )
    else:
        values = draw(_unique_floats(min_size))
    return ss.specs.List(field=field, values=tuple(values))


@st.composite
def range_loop(draw, field: str):
    start = draw(st.floats(min_value=0, max_value=1e4))
    step = draw(st.floats(min_value=1.0, max_value=1e3))
    count = draw(st.integers(min_value=1, max_value=4))
    stop = start + (count - 1) * step
    return ss.specs.Range(field=field, start=start, stop=stop, step=step)


@st.composite
def frequency_bin_range_loop(draw, field: str):
    step = draw(st.floats(min_value=1.0, max_value=1e3))
    half = draw(st.integers(min_value=0, max_value=2))
    return ss.specs.FrequencyBinRange(
        field=field, start=-half * step, stop=half * step, step=step
    )


@st.composite
def loop_sets(draw, allow_repeat: bool = True, allow_analysis: bool = True):
    fields = draw(st.lists(st.sampled_from(LIST_LOOP_FIELDS), max_size=3, unique=True))
    loops = []
    for field in fields:
        if field == 'frequency_offset':
            loop = draw(
                st.one_of(
                    list_loop(field), range_loop(field), frequency_bin_range_loop(field)
                )
            )
        else:
            loop = draw(list_loop(field))
        loops.append(loop)
    if allow_analysis and draw(st.booleans()):
        loops.append(
            ss.specs.List(field='window', isin='analysis', values=('hann', 'hamming'))
        )
    loops = draw(st.permutations(loops))
    if allow_repeat and draw(st.booleans()):
        # a Repeat is only valid as the outermost loop
        count = draw(st.integers(min_value=1, max_value=3))
        loops.insert(0, ss.specs.Repeat(count=count))
    return tuple(loops)


@st.composite
def sweeps(draw, min_captures: int = 1):
    captures = draw(capture_tuples(min_size=min_captures))
    return make_sweep(captures=captures, loops=draw(loop_sets()))
