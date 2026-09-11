"""strategies and factories for the sweep and capture helper tests.

Not a conftest: a second rootless conftest would shadow the root one that other test
directories import by name.
"""

from __future__ import annotations

import math

from hypothesis import HealthCheck, settings
from hypothesis import strategies as st

import striqt.sensor as ss

PROPERTY = settings(
    suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None
)
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


def sweep_dict(captures=(), loops=(), **kws):
    """from_dict input for SweepCls, usable even when construction should fail"""
    return {
        'source': SOURCE.to_dict(),
        'captures': [c.to_dict() for c in captures],
        'loops': [l.to_dict() for l in loops],
        **kws,
    }


gains = st.floats(min_value=-30, max_value=60)


@st.composite
def consistent_port_gain(draw):
    port = draw(port_values)
    if isinstance(port, tuple) and draw(st.booleans()):
        gain = tuple(draw(st.lists(gains, min_size=len(port), max_size=len(port))))
    else:
        gain = draw(gains)
    return port, gain


@st.composite
def scalar_port_tuple_gain(draw):
    gain = tuple(draw(st.lists(gains, min_size=1, max_size=4)))
    return draw(port_scalars), gain


@st.composite
def mismatched_port_gain(draw):
    port = draw(port_tuples)
    n = draw(st.integers(min_value=1, max_value=5).filter(lambda k: k != len(port)))
    return port, tuple(draw(st.lists(gains, min_size=n, max_size=n)))


@st.composite
def misplaced_repeat_loops(draw):
    """loop tuples with a Repeat anywhere but first"""
    loops = list(draw(loop_sets(allow_repeat=False).filter(lambda l: len(l) >= 1)))
    idx = draw(st.integers(min_value=1, max_value=len(loops)))
    loops.insert(
        idx, ss.specs.Repeat(count=draw(st.integers(min_value=1, max_value=3)))
    )
    return tuple(loops)


@st.composite
def duplicate_field_loops(draw):
    """(loops, field) where two loops target the same capture field"""
    base = draw(
        loop_sets(allow_repeat=False, allow_analysis=False).filter(
            lambda l: len(l) >= 1
        )
    )
    dup = draw(st.sampled_from(base))
    loops = draw(st.permutations([*base, draw(list_loop(dup.field))]))
    if draw(st.booleans()):
        loops.insert(0, ss.specs.Repeat(count=1))
    return tuple(loops), dup.field


CAL = ss.bindings.air7101b_calibration
CalSweepCls = CAL.sensor.sweep_spec_cls
CalCaptureCls = CAL.schema.capture
CalSourceCls = CAL.schema.source

NDE_LOOP = ss.specs.List(field='noise_diode_enabled', values=(False, True))
PORT_LOOP = ss.specs.List(field='port', values=(0, 1))
GAIN_LOOP = ss.specs.List(field='gain', values=(0, -10))
CAL_LOOP_VALUES = {'port': (0, 1), 'gain': (0, -10), 'center_frequency': (1e9, 2e9)}


def make_calibration_capture(**kws):
    kws = {
        'port': 0,
        'center_frequency': 1e9,
        'gain': 0,
        'duration': 1e-3,
        'sample_rate': 1e6,
        **kws,
    }
    return CalCaptureCls(**kws)


def make_calibration_sweep(captures=None, loops=(), source=None, **kws):
    if captures is None:
        captures = (make_calibration_capture(),)
    if source is None:
        source = CalSourceCls()
    return CalSweepCls(
        source=source, captures=tuple(captures), loops=tuple(loops), **kws
    )


def calibration_sweep_dict(captures=None, loops=(), source=None, **kws):
    if captures is None:
        captures = (make_calibration_capture(),)
    if source is None:
        source = CalSourceCls()
    return {
        'source': source.to_dict(),
        'captures': [c.to_dict() for c in captures],
        'loops': [l.to_dict() for l in loops],
        **kws,
    }


def loop_fields(sweep) -> list:
    return [loop.field for loop in sweep.loops]


@st.composite
def calibration_loop_orderings(draw):
    """(loops, accepted) for the noise diode toggle placement rule"""
    fields = draw(st.permutations(list(CAL_LOOP_VALUES)))
    fields = fields[: draw(st.integers(min_value=0, max_value=len(fields)))]
    loops = [ss.specs.List(field=f, values=CAL_LOOP_VALUES[f]) for f in fields]
    if draw(st.booleans()):
        idx = draw(st.integers(min_value=0, max_value=len(loops)))
        loops.insert(idx, NDE_LOOP)
        accepted = idx == 0 or (idx == 1 and fields[0] == 'port')
    else:
        accepted = True
    return tuple(loops), accepted
