"""per-capture helpers: port splitting, tuple coercion, grouping and description"""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from sweep_strategies import (
    capture_tuples,
    make_capture,
    port_scalars,
    port_tuples,
    port_values,
    ports_and_lo,
    scalars,
)

import striqt.sensor as ss

H = ss.specs.helpers
PROPERTY = settings(
    suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None
)


def test_convert_capture_arg_downcasts_first_argument(cw_sweep):
    @H.convert_capture_arg(ss.specs.SensorCapture)
    def probe(capture, extra):
        return capture, extra

    capture, extra = probe(cw_sweep.captures[0], 'x')
    assert type(capture) is ss.specs.SensorCapture
    assert capture.port == cw_sweep.captures[0].port
    assert extra == 'x'
    assert probe.__name__ == 'probe'


@given(port=port_scalars)
@PROPERTY
def test_split_scalar_port_is_identity(port):
    capture = make_capture(port=port)
    assert H.split_capture_ports(capture) == [capture]


@given(spec=ports_and_lo())
@PROPERTY
def test_split_tuple_ports(spec):
    port, lo = spec
    capture = make_capture(port=port, external_lo_frequency=lo)
    split = H.split_capture_ports(capture)
    assert [c.port for c in split] == list(port)
    for i, c in enumerate(split):
        assert c.external_lo_frequency == (lo[i] if isinstance(lo, tuple) else lo)
        assert c.replace(port=port, external_lo_frequency=lo) == capture


def test_split_single_port_tuple_becomes_scalar():
    (capture,) = H.split_capture_ports(make_capture(port=(2,)))
    assert capture.port == 2


def test_split_soapy_capture_fields():
    capture = ss.specs.SoapyCapture(
        port=(0, 1), center_frequency=(1e9, 2e9), gain=(0.0, 5.0)
    )
    split = H.split_capture_ports(capture)
    assert [(c.port, c.center_frequency, c.gain) for c in split] == [
        (0, 1e9, 0.0),
        (1, 2e9, 5.0),
    ]


def test_split_leaves_adjust_analysis_intact():
    capture = make_capture(port=(0, 1), adjust_analysis={'a': (1, 2, 3)})
    for c in H.split_capture_ports(capture):
        assert c.adjust_analysis == {'a': (1, 2, 3)}


def test_split_short_tuple_field_is_left_on_the_extra_port():
    # zip stops at the shorter tuple, so the third port keeps the whole 2-tuple
    capture = make_capture(port=(0, 1, 2), external_lo_frequency=(1e9, 2e9))
    split = H.split_capture_ports(capture)
    assert [c.external_lo_frequency for c in split] == [1e9, 2e9, (1e9, 2e9)]


def test_pairwise_without_previous_capture():
    capture = make_capture(port=(0, 1))
    first, second = H.split_capture_ports(capture)
    assert H.pairwise_by_port(capture, None, False) == [(first, None), (second, None)]
    assert H.pairwise_by_port(capture, capture, True) == [(first, None), (second, None)]


@given(port=port_tuples, offsets=st.tuples(st.floats(0, 1e3), st.floats(0, 1e3)))
@PROPERTY
def test_pairwise_pairs_by_index(port, offsets):
    c1 = make_capture(port=port, frequency_offset=offsets[0])
    c2 = make_capture(port=port, frequency_offset=offsets[1])
    expected = list(zip(H.split_capture_ports(c1), H.split_capture_ports(c2)))
    assert H.pairwise_by_port(c1, c2, False) == expected


def test_pairwise_truncates_to_the_shorter_port_count():
    pairs = H.pairwise_by_port(make_capture(port=(0, 1)), make_capture(port=0), False)
    assert len(pairs) == 1


@given(x=scalars, size=st.one_of(st.none(), st.integers(min_value=0, max_value=5)))
@PROPERTY
def test_ensure_tuple_wraps_scalars(x, size):
    if size is None:
        assert H.ensure_tuple(x) == (x,)
    else:
        assert H.ensure_tuple(x, size) == (x,) * size


@given(
    items=st.lists(scalars, min_size=1, max_size=4),
    size=st.integers(min_value=1, max_value=5),
)
@PROPERTY
def test_ensure_tuple_broadcasts_only_singletons(items, size):
    t = tuple(items)
    assert H.ensure_tuple(t) is t
    if len(t) == 1:
        assert H.ensure_tuple(t, size) == t * size
    else:
        assert H.ensure_tuple(t, size) is t


def test_ensure_tuple_size_zero():
    assert H.ensure_tuple(1, 0) == ()
    assert H.ensure_tuple((1,), 0) == ()


@given(
    captures=capture_tuples(min_size=0, max_size=3),
    loop_ports=st.one_of(st.none(), st.lists(port_values, min_size=1, max_size=3)),
)
@PROPERTY
def test_unique_ports_is_the_sorted_union(captures, loop_ports):
    expected = set()
    for c in captures:
        expected |= set(H.ensure_tuple(c.port))
    loops = ()
    if loop_ports is not None:
        loops = (ss.specs.List(field='port', values=tuple(loop_ports)),)
        for p in loop_ports:
            expected |= set(H.ensure_tuple(p))
    assert H.get_unique_ports(captures, loops) == tuple(sorted(expected))


@given(captures=capture_tuples())
@PROPERTY
def test_unique_ports_ignores_other_loops(captures):
    loops = (ss.specs.List(field='snr', values=(1.0, 2.0)),)
    assert H.get_unique_ports(captures, loops) == H.get_unique_ports(captures)


def test_unique_ports_of_fixtures(cw_sweep, calibration_sweep):
    assert H.get_unique_ports(cw_sweep.captures, cw_sweep.loops) == (0, 1)
    cal = calibration_sweep
    assert H.get_unique_ports(cal.captures, cal.loops) == (0, 1)


@pytest.mark.parametrize(
    'binding', ['single_tone', 'noise', 'sawtooth', 'dirac_delta', 'warmup']
)
def test_capture_type_of_bound_sweeps(binding):
    b = getattr(ss.bindings, binding)
    assert H.get_capture_type(b.sensor.sweep_spec_cls) is b.schema.capture


def test_capture_type_of_calibration_sweep(calibration_sweep):
    cls = H.get_capture_type(type(calibration_sweep))
    assert 'noise_diode_enabled' in cls.__struct_fields__
    assert isinstance(calibration_sweep.captures[0], cls)


DESCRIBE_FIELDS = ('port', 'duration', 'sample_rate', 'frequency_offset', 'lo_shift')


@given(
    captures=capture_tuples(min_size=1, max_size=1),
    fields=st.lists(st.sampled_from(DESCRIBE_FIELDS), min_size=1, unique=True),
    join=st.sampled_from((' | ', '; ')),
)
@PROPERTY
def test_describe_capture_one_segment_per_field(captures, fields, join):
    text = H.describe_capture(captures[0], tuple(fields), source_id=None, join=join)
    parts = text.split(join)
    assert len(parts) == len(fields)
    for part, field in zip(parts, fields):
        assert part.startswith(f'{field}: ')


def test_describe_capture_names_the_field_driven_by_a_key():
    from sweep_strategies import make_sweep

    remap = ss.specs.CaptureRemap(
        key='frequency_offset', lookup={100.0: 1.0}, default=-1.0
    )
    sweep = make_sweep(
        captures=(make_capture(frequency_offset=100.0),),
        adjust_captures={'defaults': {'snr': remap}},
    )
    kws = {'adjust_spec': sweep.adjust_captures, 'source_id': None}
    text = H.describe_capture(sweep.captures[0], ('frequency_offset',), **kws)
    assert text.startswith('snr: ')
    same = H.describe_capture(sweep.captures[0], ('frequency_offset',), **kws)
    other = H.describe_capture(
        sweep.captures[0],
        ('frequency_offset',),
        adjust_spec=sweep.adjust_captures,
        source_id='ab12',
    )
    assert same == other


@given(
    captures=capture_tuples(min_size=1, max_size=6),
    min_size=st.integers(min_value=1, max_value=5),
)
@PROPERTY
def test_concat_group_sizes_partition_the_captures(captures, min_size):
    sizes = H.concat_group_sizes(captures, min_size=min_size)
    assert sum(sizes) == len(captures)
    assert all(size >= 1 for size in sizes)
    assert all(size >= min_size for size in sizes[:-1])


def test_concat_group_sizes_empty():
    assert H.concat_group_sizes(()) == []


@given(
    count=st.integers(min_value=1, max_value=12),
    min_size=st.integers(min_value=1, max_value=5),
)
@PROPERTY
def test_concat_group_sizes_homogeneous(count, min_size):
    captures = (make_capture(),) * count
    expected = [min_size] * (count // min_size)
    if count % min_size:
        expected.append(count % min_size)
    assert H.concat_group_sizes(captures, min_size=min_size) == expected


@given(
    captures=capture_tuples(min_size=1, max_size=6),
    min_size=st.integers(min_value=1, max_value=4),
)
@PROPERTY
def test_concat_group_sizes_ignore_subclass_fields(captures, min_size):
    base = tuple(c.replace(frequency_offset=0.0, snr=None) for c in captures)
    assert H.concat_group_sizes(captures, min_size=min_size) == H.concat_group_sizes(
        base, min_size=min_size
    )


def test_concat_group_sizes_of_fixtures(cw_sweep, calibration_sweep):
    # a group only closes while every distinct shape is still pending *and* remaining,
    # so mixed sweeps collapse into a single group once any shape runs out
    assert H.concat_group_sizes(cw_sweep.captures) == [4]
    cal = H.loop_captures(calibration_sweep)
    assert H.concat_group_sizes(cal) == [len(cal)]
    assert H.concat_group_sizes(cw_sweep.captures, min_size=10) == [4]


def test_max_by_frequency_without_loops():
    captures = (
        ss.specs.SoapyCapture(port=(0, 1), center_frequency=1e9, gain=(0.0, 5.0)),
        ss.specs.SoapyCapture(port=0, center_frequency=2e9, gain=3.0),
    )
    assert H.max_by_frequency('gain', captures) == {
        0: {1e9: 0.0, 2e9: 3.0},
        1: {1e9: 5.0},
    }


def test_max_by_frequency_loop_replaces_the_field():
    captures = (ss.specs.SoapyCapture(port=0, center_frequency=1e9, gain=3.0),)
    loops = (ss.specs.List(field='gain', values=(1.0, 9.0)),)
    assert H.max_by_frequency('gain', captures, loops) == {0: {1e9: 9.0}}


@given(
    gains=st.lists(st.floats(min_value=0, max_value=60), min_size=1, max_size=8),
    data=st.data(),
)
@PROPERTY
def test_max_by_frequency_is_the_maximum(gains, data):
    frequencies = [data.draw(st.sampled_from((1e9, 2e9, 3e9))) for _ in gains]
    ports = [data.draw(st.integers(min_value=0, max_value=1)) for _ in gains]
    captures = tuple(
        ss.specs.SoapyCapture(port=p, center_frequency=f, gain=g)
        for p, f, g in zip(ports, frequencies, gains)
    )
    result = H.max_by_frequency('gain', captures)
    for p, f, g in zip(ports, frequencies, gains):
        peers = [
            gg for pp, ff, gg in zip(ports, frequencies, gains) if (pp, ff) == (p, f)
        ]
        assert result[p][f] == max(peers)
