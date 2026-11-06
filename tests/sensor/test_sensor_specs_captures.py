"""striqt.sensor.specs.captures: per-capture and per-port utilities"""

from __future__ import annotations

import pytest
from hypothesis import given, strategies as st
from sweep_strategies import capture_tuples, make_capture, ports_and_lo

import striqt.sensor as ss

C = ss.specs.captures


# %% split_capture_ports and pairwise_by_port


@pytest.mark.parametrize('port', [0, 1, 2, 3], ids=lambda p: f'port{p}')
def test_split_scalar_port_is_identity(port):
    capture = make_capture(port=port)
    assert C.split_capture_ports(capture) == [capture]


@given(spec=ports_and_lo())
def test_split_tuple_ports(spec):
    port, lo = spec
    capture = make_capture(port=port, external_lo_frequency=lo)
    split = C.split_capture_ports(capture)
    assert [c.port for c in split] == list(port)
    for i, c in enumerate(split):
        assert c.external_lo_frequency == (lo[i] if isinstance(lo, tuple) else lo)
        assert c.replace(port=port, external_lo_frequency=lo) == capture


def test_split_soapy_capture_fields():
    capture = ss.specs.SoapyCapture(
        port=(0, 1), center_frequency=(1e9, 2e9), gain=(0.0, 5.0)
    )
    split = C.split_capture_ports(capture)
    assert [(c.port, c.center_frequency, c.gain) for c in split] == [
        (0, 1e9, 0.0),
        (1, 2e9, 5.0),
    ]


def test_split_leaves_adjust_analysis_intact(subtests):
    capture = make_capture(port=(0, 1), adjust_analysis={'a': (1, 2, 3)})
    for c in C.split_capture_ports(capture):
        with subtests.test(msg=f'port {c.port}'):
            assert c.adjust_analysis == {'a': (1, 2, 3)}


def test_split_rejects_a_tuple_field_shorter_than_port():
    capture = make_capture(port=(0, 1, 2), external_lo_frequency=(1e9, 2e9))
    with pytest.raises(ValueError):
        C.split_capture_ports(capture)


def test_pairwise_without_previous_capture():
    capture = make_capture(port=(0, 1))
    first, second = C.split_capture_ports(capture)
    assert C.pairwise_by_port(capture, None, False) == [(first, None), (second, None)]
    assert C.pairwise_by_port(capture, capture, True) == [(first, None), (second, None)]


def test_pairwise_pairs_by_index():
    c1 = make_capture(port=(0, 1), frequency_offset=1.0)
    c2 = make_capture(port=(0, 1), frequency_offset=2.0)
    c1_by_port = [make_capture(port=p, frequency_offset=1.0) for p in (0, 1)]
    c2_by_port = [make_capture(port=p, frequency_offset=2.0) for p in (0, 1)]
    expected = list(zip(c1_by_port, c2_by_port))
    assert C.pairwise_by_port(c1, c2, False) == expected


def test_pairwise_rejects_different_port_counts():
    with pytest.raises(ValueError):
        C.pairwise_by_port(make_capture(port=(0, 1)), make_capture(port=0), False)


# %% ensure_tuple


@pytest.mark.parametrize(
    'x, size, expected',
    [
        (3, None, (3,)),
        ('a', 3, ('a', 'a', 'a')),
        (None, 0, ()),
        ((1,), 3, (1, 1, 1)),
    ],
    ids=['scalar_unsized', 'scalar_sized', 'scalar_zero', 'singleton_broadcast'],
)
def test_ensure_tuple_wraps_scalars_and_broadcasts_singletons(x, size, expected):
    assert C.ensure_tuple(x, size) == expected


@pytest.mark.parametrize('size', [None, 3], ids=['unsized', 'sized'])
def test_ensure_tuple_passes_a_longer_tuple_through(size):
    t = (1, 2)
    assert C.ensure_tuple(t, size) is t


# %% get_unique_ports


PORT_LOOP = ss.specs.List(field='port', values=((3,), 1))
SNR_LOOP = ss.specs.List(field='snr', values=(1.0, 2.0))


@pytest.mark.parametrize(
    'ports, loops, expected',
    [
        ((2, (0, 1)), (), (0, 1, 2)),
        ((2, (0, 1)), (PORT_LOOP,), (0, 1, 2, 3)),
        ((), (PORT_LOOP,), (1, 3)),
        (((1, 0),), (SNR_LOOP,), (0, 1)),
        ((), (), ()),
    ],
    ids=['captures_only', 'captures_and_loop', 'loop_only', 'other_loop', 'empty'],
)
def test_unique_ports_is_the_sorted_union_of_captures_and_port_loops(
    ports, loops, expected
):
    captures = tuple(make_capture(port=p) for p in ports)
    assert C.get_unique_ports(captures, loops) == expected


# %% get_capture_type


class _UnboundSweep(
    ss.specs.Sweep[
        ss.specs.FunctionSource,
        ss.specs.NoPeripherals,
        ss.specs.SingleToneCapture,
        ss.specs.BuiltinAnalyses,
    ],
    frozen=True,
    kw_only=True,
):
    pass


def test_capture_type_of_a_bound_sweep():
    b = ss.bindings.single_tone
    assert C.get_capture_type(b.sensor.sweep_spec_cls) is b.schema.capture


def test_capture_type_of_an_unbound_sweep():
    assert C.get_capture_type(_UnboundSweep) is ss.specs.SingleToneCapture


# %% concat_group_sizes


def test_concat_group_sizes_empty():
    assert C.concat_group_sizes(()) == []


@given(
    count=st.integers(min_value=1, max_value=12),
    min_size=st.integers(min_value=1, max_value=5),
)
def test_concat_group_sizes_homogeneous(count, min_size):
    captures = (make_capture(),) * count
    expected = [min_size] * (count // min_size)
    if count % min_size:
        expected.append(count % min_size)
    assert C.concat_group_sizes(captures, min_size=min_size) == expected


@given(
    captures=capture_tuples(min_size=1, max_size=6),
    min_size=st.integers(min_value=1, max_value=4),
)
def test_concat_group_sizes_ignore_subclass_fields(captures, min_size):
    base = tuple(c.replace(frequency_offset=0.0, snr=None) for c in captures)
    assert C.concat_group_sizes(captures, min_size=min_size) == C.concat_group_sizes(
        base, min_size=min_size
    )


def test_concat_group_sizes_of_mixed_shapes_form_one_group():
    # a group only closes while every distinct shape is still pending *and* remaining,
    # so captures of 2 shapes collapse into a single group once one shape runs out
    captures = (make_capture(), make_capture(duration=2e-3), make_capture())
    assert C.concat_group_sizes(captures, min_size=len(captures) + 1) == [3]
