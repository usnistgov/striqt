"""striqt.sensor.specs.captures: per-capture and per-port utilities"""

from __future__ import annotations

import pytest
from conftest import scalars
from hypothesis import given
from hypothesis import strategies as st
from pytest_lazy_fixtures import lf
from sweep_strategies import (
    capture_tuples,
    make_capture,
    port_scalars,
    port_values,
    ports_and_lo,
)

import striqt.sensor as ss

C = ss.specs.captures


# %% split_capture_ports and pairwise_by_port


@given(port=port_scalars)
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


@pytest.mark.xfail(
    strict=True,
    reason='split_capture_ports zips each tuple field against port, so a tuple '
    'shorter than port is copied whole onto the extra ports instead of raising',
)
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


@pytest.mark.xfail(
    strict=True,
    reason='pairwise_by_port zips the two split lists, silently truncating to the '
    'shorter port count instead of raising',
)
def test_pairwise_rejects_different_port_counts():
    with pytest.raises(ValueError):
        C.pairwise_by_port(make_capture(port=(0, 1)), make_capture(port=0), False)


# %% ensure_tuple


@given(x=scalars, size=st.one_of(st.none(), st.integers(min_value=0, max_value=5)))
def test_ensure_tuple_wraps_scalars(x, size):
    if size is None:
        assert C.ensure_tuple(x) == (x,)
    else:
        assert C.ensure_tuple(x, size) == (x,) * size


@given(
    items=st.lists(scalars, min_size=1, max_size=4),
    size=st.integers(min_value=1, max_value=5),
)
def test_ensure_tuple_broadcasts_only_singletons(items, size):
    t = tuple(items)
    assert C.ensure_tuple(t) is t
    if len(t) == 1:
        assert C.ensure_tuple(t, size) == t * size
    else:
        assert C.ensure_tuple(t, size) is t


def test_ensure_tuple_size_zero():
    assert C.ensure_tuple((1,), 0) == ()


# %% get_unique_ports


@given(
    captures=capture_tuples(min_size=0, max_size=3),
    loop_ports=st.one_of(st.none(), st.lists(port_values, min_size=1, max_size=3)),
)
def test_unique_ports_is_the_sorted_union(captures, loop_ports):
    expected = set()
    for c in captures:
        expected |= set(C.ensure_tuple(c.port))
    loops = ()
    if loop_ports is not None:
        loops = (ss.specs.List(field='port', values=tuple(loop_ports)),)
        for p in loop_ports:
            expected |= set(C.ensure_tuple(p))
    assert C.get_unique_ports(captures, loops) == tuple(sorted(expected))


@given(captures=capture_tuples())
def test_unique_ports_ignores_other_loops(captures):
    loops = (ss.specs.List(field='snr', values=(1.0, 2.0)),)
    assert C.get_unique_ports(captures, loops) == C.get_unique_ports(captures)


@pytest.mark.parametrize('sweep', [lf('synthetic_sweep'), lf('calibration_sweep')])
def test_unique_ports_of_fixtures(sweep):
    assert C.get_unique_ports(sweep.captures, sweep.loops) == (0, 1)


# %% get_capture_type


class _UnboundSweep(
    ss.specs.Sweep[
        ss.specs.FunctionSource, ss.specs.NoPeripherals, ss.specs.SingleToneCapture
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


@pytest.mark.parametrize('sweep', [lf('synthetic_sweep'), lf('calibration_sweep')])
def test_concat_group_sizes_of_fixtures(sweep):
    # a group only closes while every distinct shape is still pending *and* remaining,
    # so mixed sweeps collapse into a single group once any shape runs out
    captures = ss.specs.sequencing.loop_captures(sweep)
    assert C.concat_group_sizes(captures) == [len(captures)]
    assert C.concat_group_sizes(captures, min_size=10) == [len(captures)]
