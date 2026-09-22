"""striqt.sensor.specs.helpers: per-capture helpers, sink path formatting, and sweep
expansion (loop_captures, adjust_captures, list_capture_adjustments, adjust_analysis)

The site-shaped cases use the extension binding in sweeps/src/extensions.py and the
override files in sweeps/sites*, mirroring the downstream sensor configuration.
"""

from __future__ import annotations

import itertools
import logging
import math
import re
from pathlib import Path

import msgspec
import pytest
from conftest import SITE_SPEC, scalars
from hypothesis import given
from hypothesis import strategies as st
from pytest_lazy_fixtures import lf
from site_strategies import (
    ANTENNA_MODELS,
    OTHER_ID,
    RADIO_ID,
    SITE_ADJUST,
    SiteSweepCls,
    SurveyCaptureCls,
    SurveySweepCls,
    make_site_capture,
    make_site_capture_kws,
    make_site_sweep,
)
from sweep_strategies import (
    CaptureCls,
    capture_tuples,
    frequency_bin_range_loop,
    make_capture,
    make_capture_kws,
    make_sweep,
    port_scalars,
    port_values,
    ports_and_lo,
    range_loop,
    sweeps,
)

import striqt.analysis as sa
import striqt.sensor as ss
from striqt.analysis.specs.helpers import frozendict

H = ss.specs.helpers
Remap = ss.specs.CaptureRemap

SNR_BY_OFFSET = {'100': 1.0, '200': 2.0}
DEFAULT_SNR = Remap(key='frequency_offset', lookup=SNR_BY_OFFSET, default=-1.0)
AB12_SNR = Remap(key='frequency_offset', lookup={100: 11.0})
ADJUST = {
    'defaults': {'lo_shift': 'left', 'snr': DEFAULT_SNR},
    'ab12': {'snr': AB12_SNR, 'lo_shift': 'right'},
}
MODEL_FROM_NAME = Remap(key='antenna_name', lookup=ANTENNA_MODELS)
NAME_FROM_PORT = Remap(key='port', lookup={0: 'Omni'})


def first_site_capture(sweep, source_id, **capture_kws):
    if capture_kws:
        sweep = sweep.replace(captures=(make_site_capture(**capture_kws),))
    return H.loop_captures(sweep, source_id=source_id)[0]


# %% convert_capture_arg


def test_convert_capture_arg_downcasts_first_argument(synthetic_sweep):
    @H.convert_capture_arg(ss.specs.SensorCapture)
    def probe(capture, extra):
        return capture, extra

    capture, extra = probe(synthetic_sweep.captures[0], 'x')
    assert type(capture) is ss.specs.SensorCapture
    assert capture.port == synthetic_sweep.captures[0].port
    assert extra == 'x'
    assert probe.__name__ == 'probe'


# %% split_capture_ports and pairwise_by_port


@given(port=port_scalars)
def test_split_scalar_port_is_identity(port):
    capture = make_capture(port=port)
    assert H.split_capture_ports(capture) == [capture]


@given(spec=ports_and_lo())
def test_split_tuple_ports(spec):
    port, lo = spec
    capture = make_capture(port=port, external_lo_frequency=lo)
    split = H.split_capture_ports(capture)
    assert [c.port for c in split] == list(port)
    for i, c in enumerate(split):
        assert c.external_lo_frequency == (lo[i] if isinstance(lo, tuple) else lo)
        assert c.replace(port=port, external_lo_frequency=lo) == capture


def test_split_soapy_capture_fields():
    capture = ss.specs.SoapyCapture(
        port=(0, 1), center_frequency=(1e9, 2e9), gain=(0.0, 5.0)
    )
    split = H.split_capture_ports(capture)
    assert [(c.port, c.center_frequency, c.gain) for c in split] == [
        (0, 1e9, 0.0),
        (1, 2e9, 5.0),
    ]


def test_split_leaves_adjust_analysis_intact(subtests):
    capture = make_capture(port=(0, 1), adjust_analysis={'a': (1, 2, 3)})
    for c in H.split_capture_ports(capture):
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
        H.split_capture_ports(capture)


def test_pairwise_without_previous_capture():
    capture = make_capture(port=(0, 1))
    first, second = H.split_capture_ports(capture)
    assert H.pairwise_by_port(capture, None, False) == [(first, None), (second, None)]
    assert H.pairwise_by_port(capture, capture, True) == [(first, None), (second, None)]


def test_pairwise_pairs_by_index():
    c1 = make_capture(port=(0, 1), frequency_offset=1.0)
    c2 = make_capture(port=(0, 1), frequency_offset=2.0)
    c1_by_port = [make_capture(port=p, frequency_offset=1.0) for p in (0, 1)]
    c2_by_port = [make_capture(port=p, frequency_offset=2.0) for p in (0, 1)]
    expected = list(zip(c1_by_port, c2_by_port))
    assert H.pairwise_by_port(c1, c2, False) == expected


@pytest.mark.xfail(
    strict=True,
    reason='pairwise_by_port zips the two split lists, silently truncating to the '
    'shorter port count instead of raising',
)
def test_pairwise_rejects_different_port_counts():
    with pytest.raises(ValueError):
        H.pairwise_by_port(make_capture(port=(0, 1)), make_capture(port=0), False)


# %% ensure_tuple


@given(x=scalars, size=st.one_of(st.none(), st.integers(min_value=0, max_value=5)))
def test_ensure_tuple_wraps_scalars(x, size):
    if size is None:
        assert H.ensure_tuple(x) == (x,)
    else:
        assert H.ensure_tuple(x, size) == (x,) * size


@given(
    items=st.lists(scalars, min_size=1, max_size=4),
    size=st.integers(min_value=1, max_value=5),
)
def test_ensure_tuple_broadcasts_only_singletons(items, size):
    t = tuple(items)
    assert H.ensure_tuple(t) is t
    if len(t) == 1:
        assert H.ensure_tuple(t, size) == t * size
    else:
        assert H.ensure_tuple(t, size) is t


def test_ensure_tuple_size_zero():
    assert H.ensure_tuple((1,), 0) == ()


# %% get_unique_ports


@given(
    captures=capture_tuples(min_size=0, max_size=3),
    loop_ports=st.one_of(st.none(), st.lists(port_values, min_size=1, max_size=3)),
)
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
def test_unique_ports_ignores_other_loops(captures):
    loops = (ss.specs.List(field='snr', values=(1.0, 2.0)),)
    assert H.get_unique_ports(captures, loops) == H.get_unique_ports(captures)


@pytest.mark.parametrize('sweep', [lf('synthetic_sweep'), lf('calibration_sweep')])
def test_unique_ports_of_fixtures(sweep):
    assert H.get_unique_ports(sweep.captures, sweep.loops) == (0, 1)


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
    assert H.get_capture_type(b.sensor.sweep_spec_cls) is b.schema.capture


@pytest.mark.xfail(
    strict=True,
    reason='get_type_hints does not substitute the Generic parameters of a Sweep '
    'subclass, so the unbound branch returns the bare SC TypeVar',
)
def test_capture_type_of_an_unbound_sweep():
    assert H.get_capture_type(_UnboundSweep) is ss.specs.SingleToneCapture


# %% describe_capture

DESCRIBE_FIELDS = ('port', 'duration', 'sample_rate', 'frequency_offset', 'lo_shift')


@given(
    captures=capture_tuples(min_size=1, max_size=1),
    fields=st.lists(st.sampled_from(DESCRIBE_FIELDS), min_size=1, unique=True),
    join=st.sampled_from((' | ', '; ')),
)
def test_describe_capture_one_segment_per_field(captures, fields, join):
    text = H.describe_capture(captures[0], tuple(fields), source_id=None, join=join)
    parts = text.split(join)
    assert len(parts) == len(fields)
    for part, field in zip(parts, fields):
        assert part.startswith(f'{field}: ')


def test_describe_capture_names_the_field_driven_by_a_key():
    remap = Remap(key='frequency_offset', lookup={100.0: 1.0}, default=-1.0)
    sweep = make_sweep(
        captures=(make_capture(frequency_offset=100.0),),
        adjust_captures={'defaults': {'snr': remap}},
    )
    capture, fields = sweep.captures[0], ('frequency_offset',)
    kws = {'adjust_spec': sweep.adjust_captures, 'source_id': None}
    text = H.describe_capture(capture, fields, **kws)
    assert text.startswith('snr: ')
    other = H.describe_capture(capture, fields, **{**kws, 'source_id': 'ab12'})
    assert text == other


# %% concat_group_sizes


def test_concat_group_sizes_empty():
    assert H.concat_group_sizes(()) == []


@given(
    count=st.integers(min_value=1, max_value=12),
    min_size=st.integers(min_value=1, max_value=5),
)
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
def test_concat_group_sizes_ignore_subclass_fields(captures, min_size):
    base = tuple(c.replace(frequency_offset=0.0, snr=None) for c in captures)
    assert H.concat_group_sizes(captures, min_size=min_size) == H.concat_group_sizes(
        base, min_size=min_size
    )


@pytest.mark.parametrize('sweep', [lf('synthetic_sweep'), lf('calibration_sweep')])
def test_concat_group_sizes_of_fixtures(sweep):
    # a group only closes while every distinct shape is still pending *and* remaining,
    # so mixed sweeps collapse into a single group once any shape runs out
    captures = H.loop_captures(sweep)
    assert H.concat_group_sizes(captures) == [len(captures)]
    assert H.concat_group_sizes(captures, min_size=10) == [len(captures)]


# %% max_by_frequency


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


# %% get_format_fields, get_path_fields and PathFormatter


def test_format_fields_are_extracted_in_order():
    assert H.get_format_fields('a{x}b{y!r}c{z:04d}{{}}') == ['x', 'y', 'z']
    assert H.get_format_fields('plain') == []


@given(names=st.lists(st.from_regex(r'\A[a-z][a-z0-9_]{0,8}\Z'), max_size=5))
def test_format_fields_roundtrip(names):
    template = ''.join(f'{{{name}}}' for name in names)
    assert H.get_format_fields(template) == names


def test_path_fields_with_spec_path(synthetic_sweep, synthetic_spec_path):
    fields = H.get_path_fields(
        synthetic_sweep, source_id='abcd', spec_path=synthetic_spec_path
    )
    assert set(fields) == {
        'start_time',
        'sensor_binding',
        'spec_name',
        'parent_name',
        'source_id',
    }
    assert fields['sensor_binding'] == 'single_tone'
    assert fields['spec_name'] == 'synthetic'
    assert fields['parent_name'] == 'sweeps'
    assert fields['source_id'] == 'abcd'
    assert re.fullmatch(r'\d{8}-\d{2}h\d{2}m\d{2}', fields['start_time'])


def test_path_fields_without_spec_path(synthetic_sweep):
    fields = H.get_path_fields(synthetic_sweep, source_id='abcd')
    assert set(fields) == {'start_time', 'sensor_binding', 'source_id'}


def test_path_fields_accept_a_callable_source_id(synthetic_sweep):
    fields = H.get_path_fields(synthetic_sweep, source_id=lambda: 'abcd')
    assert fields['source_id'] == 'abcd'


def test_path_fields_include_fixed_adjustments_only():
    remap = Remap(key='frequency_offset', lookup={100: 1.0})
    sweep = make_sweep(adjust_captures={'defaults': {'lo_shift': 'left', 'snr': remap}})
    fields = H.get_path_fields(sweep, source_id='abcd')
    assert fields['lo_shift'] == 'left'
    assert 'snr' not in fields


def test_path_fields_only_string_fixed_values_from_site_files(site_sweep):
    # documented as current: numeric and null fixed values are not formattable
    fields = H.get_path_fields(site_sweep, source_id=RADIO_ID)
    assert fields['radio_name'] == 'radio02'
    assert fields['site_name'] == 'WAPA-north'
    assert not {'gain', 'mast_height', 'azimuth_offset'} & set(fields)
    assert 'mast_height' not in H.get_path_fields(site_sweep, source_id=OTHER_ID)


def test_formatter_passes_through_paths_without_fields(synthetic_sweep, monkeypatch):
    def fail(*args, **kws):
        raise AssertionError('lookup.id must not be called')

    monkeypatch.setattr(ss.lib.controller.lookup, 'id', fail)
    formatter = H.PathFormatter(synthetic_sweep)
    assert formatter('outputs/noformat.zarr') == 'outputs/noformat.zarr'


def test_formatter_substitutes_fields(
    synthetic_sweep, synthetic_spec_path, fake_source_id
):
    formatter = H.PathFormatter(synthetic_sweep, spec_path=synthetic_spec_path)
    result = formatter('outputs/{spec_name}-{source_id}.zarr')
    assert result == f'outputs/synthetic-{fake_source_id}.zarr'


def test_formatter_expands_user(synthetic_sweep, synthetic_spec_path, fake_source_id):
    result = H.PathFormatter(synthetic_sweep, spec_path=synthetic_spec_path)(
        '~/{parent_name}'
    )
    assert result == str(Path.home() / 'sweeps')


def test_formatter_names_the_unknown_field_and_the_allowed_ones(
    synthetic_sweep, fake_source_id
):
    with pytest.raises(KeyError, match="'nope'") as info:
        H.PathFormatter(synthetic_sweep)('{nope}')
    assert 'source_id' in str(info.value)


@pytest.mark.xfail(
    strict=True,
    reason="Sink.path defaults to '{yaml_name}-{start_time}' but get_path_fields "
    'provides spec_name, not yaml_name',
)
def test_formatter_accepts_the_default_sink_path(
    synthetic_sweep, synthetic_spec_path, fake_source_id
):
    result = H.PathFormatter(synthetic_sweep, spec_path=synthetic_spec_path)(
        ss.specs.Sink().path
    )
    assert 'synthetic-' in result


def test_formatter_fills_the_site_sink_template(site_sweep, fake_radio_id):
    path = H.PathFormatter(site_sweep, spec_path=SITE_SPEC)(site_sweep.sink.path)
    assert path.startswith('../outputs/WAPA-north/site-cpu_site_')
    assert path.endswith('.zarr.zip')


# %% loop_captures


@given(sweep=sweeps())
def test_loop_captures_returns_a_tuple_of_the_capture_class(sweep):
    result = H.loop_captures(sweep)
    assert isinstance(result, tuple)
    assert all(type(c) is CaptureCls for c in result)


def test_repeat_is_not_expanded(synthetic_sweep):
    repeat, *others = synthetic_sweep.loops
    assert isinstance(repeat, ss.specs.Repeat) and repeat.count > 1
    without_repeat = synthetic_sweep.replace(loops=tuple(others))
    assert H.loop_captures(synthetic_sweep) == H.loop_captures(without_repeat)


def test_loop_order_is_declaration_order_with_captures_innermost():
    ports, offsets, snrs = (0, 1), (1e3, 2e3), (0.0, 5.0, 10.0)
    captures = tuple(make_capture(port=p) for p in ports)
    loops = (
        ss.specs.List(field='frequency_offset', values=offsets),
        ss.specs.Range(field='snr', start=0, stop=10, step=5),
    )
    result = H.loop_captures(make_sweep(captures=captures, loops=loops))
    expected = list(itertools.product(offsets, snrs, ports))
    assert [(c.frequency_offset, c.snr, c.port) for c in result] == expected


@given(
    values=st.lists(
        st.floats(min_value=0, max_value=1e5), min_size=1, max_size=4, unique=True
    )
)
def test_string_loop_points_are_coerced_to_the_field_type(values):
    def sweep_with(points):
        loops = (ss.specs.List(field='frequency_offset', values=tuple(points)),)
        return make_sweep(captures=(make_capture(),), loops=loops)

    numeric = H.loop_captures(sweep_with(values))
    strings = H.loop_captures(sweep_with(repr(v) for v in values))
    assert strings == numeric
    assert all(type(c.frequency_offset) is float for c in strings)


@given(sweep=sweeps(), limit=st.integers(min_value=0, max_value=12))
def test_limit_is_a_prefix(sweep, limit):
    assert H.loop_captures(sweep, limit=limit) == H.loop_captures(sweep)[:limit]


def test_only_fields_filters_capture_loops_but_keeps_analysis_loops():
    base = make_capture(frequency_offset=7.0)
    loops = (
        ss.specs.List(field='frequency_offset', values=(1.0, 2.0, 3.0)),
        ss.specs.List(field='snr', values=(10.0, 20.0)),
        ss.specs.List(field='window', isin='analysis', values=('hann', 'hamming')),
    )
    sweep = make_sweep(captures=(base,), loops=loops)
    result = H.loop_captures(sweep, only_fields=('snr',))
    assert len(result) == 2 * 2
    assert [c.frequency_offset for c in result] == [7.0] * 4
    assert [c.snr for c in result] == [10.0, 10.0, 20.0, 20.0]
    assert [c.adjust_analysis['window'] for c in result] == ['hann', 'hamming'] * 2


@pytest.mark.parametrize(
    'only_fields',
    [None, ('analysis_bandwidth',)],
    ids=['every_loop', 'one_loop_kept'],
)
def test_a_bad_loop_point_names_its_position_in_the_declared_loops(only_fields):
    """`only_fields` drops loops, so the reported index must count the loops the user
    wrote rather than the ones that survived"""
    loops = (
        ss.specs.List(field='snr', values=(1.0,)),
        ss.specs.List(field='frequency_offset', values=(1.0,)),
        ss.specs.List(field='analysis_bandwidth', values=('nope',)),
    )
    bad_index = len(loops) - 1
    sweep = make_sweep(captures=(make_capture(),), loops=loops)

    match = re.escape(f'$.loops[{bad_index}]: Expected `float`, got `str`')
    with pytest.raises(msgspec.ValidationError, match=match):
        H.loop_captures(sweep, only_fields=only_fields)


def test_loops_without_captures_build_new_instances():
    loops = (
        ss.specs.List(field='port', values=(0, 1)),
        ss.specs.List(field='sample_rate', values=(1e6,)),
        ss.specs.List(field='duration', values=(1e-3,)),
    )
    result = H.loop_captures(make_sweep(loops=loops))
    assert [c.port for c in result] == [0, 1]
    assert all((c.sample_rate, c.duration) == (1e6, 1e-3) for c in result)


def test_no_loops_and_no_captures():
    assert H.loop_captures(make_sweep()) == ()


def test_a_repeat_alone_and_no_captures():
    """a repeat names no capture field, so it leaves nothing to build a capture from,
    exactly as an empty `loops:` does"""
    assert H.loop_captures(make_sweep(loops=(ss.specs.Repeat(count=2),))) == ()


def test_loops_that_leave_a_required_field_unset_name_the_loop_point():
    """without a `captures:` entry the loops must supply every required capture field,
    so the loop point is the only place the failure can be reported"""
    loops = (ss.specs.List(field='snr', values=(1.0, 2.0)),)

    with pytest.raises(msgspec.ValidationError) as info:
        H.loop_captures(make_sweep(loops=loops))

    message = str(info.value)
    # the expanded tuple index msgspec would report is not a place in the sweep
    assert '$[0]' not in message
    assert message == "Object missing required field `port` - at $.loops: {'snr': 1.0}"


def test_unknown_loop_field_raises():
    loops = (ss.specs.List(field='nope', values=(1,)),)
    sweep = make_sweep(captures=(make_capture(),), loops=loops)
    match = r'\$\.loops: Object contains unknown field `nope`'
    with pytest.raises(msgspec.ValidationError, match=match):
        H.loop_captures(sweep)


def test_analysis_loops_populate_adjust_analysis():
    loops = (
        ss.specs.List(field='a', isin='analysis', values=(1, 2)),
        ss.specs.List(field='b', isin='analysis', values=(3, 4, 5)),
    )
    sweep = make_sweep(captures=(make_capture(),), loops=loops)
    result = H.loop_captures(sweep)
    expected = [{'a': a, 'b': b} for a in (1, 2) for b in (3, 4, 5)]
    assert [dict(c.adjust_analysis) for c in result] == expected


def test_analysis_loop_merges_with_the_captures_adjust_analysis():
    capture = make_capture(adjust_analysis={'keep': 1})
    loops = (ss.specs.List(field='window', isin='analysis', values=('hann',)),)
    (result,) = H.loop_captures(make_sweep(captures=(capture,), loops=loops))
    assert dict(result.adjust_analysis) == {'keep': 1, 'window': 'hann'}


@pytest.mark.parametrize(
    'bandwidths, expected',
    [
        ((0.5e6, 1e6, 2e6, math.inf), [0.5e6, 1e6, math.inf]),
        ((math.inf, 2e6, 0.5e6), [math.inf, 0.5e6]),
        ((2e6,), []),
    ],
    ids=['ascending', 'order_kept', 'all_above_nyquist'],
)
def test_loop_only_nyquist_keeps_bandwidths_within_the_sample_rate(
    bandwidths, expected
):
    captures = (make_capture(sample_rate=1e6, duration=1e-3),)
    loops = (ss.specs.List(field='analysis_bandwidth', values=bandwidths),)
    options = ss.specs.SweepOptions(loop_only_nyquist=True)
    filtered = H.loop_captures(
        make_sweep(captures=captures, loops=loops, options=options)
    )
    assert [c.analysis_bandwidth for c in filtered] == expected


def test_loop_only_nyquist_keeps_inf_bandwidth():
    loops = (ss.specs.List(field='analysis_bandwidth', values=(0.5e6, math.inf)),)
    options = ss.specs.SweepOptions(loop_only_nyquist=True)
    sweep = make_site_sweep(loops=loops, options=options)
    bandwidths = [c.analysis_bandwidth for c in H.loop_captures(sweep)]
    assert bandwidths == [0.5e6, math.inf]


def test_calibration_fixture_expansion(calibration_sweep):
    result = H.loop_captures(calibration_sweep)
    assert len(result) == 384
    assert all(
        not math.isfinite(c.analysis_bandwidth) or c.sample_rate >= c.analysis_bandwidth
        for c in result
    )
    assert all(type(c.sample_rate) is float for c in result)
    # the limit applies before the nyquist filter
    assert len(H.loop_captures(calibration_sweep, limit=7)) <= 7


def test_adjustments_apply_before_loops_take_priority():
    loops = (ss.specs.List(field='lo_shift', values=('right',)),)
    adjust = {'defaults': {'lo_shift': 'left'}}
    sweep = make_sweep(captures=(make_capture(),), loops=loops, adjust_captures=adjust)
    assert H.loop_captures(sweep)[0].lo_shift == 'right'
    assert H.loop_captures(sweep.replace(loops=()))[0].lo_shift == 'left'


def test_source_specific_adjustments_in_loop_captures():
    captures = tuple(make_capture(frequency_offset=fo) for fo in (100.0, 200.0, 300.0))
    sweep = make_sweep(captures=captures, adjust_captures=ADJUST)

    def summary(source_id):
        result = H.loop_captures(sweep, source_id)
        return [(c.frequency_offset, c.snr, c.lo_shift) for c in result]

    assert summary('ab12') == [
        (100.0, 11.0, 'right'),
        (200.0, -1.0, 'right'),
        (300.0, -1.0, 'right'),
    ]
    assert summary(None) == [
        (100.0, 1.0, 'left'),
        (200.0, 2.0, 'left'),
        (300.0, -1.0, 'left'),
    ]


def test_site_adjustments_apply_to_calibration_points(site_calibration_sweep):
    captures = H.loop_captures(site_calibration_sweep, source_id=RADIO_ID)
    assert {c.radio_name for c in captures} == {'radio02'}
    assert {c.site_name for c in captures} == {'WAPA-north'}
    assert {c.channel_name for c in captures} == {
        '3750 MHz',
        '3830 MHz',
        '3900 MHz',
        None,
    }


def test_tuple_port_fields_survive_looping(subtests):
    capture = make_capture(port=(0, 1), external_lo_frequency=(1e9, 2e9))
    loops = (ss.specs.List(field='frequency_offset', values=(1.0, 2.0)),)
    for c in H.loop_captures(make_sweep(captures=(capture,), loops=loops)):
        with subtests.test(msg=f'frequency_offset {c.frequency_offset}'):
            assert c.port == (0, 1)
            assert c.external_lo_frequency == (1e9, 2e9)


@given(
    loop=st.one_of(
        range_loop('frequency_offset'), frequency_bin_range_loop('frequency_offset')
    )
)
def test_range_loops_step_from_start_to_stop(loop):
    result = H.loop_captures(make_sweep(captures=(make_capture(),), loops=(loop,)))
    count = round((loop.stop - loop.start) / loop.step) + 1
    expected = [loop.start + i * loop.step for i in range(count)]
    assert [c.frequency_offset for c in result] == pytest.approx(expected)


def test_survey_loops_expand(site_survey_sweep):
    loops = site_survey_sweep.loops
    assert [type(l).__name__ for l in loops] == [
        'Repeat',
        'Range',
        'Range',
        'List',
        'Range',
        'FrequencyBinRange',
    ]
    assert [len(l.get_points()) for l in loops[1:]] == [9, 4, 2, 4, 9]

    captures = H.loop_captures(site_survey_sweep, source_id=RADIO_ID)
    # the repeat loop is left to the sweep runner, so it does not multiply the count
    assert len(captures) == 9 * 4 * 2 * 4 * 9
    c = captures[0]
    assert (c.azimuth, c.elevation) == (-180.0, 0.0)
    assert type(c.azimuth_repeat) is int
    assert dict(c.adjust_analysis) == {
        'time_statistic': ('mean', 'max'),
        'frame_slots': None,
        'frequency_offset': -46.08e6,
    }
    assert c.frequency_offset == pytest.approx(-3e6)


def test_meta_bounds_are_enforced_when_the_sweep_is_constructed(site_survey_sweep):
    """`validate_sweep_analysis` expands the loops from `Sweep.__post_init__`, so a
    `Meta` bound that only `_expand_capture_loops_with_origins` re-checks fails at
    construction"""
    loops = (
        ss.specs.Range(field='azimuth', start=-45, stop=226, step=2.5),
        ss.specs.Range(field='elevation', start=0, stop=5, step=5),
    )
    with pytest.raises(msgspec.ValidationError, match='<= 180'):
        site_survey_sweep.replace(loops=loops)


def test_capture_post_init_is_enforced_when_the_sweep_is_constructed(site_survey_sweep):
    loops = (ss.specs.Range(field='azimuth', start=-180, stop=180, step=90),)
    with pytest.raises(msgspec.ValidationError, match='azimuth and elevation'):
        site_survey_sweep.replace(loops=loops)


def test_range_loop_on_an_int_field_accepts_integral_floats():
    loops = (ss.specs.Range(field='azimuth_repeat', start=0, stop=3, step=1),)
    sweep = make_site_sweep(cls=SurveySweepCls, loops=loops)
    assert [c.azimuth_repeat for c in H.loop_captures(sweep)] == [0, 1, 2, 3]


def test_range_loop_on_an_int_field_rejects_fractions():
    loops = (ss.specs.Range(field='azimuth_repeat', start=0, stop=3, step=1.5),)
    sweep = make_site_sweep(cls=SurveySweepCls, loops=loops)
    match = re.escape('$.loops[0]: Expected `int`') + '.*azimuth_repeat'
    with pytest.raises(msgspec.ValidationError, match=match):
        H.loop_captures(sweep)


def test_remaps_see_loop_values_given_as_numbers(site_sweep):
    loops = (ss.specs.List(field='center_frequency', values=(3750e6, 3900e6)),)
    sweep = site_sweep.replace(loops=loops)
    captures = H.loop_captures(sweep, source_id=RADIO_ID)
    assert [c.channel_name for c in captures] == ['3750 MHz', '3900 MHz']


def test_remaps_see_loop_values_given_as_yaml_strings(site_sweep):
    loops = [
        {'kind': 'list', 'field': 'center_frequency', 'values': ['3750e6', '3900e6']}
    ]
    sweep = SiteSweepCls.from_dict({**site_sweep.to_dict(), 'loops': loops})
    captures = H.loop_captures(sweep, source_id=RADIO_ID)
    assert [c.channel_name for c in captures] == ['3750 MHz', '3900 MHz']


# %% loop_capture_origins


@given(sweep=sweeps())
def test_origins_key_the_first_occurrence_of_each_capture(sweep):
    captures = H.loop_captures(sweep)
    origins = H.loop_capture_origins(sweep)

    assert isinstance(origins, frozendict)
    assert tuple(origins) == tuple(dict.fromkeys(captures))

    for capture, origin in origins.items():
        assert origin.capture_index == captures.index(capture)


def test_origin_spec_index_cycles_over_the_capture_entries():
    """captures are the innermost loop, so each loop point visits every entry"""
    captures = tuple(make_capture(port=p) for p in (0, 1))
    loops = (ss.specs.List(field='frequency_offset', values=(1e3, 2e3)),)
    origins = H.loop_capture_origins(make_sweep(captures=captures, loops=loops))
    assert [o.spec_index for o in origins.values()] == [0, 1, 0, 1]
    assert [o.capture_index for o in origins.values()] == [0, 1, 2, 3]


def test_origin_spec_index_is_none_without_a_capture_list():
    loops = (
        ss.specs.List(field='port', values=(0, 1)),
        ss.specs.List(field='sample_rate', values=(1e6,)),
        ss.specs.List(field='duration', values=(1e-3,)),
    )
    origins = H.loop_capture_origins(make_sweep(loops=loops))
    assert [o.spec_index for o in origins.values()] == [None, None]


def test_origin_loop_points_coerce_capture_values_but_not_analysis_values():
    """`_build_loop_points_dict` converts only `isin='capture'` points, so an analysis
    point reaches the origin as the YAML wrote it"""
    loops = (
        ss.specs.List(field='frequency_offset', values=('1e5',)),
        ss.specs.List(field='window', isin='analysis', values=('hann',)),
    )
    sweep = make_sweep(captures=(make_capture(),), loops=loops)
    (origin,) = H.loop_capture_origins(sweep).values()
    assert dict(origin.loop_points) == {
        ('capture', 'frequency_offset'): 1e5,
        ('analysis', 'window'): 'hann',
    }
    assert type(origin.loop_points['capture', 'frequency_offset']) is float


def test_origin_loop_points_are_hashable(site_survey_sweep):
    """the mapping hashes lazily, so an unfrozen loop point would surface late"""
    origins = H.loop_capture_origins(site_survey_sweep, source_id=RADIO_ID)
    assert isinstance(hash(origins), int)
    assert all(isinstance(hash(o.loop_points), int) for o in origins.values())


@given(sweep=sweeps(), limit=st.integers(min_value=0, max_value=12))
def test_origins_stay_aligned_under_limit(sweep, limit):
    captures = H.loop_captures(sweep, limit=limit)
    origins = H.loop_capture_origins(sweep, limit=limit)
    assert tuple(origins) == tuple(dict.fromkeys(captures))
    for capture, origin in origins.items():
        assert captures[origin.capture_index] == capture


def test_origins_are_renumbered_after_the_nyquist_filter():
    """`capture_index` is a position in the returned tuple, so dropping the 2e6 point
    must not leave a gap"""
    captures = (make_capture(sample_rate=1e6, duration=1e-3),)
    loops = (
        ss.specs.List(field='analysis_bandwidth', values=(0.5e6, 1e6, 2e6, math.inf)),
    )
    options = ss.specs.SweepOptions(loop_only_nyquist=True)
    sweep = make_sweep(captures=captures, loops=loops, options=options)

    origins = H.loop_capture_origins(sweep)
    assert tuple(origins) == H.loop_captures(sweep)
    assert [o.capture_index for o in origins.values()] == [0, 1, 2]
    bandwidths = [
        o.loop_points['capture', 'analysis_bandwidth'] for o in origins.values()
    ]
    assert bandwidths == [0.5e6, 1e6, math.inf]


def test_a_repeated_loop_value_collapses_onto_the_first_origin():
    loops = (ss.specs.List(field='snr', values=(10.0, 10.0)),)
    sweep = make_sweep(captures=(make_capture(),), loops=loops)

    assert len(H.loop_captures(sweep)) == 2
    (origin,) = H.loop_capture_origins(sweep).values()
    assert origin.capture_index == 0


def test_only_fields_omits_the_filtered_loops_from_loop_points():
    loops = (
        ss.specs.List(field='frequency_offset', values=(1.0, 2.0, 3.0)),
        ss.specs.List(field='snr', values=(10.0, 20.0)),
        ss.specs.List(field='window', isin='analysis', values=('hann', 'hamming')),
    )
    sweep = make_sweep(captures=(make_capture(frequency_offset=7.0),), loops=loops)
    origins = H.loop_capture_origins(sweep, only_fields=('snr',))
    assert all(
        set(o.loop_points) == {('capture', 'snr'), ('analysis', 'window')}
        for o in origins.values()
    )


# %% describe_capture_origin


def test_describe_capture_origin_names_the_entry_and_every_loop():
    loops = (
        ss.specs.Repeat(count=3),
        ss.specs.List(field='frequency_offset', values=('1e5',)),
        ss.specs.List(field='window', isin='analysis', values=('hann',)),
    )
    sweep = make_sweep(captures=(make_capture(),), loops=loops)
    ((capture, origin),) = H.loop_capture_origins(sweep).items()

    assert H.describe_capture_origin(sweep.loops, capture, origin) == (
        # a Repeat is left to the sweep runner, so only its first pass is validated
        ".loops: {'repeat': 0, 'frequency_offset': 100000.0, 'window': 'hann'}",
        '.captures[0]',
    )


def test_describe_capture_origin_omits_the_entry_without_a_capture_list():
    loops = (
        ss.specs.List(field='port', values=(0,)),
        ss.specs.List(field='sample_rate', values=(1e6,)),
        ss.specs.List(field='duration', values=(1e-3,)),
    )
    sweep = make_sweep(loops=loops)
    ((capture, origin),) = H.loop_capture_origins(sweep).items()

    assert H.describe_capture_origin(sweep.loops, capture, origin) == (
        ".loops: {'port': 0, 'sample_rate': 1000000.0, 'duration': 0.001}",
    )


def test_describe_capture_origin_omits_the_loops_dropped_by_only_fields():
    loops = (
        ss.specs.List(field='frequency_offset', values=(1.0,)),
        ss.specs.List(field='snr', values=(10.0,)),
    )
    sweep = make_sweep(captures=(make_capture(frequency_offset=7.0),), loops=loops)
    origins = H.loop_capture_origins(sweep, only_fields=('snr',))
    ((capture, origin),) = origins.items()

    assert H.describe_capture_origin(sweep.loops, capture, origin) == (
        ".loops: {'snr': 10.0}",
        '.captures[0]',
    )


# %% adjust_captures


def test_adjust_captures_source_overrides_and_falls_back_to_defaults():
    spec = make_sweep(adjust_captures=ADJUST).adjust_captures
    hit = H.adjust_captures(make_capture_kws(frequency_offset=100.0), spec, 'ab12')
    assert hit == {'lo_shift': 'right', 'snr': 11.0}
    miss = H.adjust_captures(make_capture_kws(frequency_offset=300.0), spec, 'ab12')
    assert miss == {'lo_shift': 'right', 'snr': -1.0}
    unknown = H.adjust_captures(make_capture_kws(frequency_offset=300.0), spec, None)
    assert unknown == {'lo_shift': 'left', 'snr': -1.0}


def test_adjust_captures_missing_required_source_lookup_raises():
    adjust = {'ab12': {'snr': Remap(key='frequency_offset', lookup={100: 1.0})}}
    spec = make_sweep(adjust_captures=adjust).adjust_captures
    match = re.escape(
        "$.adjust_captures['ab12'].snr.lookup: Object missing a lookup entry for key"
    )
    with pytest.raises(msgspec.ValidationError, match=match):
        H.adjust_captures(make_capture_kws(frequency_offset=300.0), spec, 'ab12')


@pytest.mark.xfail(
    strict=True,
    reason='do_lookup returns UNSET for a required defaults remap without a default, '
    'so the field is silently omitted; the required/not-required branches are inverted',
)
def test_adjust_captures_missing_required_default_lookup_raises():
    adjust = {'defaults': {'snr': Remap(key='frequency_offset', lookup={100: 1.0})}}
    spec = make_sweep(adjust_captures=adjust).adjust_captures
    with pytest.raises(
        msgspec.ValidationError, match='Object missing a lookup entry for key'
    ):
        H.adjust_captures(make_capture_kws(frequency_offset=300.0), spec, None)


# a remap of snr keyed on frequency_offset, the capture it is applied to, and the
# adjustment it produces. PER_PORT hits 100 and misses 200.
SCALAR_HIT = {'frequency_offset': 100.0}
SCALAR_MISS = {'frequency_offset': 300.0}
PER_PORT = {'port': (0, 1), 'frequency_offset': (100.0, 200.0)}

REMAP_CASES = {
    'optional_scalar_hit': (
        {'lookup': {100: 1.0}, 'required': False},
        SCALAR_HIT,
        {'snr': 1.0},
    ),
    'optional_scalar_miss': (
        {'lookup': {100: 1.0}, 'required': False},
        SCALAR_MISS,
        {},
    ),
    'per_port_hits': ({'lookup': {100: 5.0, 200: 6.0}}, PER_PORT, {'snr': (5.0, 6.0)}),
    'per_port_miss_default': (
        {'lookup': {100: 5.0}, 'default': 0.0},
        PER_PORT,
        {'snr': (5.0, 0.0)},
    ),
    'per_port_miss_optional': ({'lookup': {100: 5.0}, 'required': False}, PER_PORT, {}),
}


@pytest.mark.parametrize('case', list(REMAP_CASES), ids=list(REMAP_CASES))
def test_adjust_captures_remap(case):
    remap_kws, capture_kws, expected = REMAP_CASES[case]
    remap = Remap(key='frequency_offset', **remap_kws)
    spec = make_sweep(adjust_captures={'defaults': {'snr': remap}}).adjust_captures
    capture = make_capture_kws(**capture_kws)
    assert H.adjust_captures(capture, spec, None) == expected


def test_adjust_captures_multi_field_key():
    remap = Remap(key=('frequency_offset', 'lo_shift'), lookup={'[100.0, "none"]': 7.0})
    spec = make_sweep(adjust_captures={'defaults': {'snr': remap}}).adjust_captures
    assert (100.0, 'none') in spec['defaults']['snr'].lookup
    capture = make_capture_kws(frequency_offset=100.0, lo_shift='none')
    assert H.adjust_captures(capture, spec, None) == {'snr': 7.0}


def test_adjust_captures_requires_a_mapping():
    spec = make_sweep(adjust_captures=ADJUST).adjust_captures
    match = 'Expected `capture` as a mapping, got `SingleToneCapture`'
    with pytest.raises(TypeError, match=re.escape(match)):
        H.adjust_captures(make_capture(), spec, None)


def test_port_adjustments_are_rejected_when_the_sweep_is_built():
    match = 'Object contains reserved capture field `port`'
    with pytest.raises(msgspec.ValidationError, match=re.escape(match)):
        make_sweep(adjust_captures={'defaults': {'port': 3}})


# %% adjust_captures: site override files on the extension binding


def test_yaml_and_direct_adjustments_agree(site_sweep):
    direct = make_site_sweep(adjust_captures=SITE_ADJUST)
    assert site_sweep.adjust_captures == direct.adjust_captures


def test_builtin_binding_rejects_site_fields():
    match = re.escape(
        "$.adjust_captures['defaults']: Object contains unknown field `channel_name` "
        'for capture type `striqt.sensor.specs.SingleToneCapture`'
    )
    with pytest.raises(msgspec.ValidationError, match=match):
        make_sweep(adjust_captures={'defaults': {'channel_name': 'x'}})


def test_source_block_chains_through_defaults(site_sweep):
    c = first_site_capture(site_sweep, RADIO_ID, port=(0, 1))
    assert c.radio_name == 'radio02'
    assert c.site_name == 'WAPA-north'
    assert c.antenna_name == ('Omni', '1x32')
    assert c.antenna_polarization == ('Vertical', 'Linear +45')
    assert c.antenna_model == ('OmniModel', 'PanelModel')
    assert c.antenna_index == (0, 1)
    assert c.channel_name == '3750 MHz'
    assert c.gain == 0
    assert c.mast_height == pytest.approx(1.7)
    assert c.azimuth_offset == 150


def test_source_override_of_a_defaults_alias_keeps_its_evaluation_position():
    # the source block lists antenna_model before antenna_name, but antenna_name
    # was declared first in defaults, so the merged field order still resolves it first
    adjust = {
        'defaults': {'antenna_name': 'Unspecified', 'antenna_model': MODEL_FROM_NAME},
        RADIO_ID: {'antenna_model': MODEL_FROM_NAME, 'antenna_name': NAME_FROM_PORT},
    }
    sweep = make_site_sweep(adjust_captures=adjust)
    assert first_site_capture(sweep, RADIO_ID).antenna_model == 'OmniModel'


def test_scalar_miss_returns_the_null_default(site_sweep):
    c = first_site_capture(site_sweep, RADIO_ID, center_frequency=3960e6)
    assert c.channel_name is None


def test_nan_string_fixed_value_becomes_float_nan(site_survey_sweep):
    c = H.loop_captures(site_survey_sweep, source_id=RADIO_ID, limit=1)[0]
    assert isinstance(c, SurveyCaptureCls)
    assert math.isnan(c.tx1_power)
    assert c.test_case == 3


def test_per_port_miss_uses_the_default(site_sweep):
    capture = make_site_capture_kws(port=(0, 1), center_frequency=(7350e6, 7350e6))
    result = H.adjust_captures(capture, site_sweep.adjust_captures, RADIO_ID)
    assert result['channel_name'] == (None, None)


@pytest.mark.parametrize(
    'center_frequency, gain', [(3750e6, 0), (3900e6, -10)], ids=['hit', 'miss']
)
def test_single_element_tuple_key_remap_is_optional(site_sweep, center_frequency, gain):
    c = first_site_capture(site_sweep, OTHER_ID, center_frequency=center_frequency)
    assert c.gain == gain


def test_multi_field_key_with_per_port_values(site_sweep):
    c = first_site_capture(site_sweep, OTHER_ID, port=(0, 1), switch_input=1)
    assert c.antenna_index == (2, 3)


@pytest.mark.xfail(
    strict=True,
    reason='_convert_label_lookup_keys prunes remaps keyed on unknown fields with a '
    '`continue` that precedes the intended raise',
)
def test_remap_keyed_on_an_unknown_field_is_rejected():
    remap = Remap(key='centre_frequency', lookup={1.0: 'x'})
    with pytest.raises(msgspec.ValidationError, match='centre_frequency'):
        make_site_sweep(adjust_captures={'defaults': {'channel_name': remap}})


@pytest.mark.xfail(
    strict=True,
    raises=msgspec.ValidationError,
    reason='after a fixed value or remap is processed its lookup type is forced to '
    'str, so remaps keyed on numeric aliases reject numeric lookup keys',
)
def test_remap_keyed_on_a_fixed_numeric_alias_accepts_numeric_keys():
    index_from_switch = Remap(key='switch_input', lookup={1: 7})
    adjust = {'defaults': {'switch_input': 1, 'antenna_index': index_from_switch}}
    sweep = make_site_sweep(adjust_captures=adjust)
    assert first_site_capture(sweep, None).antenna_index == 7


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason='remaps are evaluated in declaration order, so a remap keyed on an alias '
    'declared after it sees the unadjusted capture value and is silently omitted',
)
def test_chained_remap_declared_before_its_key_resolves():
    adjust = {
        'defaults': {'antenna_model': MODEL_FROM_NAME, 'antenna_name': NAME_FROM_PORT}
    }
    sweep = make_site_sweep(adjust_captures=adjust)
    result = H.adjust_captures(make_site_capture_kws(), sweep.adjust_captures, None)
    assert result.get('antenna_model') == 'OmniModel'


def test_empty_string_source_key_is_valid_hex():
    # documented as current: the built-in function sources report '' as their id
    sweep = make_site_sweep(adjust_captures={'': {'radio_name': 'anon'}})
    result = H.adjust_captures(make_site_capture_kws(), sweep.adjust_captures, '')
    assert result == {'radio_name': 'anon'}


def test_source_key_error_names_defaults():
    with pytest.raises(msgspec.ValidationError, match='defaults'):
        make_site_sweep(adjust_captures={'zz': {'radio_name': 'x'}})


# %% list_capture_adjustments


def test_list_capture_adjustments_by_source():
    loops = (ss.specs.List(field='frequency_offset', values=(100, 200, 300)),)
    sweep = make_sweep(captures=(make_capture(),), loops=loops, adjust_captures=ADJUST)
    assert H.list_capture_adjustments(sweep, 'ffff') == {
        'lo_shift': ('left',),
        'snr': (1.0, 2.0, -1.0),
    }
    assert H.list_capture_adjustments(sweep, 'ab12') == {
        'lo_shift': ('right',),
        'snr': (11.0, -1.0),
    }


def test_list_capture_adjustments_empty():
    sweep = make_sweep(captures=(make_capture(),))
    assert H.list_capture_adjustments(sweep, 'ffff') == {}


def test_list_capture_adjustments_of_the_synthetic_sweep(synthetic_sweep):
    # the defaults remap of synthetic.yaml keys frequency_offset on the snr loop
    listed = H.list_capture_adjustments(synthetic_sweep, 'ffff')
    assert set(listed) == {'frequency_offset'}
    assert {float(v) for v in listed['frequency_offset']} == {-1e6, 1e6}


@pytest.mark.parametrize(
    'offsets, expected',
    [
        ((100.0,), (1.0,)),
        ((300.0, 100.0, 300.0, 200.0), (-1.0, 1.0, 2.0)),
        ((200.0, 100.0, 200.0, 100.0), (2.0, 1.0)),
    ],
    ids=['single', 'miss_first', 'repeated_pairs'],
)
def test_list_capture_adjustments_are_unique_in_first_seen_order(offsets, expected):
    loops = (ss.specs.List(field='frequency_offset', values=offsets),)
    sweep = make_sweep(captures=(make_capture(),), loops=loops, adjust_captures=ADJUST)
    assert H.list_capture_adjustments(sweep, 'ffff')['snr'] == expected


def test_list_capture_adjustments_for_a_radio(site_sweep):
    listed = H.list_capture_adjustments(site_sweep, RADIO_ID)
    assert listed['radio_name'] == ('radio02',)
    assert listed['antenna_name'] == (('Omni', '1x32'),)
    assert listed['channel_name'] == ('3750 MHz',)


# %% adjust_analysis


def test_adjust_analysis_is_identity_without_adjustments(synthetic_sweep):
    # through the lru_cache an equal analysis seen earlier in the session would be
    # returned instead, so the identity is checked on the uncached function
    adjust_analysis = H.adjust_analysis.__wrapped__
    analysis = synthetic_sweep.analysis
    assert adjust_analysis(analysis, None) is analysis
    assert adjust_analysis(analysis, frozendict()) is analysis


@given(resolution=st.floats(min_value=1e3, max_value=1e5))
def test_adjust_analysis_replaces_matching_fields_everywhere(
    synthetic_sweep, resolution
):
    analysis = synthetic_sweep.analysis
    adjustment = frozendict({'frequency_resolution': resolution})
    adjusted = H.adjust_analysis(analysis, adjustment)
    assert type(adjusted) is type(analysis)
    hash(adjusted)
    before, after = analysis.to_dict(), adjusted.to_dict()
    measurement_kws = {n: kws for n, kws in before.items() if isinstance(kws, dict)}
    touched = {n for n, kws in measurement_kws.items() if 'frequency_resolution' in kws}
    assert touched and touched < set(measurement_kws)
    for name in before:
        if name in touched:
            assert after[name] == {**before[name], 'frequency_resolution': resolution}
        else:
            assert after[name] == before[name]


def test_adjust_analysis_warns_about_unused_keys(synthetic_sweep, caplog):
    with caplog.at_level(logging.WARNING, logger='striqt.sweep'):
        result = H.adjust_analysis(
            synthetic_sweep.analysis, frozendict({'bogus_key': 1})
        )
    assert result == synthetic_sweep.analysis
    assert any('bogus_key' in record.getMessage() for record in caplog.records)


# %% validate_sweep_analysis

# 1e4 Hz divides the 1e6 sample_rate of make_capture into 100 bins; 3e4 does not
SPG = ss.specs.BundledAnalysis.from_dict({
    'spectrogram': {'window': 'hann', 'frequency_resolution': 1e4}
})
BAD_RESOLUTION = Remap(
    key='frequency_offset',
    lookup={0.0: {'frequency_resolution': 1e4}, 1e5: {'frequency_resolution': 3e4}},
)


def test_validate_sweep_analysis_accepts_the_synthetic_sweep(synthetic_sweep):
    assert H.validate_sweep_analysis(synthetic_sweep) is None


def test_validate_sweep_analysis_reports_a_per_source_override():
    """`source_id` selects an `adjust_captures` block that `Sweep.__post_init__` cannot
    reach, since it resolves only the 'defaults' block"""
    sweep = make_sweep(
        captures=(make_capture(),),
        loops=(ss.specs.List(field='frequency_offset', values=(0.0, 1e5)),),
        adjust_captures={'ab12': {'adjust_analysis': BAD_RESOLUTION}},
        analysis=SPG,
    )

    assert H.validate_sweep_analysis(sweep) is None

    with pytest.raises(msgspec.ValidationError) as excinfo:
        H.validate_sweep_analysis(sweep, 'ab12')

    message = str(excinfo.value)
    # the measurement that rejected it, the values the failed rule compared, then the
    # place in the sweep that produced them: the loop point and the `captures:` entry
    assert message.startswith('$.analysis.spectrogram: ')
    assert 'sample_rate/resolution must be a counting number' in message
    assert '(sample_rate: 1000000.0, frequency_resolution: 30000.0)' in message
    assert message.endswith(
        " - at $.loops: {'frequency_offset': 100000.0} on $.captures[0]"
    )


def test_validate_sweep_analysis_ignores_loops_over_sensor_only_fields(monkeypatch):
    """`snr` is invisible to the analysis layer, so both captures project onto one
    `AnalysisCapture` and the pair is validated once"""
    # build before patching: Sweep.__post_init__ validates too, and would be counted
    sweep = make_sweep(
        captures=(make_capture(),),
        loops=(ss.specs.List(field='snr', values=(10.0, 20.0)),),
        analysis=SPG,
    )
    assert len(H.loop_captures(sweep)) == 2

    seen = []
    monkeypatch.setattr(
        sa.registry, 'validate', lambda capture, analysis: seen.append(capture)
    )

    H.validate_sweep_analysis(sweep)

    assert len(seen) == 1


def test_validate_sweep_analysis_skips_an_empty_analysis(monkeypatch):
    sweep = make_sweep(captures=(make_capture(),))

    seen = []
    monkeypatch.setattr(
        sa.registry, 'validate', lambda capture, analysis: seen.append(capture)
    )

    H.validate_sweep_analysis(sweep)

    assert seen == []
