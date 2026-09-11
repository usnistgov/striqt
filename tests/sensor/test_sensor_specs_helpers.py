"""striqt.sensor.specs.helpers: per-capture helpers, sink path formatting, and sweep
expansion (loop_captures, adjust_captures, list_capture_adjustments, adjust_analysis)

The site-shaped cases use the extension binding in sweeps/src/extensions.py and the
override files in sweeps/sites*, mirroring the downstream sensor configuration.
"""

from __future__ import annotations

import logging
import math
import re
from pathlib import Path

import msgspec
import pytest
from hypothesis import given
from hypothesis import strategies as st
from site_strategies import (
    ANTENNA_MODELS,
    OTHER_ID,
    RADIO_ID,
    SITE_ADJUST,
    SiteSweepCls,
    SurveyCaptureCls,
    SurveySweepCls,
    make_site_capture,
    make_site_sweep,
)
from site_strategies import capture_dict as site_capture_dict
from sweep_strategies import (
    LIST_LOOP_FIELDS,
    PROPERTY,
    CaptureCls,
    capture_tuples,
    frequency_bin_range_loop,
    loop_point_count,
    make_capture,
    make_sweep,
    port_scalars,
    port_tuples,
    port_values,
    ports_and_lo,
    range_loop,
    scalars,
    sweeps,
)

import striqt.sensor as ss
from striqt.analysis.specs.helpers import frozendict

H = ss.specs.helpers
Remap = ss.specs.CaptureRemap

ADJUST = {
    'defaults': {
        'lo_shift': 'left',
        'snr': Remap(
            key='frequency_offset', lookup={'100': 1.0, '200': 2.0}, default=-1.0
        ),
    },
    'ab12': {
        'snr': Remap(key='frequency_offset', lookup={100: 11.0}),
        'lo_shift': 'right',
    },
}


def capture_dict(**kws):
    return make_capture(**kws).to_dict()


def first_site_capture(sweep, source_id, **capture_kws):
    if capture_kws:
        sweep = sweep.replace(captures=(make_site_capture(**capture_kws),))
    return H.loop_captures(sweep, source_id=source_id)[0]


# %% convert_capture_arg


def test_convert_capture_arg_downcasts_first_argument(cw_sweep):
    @H.convert_capture_arg(ss.specs.SensorCapture)
    def probe(capture, extra):
        return capture, extra

    capture, extra = probe(cw_sweep.captures[0], 'x')
    assert type(capture) is ss.specs.SensorCapture
    assert capture.port == cw_sweep.captures[0].port
    assert extra == 'x'
    assert probe.__name__ == 'probe'


# %% split_capture_ports and pairwise_by_port


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


# %% ensure_tuple


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
    assert H.ensure_tuple((1,), 0) == ()


# %% get_unique_ports


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


# %% get_capture_type


@pytest.mark.parametrize(
    'binding',
    [
        'single_tone',
        'noise',
        'sawtooth',
        'dirac_delta',
        'warmup',
        'air7101b_calibration',
    ],
)
def test_capture_type_of_bound_sweeps(binding):
    b = getattr(ss.bindings, binding)
    assert H.get_capture_type(b.sensor.sweep_spec_cls) is b.schema.capture


# %% describe_capture

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
    remap = Remap(key='frequency_offset', lookup={100.0: 1.0}, default=-1.0)
    sweep = make_sweep(
        captures=(make_capture(frequency_offset=100.0),),
        adjust_captures={'defaults': {'snr': remap}},
    )
    kws = {'adjust_spec': sweep.adjust_captures, 'source_id': None}
    text = H.describe_capture(sweep.captures[0], ('frequency_offset',), **kws)
    assert text.startswith('snr: ')
    other = H.describe_capture(
        sweep.captures[0],
        ('frequency_offset',),
        adjust_spec=sweep.adjust_captures,
        source_id='ab12',
    )
    assert text == other


# %% concat_group_sizes


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


# %% get_format_fields, get_path_fields and PathFormatter


def test_format_fields_are_extracted_in_order():
    assert H.get_format_fields('a{x}b{y!r}c{z:04d}{{}}') == ['x', 'y', 'z']
    assert H.get_format_fields('plain') == []


@given(names=st.lists(st.from_regex(r'\A[a-z][a-z0-9_]{0,8}\Z'), max_size=5))
@PROPERTY
def test_format_fields_roundtrip(names):
    template = ''.join(f'{{{name}}}' for name in names)
    assert H.get_format_fields(template) == names


def test_path_fields_with_spec_path(cw_sweep, cw_spec_path):
    fields = H.get_path_fields(cw_sweep, source_id='abcd', spec_path=cw_spec_path)
    assert set(fields) == {
        'start_time',
        'sensor_binding',
        'spec_name',
        'parent_name',
        'source_id',
    }
    assert fields['sensor_binding'] == 'single_tone'
    assert fields['spec_name'] == 'cw-cpu'
    assert fields['parent_name'] == 'sweeps'
    assert fields['source_id'] == 'abcd'
    assert re.fullmatch(r'\d{8}-\d{2}h\d{2}m\d{2}', fields['start_time'])


def test_path_fields_without_spec_path(cw_sweep):
    fields = H.get_path_fields(cw_sweep, source_id='abcd')
    assert set(fields) == {'start_time', 'sensor_binding', 'source_id'}


def test_path_fields_accept_a_callable_source_id(cw_sweep):
    fields = H.get_path_fields(cw_sweep, source_id=lambda: 'abcd')
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


def test_path_fields_are_cached_per_argument_set(cw_sweep):
    # the lru_cache also freezes start_time for repeated identical arguments
    first = H.get_path_fields(cw_sweep, source_id='cafe')
    assert H.get_path_fields(cw_sweep, source_id='cafe') is first


def test_formatter_passes_through_paths_without_fields(cw_sweep, monkeypatch):
    def fail(*args, **kws):
        raise AssertionError('lookup.id must not be called')

    monkeypatch.setattr(ss.lib.controller.lookup, 'id', fail)
    assert H.PathFormatter(cw_sweep)('outputs/noformat.zarr') == 'outputs/noformat.zarr'


def test_formatter_substitutes_fields(cw_sweep, cw_spec_path, fake_source_id):
    formatter = H.PathFormatter(cw_sweep, spec_path=cw_spec_path)
    result = formatter('outputs/{spec_name}-{source_id}.zarr')
    assert result == f'outputs/cw-cpu-{fake_source_id}.zarr'


def test_formatter_expands_user(cw_sweep, cw_spec_path, fake_source_id):
    result = H.PathFormatter(cw_sweep, spec_path=cw_spec_path)('~/{parent_name}')
    assert result == str(Path.home() / 'sweeps')


def test_formatter_names_the_unknown_field_and_the_allowed_ones(
    cw_sweep, fake_source_id
):
    with pytest.raises(KeyError, match="'nope'") as info:
        H.PathFormatter(cw_sweep)('{nope}')
    assert 'source_id' in str(info.value)


def test_formatter_resolves_spec_path(cw_sweep):
    relative = 'tests/sensor/sweeps/cw-cpu.yaml'
    assert H.PathFormatter(cw_sweep, spec_path=relative).spec_path.is_absolute()
    assert H.PathFormatter(cw_sweep).spec_path is None


@pytest.mark.xfail(
    strict=True,
    reason="Sink.path defaults to '{yaml_name}-{start_time}' but get_path_fields "
    'provides spec_name, not yaml_name',
)
def test_formatter_accepts_the_default_sink_path(
    cw_sweep, cw_spec_path, fake_source_id
):
    result = H.PathFormatter(cw_sweep, spec_path=cw_spec_path)(ss.specs.Sink().path)
    assert 'cw-cpu-' in result


def test_formatter_fills_the_site_sink_template(
    site_sweep, site_spec_path, fake_radio_id
):
    path = H.PathFormatter(site_sweep, spec_path=site_spec_path)(site_sweep.sink.path)
    assert path.startswith('../outputs/WAPA-north/site-cpu_site_')
    assert path.endswith('.zarr.zip')


# %% loop_captures


@given(sweep=sweeps())
@PROPERTY
def test_loop_count_is_the_product_of_points_and_captures(sweep):
    result = H.loop_captures(sweep)
    assert isinstance(result, tuple)
    assert len(result) == loop_point_count(sweep.loops) * len(sweep.captures)
    assert all(type(c) is CaptureCls for c in result)


def test_repeat_is_not_expanded(cw_sweep):
    assert isinstance(cw_sweep.loops[0], ss.specs.Repeat)
    assert H.loop_captures(cw_sweep) == cw_sweep.captures


def test_loop_order_is_declaration_order_with_captures_innermost():
    captures = (make_capture(port=0), make_capture(port=1))
    loops = (
        ss.specs.List(field='frequency_offset', values=(1e3, 2e3)),
        ss.specs.Range(field='snr', start=0, stop=10, step=5),
    )
    result = H.loop_captures(make_sweep(captures=captures, loops=loops))
    expected = [
        (fo, snr, port)
        for fo in (1e3, 2e3)
        for snr in (0.0, 5.0, 10.0)
        for port in (0, 1)
    ]
    assert [(c.frequency_offset, c.snr, c.port) for c in result] == expected


@given(
    values=st.lists(
        st.floats(min_value=0, max_value=1e5), min_size=1, max_size=4, unique=True
    )
)
@PROPERTY
def test_string_loop_points_are_coerced_to_the_field_type(values):
    def sweep_with(points):
        loops = (ss.specs.List(field='frequency_offset', values=tuple(points)),)
        return make_sweep(captures=(make_capture(),), loops=loops)

    numeric = H.loop_captures(sweep_with(values))
    strings = H.loop_captures(sweep_with(repr(v) for v in values))
    assert strings == numeric
    assert all(type(c.frequency_offset) is float for c in strings)


@given(sweep=sweeps(), limit=st.integers(min_value=0, max_value=12))
@PROPERTY
def test_limit_is_a_prefix(sweep, limit):
    assert H.loop_captures(sweep, limit=limit) == H.loop_captures(sweep)[:limit]


@given(sweep=sweeps(), data=st.data())
@PROPERTY
def test_only_fields_filters_capture_loops_but_keeps_analysis_loops(sweep, data):
    fields = tuple(data.draw(st.lists(st.sampled_from(LIST_LOOP_FIELDS), unique=True)))
    kept = tuple(l for l in sweep.loops if l.isin == 'analysis' or l.field in fields)
    expected = H.loop_captures(make_sweep(captures=sweep.captures, loops=kept))
    assert H.loop_captures(sweep, only_fields=fields) == expected


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


def test_unknown_loop_field_raises():
    loops = (ss.specs.List(field='nope', values=(1,)),)
    sweep = make_sweep(captures=(make_capture(),), loops=loops)
    with pytest.raises(TypeError, match='invalid capture fields'):
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


@pytest.mark.xfail(
    strict=True,
    reason='the final `c | adjust | loop_point` in _expand_capture_loops replaces '
    "adjust_analysis wholesale, dropping the capture's own entries",
)
def test_analysis_loop_merges_with_the_captures_adjust_analysis():
    capture = make_capture(adjust_analysis={'keep': 1})
    loops = (ss.specs.List(field='window', isin='analysis', values=('hann',)),)
    (result,) = H.loop_captures(make_sweep(captures=(capture,), loops=loops))
    assert dict(result.adjust_analysis) == {'keep': 1, 'window': 'hann'}


@given(
    bandwidths=st.lists(
        st.sampled_from((0.5e6, 1e6, 2e6, float('inf'))),
        min_size=1,
        max_size=4,
        unique=True,
    )
)
@PROPERTY
def test_loop_only_nyquist_keeps_bandwidths_within_the_sample_rate(bandwidths):
    captures = (make_capture(sample_rate=1e6, duration=1e-3),)
    loops = (ss.specs.List(field='analysis_bandwidth', values=tuple(bandwidths)),)
    unfiltered = H.loop_captures(make_sweep(captures=captures, loops=loops))
    options = ss.specs.SweepOptions(loop_only_nyquist=True)
    filtered = H.loop_captures(
        make_sweep(captures=captures, loops=loops, options=options)
    )
    expected = tuple(c for c in unfiltered if c.sample_rate >= c.analysis_bandwidth)
    assert filtered == expected


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason='loop_only_nyquist keeps sample_rate >= analysis_bandwidth, which drops '
    'inf even though it is the field default meaning "no analysis filter"',
)
def test_loop_only_nyquist_keeps_inf_bandwidth():
    loops = (ss.specs.List(field='analysis_bandwidth', values=(0.5e6, math.inf)),)
    options = ss.specs.SweepOptions(loop_only_nyquist=True)
    sweep = make_site_sweep(loops=loops, options=options)
    bandwidths = [c.analysis_bandwidth for c in H.loop_captures(sweep)]
    assert bandwidths == [0.5e6, math.inf]


def test_calibration_fixture_expansion(calibration_sweep):
    result = H.loop_captures(calibration_sweep)
    assert len(result) == 288
    assert all(c.sample_rate >= c.analysis_bandwidth for c in result)
    assert all(type(c.sample_rate) is float for c in result)
    # the limit applies before the nyquist filter
    assert len(H.loop_captures(calibration_sweep, limit=7)) <= 7


def test_adjustments_apply_before_loops_take_priority():
    loops = (ss.specs.List(field='lo_shift', values=('right',)),)
    sweep = make_sweep(
        captures=(make_capture(),),
        loops=loops,
        adjust_captures={'defaults': {'lo_shift': 'left'}},
    )
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
    # documented as current: the looped center_frequency values are still YAML
    # strings when the channel_name remap is evaluated, so every lookup misses
    # (see test_remaps_see_loop_values_given_as_yaml_strings)
    assert {c.channel_name for c in captures} == {None}


def test_tuple_port_fields_survive_looping():
    capture = make_capture(port=(0, 1), external_lo_frequency=(1e9, 2e9))
    loops = (ss.specs.List(field='frequency_offset', values=(1.0, 2.0)),)
    for c in H.loop_captures(make_sweep(captures=(capture,), loops=loops)):
        assert c.port == (0, 1)
        assert c.external_lo_frequency == (1e9, 2e9)


@given(
    loop=st.one_of(
        range_loop('frequency_offset'), frequency_bin_range_loop('frequency_offset')
    )
)
@PROPERTY
def test_range_loops_follow_their_own_points(loop):
    result = H.loop_captures(make_sweep(captures=(make_capture(),), loops=(loop,)))
    assert [c.frequency_offset for c in result] == loop.get_points()


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
    assert dict(c.adjust_analysis) == {'frequency_offset': -46.08e6}
    assert c.frequency_offset == pytest.approx(-3e6)


def test_meta_bounds_are_enforced_at_expansion_not_decode(site_survey_sweep):
    loops = (
        ss.specs.Range(field='azimuth', start=-45, stop=226, step=2.5),
        ss.specs.Range(field='elevation', start=0, stop=5, step=5),
    )
    sweep = site_survey_sweep.replace(loops=loops)
    with pytest.raises(msgspec.ValidationError, match='<= 180'):
        H.loop_captures(sweep, source_id=RADIO_ID)


def test_capture_post_init_is_enforced_at_expansion(site_survey_sweep):
    loops = (ss.specs.Range(field='azimuth', start=-180, stop=180, step=90),)
    sweep = site_survey_sweep.replace(loops=loops)
    with pytest.raises(msgspec.ValidationError, match='azimuth and elevation'):
        H.loop_captures(sweep, source_id=RADIO_ID)


def test_range_loop_on_an_int_field_accepts_integral_floats():
    loops = (ss.specs.Range(field='azimuth_repeat', start=0, stop=3, step=1),)
    sweep = make_site_sweep(cls=SurveySweepCls, loops=loops)
    assert [c.azimuth_repeat for c in H.loop_captures(sweep)] == [0, 1, 2, 3]


def test_range_loop_on_an_int_field_rejects_fractions():
    loops = (ss.specs.Range(field='azimuth_repeat', start=0, stop=3, step=1.5),)
    sweep = make_site_sweep(cls=SurveySweepCls, loops=loops)
    with pytest.raises(msgspec.ValidationError, match='Expected `int`'):
        H.loop_captures(sweep)


def test_remaps_see_loop_values_given_as_numbers(site_sweep):
    loops = (ss.specs.List(field='center_frequency', values=(3750e6, 3900e6)),)
    sweep = site_sweep.replace(loops=loops)
    captures = H.loop_captures(sweep, source_id=RADIO_ID)
    assert [c.channel_name for c in captures] == ['3750 MHz', '3900 MHz']


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason='loop points are coerced to the field type only after adjust_captures has '
    'run, so remaps keyed on a looped field see the raw YAML strings and miss',
)
def test_remaps_see_loop_values_given_as_yaml_strings(site_sweep):
    loops = [
        {'kind': 'list', 'field': 'center_frequency', 'values': ['3750e6', '3900e6']}
    ]
    sweep = SiteSweepCls.from_dict({**site_sweep.to_dict(), 'loops': loops})
    captures = H.loop_captures(sweep, source_id=RADIO_ID)
    assert [c.channel_name for c in captures] == ['3750 MHz', '3900 MHz']


# %% adjust_captures


def test_adjust_captures_source_overrides_and_falls_back_to_defaults():
    spec = make_sweep(adjust_captures=ADJUST).adjust_captures
    hit = H.adjust_captures(capture_dict(frequency_offset=100.0), spec, 'ab12')
    assert hit == {'lo_shift': 'right', 'snr': 11.0}
    miss = H.adjust_captures(capture_dict(frequency_offset=300.0), spec, 'ab12')
    assert miss == {'lo_shift': 'right', 'snr': -1.0}
    unknown = H.adjust_captures(capture_dict(frequency_offset=300.0), spec, None)
    assert unknown == {'lo_shift': 'left', 'snr': -1.0}


def test_adjust_captures_missing_required_source_lookup_raises():
    adjust = {'ab12': {'snr': Remap(key='frequency_offset', lookup={100: 1.0})}}
    spec = make_sweep(adjust_captures=adjust).adjust_captures
    with pytest.raises(KeyError, match='is missing a lookup for key'):
        H.adjust_captures(capture_dict(frequency_offset=300.0), spec, 'ab12')


@pytest.mark.xfail(
    strict=True,
    reason='do_lookup returns UNSET for a required defaults remap without a default, '
    'so the field is silently omitted; the required/not-required branches are inverted',
)
def test_adjust_captures_missing_required_default_lookup_raises():
    adjust = {'defaults': {'snr': Remap(key='frequency_offset', lookup={100: 1.0})}}
    spec = make_sweep(adjust_captures=adjust).adjust_captures
    with pytest.raises(KeyError, match='is missing a lookup for key'):
        H.adjust_captures(capture_dict(frequency_offset=300.0), spec, None)


def test_adjust_captures_omits_optional_misses():
    remap = Remap(key='frequency_offset', lookup={100: 1.0}, required=False)
    spec = make_sweep(adjust_captures={'defaults': {'snr': remap}}).adjust_captures
    assert H.adjust_captures(capture_dict(frequency_offset=300.0), spec, None) == {}
    hit = H.adjust_captures(capture_dict(frequency_offset=100.0), spec, None)
    assert hit == {'snr': 1.0}


def test_adjust_captures_per_port_key():
    remap = Remap(key='frequency_offset', lookup={100: 5.0, 200: 6.0})
    spec = make_sweep(adjust_captures={'defaults': {'snr': remap}}).adjust_captures
    capture = capture_dict(port=(0, 1), frequency_offset=(100.0, 200.0))
    assert H.adjust_captures(capture, spec, None) == {'snr': (5.0, 6.0)}


def test_adjust_captures_per_port_miss_bypasses_default():
    # the per-port path indexes the lookup directly, so `default` does not apply
    remap = Remap(key='frequency_offset', lookup={100: 5.0}, default=0.0)
    spec = make_sweep(adjust_captures={'defaults': {'snr': remap}}).adjust_captures
    capture = capture_dict(port=(0, 1), frequency_offset=(100.0, 200.0))
    with pytest.raises(KeyError):
        H.adjust_captures(capture, spec, None)


def test_adjust_captures_multi_field_key():
    remap = Remap(key=('frequency_offset', 'lo_shift'), lookup={'[100.0, "none"]': 7.0})
    spec = make_sweep(adjust_captures={'defaults': {'snr': remap}}).adjust_captures
    assert (100.0, 'none') in spec['defaults']['snr'].lookup
    capture = capture_dict(frequency_offset=100.0, lo_shift='none')
    assert H.adjust_captures(capture, spec, None) == {'snr': 7.0}


def test_adjust_captures_requires_a_mapping():
    spec = make_sweep(adjust_captures=ADJUST).adjust_captures
    with pytest.raises(TypeError, match='capture must be a dict or mapping'):
        H.adjust_captures(make_capture(), spec, None)


def test_port_adjustments_are_rejected_when_the_sweep_is_built():
    with pytest.raises(msgspec.ValidationError, match='not allowed by adjust_captures'):
        make_sweep(adjust_captures={'defaults': {'port': 3}})


# %% adjust_captures: site override files on the extension binding


def test_site_fields_are_accepted_by_the_bound_capture_class():
    sweep = make_site_sweep(adjust_captures=SITE_ADJUST)
    assert SiteSweepCls.from_dict(sweep.to_dict()) == sweep


def test_yaml_and_direct_adjustments_agree(site_sweep):
    direct = make_site_sweep(adjust_captures=SITE_ADJUST)
    assert site_sweep.adjust_captures == direct.adjust_captures


def test_builtin_binding_rejects_site_fields():
    match = (
        "adjust_captures field 'channel_name' was not defined in capture class "
        "'striqt.sensor.specs.SingleToneCapture'"
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
        'defaults': {
            'antenna_name': 'Unspecified',
            'antenna_model': Remap(key='antenna_name', lookup=ANTENNA_MODELS),
        },
        RADIO_ID: {
            'antenna_model': Remap(key='antenna_name', lookup=ANTENNA_MODELS),
            'antenna_name': Remap(key='port', lookup={0: 'Omni'}),
        },
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


@pytest.mark.xfail(
    strict=True,
    raises=KeyError,
    reason='the per-port branch of do_lookup indexes the lookup directly, bypassing '
    '`default`; this is how per-port center_frequency captures fail downstream',
)
def test_per_port_miss_uses_the_default(site_sweep):
    capture = site_capture_dict(port=(0, 1), center_frequency=(7350e6, 7350e6))
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


def test_remap_keyed_on_an_unknown_field_is_dropped():
    # documented as current; see the xfail below
    remap = Remap(key='centre_frequency', lookup={1.0: 'x'})
    sweep = make_site_sweep(adjust_captures={'defaults': {'channel_name': remap}})
    assert dict(sweep.adjust_captures['defaults']) == {}


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
    adjust = {
        'defaults': {
            'switch_input': 1,
            'antenna_index': Remap(key='switch_input', lookup={1: 7}),
        }
    }
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
        'defaults': {
            'antenna_model': Remap(key='antenna_name', lookup=ANTENNA_MODELS),
            'antenna_name': Remap(key='port', lookup={0: 'Omni'}),
        }
    }
    sweep = make_site_sweep(adjust_captures=adjust)
    result = H.adjust_captures(site_capture_dict(), sweep.adjust_captures, None)
    assert result.get('antenna_model') == 'OmniModel'


def test_empty_string_source_key_is_valid_hex():
    # documented as current: the built-in function sources report '' as their id
    sweep = make_site_sweep(adjust_captures={'': {'radio_name': 'anon'}})
    result = H.adjust_captures(site_capture_dict(), sweep.adjust_captures, '')
    assert result == {'radio_name': 'anon'}


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason='the error text says "global" but the accepted key is "defaults"',
)
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


def test_list_capture_adjustments_empty(cw_sweep):
    assert H.list_capture_adjustments(cw_sweep, 'ffff') == {}


@given(offsets=st.lists(st.sampled_from((100.0, 200.0, 300.0)), min_size=1, max_size=6))
@PROPERTY
def test_list_capture_adjustments_are_unique_in_first_seen_order(offsets):
    loops = (ss.specs.List(field='frequency_offset', values=tuple(offsets)),)
    sweep = make_sweep(captures=(make_capture(),), loops=loops, adjust_captures=ADJUST)
    table = {100.0: 1.0, 200.0: 2.0}
    expected = tuple(dict.fromkeys(table.get(fo, -1.0) for fo in offsets))
    assert H.list_capture_adjustments(sweep, 'ffff')['snr'] == expected


def test_list_capture_adjustments_for_a_radio(site_sweep):
    listed = H.list_capture_adjustments(site_sweep, RADIO_ID)
    assert listed['radio_name'] == ('radio02',)
    assert listed['antenna_name'] == (('Omni', '1x32'),)
    assert listed['channel_name'] == ('3750 MHz',)


# %% adjust_analysis


def test_adjust_analysis_is_identity_without_adjustments(cw_sweep):
    analysis = cw_sweep.analysis
    assert H.adjust_analysis(analysis, None) is analysis
    assert H.adjust_analysis(analysis, frozendict()) is analysis


@given(fo=st.floats(min_value=-1e4, max_value=1e4))
@PROPERTY
def test_adjust_analysis_replaces_matching_fields_everywhere(cw_sweep, fo):
    analysis = cw_sweep.analysis
    adjusted = H.adjust_analysis(analysis, frozendict({'frequency_offset': fo}))
    assert type(adjusted) is type(analysis)
    hash(adjusted)
    before, after = analysis.to_dict(), adjusted.to_dict()
    touched = {
        name
        for name, kws in before.items()
        if isinstance(kws, dict) and 'frequency_offset' in kws
    }
    assert touched
    for name in before:
        if name in touched:
            assert after[name] == {**before[name], 'frequency_offset': fo}
        else:
            assert after[name] == before[name]


def test_adjust_analysis_warns_about_unused_keys(cw_sweep, caplog):
    with caplog.at_level(logging.WARNING, logger='sweep'):
        result = H.adjust_analysis(cw_sweep.analysis, frozendict({'bogus_key': 1}))
    assert result == cw_sweep.analysis
    assert any('bogus_key' in record.getMessage() for record in caplog.records)
