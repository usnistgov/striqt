"""sweep expansion: loop_captures, adjust_captures, list_capture_adjustments, adjust_analysis"""

from __future__ import annotations

import logging

import msgspec
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from sweep_strategies import (
    LIST_LOOP_FIELDS,
    LO_SHIFTS,
    CaptureCls,
    frequency_bin_range_loop,
    loop_point_count,
    make_capture,
    make_sweep,
    range_loop,
    sweeps,
)

import striqt.sensor as ss
from striqt.analysis.specs.helpers import frozendict

H = ss.specs.helpers
Remap = ss.specs.CaptureRemap
PROPERTY = settings(
    suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None
)

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
    sweep = make_sweep(captures=(make_capture(),), loops=(ss.specs.Repeat(count=5),))
    assert len(H.loop_captures(sweep)) == 1


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


def test_zero_point_loop_yields_nothing():
    loops = (ss.specs.List(field='snr', values=()),)
    assert H.loop_captures(make_sweep(captures=(make_capture(),), loops=loops)) == ()


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
    assert len(H.loop_captures(sweep, only_fields=('duration',))) == 6


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


# %% adjust_captures


def test_adjust_captures_defaults():
    spec = make_sweep(adjust_captures=ADJUST).adjust_captures
    result = H.adjust_captures(capture_dict(frequency_offset=100.0), spec, None)
    assert result == {'lo_shift': 'left', 'snr': 1.0}


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


@given(shift=st.sampled_from(LO_SHIFTS), fo=st.floats(min_value=0, max_value=1e3))
@PROPERTY
def test_adjust_captures_passes_fixed_values_through(shift, fo):
    spec = make_sweep(adjust_captures={'defaults': {'lo_shift': shift}}).adjust_captures
    result = H.adjust_captures(capture_dict(frequency_offset=fo), spec, None)
    assert result == {'lo_shift': shift}


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


@pytest.mark.parametrize(
    'adjust, message',
    [
        ({'defaults': {'port': 3}}, 'not allowed by adjust_captures'),
        ({'defaults': {'duration': 1.0}}, 'not allowed by adjust_captures'),
        ({'defaults': {'nope': 1}}, 'was not defined in capture class'),
        ({'zz': {'lo_shift': 'left'}}, 'is not "global" or a hex string'),
    ],
)
def test_adjust_captures_are_validated_when_the_sweep_is_built(adjust, message):
    with pytest.raises(msgspec.ValidationError, match=message):
        make_sweep(adjust_captures=adjust)


def test_adjust_captures_lookup_keys_take_the_key_fields_type():
    lookup = (
        make_sweep(adjust_captures=ADJUST).adjust_captures['defaults']['snr'].lookup
    )
    assert isinstance(lookup, frozendict)
    assert lookup == {100.0: 1.0, 200.0: 2.0}
    assert all(type(k) is float for k in lookup)


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
