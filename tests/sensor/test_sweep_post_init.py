"""Sweep and BoundSweep __post_init__ validation"""

from __future__ import annotations

import re

import msgspec
import pytest
from hypothesis import HealthCheck, given, settings
from sweep_strategies import (
    SOURCE,
    SweepCls,
    duplicate_field_loops,
    loop_sets,
    make_capture,
    make_sweep,
    misplaced_repeat_loops,
    sweep_dict,
)

import striqt.sensor as ss
from striqt.analysis.specs.helpers import frozendict

List = ss.specs.List
Repeat = ss.specs.Repeat
Remap = ss.specs.CaptureRemap
REPEAT_MSG = r'a repeat may only be the outermost \(first\) loop'
DUPLICATE_MSG = 'more than one loop specified for capture field'
MOCK_MSG = 'no sensor was bound with this name'
PROPERTY = settings(
    suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None
)


class _ConflictingCapture(ss.specs.SingleToneCapture, frozen=True, kw_only=True):
    spectrogram: int = 0


class TestLoops:
    @given(loops=loop_sets())
    @PROPERTY
    def test_valid_loop_sets_accepted(self, loops):
        sweep = make_sweep(captures=(make_capture(),), loops=loops)
        assert sweep.loops == loops
        assert SweepCls.from_dict(sweep.to_dict()) == sweep

    @given(loops=misplaced_repeat_loops())
    @PROPERTY
    def test_repeat_must_be_first(self, loops):
        captures = (make_capture(),)
        with pytest.raises(msgspec.ValidationError, match=REPEAT_MSG):
            make_sweep(captures=captures, loops=loops)
        with pytest.raises(msgspec.ValidationError, match=REPEAT_MSG):
            SweepCls.from_dict(sweep_dict(captures, loops))

    def test_two_repeats_rejected(self):
        with pytest.raises(msgspec.ValidationError, match=REPEAT_MSG):
            make_sweep(loops=(Repeat(count=2), Repeat(count=3)))

    @given(loops_field=duplicate_field_loops())
    @PROPERTY
    def test_duplicate_loop_field_rejected(self, loops_field):
        loops, field = loops_field
        match = f"{DUPLICATE_MSG} '{field}'"
        with pytest.raises(msgspec.ValidationError, match=match):
            make_sweep(loops=loops)
        with pytest.raises(msgspec.ValidationError, match=match):
            SweepCls.from_dict(sweep_dict(loops=loops))

    def test_replace_loops_revalidates(self):
        sweep = make_sweep(captures=(make_capture(),))
        with pytest.raises(msgspec.ValidationError, match=REPEAT_MSG):
            sweep.replace(loops=(List(field='snr', values=(1.0,)), Repeat(count=2)))


class TestCaptures:
    @pytest.mark.parametrize('cls', [ss.specs.Sweep, ss.specs.CalibrationSweep])
    def test_bare_sweep_is_not_constructible(self, cls):
        # the capture type comes from a bound schema
        with pytest.raises(TypeError, match='Must be called with a struct type'):
            cls(source=SOURCE)

    def test_capture_field_conflicting_with_measurement_name(self):
        capture = _ConflictingCapture(port=0, sample_rate=1e6, duration=1e-3)
        match = r"capture fields \('spectrogram',\) conflict with measurements"
        with pytest.raises(AttributeError, match=match):
            make_sweep(captures=(capture,))


class TestAdjustCaptures:
    @pytest.mark.parametrize('path', ['direct', 'from_dict'])
    def test_lookup_keys_are_converted_before_freezing(self, path):
        remap = Remap(key='frequency_offset', lookup={'1': 2.0})
        if path == 'direct':
            sweep = make_sweep(adjust_captures={'defaults': {'snr': remap}})
        else:
            adjust = {'defaults': {'snr': remap.to_dict()}}
            sweep = SweepCls.from_dict(sweep_dict(adjust_captures=adjust))
        assert isinstance(sweep.adjust_captures, frozendict)
        lookup = sweep.adjust_captures['defaults']['snr'].lookup
        assert lookup == {1.0: 2.0}
        assert isinstance(next(iter(lookup)), float)

    @pytest.mark.parametrize(
        'adjust, match',
        [
            ({'zz': {'snr': 1.0}}, 'is not "global" or a hex string'),
            (
                {'defaults': {'duration': 1.0}},
                "capture field 'duration' is not allowed by adjust_captures",
            ),
        ],
    )
    def test_invalid_adjustments_rejected(self, adjust, match):
        with pytest.raises(msgspec.ValidationError, match=match):
            make_sweep(adjust_captures=adjust)
        with pytest.raises(msgspec.ValidationError, match=match):
            SweepCls.from_dict(sweep_dict(adjust_captures=adjust))

    def test_tuple_form_is_broken(self):
        # the tuple branch of _get_capture_adjust_map references an undefined name
        with pytest.raises(NameError, match='source_fields'):
            make_sweep(adjust_captures=(('defaults', {'snr': 1.0}),))


class TestMockSource:
    def test_must_be_a_registered_binding(self):
        with pytest.raises(TypeError, match=MOCK_MSG):
            make_sweep(mock_source='bogus')
        with pytest.raises(msgspec.ValidationError, match=MOCK_MSG):
            SweepCls.from_dict(sweep_dict(mock_source='bogus'))

    def test_check_runs_before_loop_validation(self):
        loops = (List(field='snr', values=(1.0,)), Repeat(count=2))
        with pytest.raises(TypeError, match=MOCK_MSG):
            make_sweep(mock_source='bogus', loops=loops)

    def test_message_lists_registered_bindings(self):
        names = re.escape(repr(tuple(ss.lib.bindings.registry)))
        with pytest.raises(TypeError, match=names):
            make_sweep(mock_source='bogus')

    def test_registered_binding_accepted(self, construct):
        sweep = construct(SweepCls, source=SOURCE.to_dict(), mock_source='warmup')
        assert sweep.mock_source == 'warmup'

    @pytest.mark.xfail(
        strict=True,
        reason='BoundSweep.__post_init__ error text says mock_sensor, not mock_source',
    )
    def test_message_names_the_field(self):
        with pytest.raises(TypeError, match=r"mock_source 'bogus'"):
            make_sweep(mock_source='bogus')
