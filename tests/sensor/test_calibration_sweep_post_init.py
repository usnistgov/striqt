"""CalibrationSweep and the calibration binding's noise diode loop insertion"""

from __future__ import annotations

import msgspec
import pytest
from hypothesis import HealthCheck, given, settings
from sweep_strategies import (
    GAIN_LOOP,
    NDE_LOOP,
    PORT_LOOP,
    CalSourceCls,
    CalSweepCls,
    calibration_loop_orderings,
    calibration_sweep_dict,
    loop_fields,
    make_calibration_capture,
    make_calibration_sweep,
)

import striqt.sensor as ss

Repeat = ss.specs.Repeat
TOGGLE = 'noise_diode_enabled'
TOGGLE_MSG = 'noise_diode_enabled must be the first specified loop'
IMPLIED_MSG = (
    'calibration sweeps may only include explicit capture sequences '
    'if implied_loops are specified'
)
SOURCE_CAL_MSG = 'source.calibration must be None for a calibration sweep'
PROPERTY = settings(
    suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None
)


def test_fixture_calibration_sweep_is_happy_path(calibration_sweep):
    assert isinstance(calibration_sweep, CalSweepCls)
    assert loop_fields(calibration_sweep)[:2] == ['port', TOGGLE]


class TestNoiseDiodeToggle:
    @pytest.mark.parametrize(
        'loops, expected',
        [
            ((), [TOGGLE]),
            ((GAIN_LOOP, PORT_LOOP), [TOGGLE, 'gain', 'port']),
            ((PORT_LOOP, GAIN_LOOP), ['port', TOGGLE, 'gain']),
        ],
    )
    def test_toggle_inserted(self, loops, expected):
        assert loop_fields(make_calibration_sweep(loops=loops)) == expected
        from_dict = CalSweepCls.from_dict(calibration_sweep_dict(loops=loops))
        assert loop_fields(from_dict) == expected

    @given(loops_accepted=calibration_loop_orderings())
    @PROPERTY
    def test_explicit_toggle_position(self, loops_accepted):
        loops, accepted = loops_accepted
        if accepted:
            fields = loop_fields(make_calibration_sweep(loops=loops))
            assert fields == loop_fields(
                CalSweepCls.from_dict(calibration_sweep_dict(loops=loops))
            )
            idx = fields.index(TOGGLE)
            assert idx == 0 or (idx == 1 and fields[0] == 'port')
            assert [f for f in fields if f != TOGGLE] == [
                l.field for l in loops if l.field != TOGGLE
            ]
        else:
            with pytest.raises(TypeError, match=TOGGLE_MSG):
                make_calibration_sweep(loops=loops)
            with pytest.raises(msgspec.ValidationError, match=TOGGLE_MSG):
                CalSweepCls.from_dict(calibration_sweep_dict(loops=loops))

    def test_replace_loops_reinserts_or_rejects(self):
        sweep = make_calibration_sweep()
        with pytest.raises(TypeError, match=TOGGLE_MSG):
            sweep.replace(loops=(GAIN_LOOP, NDE_LOOP))
        assert loop_fields(sweep.replace(loops=(PORT_LOOP,))) == ['port', TOGGLE]

    @pytest.mark.xfail(
        strict=True,
        reason=(
            '_ensure_loop_at_position does not account for a leading Repeat, '
            'which _validate_loops requires to be first'
        ),
    )
    @pytest.mark.parametrize(
        'loops, expected',
        [
            ((Repeat(count=2), GAIN_LOOP), [None, TOGGLE, 'gain']),
            ((Repeat(count=2), NDE_LOOP, GAIN_LOOP), [None, TOGGLE, 'gain']),
        ],
    )
    def test_leading_repeat_is_accepted(self, loops, expected):
        assert loop_fields(make_calibration_sweep(loops=loops)) == expected


class TestCaptures:
    def test_single_capture_needs_no_implied_loops(self):
        sweep = make_calibration_sweep()
        assert len(sweep.captures) == 1
        CalSweepCls.from_dict(calibration_sweep_dict())

    def test_multiple_captures_need_implied_loops(self):
        captures = (make_calibration_capture(), make_calibration_capture(gain=-10))
        with pytest.raises(TypeError, match=IMPLIED_MSG):
            make_calibration_sweep(captures=captures)
        with pytest.raises(msgspec.ValidationError, match=IMPLIED_MSG):
            CalSweepCls.from_dict(calibration_sweep_dict(captures=captures))

        peripheral = ss.specs.ManualYFactorPeripheral(
            enr=10, ambient_temperature=290, implied_loops=('gain',)
        )
        sweep = make_calibration_sweep(captures=captures, calibration=peripheral)
        assert len(sweep.captures) == 2
        CalSweepCls.from_dict(
            calibration_sweep_dict(captures=captures, calibration=peripheral.to_dict())
        )

    def test_source_calibration_must_be_none(self):
        source = CalSourceCls(calibration='cal.nc')
        with pytest.raises(ValueError, match=SOURCE_CAL_MSG):
            make_calibration_sweep(source=source)
        with pytest.raises(msgspec.ValidationError, match=SOURCE_CAL_MSG):
            CalSweepCls.from_dict(calibration_sweep_dict(source=source))

    def test_default_options_are_calibration_defaults(self):
        expected = ss.specs.SweepOptions(
            reuse_iq=True, loop_only_nyquist=True, skip_warmup=True
        )
        assert make_calibration_sweep().options == expected
