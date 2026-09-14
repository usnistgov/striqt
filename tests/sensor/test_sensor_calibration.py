"""striqt.sensor.lib.calibration.bind_manual_yfactor_calibration: the generated
calibration sweep and capture classes, and noise diode loop insertion"""

from __future__ import annotations

import msgspec
import pytest
from conftest import SITE_DIR
from hypothesis import given
from site_strategies import SiteCalCaptureCls
from sweep_strategies import (
    GAIN_LOOP,
    NDE_LOOP,
    PORT_LOOP,
    CalSweepCls,
    calibration_loop_orderings,
    calibration_sweep_dict,
    loop_fields,
    make_calibration_sweep,
)

import striqt.sensor as ss

H = ss.specs.helpers
Repeat = ss.specs.Repeat
TOGGLE = 'noise_diode_enabled'
TOGGLE_MSG = 'noise_diode_enabled must be the first specified loop'


# %% noise diode loop insertion


class TestNoiseDiodeToggle:
    @given(loops_accepted=calibration_loop_orderings())
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

    def test_explicit_captures_with_implied_loops(self):
        sweep = ss.read_yaml_spec(SITE_DIR / 'site-calibration-explicit.yaml')
        assert loop_fields(sweep) == [TOGGLE, 'lo_shift']
        assert 'external_lo_frequency' in sweep.calibration.implied_loops
        assert len(H.loop_captures(sweep)) == 4


# %% generated classes


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason='bind_manual_yfactor_calibration leaves the generated capture class named '
    'capture_spec_cls (it renames only the peripherals class)',
)
def test_calibration_capture_class_has_a_descriptive_name():
    assert SiteCalCaptureCls.__name__ != 'capture_spec_cls'
