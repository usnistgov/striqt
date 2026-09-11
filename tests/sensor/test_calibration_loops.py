"""y-factor calibration sweeps shaped like the downstream calibration YAML files"""

from __future__ import annotations

import math

import msgspec
import pytest
from conftest import SITE_DIR
from site_strategies import RADIO_ID, SiteCalCaptureCls, make_site_sweep
from sweep_strategies import loop_fields, make_calibration_sweep

import striqt.sensor as ss

H = ss.specs.helpers


def test_site_calibration_decodes(site_calibration_sweep):
    sweep = site_calibration_sweep
    assert type(sweep).__name__ == 'site_single_tone_calibration'
    assert sweep.options == ss.specs.SweepOptions(
        reuse_iq=True, loop_only_nyquist=True, skip_warmup=True
    )
    assert sweep.calibration.enr == pytest.approx(20.9)
    assert sweep.source.calibration is None
    assert isinstance(sweep.captures[0], SiteCalCaptureCls)
    assert sweep.captures[0].noise_diode_enabled is False


def test_toggle_is_inserted_after_the_port_loop(site_calibration_sweep):
    assert loop_fields(site_calibration_sweep) == [
        'port',
        'noise_diode_enabled',
        'sample_rate',
        'center_frequency',
        'gain',
        'analysis_bandwidth',
        'lo_shift',
    ]


def test_expanded_points_are_floats_within_nyquist(site_calibration_sweep):
    captures = H.loop_captures(site_calibration_sweep, source_id=RADIO_ID)
    assert len(captures) == 288
    assert all(type(c.sample_rate) is float for c in captures)
    assert all(c.sample_rate >= c.analysis_bandwidth for c in captures)
    # the `inf` loop value never survives the filter; see the xfail below
    assert not any(math.isinf(c.analysis_bandwidth) for c in captures)


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


def test_explicit_captures_with_implied_loops():
    sweep = ss.read_yaml_spec(SITE_DIR / 'site-calibration-explicit.yaml')
    assert loop_fields(sweep) == ['noise_diode_enabled', 'lo_shift']
    assert 'external_lo_frequency' in sweep.calibration.implied_loops
    assert len(H.loop_captures(sweep)) == 4


def test_site_adjustments_apply_to_calibration_points(site_calibration_sweep):
    captures = H.loop_captures(site_calibration_sweep, source_id=RADIO_ID)
    assert {c.radio_name for c in captures} == {'radio02'}
    assert {c.site_name for c in captures} == {'WAPA-north'}
    # documented as current: the looped center_frequency values are still YAML
    # strings when the channel_name remap is evaluated, so every lookup misses
    # (see test_remaps_see_loop_values_given_as_yaml_strings)
    assert {c.channel_name for c in captures} == {None}


def test_calibration_sink_path_uses_site_fields(site_calibration_sweep, fake_radio_id):
    fmt = H.PathFormatter(site_calibration_sweep, SITE_DIR / 'site-calibration.yaml')
    assert (
        fmt(site_calibration_sweep.sink.path) == '../../cals/WAPA-north-site-radio02.nc'
    )


def test_builtin_calibration_binding_rejects_site_fields():
    # documented as current: the generated capture class name leaks into the message
    adjust = {'defaults': {'channel_name': 'x'}}
    with pytest.raises(msgspec.ValidationError) as excinfo:
        make_calibration_sweep(adjust_captures=adjust)
    assert 'striqt.sensor.lib.calibration.capture_spec_cls' in str(excinfo.value)


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason='bind_manual_yfactor_calibration leaves the generated capture class named '
    'capture_spec_cls (it renames only the peripherals class)',
)
def test_calibration_capture_class_has_a_descriptive_name():
    assert SiteCalCaptureCls.__name__ != 'capture_spec_cls'
