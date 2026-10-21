"""striqt.sensor.lib.actions.yfactor: the sweep, passes, analysis, sink, and operator
interaction a manual Y-factor calibration runner derives from its document
"""

from __future__ import annotations

import math
from fractions import Fraction
from pathlib import Path

import msgspec
import pytest
from sweep_strategies import (
    GAIN_LOOP,
    PORT_LOOP,
    CalSweepCls,
    SweepCls,
    make_calibration_capture,
    make_calibration_runner,
    make_calibration_sweep,
)

import striqt.sensor as ss
from striqt.sensor.lib.actions import ManualYFactorRunner, registry, yfactor

Repeat = ss.specs.Repeat
TOGGLE = 'noise_diode_enabled'
DURATION_LOOP = ss.specs.List(field='duration', values=(1e-3, 2e-3))


# %% calibration_analysis


def test_explicit_detector_period_is_used():
    cal = ss.specs.ManualYFactorSpec(
        enr=20.0, ambient_temperature=290.0, detector_period=Fraction(1, 4000)
    )
    analysis = yfactor.calibration_analysis(make_calibration_sweep(calibration=cal))
    meas = analysis.channel_power_time_series
    assert meas.detector_period == Fraction(1, 4000)
    assert set(meas.power_detectors) == {'rms', 'peak'}
    assert [k for k, v in analysis.to_dict().items() if v is not None] == [
        'channel_power_time_series'
    ]


def test_default_detector_period_is_the_capture_duration():
    """one bin per capture: the rms mean that the Y-factor math uses is unchanged,
    and the PAPR diagnostics collapse to 0 dB"""
    sweep = make_calibration_sweep(captures=(make_calibration_capture(duration=2e-3),))
    meas = yfactor.calibration_analysis(sweep).channel_power_time_series
    assert meas.detector_period == Fraction(1, 500)


def test_default_detector_period_needs_one_duration():
    sweep = make_calibration_sweep(loops=(DURATION_LOOP,))
    with pytest.raises(msgspec.ValidationError, match=r'calibration\.detector_period'):
        yfactor.calibration_analysis(sweep)


# %% ManualYFactorRunner.build_sweep


def test_build_sweep_is_the_plan_with_generated_analysis_and_fixed_options():
    runner = make_calibration_runner(loops=(PORT_LOOP, GAIN_LOOP))
    doc, sweep = runner.spec, runner.sweep
    assert type(sweep) is type(doc).sweep_cls
    assert isinstance(sweep, ss.specs.Sweep)
    assert (sweep.source, sweep.captures, sweep.loops) == (
        doc.source,
        doc.captures,
        doc.loops,
    )
    assert sweep.analysis == yfactor.calibration_analysis(doc)
    assert sweep.options == ss.specs.SweepOptions(
        reuse_iq=False, loop_only_nyquist=True, skip_warmup=True
    )
    assert runner.warmup_sweep() is None


def test_build_sweep_applies_the_nyquist_filter():
    loops = (ss.specs.List(field='analysis_bandwidth', values=(0.5e6, 2e6, math.inf)),)
    runner = make_calibration_runner(loops=loops)
    bandwidths = {c.analysis_bandwidth for c in H.loop_captures(runner.sweep)}
    assert bandwidths == {0.5e6, math.inf}


# %% passes and pass_sweep

PASS_CASES = {
    'port_loop': ({'loops': (PORT_LOOP, GAIN_LOOP)}, (0, 1)),
    'single_capture': ({}, (0,)),
    'explicit_ports': (
        {
            'captures': (
                make_calibration_capture(port=1),
                make_calibration_capture(port=0),
            ),
            'calibration': ss.specs.ManualYFactorSpec(
                enr=20.0, ambient_temperature=290.0, implied_loops=('port',)
            ),
        },
        (1, 0),
    ),
}


@pytest.mark.parametrize('case', list(PASS_CASES), ids=list(PASS_CASES))
def test_passes_are_port_major_with_the_diode_off_then_on(case):
    kws, ports = PASS_CASES[case]
    runner = make_calibration_runner(**kws)
    assert runner.ports() == ports
    assert runner.passes() == [(p, on) for p in ports for on in (False, True)]


def test_pass_sweep_fixes_the_port_and_drops_the_port_loop():
    runner = make_calibration_runner(loops=(Repeat(count=2), PORT_LOOP, GAIN_LOOP))
    sweep = runner.pass_sweep(1)
    assert [loop.field for loop in sweep.loops] == [None, 'gain']
    assert {c.port for c in sweep.captures} == {1}
    assert {c.port for c in H.loop_captures(sweep)} == {1}


def test_pass_sweep_selects_the_captures_of_the_port():
    kws, _ = PASS_CASES['explicit_ports']
    runner = make_calibration_runner(**kws)
    assert [c.port for c in runner.pass_sweep(0).captures] == [0]
    assert [c.port for c in runner.pass_sweep(1).captures] == [1]


# %% build_sink and calibration_fields


def test_calibration_fields_follow_the_loops_with_the_diode_after_port():
    loops = (
        PORT_LOOP,
        ss.specs.List(field='sample_rate', values=(1e6, 2e6)),
        GAIN_LOOP,
    )
    cal = ss.specs.ManualYFactorSpec(
        enr=20.0, ambient_temperature=290.0, implied_loops=('center_frequency',)
    )
    runner = make_calibration_runner(loops=loops, calibration=cal)
    assert runner.calibration_fields() == [
        'center_frequency',
        'port',
        TOGGLE,
        'backend_sample_rate',
        'gain',
    ]


def test_calibration_fields_put_the_diode_first_without_a_port_loop():
    runner = make_calibration_runner(loops=(GAIN_LOOP,))
    assert runner.calibration_fields() == [TOGGLE, 'gain']


def test_build_sink_collects_in_memory(fake_source_id):
    runner = make_calibration_runner(loops=(PORT_LOOP,))
    assert type(runner.build_sink(None)) is ss.sinks.MemoryCacheSink
    assert runner.calibration_fields() == ['port', TOGGLE]


def test_corrections_path_is_the_formatted_sink_path(fake_source_id):
    runner = make_calibration_runner(sink=ss.specs.Sink(path='cal-{sensor_binding}.nc'))
    fmt = ss.specs.paths.PathFormatter(runner.spec)
    assert runner.corrections_path(None) == Path('cal-{sensor_binding}.nc')
    assert runner.corrections_path(fmt) == Path('cal-air7101b_calibration.nc')


def test_build_sink_honors_the_extensions_override(fake_source_id):
    ext = ss.specs.Extension(sink='striqt.sensor.sinks.NoSink')
    runner = make_calibration_runner(extensions=ext)
    assert type(runner.build_sink(None)) is ss.sinks.NoSink


# %% registry and binding


def test_registered_for_calibration_documents_only():
    assert registry.lookup(CalSweepCls) is ManualYFactorRunner
    assert registry.lookup(SweepCls) is not ManualYFactorRunner


def test_bind_sensor_with_the_action_matches_the_derived_binding(isolated_bindings):
    """`bind_manual_yfactor_calibration` is `bind_sensor` with
    `Schema(action=ManualYFactorCalibration)`"""
    parent = ss.bindings.air7101b
    derived = ss.bindings.air7101b_calibration
    schema = ss.bindings.Schema(
        source=parent.schema.source,
        init_like=parent.schema.init_like,
        capture=parent.schema.capture,
        arm_like=parent.schema.arm_like,
        peripherals=parent.schema.peripherals,
        action=ss.specs.ManualYFactorCalibration,
    )
    direct = ss.bindings.bind_sensor(
        'direct_cal_test', parent.sensor, schema, register=False
    )
    assert direct.schema.capture is derived.schema.capture
    assert direct.sensor.peripherals_cls is derived.sensor.peripherals_cls
    assert direct.sensor.sweep_spec_cls.__struct_fields__ == (
        derived.sensor.sweep_spec_cls.__struct_fields__
    )


def test_iterate_does_not_loop():
    runner = make_calibration_runner()
    with pytest.raises(ValueError, match='loop'):
        next(runner.iterate({}, loop=True))


H = ss.specs.sequencing
