"""striqt.sensor.lib.actions.yfactor: the Y-factor corrections, and the sweep, passes,
analysis, sink, and operator interaction a calibration runner derives from its spec
"""

from __future__ import annotations

import logging
import math
import re
from fractions import Fraction
from pathlib import Path

import msgspec
import numpy as np
import pytest
import xarray as xr
from conftest import SITE_DIR
from sweep_strategies import (
    CAL,
    GAIN_LOOP,
    INF,
    PORT_LOOP,
    YFACTOR_ENR_DB,
    YFACTOR_GRID,
    CalSweepCls,
    SweepCls,
    assert_corrections_match,
    expected_yfactor,
    make_calibration_capture,
    make_calibration_runner,
    make_calibration_sweep,
    make_yfactor_dataset,
    yfactor_rms_power,
)

import striqt.sensor as ss
from striqt.sensor.lib.actions import ManualYFactorRunner, registry, warmup, yfactor

H = ss.specs.sequencing
Repeat = ss.specs.Repeat
TOGGLE = 'noise_diode_enabled'
DURATION_LOOP = ss.specs.List(field='duration', values=(1e-3, 2e-3))


# %% compute_y_factor_corrections


class TestYFactorCorrections:
    def test_recovers_the_receiver_model(self):
        corrections = yfactor.compute_y_factor_corrections(
            make_yfactor_dataset(), YFACTOR_ENR_DB
        )
        assert_corrections_match(corrections, expected_yfactor())

    def test_frequency_dependent_noise_figure(self):
        nf = {0: {1e9: 5.0, 2e9: 7.0}, 1: 8.0}
        corrections = yfactor.compute_y_factor_corrections(
            make_yfactor_dataset(nf_dB=nf), YFACTOR_ENR_DB
        )
        assert_corrections_match(corrections, expected_yfactor(nf_dB=nf))

    def test_output_is_indexed_by_the_calibration_fields(self):
        corrections = yfactor.compute_y_factor_corrections(
            make_yfactor_dataset(), YFACTOR_ENR_DB
        )
        assert set(corrections.dims) == set(YFACTOR_GRID)
        assert 'noise_diode_enabled' not in corrections.coords
        assert corrections.noise_figure.attrs['units'] == 'dB'
        assert corrections.power_correction.attrs['units'] == 'mW/fs'

    def test_papr_and_raw_powers(self):
        ds = make_yfactor_dataset(papr_dB=7.5)
        corrections = yfactor.compute_y_factor_corrections(ds, YFACTOR_ENR_DB)
        assert corrections.papr_on.values == pytest.approx(7.5)
        assert corrections.papr_off.values == pytest.approx(7.5)
        p_off = yfactor_rms_power(0.0, 5.0, 40e6, diode_on=False)
        p_on = yfactor_rms_power(0.0, 5.0, 40e6, diode_on=True)
        point = {'port': 0, 'gain': 0.0, 'center_frequency': 1e9}
        assert corrections.p_off.sel(point).item() == pytest.approx(p_off, rel=1e-9)
        assert corrections.p_on.sel(point).item() == pytest.approx(p_on, rel=1e-9)

    def test_infinite_bandwidth_uses_the_nyquist_bandwidth(self):
        grid = dict(YFACTOR_GRID, analysis_bandwidth=(40e6, INF))
        corrections = yfactor.compute_y_factor_corrections(
            make_yfactor_dataset(grid), YFACTOR_ENR_DB
        )
        assert_corrections_match(corrections, expected_yfactor(grid))


# %% _limit_nyquist_bandwidth


def _bandwidth_grid():
    return xr.DataArray(
        np.zeros((2, 2)),
        dims=('analysis_bandwidth', 'backend_sample_rate'),
        coords={
            'analysis_bandwidth': [40e6, INF],
            'backend_sample_rate': [125e6, 62.5e6],
        },
    )


@pytest.mark.parametrize(
    'analysis_bandwidth, expected',
    [(40e6, [40e6, 40e6]), (INF, [125e6, 62.5e6])],
    ids=['finite', 'infinite'],
)
def test_limit_nyquist_bandwidth(analysis_bandwidth, expected):
    """a finite bandwidth is kept as it is, and an infinite one becomes each
    backend sample rate"""
    bw = yfactor._limit_nyquist_bandwidth(_bandwidth_grid())
    assert bw.sel(analysis_bandwidth=analysis_bandwidth).values.tolist() == expected


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


def test_default_detector_period_follows_a_duration_loop():
    """a looped duration overrides the capture's own, so one loop point is one
    duration even when the capture says otherwise"""
    loop = ss.specs.List(field='duration', values=(5e-3,))
    sweep = make_calibration_sweep(
        captures=(make_calibration_capture(duration=1e-3),), loops=(loop,)
    )
    meas = yfactor.calibration_analysis(sweep).channel_power_time_series
    assert meas.detector_period == Fraction(1, 200)


def test_default_detector_period_needs_one_duration():
    """the runner's `validate` runs when the bound spec is constructed, so the
    `detector_period` error is raised there"""
    with pytest.raises(msgspec.ValidationError, match=r'calibration\.detector_period'):
        make_calibration_sweep(loops=(DURATION_LOOP,))


# %% build_calibration_sweep and validate


def test_build_sweep_is_cached_on_the_spec():
    runner = make_calibration_runner()
    assert yfactor.build_calibration_sweep(runner.spec) is runner.sweep


def test_validate_accepts_a_spec_whose_sweep_builds():
    doc = make_calibration_sweep(loops=(PORT_LOOP,))
    assert ss.lib.actions.ManualYFactorRunner.validate(doc) is None


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
        reuse_iq=True, loop_only_nyquist=True, skip_warmup=True
    )
    assert warmup.warmup_sweep_for(runner.sweep) is None


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
    assert yfactor.calibration_ports(runner.spec) == ports
    assert yfactor.calibration_passes(runner.spec) == [
        (p, on) for p in ports for on in (False, True)
    ]


def test_pass_sweep_fixes_the_port_and_drops_the_port_loop():
    runner = make_calibration_runner(loops=(Repeat(count=2), PORT_LOOP, GAIN_LOOP))
    sweep = yfactor.pass_sweep(runner.sweep, 1)
    assert [loop.field for loop in sweep.loops] == [None, 'gain']
    assert {c.port for c in sweep.captures} == {1}
    assert {c.port for c in H.loop_captures(sweep)} == {1}


def test_pass_sweep_selects_the_captures_of_the_port():
    kws, _ = PASS_CASES['explicit_ports']
    runner = make_calibration_runner(**kws)
    assert [c.port for c in yfactor.pass_sweep(runner.sweep, 0).captures] == [0]
    assert [c.port for c in yfactor.pass_sweep(runner.sweep, 1).captures] == [1]


def test_explicit_captures_with_implied_loops():
    """two explicit captures at different ports give one pass per port, each over
    its own capture, so the whole calibration acquires 2 ports x 2 states x 1"""
    sweep = ss.read_yaml_spec(SITE_DIR / 'site-calibration-explicit.yaml')
    runner = ManualYFactorRunner(sweep)
    assert 'external_lo_frequency' in sweep.calibration.implied_loops
    assert yfactor.calibration_ports(sweep) == (0, 1)
    pass_sizes = [
        len(H.loop_captures(yfactor.pass_sweep(runner.sweep, p))) for p in (0, 1)
    ]
    assert pass_sizes == [1, 1]
    assert len(yfactor.calibration_passes(sweep)) == 4


# %% open_sink and calibration_fields


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
    assert yfactor.calibration_fields(runner.spec, runner.sweep) == [
        'center_frequency',
        'port',
        TOGGLE,
        'backend_sample_rate',
        'gain',
    ]


def test_calibration_fields_put_the_diode_first_without_a_port_loop():
    runner = make_calibration_runner(loops=(GAIN_LOOP,))
    assert yfactor.calibration_fields(runner.spec, runner.sweep) == [TOGGLE, 'gain']


def test_sink_collects_in_memory():
    runner = make_calibration_runner(loops=(PORT_LOOP,))
    assert isinstance(runner.open_sink(None), ss.sinks.MemoryCacheSink)
    assert yfactor.calibration_fields(runner.spec, runner.sweep) == ['port', TOGGLE]


def test_calibration_sweep_widens_the_sink_to_a_zarr_sink():
    sink = ss.specs.Sink(path='cal.nc', log_path='log.json')
    runner = make_calibration_runner(sink=sink)
    assert runner.spec.sink == ss.specs.Sink(path='cal.nc', log_path='log.json')
    assert runner.sweep.sink == ss.specs.ZarrSink(path='cal.nc', log_path='log.json')


def test_corrections_path_is_the_formatted_sink_path(fake_source_id):
    runner = make_calibration_runner(sink=ss.specs.Sink(path='cal-{sensor_binding}.nc'))
    fmt = ss.specs.paths.PathFormatter(runner.spec)
    assert yfactor.corrections_path(runner.sweep, None) == Path(
        'cal-{sensor_binding}.nc'
    )
    assert yfactor.corrections_path(runner.sweep, fmt) == Path(
        'cal-air7101b_calibration.nc'
    )


def test_corrections_path_rejects_a_null_path():
    runner = make_calibration_runner(sink=ss.specs.Sink(path=None))
    with pytest.raises(ValueError, match='path'):
        yfactor.corrections_path(runner.sweep, None)


# %% operator prompts


def _calibration_runner():
    return make_calibration_runner(
        calibration=ss.specs.ManualYFactorSpec(enr=20.87, ambient_temperature=290.0)
    )


class TestOperatorPrompts:
    def test_confirm_enr_names_the_value(self, scripted_input):
        scripted_input['answers'] += ['y']
        yfactor.confirm_enr(20.87)
        (prompt,) = scripted_input['prompts']
        assert 'ENR' in prompt and '20.87' in prompt

    def test_confirm_enr_reprompts_until_answered(self, scripted_input):
        scripted_input['answers'] += ['maybe', '', 'Y']
        yfactor.confirm_enr(20.87)
        assert len(scripted_input['prompts']) == 3

    def test_confirm_enr_rejection_raises(self, scripted_input):
        scripted_input['answers'] += ['N']
        with pytest.raises(RuntimeError, match='ENR'):
            yfactor.confirm_enr(20.87)

    def test_switch_diode_prompts_per_pass(self, scripted_input):
        scripted_input['answers'] += ['', '', '']
        yfactor.switch_diode(0, False)
        yfactor.switch_diode(0, True)
        yfactor.switch_diode(1, True)

        # conftest.fake_sweep_runner parses this wording to switch the fake diode
        matches = [
            re.match(r'(enable|disable) noise diode at port (\d+)', prompt)
            for prompt in scripted_input['prompts']
        ]
        assert [m.groups() for m in matches if m is not None] == [
            ('disable', '0'),
            ('enable', '0'),
            ('enable', '1'),
        ]


# %% registry and binding


def test_binding_derives_its_classes_from_the_parent():
    """the calibration binding is the parent's sensor and spec types under the
    calibration action: no derived capture, sink, or peripherals class"""
    parent = ss.bindings.air7101b
    assert CAL.schema.capture is parent.schema.capture
    assert CAL.schema.source is parent.schema.source
    assert CAL.sensor.source_cls is parent.sensor.source_cls
    assert CAL.sensor.peripherals_cls is parent.sensor.peripherals_cls
    action_cls = CAL.sensor.sweep_spec_cls
    assert issubclass(action_cls, ss.specs.ManualYFactorCalibration)
    assert not issubclass(action_cls, ss.specs.Sweep)
    assert issubclass(action_cls.sweep_cls, ss.specs.Sweep)
    assert action_cls.controller is CAL


def test_registered_for_calibration_specs_only():
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


class _RecordingSource:
    source_id = 'beef'

    def __init__(self):
        self.discards = 0

    def discard_iq(self):
        self.discards += 1


def test_each_pass_begins_with_a_fresh_acquisition(monkeypatch, scripted_input):
    """reuse_iq is on, so the controller would serve the diode-off IQ to an equal
    diode-on capture; the runner discards the cached acquisition at every switch and
    hands the pipeline the new state"""
    runner = make_calibration_runner(loops=(PORT_LOOP,))
    scripted_input['answers'] += ['y'] + [''] * len(
        yfactor.calibration_passes(runner.spec)
    )
    source = _RecordingSource()
    seen_passes = []

    def fake_iterate_captures(resources, **kws):
        seen_passes.append((source.discards, dict(kws['extra_data'])))
        yield from ()

    monkeypatch.setattr(yfactor.core, 'iterate_captures', fake_iterate_captures)
    resources = {'source': source, 'sink': ss.sinks.NoSink, 'format_path': None}
    list(runner.iterate(resources))

    assert seen_passes == [
        (1, {TOGGLE: False}),
        (2, {TOGGLE: True}),
        (3, {TOGGLE: False}),
        (4, {TOGGLE: True}),
    ]


def _run_passes_into_a_memory_sink(runner, monkeypatch, scripted_input):
    """drive every pass of `runner` with no acquisitions into a `MemoryCacheSink`
    and return the calls made to `save_yfactor_corrections`"""
    scripted_input['answers'] += ['y'] + [''] * len(
        yfactor.calibration_passes(runner.spec)
    )
    saves = []

    def fake_iterate_captures(resources, **kws):
        yield from ()

    def record_save(data, calibration_fields, path, *, enr_dB):
        saves.append({'path': path, 'enr_dB': enr_dB})

    monkeypatch.setattr(yfactor.core, 'iterate_captures', fake_iterate_captures)
    monkeypatch.setattr(yfactor, 'save_yfactor_corrections', record_save)
    resources = {
        'source': _RecordingSource(),
        'sink': ss.sinks.MemoryCacheSink(runner.sweep),
        'format_path': None,
    }
    list(runner.iterate(resources))
    return saves


def test_null_sink_path_skips_the_save(monkeypatch, scripted_input, caplog):
    runner = make_calibration_runner(sink=ss.specs.Sink(path=None))
    caplog.set_level(logging.INFO, logger='striqt.sink')
    saves = _run_passes_into_a_memory_sink(runner, monkeypatch, scripted_input)
    assert saves == []
    assert any('not written' in record.getMessage() for record in caplog.records)


def test_sink_path_is_saved_once_after_the_last_pass(
    monkeypatch, scripted_input, tmp_path
):
    path = tmp_path / 'cal.nc'
    runner = make_calibration_runner(sink=ss.specs.Sink(path=str(path)))
    saves = _run_passes_into_a_memory_sink(runner, monkeypatch, scripted_input)
    assert saves == [{'path': path, 'enr_dB': runner.spec.calibration.enr}]


def test_iterate_does_not_loop():
    runner = make_calibration_runner()
    with pytest.raises(ValueError, match='loop'):
        next(runner.iterate({}, loop=True))
