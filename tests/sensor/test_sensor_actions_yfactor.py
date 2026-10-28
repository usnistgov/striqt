"""striqt.sensor.lib.actions.yfactor: the Y-factor corrections, and the calibration
runner that derives its sweep, passes, analysis, and operator prompts from a
`ManualYFactorCalibration` spec
"""

from __future__ import annotations

import logging
import re
from fractions import Fraction
from pathlib import Path

import msgspec
import numpy as np
import pytest
import xarray as xr
from conftest import SITE_DIR
from sweep_strategies import (
    GAIN_LOOP,
    INF,
    PORT_LOOP,
    SOURCE,
    YFACTOR_ENR_DB,
    YFACTOR_GRID,
    CalSweepCls,
    QuickAnalyses,
    SweepCls,
    assert_corrections_match,
    expected_yfactor,
    make_calibration_capture,
    make_calibration_runner,
    make_calibration_sweep,
    make_capture,
    make_yfactor_dataset,
    quick_binding,
    yfactor_rms_power,
)

import striqt.sensor as ss
from striqt.sensor.lib.actions import ManualYFactorRunner, registry, yfactor

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
        assert corrections.power_correction.attrs['units'] == 'mW/FS'

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


@pytest.mark.parametrize(
    'loops, period',
    [
        ((), Fraction(1, 500)),
        ((ss.specs.List(field='duration', values=(5e-3,)),), Fraction(1, 200)),
    ],
    ids=['capture', 'loop'],
)
def test_default_detector_period_is_the_one_duration(loops, period):
    """the default period is one bin per capture. The rms mean that the Y-factor
    math uses does not depend on the bin count, and the PAPR diagnostics collapse
    to 0 dB. A looped duration overrides the capture's own."""
    sweep = make_calibration_sweep(
        captures=(make_calibration_capture(duration=2e-3),), loops=loops
    )
    meas = yfactor.calibration_analysis(sweep).channel_power_time_series
    assert meas.detector_period == period


def test_calibration_analysis_is_the_group_of_the_binding():
    """the generated analysis is an instance of `schema.analysis`, which the
    binding's `check_schema` has required to declare `channel_power_time_series`"""
    cls = quick_binding(ss.specs.ManualYFactorCalibration).sensor.sweep_spec_cls
    spec = cls(
        source=SOURCE,
        captures=(make_capture(),),
        calibration=ss.specs.ManualYFactorSpec(enr=20.0, ambient_temperature=290.0),
    )
    assert type(yfactor.calibration_analysis(spec)) is QuickAnalyses
    assert type(yfactor.build_calibration_sweep(spec).analysis) is QuickAnalyses


def test_default_detector_period_needs_one_duration():
    """the bound spec's constructor reads the runner's `sweep`, which raises the
    `detector_period` error"""
    with pytest.raises(msgspec.ValidationError, match=r'calibration\.detector_period'):
        make_calibration_sweep(loops=(DURATION_LOOP,))


# %% build_calibration_sweep and validate


def test_build_sweep_is_cached_on_the_spec():
    runner = make_calibration_runner()
    assert yfactor.build_calibration_sweep(runner.spec) is runner.sweep


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


def test_explicit_captures_with_implied_loops():
    """2 explicit captures at different ports give one pass per port over that
    port's capture. The whole calibration acquires 2 ports x 2 states x 1 capture."""
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


# %% registry


def test_registered_for_calibration_specs_only():
    assert registry.lookup(CalSweepCls) is ManualYFactorRunner
    assert registry.lookup(SweepCls) is not ManualYFactorRunner


class _RecordingSource:
    source_id = 'beef'

    def __init__(self):
        self.discards = 0

    def discard_iq(self):
        self.discards += 1


def test_each_pass_begins_with_a_fresh_acquisition(monkeypatch, scripted_input):
    """the runner calls `discard_iq` before every pass and passes the diode state to
    the pipeline. With `reuse_iq` on, the controller would otherwise serve the
    diode-off IQ to the equal diode-on capture."""
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
