"""striqt.sensor.lib.calibration: the generated calibration sweep and capture classes,
noise diode loop insertion, the Y-factor corrections and their lookups, YFactorSink,
and the manual noise diode peripheral"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from conftest import (
    FAKE_SOAPY_CALIBRATION_SPEC,
    FAKE_SOAPY_SPEC,
    SITE_DIR,
    FakeRun,
    construct_both,
    raises_on_both_paths,
)
from hypothesis import given
from pytest_lazy_fixtures import lf
from site_strategies import SiteCalCaptureCls
from soapy_factories import MCR, soapy_capture
from sweep_strategies import (
    BOLTZMANN_MW,
    GAIN_LOOP,
    NDE_LOOP,
    PORT_LOOP,
    YFACTOR_ENR_DB,
    YFACTOR_GRID,
    YFACTOR_NF_DB,
    CalSourceCls,
    CalSweepCls,
    calibration_loop_orderings,
    expected_yfactor,
    loop_fields,
    make_calibration_capture,
    make_calibration_sweep,
    make_calibration_sweep_kws,
    make_yfactor_dataset,
    noise_figure_lookup,
    receiver_gain,
    save_yfactor_calibration,
    yfactor_rms_power,
)

import striqt.analysis as sa
import striqt.sensor as ss
from striqt.sensor.lib import calibration
from striqt.sensor.lib.compute.datasets import build_capture_coords

H = ss.specs.helpers
Repeat = ss.specs.Repeat
TOGGLE = 'noise_diode_enabled'
TOGGLE_MSG = 'noise_diode_enabled must be the first specified loop'


# %% noise diode loop insertion


class TestNoiseDiodeToggle:
    @given(loops_accepted=calibration_loop_orderings())
    def test_explicit_toggle_position(self, loops_accepted):
        loops, accepted = loops_accepted
        kws = make_calibration_sweep_kws(loops=loops)
        if accepted:
            direct, converted = construct_both(CalSweepCls, **kws)
            fields = loop_fields(direct)
            assert fields == loop_fields(converted)
            assert [f for f in fields if f != TOGGLE] == [
                l.field for l in loops if l.field != TOGGLE
            ]
        else:
            raises_on_both_paths(CalSweepCls, TypeError, TOGGLE_MSG, **kws)

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


# %% _y_factor_power_corrections

INF = float('inf')
CORRECTION_VARS = ('noise_figure', 'temperature', 'power_correction')
# float64 roundoff of the closed-form corrections, expressed in dB
EXACT_DB = 1e-8


def assert_corrections_match(actual, expected, tol_dB=EXACT_DB, names=CORRECTION_VARS):
    """each correction variable of `actual` matches the model `expected`, which may
    span fewer dimensions, to within `tol_dB`; temperature and power_correction are
    compared in dB"""
    for name in names:
        result = actual[name].reset_coords(drop=True)
        model = expected[name].reset_coords(drop=True).broadcast_like(result)
        if name != 'noise_figure':
            result, model = 10 * np.log10(result), 10 * np.log10(model)
        xr.testing.assert_allclose(result, model, rtol=0, atol=tol_dB)


class TestYFactorCorrections:
    def test_recovers_the_receiver_model(self):
        corrections = calibration._y_factor_power_corrections(make_yfactor_dataset())
        assert_corrections_match(corrections, expected_yfactor())

    def test_frequency_dependent_noise_figure(self):
        nf = {0: {1e9: 5.0, 2e9: 7.0}, 1: 8.0}
        corrections = calibration._y_factor_power_corrections(
            make_yfactor_dataset(nf_dB=nf)
        )
        assert_corrections_match(corrections, expected_yfactor(nf_dB=nf))

    def test_output_is_indexed_by_the_calibration_fields(self):
        corrections = calibration._y_factor_power_corrections(make_yfactor_dataset())
        assert set(corrections.dims) == set(YFACTOR_GRID)
        assert 'noise_diode_enabled' not in corrections.coords
        assert corrections.noise_figure.attrs['units'] == 'dB'
        assert corrections.power_correction.attrs['units'] == 'mW/fs'

    def test_papr_and_raw_powers(self):
        ds = make_yfactor_dataset(papr_dB=7.5)
        corrections = calibration._y_factor_power_corrections(ds)
        assert corrections.papr_on.values == pytest.approx(7.5)
        assert corrections.papr_off.values == pytest.approx(7.5)
        p_off = yfactor_rms_power(0.0, 5.0, 40e6, diode_on=False)
        p_on = yfactor_rms_power(0.0, 5.0, 40e6, diode_on=True)
        point = {'port': 0, 'gain': 0.0, 'center_frequency': 1e9}
        assert corrections.p_off.sel(point).item() == pytest.approx(p_off, rel=1e-9)
        assert corrections.p_on.sel(point).item() == pytest.approx(p_on, rel=1e-9)

    def test_infinite_bandwidth_uses_the_nyquist_bandwidth(self):
        grid = dict(YFACTOR_GRID, analysis_bandwidth=(40e6, INF))
        corrections = calibration._y_factor_power_corrections(
            make_yfactor_dataset(grid)
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


def test_limit_nyquist_bandwidth_keeps_finite_values():
    bw = calibration._limit_nyquist_bandwidth(_bandwidth_grid())
    assert bw.sel(analysis_bandwidth=40e6).values.tolist() == [40e6, 40e6]


def test_limit_nyquist_bandwidth_replaces_inf_with_the_sample_rate():
    bw = calibration._limit_nyquist_bandwidth(_bandwidth_grid())
    assert bw.sel(analysis_bandwidth=INF).values.tolist() == [125e6, 62.5e6]


# %% summarize_calibration


@pytest.fixture(scope='module')
def model_corrections():
    nf = {0: {1e9: 5.0, 2e9: 7.0}, 1: 8.0}
    return calibration._y_factor_power_corrections(make_yfactor_dataset(nf_dB=nf))


def test_summarize_selects_the_port_at_maximum_gain(model_corrections):
    summary = calibration.summarize_calibration(model_corrections, port=1)
    assert list(summary.columns) == ['NF (dB)', 'Power Corr (dB)']
    assert summary['NF (dB)'].tolist() == [8.0, 8.0]
    assert 'gain' not in summary.index.names
    assert 'port' not in summary.index.names
    assert set(summary.index.names) >= {'center_frequency'}


def test_summarize_noise_figure_per_frequency(model_corrections):
    summary = calibration.summarize_calibration(model_corrections, port=0)
    by_frequency = summary['NF (dB)'].groupby(level='center_frequency').first()
    assert by_frequency.to_dict() == pytest.approx({1e9: 5.0, 2e9: 7.0})


# %% calibration lookups


def _lookup_pc(path, capture, **kws):
    return calibration.lookup_power_correction(path, capture, MCR, **kws)


@pytest.fixture
def legacy_channel_nc(tmp_path, calibration_nc):
    """calibration_nc with the port coordinate under its former name, channel"""
    path = tmp_path / 'legacy.nc'
    saved = ss.lib.io.read_calibration(calibration_nc)
    ss.lib.io.save_calibration(path, saved.rename(port='channel'))
    return str(path)


@pytest.fixture
def squeezed_lo_shift_nc(tmp_path, calibration_nc):
    """calibration_nc with the single-valued lo_shift dimension squeezed away"""
    path = tmp_path / 'squeezed.nc'
    saved = ss.lib.io.read_calibration(calibration_nc)
    ss.lib.io.save_calibration(path, saved.squeeze('lo_shift'))
    return str(path)


@pytest.fixture
def single_frequency_nc(tmp_path, clear_function_caches):
    grid = dict(YFACTOR_GRID, center_frequency=(1e9,))
    return save_yfactor_calibration(tmp_path / 'one.nc', grid)


def test_read_calibration_does_not_depend_on_the_open_file(calibration_nc):
    saved = ss.read_calibration(calibration_nc)
    Path(calibration_nc).write_bytes(b'')
    assert saved.noise_figure.max().item() == pytest.approx(8.0)
    saved.close()


class TestLookupPowerCorrection:
    @pytest.mark.parametrize(
        'cal_path, capture_kws',
        [
            pytest.param(lf('calibration_nc'), {'gain': -10}, id='exact_grid_point'),
            pytest.param(
                lf('legacy_channel_nc'), {'port': 1}, id='legacy_channel_coordinate'
            ),
            pytest.param(
                lf('single_frequency_nc'),
                {},
                id='single_frequency_calibration',
                marks=pytest.mark.xfail(
                    strict=True,
                    raises=ValueError,
                    reason='_lookup_calibration_var squeezes before '
                    'dropna(center_frequency), so a calibration at one center '
                    'frequency has no such dimension',
                ),
            ),
            pytest.param(
                lf('squeezed_lo_shift_nc'),
                {},
                id='calibration_without_a_lo_shift_loop',
                marks=pytest.mark.xfail(
                    strict=True,
                    raises=(TypeError, ValueError),
                    reason='exact-match fields are selected with sel(), which needs '
                    'an index; a scalar lo_shift coordinate makes '
                    '_describe_missing_data iterate a 0-d array (xarray >= 2025 '
                    'raises ValueError from sel() before that)',
                ),
            ),
        ],
    )
    def test_recovers_the_receiver_gain(self, cal_path, capture_kws):
        result = _lookup_pc(cal_path, soapy_capture(**capture_kws))
        assert result.dtype == np.float32
        expected = 1 / receiver_gain(capture_kws.get('gain', 0))
        assert result == pytest.approx([expected], rel=1e-6)

    def test_multiport_capture_gives_one_value_per_port(self, calibration_nc):
        capture = soapy_capture(port=(0, 1), gain=(0, -10))
        result = _lookup_pc(calibration_nc, capture)
        expected = [1 / receiver_gain(0), 1 / receiver_gain(-10)]
        assert result == pytest.approx(expected, rel=1e-6)

    def test_array_namespace(self, calibration_nc, xp):
        result = _lookup_pc(calibration_nc, soapy_capture(), xp=xp)
        assert isinstance(result, xp.ndarray)

    def test_host_resampled_capture_matches_on_backend_sample_rate(
        self, calibration_nc
    ):
        # 125 MS/s -> 62.5 MS/s designs fs_sdr = 62.5 MS/s, which is not calibrated
        capture = soapy_capture(sample_rate=62.5e6, host_resample=True)
        with pytest.raises(KeyError, match=r"62500000\.0 in 'backend_sample_rate'"):
            _lookup_pc(calibration_nc, capture)

    def test_none_and_invalid_inputs(self):
        assert _lookup_pc(None, soapy_capture()) is None
        with pytest.raises(TypeError, match='cal_data'):
            _lookup_pc(3, soapy_capture())

    def test_missing_grid_point_names_the_field(self, calibration_nc):
        with pytest.raises(KeyError) as exc_info:
            _lookup_pc(calibration_nc, soapy_capture(gain=5.0))
        message = str(exc_info.value)
        assert "'gain'" in message and '5.0' in message
        assert "'port'" not in message

    def test_out_of_range_frequency(self, calibration_nc):
        with pytest.raises(ValueError, match='exceeds calibration max'):
            _lookup_pc(calibration_nc, soapy_capture(center_frequency=3e9))
        with pytest.raises(ValueError, match='below calibration min'):
            _lookup_pc(calibration_nc, soapy_capture(center_frequency=0.5e9))

    @pytest.mark.xfail(
        strict=True,
        reason='the range error message interpolates a DataArray repr rather than '
        'the frequency value',
    )
    def test_out_of_range_message_shows_the_limit_in_mhz(self, calibration_nc):
        with pytest.raises(ValueError, match=r'exceeds calibration max 2000\.0 MHz'):
            _lookup_pc(calibration_nc, soapy_capture(center_frequency=3e9))


class TestLookupSystemNoisePower:
    def _lookup(self, path, capture, **kws):
        return calibration.lookup_system_noise_power(path, capture, MCR, **kws)

    def test_spectral_density_from_the_noise_figure(self, calibration_nc):
        noise = self._lookup(calibration_nc, soapy_capture(port=1))
        assert noise.dims == ('capture',)
        assert noise.attrs['units'] == 'dBm/Hz'
        assert noise.item() == pytest.approx(8.0 + 10 * np.log10(BOLTZMANN_MW * 290))

    def test_bandwidth_and_temperature_scale_the_result(self, calibration_nc):
        noise = self._lookup(calibration_nc, soapy_capture(), B=1e6, T=300.0)
        expected = 5.0 + 10 * np.log10(BOLTZMANN_MW * 300.0 * 1e6)
        assert noise.item() == pytest.approx(expected)

    def test_noise_figure_is_interpolated_in_frequency(self, calibration_nc):
        noise = self._lookup(calibration_nc, soapy_capture(center_frequency=1.5e9))
        assert noise.item() == pytest.approx(6.0 + 10 * np.log10(BOLTZMANN_MW * 290))

    def test_none_input(self):
        assert self._lookup(None, soapy_capture()) is None


# %% _get_port_variable and _describe_missing_data


def test_get_port_variable():
    ds = xr.Dataset(coords={'port': [0, 1]})
    assert calibration._get_port_variable(ds) == 'port'
    assert calibration._get_port_variable(ds.rename(port='channel')) == 'channel'
    assert calibration._get_port_variable(None) == 'port'


def test_describe_missing_data_lists_only_the_misses(model_corrections):
    exact = {'port': 0, 'gain': 5.0, 'lo_shift': 'left', 'center_frequency': 1e9}
    text = calibration._describe_missing_data(model_corrections.noise_figure, exact)
    assert "'gain'" in text and "'lo_shift'" in text
    assert "'port'" not in text and "'center_frequency'" not in text


def test_describe_missing_data_is_empty_when_everything_matches(model_corrections):
    exact = {'port': 0, 'gain': 0.0}
    assert (
        calibration._describe_missing_data(model_corrections.noise_figure, exact) == ''
    )


# %% YFactorSink

# the grid covered both by the flushed model captures and by
# sweeps/fake_soapy-calibration-cpu.yaml
SWEEP_GRID = dict(
    YFACTOR_GRID, backend_sample_rate=(125e6, 62.5e6), analysis_bandwidth=(40e6, INF)
)
FLUSH_LOOPS = (
    PORT_LOOP,
    ss.specs.List(field='sample_rate', values=SWEEP_GRID['backend_sample_rate']),
    ss.specs.List(field='center_frequency', values=SWEEP_GRID['center_frequency']),
    GAIN_LOOP,
    ss.specs.List(field='analysis_bandwidth', values=SWEEP_GRID['analysis_bandwidth']),
    ss.specs.List(field='lo_shift', values=SWEEP_GRID['lo_shift']),
)


def _capture_dataset(sweep, capture, index, start):
    """one capture's result as from_delayed shapes it, with the receiver model's power"""
    info = ss.specs.SoapyAcquisitionInfo(
        sweep_start_time=start,
        start_time=start + pd.Timedelta(seconds=index),
        backend_sample_rate=capture.sample_rate,
        source_id='beef',
        sweep_index=0,
        capture_index=index,
    )
    bandwidth = capture.analysis_bandwidth
    if not np.isfinite(bandwidth):
        bandwidth = capture.sample_rate
    nf = noise_figure_lookup(YFACTOR_NF_DB, capture.port, capture.center_frequency)
    diode_on = capture.noise_diode_enabled
    rms_power = yfactor_rms_power(capture.gain, nf, bandwidth, diode_on=diode_on)
    rms = 10 * np.log10(rms_power)
    values = np.array([[rms] * 4, [rms + 10.0] * 4])[np.newaxis]

    pvt_dims = ('capture', 'power_detector', 'time_elapsed')
    coords = {'power_detector': ['rms', 'peak'], 'time_elapsed': np.arange(4) * 1e-3}
    ds = xr.Dataset({'channel_power_time_series': (pvt_dims, values)}, coords=coords)
    ds = ds.assign_coords(build_capture_coords(capture, info, sweep.loops))
    peripheral_data = {
        k: xr.DataArray(v, dims=[]).expand_dims({'capture': 1})
        for k, v in sweep.calibration.to_dict().items()
    }
    ds = ds.assign(peripheral_data)
    ds.attrs.update(sweep.to_dict(True, allow_tuple_keys=False))
    return ds


def _flush_model_captures(path):
    """YFactorSink.flush on a full grid of model captures; returns the sweep"""
    captures = (make_calibration_capture(sample_rate=MCR, host_resample=False),)
    cal = ss.specs.ManualYFactorPeripheral(
        enr=YFACTOR_ENR_DB, ambient_temperature=290.0
    )
    source = CalSourceCls(array_backend='numpy')
    sink = ss.specs.Sink(path=str(path))
    sweep = make_calibration_sweep(
        captures=captures, loops=FLUSH_LOOPS, source=source, calibration=cal, sink=sink
    )
    start = pd.Timestamp('2026-09-15T00:00:00')

    sink = calibration.YFactorSink(sweep)
    sink.open()
    sink._pending_data = [
        _capture_dataset(sweep, c, i, start)
        for i, c in enumerate(H.loop_captures(sweep, 'beef'))
    ]
    sink.flush()
    return sweep


@pytest.fixture(scope='module')
def flushed_calibration(fake_sweep_runner, tmp_path_factory) -> FakeRun:
    """the calibration YFactorSink.flush saves from a full grid of model captures,
    as a FakeRun with the sweep but no prompts, fake device or results"""
    # the runner stubs the source id lookup that the sink's path formatting performs
    path = tmp_path_factory.mktemp('yfactor') / 'flushed.nc'
    sweep = _flush_model_captures(path)
    return FakeRun(str(path), [], sweep, None, None)


def test_flush_saves_a_file(flushed_calibration):
    assert Path(flushed_calibration.path).exists()


class TestYFactorSinkFlush:
    def test_saved_file_is_usable_for_lookups(self, flushed_calibration):
        capture = soapy_capture(port=1, gain=-10, sample_rate=62.5e6)
        result = _lookup_pc(flushed_calibration.path, capture)
        assert result == pytest.approx([1 / receiver_gain(-10)], rel=1e-6)

    @pytest.mark.xfail(
        strict=True,
        raises=KeyError,
        reason='flush assigns calibration_fields and sweep_start_time attrs to the '
        'capture data, but compute_y_factor_corrections builds a new Dataset '
        'without them',
    )
    def test_saved_attrs_record_the_calibration_fields(self, flushed_calibration):
        saved = ss.lib.io.read_calibration(flushed_calibration.path)
        assert list(saved.attrs['calibration_fields']) == [
            'port',
            'noise_diode_enabled',
            'backend_sample_rate',
            'center_frequency',
            'gain',
            'analysis_bandwidth',
            'lo_shift',
        ]


# %% ManualYFactorPeripheral


@pytest.fixture
def scripted_input(monkeypatch):
    """answers sa.util.blocking_input from a queue and records the prompts"""
    script = {'answers': [], 'prompts': []}

    def blocking_input(prompt=None):
        script['prompts'].append(prompt)
        return script['answers'].pop(0)

    monkeypatch.setattr(sa.util, 'blocking_input', blocking_input)
    return script


def _calibration_sweep():
    return make_calibration_sweep(
        calibration=ss.specs.ManualYFactorPeripheral(
            enr=20.87, ambient_temperature=290.0
        )
    )


class TestManualYFactorPeripheral:
    def test_open_confirms_the_enr(self, scripted_input):
        scripted_input['answers'] += ['y']
        calibration.ManualYFactorPeripheral(_calibration_sweep())
        (prompt,) = scripted_input['prompts']
        assert 'ENR' in prompt and '20.87' in prompt

    def test_open_reprompts_until_answered(self, scripted_input):
        scripted_input['answers'] += ['maybe', '', 'Y']
        calibration.ManualYFactorPeripheral(_calibration_sweep())
        assert len(scripted_input['prompts']) == 3

    def test_open_rejects_the_enr(self, scripted_input):
        scripted_input['answers'] += ['N']
        with pytest.raises(RuntimeError, match='ENR'):
            calibration.ManualYFactorPeripheral(_calibration_sweep())

    def test_arm_prompts_only_when_the_diode_state_changes(self, scripted_input):
        scripted_input['answers'] += ['y', '', '', '']
        periph = calibration.ManualYFactorPeripheral(_calibration_sweep())
        del scripted_input['prompts'][:]

        periph.arm(make_calibration_capture(port=0, noise_diode_enabled=False))
        periph.arm(make_calibration_capture(port=0, noise_diode_enabled=False))
        periph.arm(make_calibration_capture(port=0, noise_diode_enabled=True))
        periph.arm(make_calibration_capture(port=1, noise_diode_enabled=True))

        # conftest.answer_calibration_prompts parses this wording to switch the fake
        # diode
        matches = [
            re.match(r'(enable|disable) noise diode at port (\d+)', prompt)
            for prompt in scripted_input['prompts']
        ]
        assert [m.groups() for m in matches if m is not None] == [
            ('disable', '0'),
            ('enable', '0'),
            ('enable', '1'),
        ]
        assert len(matches) == 3

    def test_acquire_returns_the_calibration_spec(self, scripted_input):
        scripted_input['answers'] += ['y']
        periph = calibration.ManualYFactorPeripheral(_calibration_sweep())
        assert periph.acquire(make_calibration_capture()) == {
            'enr': 20.87,
            'ambient_temperature': 290.0,
            'implied_loops': (),
        }


# %% closed loop through the fake SoapySDR device

FAKE_TONE = (1.505e9, -60.0)
# 5 ms at 40 MHz gives a 1-sigma statistical error near 0.015 dB
FAKE_TOL_DB = 0.1


@pytest.fixture(scope='module')
def fake_calibration_run(fake_sweep_runner) -> FakeRun:
    return fake_sweep_runner(FAKE_SOAPY_CALIBRATION_SPEC, output_name='calibration.nc')


@pytest.fixture
def fake_model(fake_calibration_run):
    """the fake device's signal model with the noise diodes, which the calibration
    sweep leaves on, switched off; tones set by the test are cleared afterwards"""
    model = fake_calibration_run.fake.model
    model.diode_on.clear()
    yield model
    model.tones.clear()


class TestFakeCalibrationSweep:
    def test_prompts_follow_the_diode_state(self, fake_calibration_run):
        prompts = fake_calibration_run.prompts
        assert prompts[0].startswith('Confirm that the noise diode ENR is 20.87 dB')
        # one prompt per (port, state) change; the loop toggles the diode per port
        assert prompts[1:] == [
            'disable noise diode at port 0 and press enter: ',
            'enable noise diode at port 0 and press enter: ',
            'disable noise diode at port 1 and press enter: ',
            'enable noise diode at port 1 and press enter: ',
        ]


SAVED_CALIBRATIONS = [
    pytest.param(lf('flushed_calibration'), EXACT_DB, id='flushed'),
    pytest.param(lf('fake_calibration_run'), FAKE_TOL_DB, id='fake_sweep'),
]


class TestSavedCalibration:
    @pytest.mark.parametrize('run, tol_dB', SAVED_CALIBRATIONS)
    def test_indexed_by_the_looped_fields(self, run, tol_dB):
        saved = ss.read_calibration(run.path)
        assert set(saved.dims) == set(SWEEP_GRID)
        assert sorted(saved.backend_sample_rate.values) == [62.5e6, 125e6]
        assert 'noise_diode_enabled' not in saved.coords
        assert set(saved.data_vars) >= set(CORRECTION_VARS)

    @pytest.mark.parametrize('run, tol_dB', SAVED_CALIBRATIONS)
    @pytest.mark.parametrize('analysis_bandwidth', [40e6, INF])
    def test_points_match_the_model(self, run, tol_dB, analysis_bandwidth):
        saved = run.calibration(analysis_bandwidth=analysis_bandwidth)
        grid = dict(SWEEP_GRID, analysis_bandwidth=(analysis_bandwidth,))
        assert_corrections_match(saved, expected_yfactor(grid), tol_dB)


def _rms_dBm(ds):
    return ds.channel_power_time_series.sel(power_detector='rms').mean('time_elapsed')


def _expected_dBm(port, bandwidth, tone_mW=0.0, nf_dB=YFACTOR_NF_DB):
    Te = 290.0 * (10 ** (nf_dB[port] / 10) - 1)
    return 10 * np.log10(tone_mW + BOLTZMANN_MW * (290.0 + Te) * bandwidth)


class TestFakeMeasurementSweep:
    def test_uncalibrated_power_is_in_full_scale_units(
        self, fake_sweep_runner, fake_model
    ):
        fake_model.tones[0] = FAKE_TONE
        ds = fake_sweep_runner(FAKE_SOAPY_SPEC, output_name='raw.zarr.zip').results

        assert ds.port.values.tolist() == [0, 1, 0]
        assert ds.gain.values.tolist() == [0, 0, -10]
        gain_dB = np.array([50.0, 50.0, 40.0])
        tone_dBm = _expected_dBm(0, 40e6, tone_mW=1e-6)
        noise_dBm = _expected_dBm(1, 40e6)
        expected = [tone_dBm, noise_dBm, tone_dBm]
        assert _rms_dBm(ds).values == pytest.approx(expected + gain_dB, abs=0.1)
        assert 'system_noise' not in ds

    def test_calibrated_power_recovers_the_input(
        self, fake_sweep_runner, fake_calibration_run, fake_model
    ):
        fake_model.tones[0] = FAKE_TONE
        spec = ss.read_yaml_spec(FAKE_SOAPY_SPEC)
        source = spec.source.replace(calibration=fake_calibration_run.path)

        ds = fake_sweep_runner(
            FAKE_SOAPY_SPEC, output_name='cal.zarr.zip', source=source
        ).results

        rms = _rms_dBm(ds).values
        assert rms[0] == pytest.approx(_expected_dBm(0, 40e6, tone_mW=1e-6), abs=0.05)
        assert rms[2] == pytest.approx(_expected_dBm(0, 40e6, tone_mW=1e-6), abs=0.05)
        # noise only: the FIR passband is not exactly 40 MHz wide
        assert rms[1] == pytest.approx(_expected_dBm(1, 40e6), abs=0.15)
        assert ds.system_noise.attrs['units'] == 'dBm/Hz'
        assert ds.system_noise.values == pytest.approx(
            [5.0, 8.0, 5.0] + 10 * np.log10(BOLTZMANN_MW * 290.0), abs=0.1
        )

    def test_calibrated_power_at_infinite_bandwidth(
        self, fake_sweep_runner, fake_calibration_run, fake_model
    ):
        spec = ss.read_yaml_spec(FAKE_SOAPY_SPEC)
        capture = spec.captures[0].replace(analysis_bandwidth=INF)

        source = spec.source.replace(calibration=fake_calibration_run.path)
        overrides = {'source': source, 'captures': (capture,)}
        ds = fake_sweep_runner(
            FAKE_SOAPY_SPEC, output_name='inf.zarr.zip', **overrides
        ).results

        expected = [_expected_dBm(0, 125e6), _expected_dBm(1, 125e6)]
        assert _rms_dBm(ds).values == pytest.approx(expected, abs=0.1)
