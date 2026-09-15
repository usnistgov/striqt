"""striqt.sensor.lib.calibration: the generated calibration sweep and capture classes,
noise diode loop insertion, the Y-factor corrections and their lookups, YFactorSink,
and the manual noise diode peripheral"""

from __future__ import annotations

import msgspec
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from conftest import FAKE_SOAPY_CALIBRATION_SPEC, FAKE_SOAPY_SPEC, SITE_DIR
from hypothesis import given
from site_strategies import SiteCalCaptureCls
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
    calibration_sweep_dict,
    expected_yfactor,
    loop_fields,
    make_calibration_capture,
    make_calibration_sweep,
    make_yfactor_dataset,
    noise_figure_lookup,
    receiver_gain,
    save_yfactor_calibration,
    yfactor_rms_power,
)

import striqt.analysis as sa
import striqt.sensor as ss
import striqt.waveform as sw
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


# %% _y_factor_power_corrections

INF = float('inf')
CORRECTION_VARS = ('noise_figure', 'temperature', 'power_correction')


def _assert_matches_model(corrections, expected, rtol=1e-9):
    for name in CORRECTION_VARS:
        actual = corrections[name]
        model = expected[name].broadcast_like(actual).transpose(*actual.dims)
        np.testing.assert_allclose(actual.values, model.values, rtol=rtol)


class TestYFactorCorrections:
    def test_recovers_the_receiver_model(self):
        corrections = calibration._y_factor_power_corrections(make_yfactor_dataset())
        _assert_matches_model(corrections, expected_yfactor())

    def test_frequency_dependent_noise_figure(self):
        nf = {0: {1e9: 5.0, 2e9: 7.0}, 1: 8.0}
        corrections = calibration._y_factor_power_corrections(
            make_yfactor_dataset(nf_dB=nf)
        )
        _assert_matches_model(corrections, expected_yfactor(nf_dB=nf))

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
        assert float(corrections.p_off.sel(point)) == pytest.approx(p_off, rel=1e-9)
        assert float(corrections.p_on.sel(point)) == pytest.approx(p_on, rel=1e-9)

    @pytest.mark.xfail(
        strict=True,
        reason='_limit_nyquist_bandwidth never replaces inf, so power_correction is '
        'inf wherever analysis_bandwidth is',
    )
    def test_infinite_bandwidth_uses_the_nyquist_bandwidth(self):
        grid = dict(YFACTOR_GRID, analysis_bandwidth=(40e6, INF))
        corrections = calibration._y_factor_power_corrections(
            make_yfactor_dataset(grid)
        )
        _assert_matches_model(corrections, expected_yfactor(grid))


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


@pytest.mark.xfail(
    strict=True,
    reason='the mask ~np.isfinite(bw.values == inf) is a boolean comparison passed to '
    'isfinite, so it is False everywhere',
)
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


@pytest.mark.xfail(
    strict=True,
    reason='the summary column is labelled dB but holds the linear power_correction',
)
def test_summarize_power_correction_column_is_in_dB(model_corrections):
    summary = calibration.summarize_calibration(model_corrections, port=0)
    expected = 10 * np.log10(1 / receiver_gain(0.0))
    assert summary['Power Corr (dB)'].tolist() == pytest.approx([expected] * 2)


# %% calibration lookups

MCR = 125e6


@pytest.fixture
def calibration_nc(tmp_path):
    nf = {0: {1e9: 5.0, 2e9: 7.0}, 1: 8.0}
    yield save_yfactor_calibration(tmp_path / 'cal.nc', nf_dB=nf)
    # read_calibration and the lookups cache by path
    sw.util.clear_caches()


def _capture(**kws):
    kws = {
        'port': 0,
        'center_frequency': 1e9,
        'gain': 0,
        'sample_rate': MCR,
        'duration': 1e-3,
        'analysis_bandwidth': 40e6,
        'host_resample': False,
        **kws,
    }
    return ss.specs.SoapyCapture(**kws)


def _lookup_pc(path, capture, **kws):
    return calibration.lookup_power_correction(path, capture, MCR, **kws)


class TestLookupPowerCorrection:
    def test_exact_grid_point(self, calibration_nc):
        result = _lookup_pc(calibration_nc, _capture(gain=-10))
        assert result.dtype == np.float32
        assert result == pytest.approx([1 / receiver_gain(-10)], rel=1e-6)

    def test_multiport_capture_gives_one_value_per_port(self, calibration_nc):
        result = _lookup_pc(calibration_nc, _capture(port=(0, 1), gain=(0, -10)))
        expected = [1 / receiver_gain(0), 1 / receiver_gain(-10)]
        assert result == pytest.approx(expected, rel=1e-6)

    def test_array_namespace(self, calibration_nc, xp):
        result = _lookup_pc(calibration_nc, _capture(), xp=xp)
        assert isinstance(result, xp.ndarray)

    def test_host_resampled_capture_matches_on_backend_sample_rate(
        self, calibration_nc
    ):
        # 125 MS/s -> 62.5 MS/s designs fs_sdr = 62.5 MS/s, which is not calibrated
        with pytest.raises(KeyError, match=r"62500000\.0 in 'backend_sample_rate'"):
            _lookup_pc(calibration_nc, _capture(sample_rate=62.5e6, host_resample=True))

    def test_none_and_invalid_inputs(self, calibration_nc):
        assert _lookup_pc(None, _capture()) is None
        with pytest.raises(TypeError, match='cal_data'):
            _lookup_pc(3, _capture())

    def test_missing_grid_point_names_the_field(self, calibration_nc):
        with pytest.raises(KeyError) as exc_info:
            _lookup_pc(calibration_nc, _capture(gain=5.0))
        message = str(exc_info.value)
        assert "5.0 in 'gain' (available: 0.0, -10.0)" in message
        assert "'port'" not in message

    def test_out_of_range_frequency(self, calibration_nc):
        with pytest.raises(ValueError, match='exceeds calibration max'):
            _lookup_pc(calibration_nc, _capture(center_frequency=3e9))
        with pytest.raises(ValueError, match='below calibration min'):
            _lookup_pc(calibration_nc, _capture(center_frequency=0.5e9))

    @pytest.mark.xfail(
        strict=True,
        reason='the range error message interpolates a DataArray repr rather than '
        'the frequency value',
    )
    def test_out_of_range_message_shows_the_limit_in_mhz(self, calibration_nc):
        with pytest.raises(ValueError, match=r'exceeds calibration max 2000\.0 MHz'):
            _lookup_pc(calibration_nc, _capture(center_frequency=3e9))

    def test_legacy_channel_coordinate(self, tmp_path, calibration_nc):
        legacy = tmp_path / 'legacy.nc'
        ss.lib.io.save_calibration(
            legacy, ss.lib.io.read_calibration(calibration_nc).rename(port='channel')
        )
        result = _lookup_pc(str(legacy), _capture(port=1, gain=0))
        assert result == pytest.approx([1 / receiver_gain(0)], rel=1e-6)

    @pytest.mark.xfail(
        strict=True,
        raises=ValueError,
        reason='_lookup_calibration_var squeezes before dropna(center_frequency), '
        'so a calibration at one center frequency has no such dimension',
    )
    def test_single_frequency_calibration(self, tmp_path):
        grid = dict(YFACTOR_GRID, center_frequency=(1e9,))
        path = save_yfactor_calibration(tmp_path / 'one.nc', grid)
        result = _lookup_pc(path, _capture())
        assert result == pytest.approx([1 / receiver_gain(0)], rel=1e-6)

    @pytest.mark.xfail(
        strict=True,
        raises=TypeError,
        reason='exact-match fields are selected with sel(), which needs an index; a '
        'scalar lo_shift coordinate makes _describe_missing_data iterate a 0-d array',
    )
    def test_calibration_without_a_lo_shift_loop(self, tmp_path, calibration_nc):
        squeezed = tmp_path / 'squeezed.nc'
        ss.lib.io.save_calibration(
            squeezed, ss.lib.io.read_calibration(calibration_nc).squeeze('lo_shift')
        )
        result = _lookup_pc(str(squeezed), _capture())
        assert result == pytest.approx([1 / receiver_gain(0)], rel=1e-6)


class TestLookupSystemNoisePower:
    def _lookup(self, path, capture, **kws):
        return calibration.lookup_system_noise_power(path, capture, MCR, **kws)

    def test_spectral_density_from_the_noise_figure(self, calibration_nc):
        noise = self._lookup(calibration_nc, _capture(port=1))
        assert noise.dims == ('capture',)
        assert noise.attrs['units'] == 'dBm/Hz'
        assert float(noise) == pytest.approx(8.0 + 10 * np.log10(BOLTZMANN_MW * 290))

    def test_bandwidth_and_temperature_scale_the_result(self, calibration_nc):
        noise = self._lookup(calibration_nc, _capture(), B=1e6, T=300.0)
        assert noise.attrs['units'] == 'dBm/1000000 Hz'
        expected = 5.0 + 10 * np.log10(BOLTZMANN_MW * 300.0 * 1e6)
        assert float(noise) == pytest.approx(expected)

    def test_noise_figure_is_interpolated_in_frequency(self, calibration_nc):
        noise = self._lookup(calibration_nc, _capture(center_frequency=1.5e9))
        assert float(noise) == pytest.approx(6.0 + 10 * np.log10(BOLTZMANN_MW * 290))

    def test_none_input(self):
        assert self._lookup(None, _capture()) is None


# %% _get_port_variable and _describe_missing_data


def test_get_port_variable():
    ds = xr.Dataset(coords={'port': [0, 1]})
    assert calibration._get_port_variable(ds) == 'port'
    assert calibration._get_port_variable(ds.rename(port='channel')) == 'channel'
    assert calibration._get_port_variable(None) == 'port'


def test_describe_missing_data_lists_only_the_misses(model_corrections):
    exact = {'port': 0, 'gain': 5.0, 'lo_shift': 'left', 'center_frequency': 1e9}
    text = calibration._describe_missing_data(model_corrections.noise_figure, exact)
    misses = text.split('; ')
    assert len(misses) == 2
    assert "5.0 in 'gain' (available: 0.0, -10.0)" in misses
    assert "'left' in 'lo_shift' (available: none)" in misses


def test_describe_missing_data_is_empty_when_everything_matches(model_corrections):
    exact = {'port': 0, 'gain': 0.0}
    assert (
        calibration._describe_missing_data(model_corrections.noise_figure, exact) == ''
    )


# %% YFactorSink

FLUSH_GRID = dict(
    YFACTOR_GRID, backend_sample_rate=(125e6, 62.5e6), analysis_bandwidth=(40e6, INF)
)
FLUSH_LOOPS = (
    PORT_LOOP,
    ss.specs.List(field='sample_rate', values=FLUSH_GRID['backend_sample_rate']),
    ss.specs.List(field='center_frequency', values=FLUSH_GRID['center_frequency']),
    GAIN_LOOP,
    ss.specs.List(field='analysis_bandwidth', values=FLUSH_GRID['analysis_bandwidth']),
    ss.specs.List(field='lo_shift', values=FLUSH_GRID['lo_shift']),
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
    rms = 10 * np.log10(
        yfactor_rms_power(
            capture.gain, nf, bandwidth, diode_on=capture.noise_diode_enabled
        )
    )
    values = np.array([[rms] * 4, [rms + 10.0] * 4])[np.newaxis]

    ds = xr.Dataset(
        {
            'channel_power_time_series': (
                ('capture', 'power_detector', 'time_elapsed'),
                values,
            )
        },
        coords={'power_detector': ['rms', 'peak'], 'time_elapsed': np.arange(4) * 1e-3},
    )
    ds = ds.assign_coords(build_capture_coords(capture, info, sweep.loops))
    peripheral_data = {
        k: xr.DataArray(v, dims=[]).expand_dims({'capture': 1})
        for k, v in sweep.calibration.to_dict().items()
    }
    ds = ds.assign(peripheral_data)
    ds.attrs.update(sweep.to_dict(True, allow_tuple_keys=False))
    return ds


@pytest.fixture(scope='module')
def flushed_calibration(tmp_path_factory):
    """(saved path, sweep) after YFactorSink.flush on a full grid of model captures"""
    path = tmp_path_factory.mktemp('yfactor') / 'flushed.nc'
    sweep = make_calibration_sweep(
        captures=(make_calibration_capture(sample_rate=125e6, host_resample=False),),
        loops=FLUSH_LOOPS,
        source=CalSourceCls(array_backend='numpy'),
        calibration=ss.specs.ManualYFactorPeripheral(
            enr=YFACTOR_ENR_DB, ambient_temperature=290.0
        ),
        sink=ss.specs.Sink(path=str(path)),
    )
    start = pd.Timestamp('2026-09-15T00:00:00')

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ss.lib.controller.lookup, 'id', lambda spec, timeout=0.5: 'beef')
        sink = calibration.YFactorSink(sweep)
        sink.open()
        sink._pending_data = [
            _capture_dataset(sweep, c, i, start)
            for i, c in enumerate(H.loop_captures(sweep, 'beef'))
        ]
        sink.flush()

    yield str(path), sweep
    sw.util.clear_caches()


class TestYFactorSinkFlush:
    def test_writes_corrections_indexed_by_the_looped_fields(self, flushed_calibration):
        path, _ = flushed_calibration
        saved = ss.lib.io.read_calibration(path)
        assert set(saved.dims) == set(FLUSH_GRID)
        assert saved.sizes['backend_sample_rate'] == 2
        assert 'noise_diode_enabled' not in saved.coords
        assert set(saved.data_vars) >= set(CORRECTION_VARS)

    def test_finite_bandwidth_points_match_the_model(self, flushed_calibration):
        path, _ = flushed_calibration
        saved = ss.lib.io.read_calibration(path).sel(analysis_bandwidth=40e6)
        grid = dict(FLUSH_GRID, analysis_bandwidth=(40e6,))
        _assert_matches_model(saved, expected_yfactor(grid))

    def test_saved_file_is_usable_for_lookups(self, flushed_calibration):
        path, _ = flushed_calibration
        capture = _capture(port=1, gain=-10, sample_rate=62.5e6)
        result = calibration.lookup_power_correction(path, capture, MCR)
        assert result == pytest.approx([1 / receiver_gain(-10)], rel=1e-6)

    @pytest.mark.xfail(
        strict=True,
        reason='_limit_nyquist_bandwidth never replaces inf (see the unit test above)',
    )
    def test_infinite_bandwidth_points_match_the_model(self, flushed_calibration):
        path, _ = flushed_calibration
        saved = ss.lib.io.read_calibration(path).sel(analysis_bandwidth=INF)
        grid = dict(FLUSH_GRID, analysis_bandwidth=(INF,))
        _assert_matches_model(saved, expected_yfactor(grid))

    @pytest.mark.xfail(
        strict=True,
        raises=KeyError,
        reason='flush assigns calibration_fields and sweep_start_time attrs to the '
        'capture data, but compute_y_factor_corrections builds a new Dataset '
        'without them',
    )
    def test_saved_attrs_record_the_calibration_fields(self, flushed_calibration):
        path, _ = flushed_calibration
        saved = ss.lib.io.read_calibration(path)
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
        assert scripted_input['prompts'] == [
            'Confirm that the noise diode ENR is 20.87 dB (y/n): '
        ]

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

        assert scripted_input['prompts'] == [
            'disable noise diode at port 0 and press enter: ',
            'enable noise diode at port 0 and press enter: ',
            'enable noise diode at port 1 and press enter: ',
        ]

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


def run_fake_sweep(spec_path, output_path, **replace):
    """run a sweep YAML against the installed fake, returning the sink results"""
    spec = ss.read_yaml_spec(spec_path)
    spec = spec.replace(sink=spec.sink.replace(path=str(output_path)), **replace)
    with ss.open_resources(spec, spec_path) as resources:
        results = [ds for ds in ss.iterate_sweep(resources) if ds is not None]
    return xr.concat(results, 'capture') if results else results


@pytest.fixture(scope='module')
def fake_calibration_run(tmp_path_factory):
    """(netCDF path, prompt log) after the calibration sweep runs against the fake"""
    import re

    from fake_soapy import install_fake_soapy

    path = tmp_path_factory.mktemp('fake-cal') / 'calibration.nc'
    prompts = []

    with pytest.MonkeyPatch.context() as mp:
        fake = install_fake_soapy(mp)

        def blocking_input(prompt=None):
            prompts.append(prompt)
            match = re.match(
                r'(enable|disable) noise diode at port (\d+)', prompt or ''
            )
            if match:
                fake.model.diode_on[int(match.group(2))] = match.group(1) == 'enable'
                return ''
            return 'y'

        mp.setattr(sa.util, 'blocking_input', blocking_input)
        run_fake_sweep(FAKE_SOAPY_CALIBRATION_SPEC, path)

    yield str(path), prompts
    sw.util.clear_caches()


FAKE_CAL_GRID = dict(
    YFACTOR_GRID, backend_sample_rate=(125e6, 62.5e6), analysis_bandwidth=(40e6, INF)
)


def _assert_within_dB(actual, expected, tol_dB, *, in_dB):
    model = expected.broadcast_like(actual).transpose(*actual.dims)
    if in_dB:
        err = actual.values - model.values
    else:
        err = 10 * np.log10(actual.values / model.values)
    assert np.abs(err).max() < tol_dB, err


class TestFakeCalibrationSweep:
    def test_prompts_follow_the_diode_state(self, fake_calibration_run):
        _, prompts = fake_calibration_run
        assert prompts[0].startswith('Confirm that the noise diode ENR is 20.87 dB')
        # one prompt per (port, state) change; the loop toggles the diode per port
        assert prompts[1:] == [
            'disable noise diode at port 0 and press enter: ',
            'enable noise diode at port 0 and press enter: ',
            'disable noise diode at port 1 and press enter: ',
            'enable noise diode at port 1 and press enter: ',
        ]

    def test_file_is_indexed_by_the_looped_fields(self, fake_calibration_run):
        path, _ = fake_calibration_run
        saved = ss.read_calibration(path)
        assert set(saved.dims) == set(FAKE_CAL_GRID)
        assert sorted(saved.backend_sample_rate.values) == [62.5e6, 125e6]
        assert set(saved.data_vars) >= set(CORRECTION_VARS)

    def test_finite_bandwidth_points_recover_the_model(self, fake_calibration_run):
        # 5 ms at 40 MHz gives a 1-sigma statistical error near 0.015 dB
        path, _ = fake_calibration_run
        saved = ss.read_calibration(path).sel(analysis_bandwidth=40e6)
        expected = expected_yfactor(dict(FAKE_CAL_GRID, analysis_bandwidth=(40e6,)))
        _assert_within_dB(saved.noise_figure, expected.noise_figure, 0.1, in_dB=True)
        _assert_within_dB(
            saved.power_correction, expected.power_correction, 0.1, in_dB=False
        )

    @pytest.mark.xfail(
        strict=True,
        reason='_limit_nyquist_bandwidth never replaces inf (see the unit test above)',
    )
    def test_infinite_bandwidth_points_recover_the_model(self, fake_calibration_run):
        path, _ = fake_calibration_run
        saved = ss.read_calibration(path).sel(analysis_bandwidth=INF)
        expected = expected_yfactor(dict(FAKE_CAL_GRID, analysis_bandwidth=(INF,)))
        _assert_within_dB(
            saved.power_correction, expected.power_correction, 0.1, in_dB=False
        )


def _rms_dBm(ds):
    return ds.channel_power_time_series.sel(power_detector='rms').mean('time_elapsed')


def _expected_dBm(port, bandwidth, tone_mW=0.0, nf_dB=YFACTOR_NF_DB):
    Te = 290.0 * (10 ** (nf_dB[port] / 10) - 1)
    return 10 * np.log10(tone_mW + BOLTZMANN_MW * (290.0 + Te) * bandwidth)


class TestFakeMeasurementSweep:
    def test_uncalibrated_power_is_in_full_scale_units(self, fake_soapy, tmp_path):
        fake_soapy.model.tones[0] = FAKE_TONE
        ds = run_fake_sweep(FAKE_SOAPY_SPEC, tmp_path / 'raw.zarr.zip')

        assert ds.port.values.tolist() == [0, 1, 0]
        assert ds.gain.values.tolist() == [0, 0, -10]
        gain_dB = np.array([50.0, 50.0, 40.0])
        expected = [
            _expected_dBm(0, 40e6, tone_mW=1e-6),
            _expected_dBm(1, 40e6),
            _expected_dBm(0, 40e6, tone_mW=1e-6),
        ]
        assert _rms_dBm(ds).values == pytest.approx(expected + gain_dB, abs=0.1)
        assert 'system_noise' not in ds

    def test_calibrated_power_recovers_the_input(
        self, fake_soapy, fake_calibration_run, tmp_path
    ):
        cal_path, _ = fake_calibration_run
        fake_soapy.model.tones[0] = FAKE_TONE
        spec = ss.read_yaml_spec(FAKE_SOAPY_SPEC)
        source = spec.source.replace(calibration=cal_path)

        ds = run_fake_sweep(FAKE_SOAPY_SPEC, tmp_path / 'cal.zarr.zip', source=source)

        rms = _rms_dBm(ds).values
        assert rms[0] == pytest.approx(_expected_dBm(0, 40e6, tone_mW=1e-6), abs=0.05)
        assert rms[2] == pytest.approx(_expected_dBm(0, 40e6, tone_mW=1e-6), abs=0.05)
        # noise only: the FIR passband is not exactly 40 MHz wide
        assert rms[1] == pytest.approx(_expected_dBm(1, 40e6), abs=0.15)
        assert ds.system_noise.attrs['units'] == 'dBm/Hz'
        assert ds.system_noise.values == pytest.approx(
            [5.0, 8.0, 5.0] + 10 * np.log10(BOLTZMANN_MW * 290.0), abs=0.1
        )

    @pytest.mark.xfail(
        strict=True,
        reason='the calibration file holds an infinite power_correction at '
        'analysis_bandwidth inf (see _limit_nyquist_bandwidth)',
    )
    def test_calibrated_power_at_infinite_bandwidth(
        self, fake_soapy, fake_calibration_run, tmp_path
    ):
        cal_path, _ = fake_calibration_run
        spec = ss.read_yaml_spec(FAKE_SOAPY_SPEC)
        capture = spec.captures[0].replace(analysis_bandwidth=INF)

        ds = run_fake_sweep(
            FAKE_SOAPY_SPEC,
            tmp_path / 'inf.zarr.zip',
            source=spec.source.replace(calibration=cal_path),
            captures=(capture,),
        )

        expected = [_expected_dBm(0, 125e6), _expected_dBm(1, 125e6)]
        assert _rms_dBm(ds).values == pytest.approx(expected, abs=0.1)
