"""end-to-end sweeps from the CPU_RUNS YAML files, checked against the synthetic
source models in striqt.sensor.lib.sources.function"""

from __future__ import annotations

import math
import os
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

import striqt.analysis as sa
import striqt.sensor as ss
from striqt.cli import sensor_sweep

CORE_VARIABLES = {'power_spectral_density', 'channel_power_time_series', 'spectrogram'}

# capture and analysis values shared by every CPU_RUNS file
TONE_OFFSET = -3e6
PULSE_TIME = 1e-3
NOISE_PSD = 1e-17
INTEGRATION_BANDWIDTH = 360e3


def _mean_psd(ds, capture: int):
    row = ds.power_spectral_density.sel(time_statistic='mean').isel(capture=capture)
    return row.dropna('baseband_frequency')


def check_cw(ds, subtests):
    assert set(ds.frequency_offset.values) == {TONE_OFFSET}
    for i in range(ds.sizes['capture']):
        with subtests.test('tone at the frequency offset', capture=i):
            psd = _mean_psd(ds, i)
            tone_bin = np.argmin(abs(psd.baseband_frequency.values - TONE_OFFSET))
            assert np.argmax(psd.values) == tone_bin


def check_cw_cyclic_power(ds, subtests):
    check_cw(ds, subtests)
    cyclic = ds.cyclic_channel_power
    detector_period = Fraction(cyclic.attrs['detector_period'])
    lag_count = round(cyclic.attrs['cyclic_period'] / detector_period)
    expected_lags = np.arange(lag_count) * float(detector_period)
    assert np.allclose(cyclic.cyclic_lag.values, expected_lags)
    for i in range(ds.sizes['capture']):
        with subtests.test('constant tone power at every cycle lag', capture=i):
            trace = cyclic.isel(capture=i).sel(power_detector='rms')
            assert float(trace.max() - trace.min()) < 0.1


def check_dirac_delta(ds, subtests):
    assert set(ds.time.values) == {PULSE_TIME}
    # the pulse lands on a detector bin edge (time == detector_period), so it is the
    # first sample of the bin that starts at PULSE_TIME
    peak = ds.channel_power_time_series.sel(power_detector='peak')
    pulse_bin = np.argmin(abs(peak.time_elapsed.values - PULSE_TIME))
    for i in range(ds.sizes['capture']):
        with subtests.test('peak at the pulse time', capture=i):
            assert np.argmax(peak.isel(capture=i).values) == pulse_bin


def check_noise(ds, subtests):
    assert set(ds.noise_psd.values) == {NOISE_PSD}
    expected = 10 * math.log10(NOISE_PSD * INTEGRATION_BANDWIDTH)
    for i in range(ds.sizes['capture']):
        with subtests.test('flat psd at the noise level', capture=i):
            psd = _mean_psd(ds, i)
            inner = psd.where(
                abs(psd.baseband_frequency) < 0.4 * float(ds.analysis_bandwidth[i]),
                drop=True,
            )
            assert float(inner.max() - inner.min()) < 2.0
            assert float(inner.mean()) == pytest.approx(expected, abs=1.0)


def check_sawtooth(ds, subtests):
    # period == duration and power == 0: a single ramp from 0 to full scale, so the
    # detected power rises across the time bins and ends at 0 dBm
    assert np.all(ds.period.values == ds.duration.values)
    assert set(ds.power.values) == {0.0}
    detectors = ds.channel_power_time_series
    for i in range(ds.sizes['capture']):
        with subtests.test('rising ramp to full scale', capture=i):
            peak = detectors.sel(power_detector='peak').isel(capture=i).values
            rms = detectors.sel(power_detector='rms').isel(capture=i).values
            assert np.all(np.diff(peak) > 0)
            assert np.all(rms < peak)
            assert peak[-1] == pytest.approx(0.0, abs=1.0)


def check_site(ds, subtests):
    # sites/global.yaml and sites/radio02.yaml keyed on the extension module's RADIO_ID
    assert set(ds.source_id.values) == {'48b02d17a587'}
    assert set(ds.site_name.values) == {'WAPA-north'}
    assert set(ds.radio_name.values) == {'radio02'}
    assert set(ds.channel_name.values) == {'3750 MHz'}
    assert ds.antenna_name.values.tolist() == ['Omni', '1x32']
    assert ds.antenna_polarization.values.tolist() == ['Vertical', 'Linear +45']
    assert ds.antenna_model.values.tolist() == ['OmniModel', 'PanelModel']
    assert ds.antenna_index.values.tolist() == [0, 1]
    assert set(ds.gain.values) == {0.0}
    assert set(ds.mast_height.values) == {1.7}
    check_cw(ds, subtests)


CHECKS = {
    'cw-cpu': check_cw_cyclic_power,
    'dirac_delta-cpu': check_dirac_delta,
    'noise-cpu': check_noise,
    'sawtooth-cpu': check_sawtooth,
    'site-cpu': check_site,
}


def test_run(cpu_sweep_file, tmp_path, monkeypatch, subtests):
    # open_resources chdirs into the spec directory and does not change back
    monkeypatch.chdir(os.getcwd())
    out_path = tmp_path / 'out.zarr.zip'

    sensor_sweep.run(cpu_sweep_file, output_path=str(out_path))

    assert out_path.exists()
    spec = ss.read_yaml_spec(cpu_sweep_file, output_path=str(out_path))
    rebuilt = ss.read_zarr_spec(out_path, extension_root=Path(cpu_sweep_file).parent)
    assert rebuilt == spec
    ds = sa.load(out_path)

    # one row per (capture, port); the repeat loops in these files all count 1
    assert ds.port.values.tolist() == [p for c in spec.captures for p in c.port]
    assert len(set(ds.capture_index.values)) == len(spec.captures)
    assert CORE_VARIABLES <= set(ds.data_vars)
    CHECKS[Path(cpu_sweep_file).stem](ds, subtests)
