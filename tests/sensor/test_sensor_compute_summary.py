"""striqt.sensor.lib.compute.summary: the per-port overload state derived from the
headroom that a SoapySDR source reports, and the spectrogram SNR above the calibrated
noise floor"""

from __future__ import annotations

import math

import numpy as np
import pytest
from soapy_factories import soapy_capture
from sweep_strategies import BOLTZMANN_MW, T_REF
from synthetic_sources import SCALE_ONLY, preset_capture, preset_sweep

import striqt.analysis as sa
import striqt.sensor as ss
from striqt.sensor.lib import compute
from striqt.sensor.lib.compute import summary

AIR = ss.bindings.air7101b

# the capture under test spans 2 ports at 1 GHz and 2 GHz, and the sweep also
# visits 2 GHz on port 0 at 30 dB more gain
CAPTURE = soapy_capture(port=(0, 1), center_frequency=(1e9, 2e9), gain=0.0)
HIGH_GAIN = soapy_capture(port=0, center_frequency=2e9, gain=30.0)
SWEEP = AIR.sensor.sweep_spec_cls(
    source=AIR.schema.source(), captures=(CAPTURE, HIGH_GAIN)
)
TONE = preset_capture('single_tone', **SCALE_ONLY)
TONE_SWEEP = preset_sweep('single_tone', (TONE,))


def delayed(extra_data, capture=CAPTURE, sweep=SWEEP):
    config = compute.EvaluationOptions(
        sweep_spec=sweep, registry=sa.registry, as_xarray='delayed', extra_attrs={}
    )
    return compute.DelayedDataset(
        delayed={},
        capture=capture,
        extra_coords=ss.specs.AcquisitionInfo(),
        extra_data=extra_data,
        config=config,
    )


# %% overloads


def test_overloads_one_entry_per_port_in_split_order():
    ports = summary.overloads(delayed({'adc_headroom': (1.0, 1.0)}))
    assert [p.capture.port for p in ports] == [0, 1]
    assert [p.capture.center_frequency for p in ports] == [1e9, 2e9]


@pytest.mark.parametrize(
    'headroom, adc',
    [
        ((-1.0, 3.0), [True, False]),
        ((3.0, 0.0), [False, True]),
        ((1.0, 3.0), [False, False]),
    ],
    ids=['negative_on_port_0', 'zero_on_port_1', 'positive'],
)
def test_adc_overload_is_headroom_at_or_below_zero(headroom, adc):
    ports = summary.overloads(delayed({'adc_headroom': headroom}))
    assert [p.adc for p in ports] == adc
    assert [p.im3_victims for p in ports] == [(), ()]


def test_im3_victims_follow_the_im3_rule():
    """third-order products of a channel visited at higher gain grow 2 dB per dB of
    gain difference relative to the signal, so 10 dB of IF headroom at 0 dB gain is
    overrun by the 30 dB gain capture at 2 GHz on port 0 (10 + 2/3 * (0 - 30) < 0)
    but not by the 1 GHz capture at equal gain, nor on port 1"""
    ports = summary.overloads(delayed({'if_headroom': (10.0, 10.0)}))
    assert [p.im3_victims for p in ports] == [(2e9,), ()]
    assert [p.adc for p in ports] == [False, False]


def test_im3_victims_are_empty_with_ample_headroom():
    ports = summary.overloads(delayed({'if_headroom': (21.0, 21.0)}))
    assert [p.im3_victims for p in ports] == [(), ()]


@pytest.mark.parametrize(
    'extra, capture, sweep',
    [
        ({}, CAPTURE, SWEEP),
        ({'adc_headroom': (-1.0, -1.0), 'if_headroom': (-1.0, -1.0)}, TONE, TONE_SWEEP),
    ],
    ids=['no_headroom_keys', 'not_a_soapy_capture'],
)
def test_overloads_is_empty(extra, capture, sweep):
    assert summary.overloads(delayed(extra, capture, sweep)) == ()


# %% describe_overloads


def test_describe_overloads_is_none_without_an_overload():
    ports = summary.overloads(delayed({'adc_headroom': (1.0, 3.0)}))
    assert summary.describe_overloads(ports) is None
    assert summary.describe_overloads(()) is None


def test_describe_overloads_names_the_overloaded_ports():
    ports = summary.overloads(
        delayed({'adc_headroom': (-1.0, 3.0), 'if_headroom': (10.0, 10.0)})
    )
    text = summary.describe_overloads(ports)
    assert isinstance(text, str)
    assert 'port 0' in text
    assert 'port 1' not in text


# %% spectrogram_snr

NOISE_BANDWIDTH = 1e4
PEAK_DBM = -100.0


def air_sweep(capture, calibration=None):
    source = AIR.schema.source(calibration=calibration)
    return AIR.sensor.sweep_spec_cls(source=source, captures=(capture,))


def spectrogram_result(peak_dBm, shape=(1, 4, 8)):
    """a linear-power spectrogram whose only feature is one bin at `peak_dBm`"""
    spg = np.full(shape, 1e-30, dtype='float32')
    spg[0, shape[1] // 2, shape[2] // 2] = 10 ** (peak_dBm / 10)
    return spg, {'noise_bandwidth': NOISE_BANDWIDTH}


def test_spectrogram_snr_is_the_peak_above_the_calibrated_noise(calibration_nc):
    """the SNR is the spectrogram peak above the calibrated system noise k*T*B raised
    by the noise figure the calibration file records for the capture's port and
    center frequency (5 dB at port 0, 1 GHz)"""
    capture = soapy_capture()
    noise_dBm = 5.0 + 10 * math.log10(BOLTZMANN_MW * T_REF * NOISE_BANDWIDTH)

    entries = summary.spectrogram_snr(
        air_sweep(capture, calibration_nc), capture, spectrogram_result(PEAK_DBM)
    )

    assert entries is not None and len(entries) == 1
    assert entries[0].capture.port == 0
    assert entries[0].snr_dB == pytest.approx(PEAK_DBM - noise_dBm, abs=0.5)


def test_spectrogram_snr_is_none_without_a_calibration():
    capture = soapy_capture()
    result = spectrogram_result(PEAK_DBM)
    assert summary.spectrogram_snr(air_sweep(capture), capture, result) is None


def test_spectrogram_snr_skips_a_port_with_no_power(calibration_nc):
    capture = soapy_capture()
    spg, attrs = spectrogram_result(PEAK_DBM)
    spg[:] = 0.0
    entries = summary.spectrogram_snr(
        air_sweep(capture, calibration_nc), capture, (spg, attrs)
    )
    assert entries == ()


def test_describe_spectrogram_snr_names_the_port(calibration_nc):
    capture = soapy_capture()
    (entry,) = summary.spectrogram_snr(
        air_sweep(capture, calibration_nc), capture, spectrogram_result(PEAK_DBM)
    )
    text = summary.describe_spectrogram_snr(entry, source_id='beef')
    assert isinstance(text, str)
    assert 'port' in text
