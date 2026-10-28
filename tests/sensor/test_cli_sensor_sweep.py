"""striqt.cli.sensor_sweep: the logging callbacks the sweep CLI registers. The record
counts and levels are the contract; the message text is not."""

from __future__ import annotations

import logging
import types

import numpy as np
from soapy_factories import soapy_capture

import striqt.analysis as sa
import striqt.sensor as ss
from striqt.cli.sensor_sweep import log_overloads, log_spectrogram_snr
from striqt.sensor.lib import compute

AIR = ss.bindings.air7101b
CAPTURE = soapy_capture(port=(0, 1), center_frequency=(1e9, 2e9), gain=0.0)


def delayed(extra_data, calibration=None):
    sweep = AIR.sensor.sweep_spec_cls(
        source=AIR.schema.source(calibration=calibration), captures=(CAPTURE,)
    )
    config = compute.EvaluationOptions(
        sweep_spec=sweep, registry=sa.registry, as_xarray='delayed', extra_attrs={}
    )
    return compute.DelayedDataset(
        delayed={},
        capture=CAPTURE,
        extra_coords=ss.specs.AcquisitionInfo(),
        extra_data=extra_data,
        config=config,
    )


# %% log_overloads


def test_log_overloads_warns_once_for_an_overload(caplog):
    with caplog.at_level(logging.DEBUG, logger='striqt.analysis'):
        log_overloads(delayed({'adc_headroom': (-1.0, 3.0)}))
    assert [r.levelno for r in caplog.records] == [logging.WARNING]


def test_log_overloads_is_silent_without_an_overload(caplog):
    with caplog.at_level(logging.DEBUG, logger='striqt.analysis'):
        log_overloads(delayed({'adc_headroom': (1.0, 3.0)}))
    assert caplog.records == []


# %% log_spectrogram_snr


def spectrogram_result(shape=(2, 4, 8)):
    spg = np.full(shape, 1e-30, dtype='float32')
    spg[:, shape[1] // 2, shape[2] // 2] = 1e-10
    return spg, {'noise_bandwidth': 1e4}


def resources(calibration):
    dd = delayed({}, calibration)
    return {
        'sweep_spec': dd.config.sweep_spec,
        'source': types.SimpleNamespace(source_id='beef'),
        'format_path': None,
    }


def test_log_spectrogram_snr_logs_one_info_record_per_port(calibration_nc, caplog):
    cache = types.SimpleNamespace(name='_cached_spectrogram')
    with caplog.at_level(logging.DEBUG, logger='striqt.analysis'):
        log_spectrogram_snr(
            resources(calibration_nc), cache, CAPTURE, spectrogram_result()
        )
    assert [r.levelno for r in caplog.records] == [logging.INFO, logging.INFO]


def test_log_spectrogram_snr_is_silent_for_another_cache(calibration_nc, caplog):
    cache = types.SimpleNamespace(name='ssb_iq_cache')
    with caplog.at_level(logging.DEBUG, logger='striqt.analysis'):
        log_spectrogram_snr(
            resources(calibration_nc), cache, CAPTURE, spectrogram_result()
        )
    assert caplog.records == []
