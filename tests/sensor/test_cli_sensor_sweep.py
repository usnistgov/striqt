"""striqt.cli.sensor_sweep: the logging callbacks of the sweep CLI

The tests assert record counts and levels, not message text.
"""

from __future__ import annotations

import logging
import types

from soapy_factories import soapy_capture
from sweep_strategies import air_sweep, make_delayed, spectrogram_result

import striqt.sensor as ss
from striqt.cli.sensor_sweep import log_overloads, log_spectrogram_snr

AIR = ss.bindings.air7101b
CAPTURE = soapy_capture(port=(0, 1), center_frequency=(1e9, 2e9), gain=0.0)


def delayed(extra_data, calibration=None):
    sweep = air_sweep(CAPTURE, calibration=calibration)
    return make_delayed(extra_data, sweep=sweep, capture=CAPTURE)


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
            resources(calibration_nc),
            cache,
            CAPTURE,
            spectrogram_result(shape=(2, 4, 8)),
        )
    assert [r.levelno for r in caplog.records] == [logging.INFO, logging.INFO]


def test_log_spectrogram_snr_is_silent_for_another_cache(calibration_nc, caplog):
    cache = types.SimpleNamespace(name='ssb_iq_cache')
    with caplog.at_level(logging.DEBUG, logger='striqt.analysis'):
        log_spectrogram_snr(
            resources(calibration_nc),
            cache,
            CAPTURE,
            spectrogram_result(shape=(2, 4, 8)),
        )
    assert caplog.records == []
