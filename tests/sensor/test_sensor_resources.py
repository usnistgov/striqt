"""striqt.sensor.lib.resources: the open_resources lifecycle"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path

import pytest
from conftest import FAKE_SOAPY_SPEC, assert_source_released
from synthetic_sources import SCALE_ONLY, preset_capture, preset_sweep, tone_sweep

import striqt.sensor as ss

OFFSETS = (1e6, 2e6)


# %% working directory


def test_cwd_is_unchanged_without_a_spec_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with ss.open_resources(tone_sweep(OFFSETS), None):
        assert Path.cwd().resolve() == tmp_path.resolve()


def test_cwd_follows_the_spec_directory(tmp_path, monkeypatch):
    # open_resources does not change back; monkeypatch restores at teardown
    monkeypatch.chdir(os.getcwd())
    spec_dir = tmp_path / 'sub'
    spec_dir.mkdir()
    with ss.open_resources(tone_sweep(OFFSETS), spec_dir / 'sweep.yaml'):
        assert Path.cwd().resolve() == spec_dir.resolve()


# %% test_only


class BindingPeripherals(ss.peripherals.NoPeripherals):
    def open(self):
        raise RuntimeError("the binding's peripherals were opened")


def test_test_only_substitutes_no_peripherals(monkeypatch):
    """`test_only` substitutes the no-op `NoPeripherals` for the peripherals of the
    binding"""
    sweep = tone_sweep(OFFSETS)
    binding = ss.lib.bindings.get_controller(sweep).sensor
    monkeypatch.setattr(binding, 'peripherals_cls', BindingPeripherals)
    with ss.open_resources(sweep, None, test_only=True) as res:
        assert type(res['peripherals']) is ss.peripherals.NoPeripherals


# %% sink failure


class FailingInitSink(ss.sinks.NoSink):
    def __init__(self, *args, **kws):
        raise RuntimeError('simulated sink failure')


class FailingOpenSink(ss.sinks.NoSink):
    def open(self):
        raise RuntimeError('simulated sink failure')


@pytest.mark.parametrize(
    'cls', [FailingInitSink, FailingOpenSink], ids=['init', 'open']
)
def test_sink_failure_surfaces_and_closes_the_source(cls, isolated_lookup, monkeypatch):
    """a sink that fails in `open_sink` of the runner or on entry propagates out of
    `open_resources`. The source opened in parallel is released."""
    monkeypatch.setattr(
        ss.lib.actions.SweepRunner, 'open_sink', lambda self, fmt: cls(self.sweep)
    )
    sweep = tone_sweep(OFFSETS)

    with (
        pytest.raises(RuntimeError, match='simulated sink failure'),
        ss.open_resources(sweep, None),
    ):
        pass

    assert_source_released(isolated_lookup, sweep)


# %% calibration


def test_calibration_is_loaded_from_the_source_spec(fake_soapy_ext, calibration_nc):
    import xarray as xr

    spec = ss.read_yaml_spec(FAKE_SOAPY_SPEC)
    spec = spec.replace(
        source=spec.source.replace(calibration=calibration_nc),
        sink=spec.sink.replace(path=None),
    )
    with ss.open_resources(spec, None) as res:
        assert res['calibration'].equals(xr.load_dataset(calibration_nc))


# %% log file


def test_log_path_installs_a_file_handler_at_the_log_level(tmp_path):
    """a sink with `log_path` installs a file handler for that path on the ``striqt``
    logger at the `log_level` of the sink"""
    log_path = tmp_path / 'logs' / 'sweep.log'
    sweep = tone_sweep(OFFSETS)
    sink = sweep.sink.replace(log_path=str(log_path), log_level='warning')
    sweep = sweep.replace(sink=sink)

    with ss.open_resources(sweep, None):
        handlers = [
            h
            for h in logging.getLogger('striqt').handlers
            if getattr(h, 'baseFilename', None) == str(log_path)
        ]

    assert len(handlers) == 1
    assert handlers[0].level == logging.WARNING
    assert log_path.exists()


# %% resident_bytes_bound


def two_port_sweep(buffers: int):
    """return a sweep of 2 scale-only 2-port captures of 1 ms and 2 ms with `buffers`
    acquisitions queued ahead of analysis"""
    captures = (
        preset_capture('single_tone', **SCALE_ONLY),
        preset_capture('single_tone', **{**SCALE_ONLY, 'duration': 2e-3}),
    )
    sweep = preset_sweep('single_tone', captures)
    bursts = sweep.source.bursts.replace(buffers=buffers)
    return sweep.replace(source=sweep.source.replace(bursts=bursts))


def test_resident_bytes_bound_counts_queued_analyzed_and_written_segments():
    """`buffers + 2` copies of the largest read: 2 ports of complex64 over the
    record of the 2 ms capture and its correction overlaps"""
    buffers = 3
    sweep = two_port_sweep(buffers)
    longest = sweep.captures[1]
    overlaps = ss.lib.compute.get_correction_overlaps(
        longest, sweep.source, sweep.analysis
    )
    samples = round(longest.sample_rate * longest.duration) + sum(overlaps)
    expected = (buffers + 2) * 2 * 8 * samples

    assert ss.lib.resources.resident_bytes_bound(sweep, None) == expected
    one = ss.lib.resources.resident_bytes_bound(two_port_sweep(1), None)
    assert one == 3 * 2 * 8 * samples


def test_resident_bytes_bound_includes_the_holdoff_of_a_first_segment():
    """a strobe pads the read of a first segment by 2 strobe periods per port"""
    buffers = 1
    strobe = 0.25e-3
    sweep = two_port_sweep(buffers)
    strobed = sweep.replace(source=sweep.source.replace(trigger_strobe=strobe))
    longest = sweep.captures[1]
    fs = ss.lib.compute.design_resampler(longest, sweep.source.master_clock_rate)[
        'fs_sdr'
    ]
    pad = math.ceil(fs * 2 * strobe)
    plain = ss.lib.resources.resident_bytes_bound(sweep, None)
    assert (
        ss.lib.resources.resident_bytes_bound(strobed, None) == plain + 3 * 2 * 8 * pad
    )


def test_open_resources_logs_the_resident_bound_once(caplog):
    sweep = two_port_sweep(2)
    bound = ss.lib.resources.resident_bytes_bound(sweep, None)
    with (
        caplog.at_level(logging.INFO, logger='striqt.sweep'),
        ss.open_resources(sweep, None),
    ):
        pass
    records = [
        r
        for r in caplog.records
        if r.levelno == logging.INFO and str(bound) in r.getMessage()
    ]
    assert len(records) == 1


# %% callbacks


def test_on_source_opened_is_called_once_with_the_spec_and_id():
    calls = []
    sweep = tone_sweep(OFFSETS)
    on_opened = lambda *a: calls.append(a)
    with ss.open_resources(sweep, None, on_source_opened=on_opened) as res:
        source_id = ss.lib.controller.lookup.id(sweep.source)
        assert res['callbacks'].on_source_opened is on_opened
    assert calls == [(sweep, source_id)]


def test_resources_carry_empty_callbacks_by_default():
    with ss.open_resources(tone_sweep(OFFSETS), None) as res:
        assert res['callbacks'] == ss.lib.resources.SweepCallbacks()
