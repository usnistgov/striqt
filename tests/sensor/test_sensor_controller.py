"""striqt.sensor.lib.controller: registry (lookup) lifetime, the raw acquisition
layout against the synthetic generators, IQ reuse, and the chunked read loop"""

from __future__ import annotations

import gc
from threading import Event

import numpy as np
import pytest
from numeric_checks import assert_close
from soapy_factories import soapy_capture
from sweep_strategies import SOURCE
from synthetic_sources import (
    BINDINGS,
    RESAMPLE_FILTER,
    SCALE_ONLY,
    expected_raw,
    make_capture,
)

import striqt.sensor as ss
import striqt.waveform as sw

TONE = ss.bindings.single_tone


def test_close_is_idempotent():
    closes = []
    ctrl = TONE.from_source_spec(SOURCE)
    backend_close = ctrl.backend.close
    ctrl.backend.close = lambda: (closes.append(1), backend_close())
    ctrl.close()
    ctrl.close()
    assert closes == [1]


def test_repeated_close_of_a_stale_controller_keeps_the_live_one_open():
    stale = TONE.from_source_spec(SOURCE)
    stale.close()
    with TONE.from_source_spec(SOURCE) as live:
        assert live.is_open()
        stale.close()
        assert live.is_open()


def test_garbage_collected_stale_controller_keeps_the_live_one_open():
    gc.disable()
    try:
        with TONE.from_source_spec(SOURCE):
            pass
        # the closed controller is unreachable but survives in a reference
        # cycle with its ReceiveBuffers until the collector runs
        with TONE.from_source_spec(SOURCE) as live:
            assert live.is_open()
            gc.collect()
            assert live.is_open()
    finally:
        gc.enable()


def test_failed_setup_entry_is_cleared_for_the_next_open(monkeypatch):
    lookup = ss.lib.controller.lookup

    def fail(spec):
        raise RuntimeError('simulated backend failure')

    monkeypatch.setattr(TONE.sensor, 'source_cls', fail)
    with pytest.raises(RuntimeError):
        TONE.from_source_spec(SOURCE)
    assert isinstance(lookup._obj[SOURCE], Event)
    monkeypatch.undo()

    with TONE.from_source_spec(SOURCE) as live:
        assert live.is_open()


# %% read_retries (fake SoapySDR device)


class TestReadRetries:
    def test_overflow_after_the_first_read_retriggers(
        self, fake_soapy, fake_controller
    ):
        with fake_controller(receive_retries=3) as ctrl:
            device = fake_soapy.devices[0]
            ctrl._arm_spec(soapy_capture(port=(0, 1), gain=(0, 0)))
            # the first read tolerates overflow; the second (holdoff) read does not
            device.fault_queue = [None, fake_soapy.SOAPY_SDR_OVERFLOW]

            iq = ctrl.acquire()

            assert len(device.calls_named('activateStream')) == 2
            assert iq.pre_align.shape[0] == 2
            assert iq.pre_align.all()
            # the retried acquisition restarted the holdoff from its own activation
            assert iq.info.start_time.value > device.streams[0].activate_time_ns

    def test_persistent_timeout_exhausts_the_retries(self, fake_soapy, fake_controller):
        with fake_controller(receive_retries=1) as ctrl:
            device = fake_soapy.devices[0]
            ctrl._arm_spec(soapy_capture(port=(0, 1), gain=(0, 0)))
            device.fault_queue = [fake_soapy.SOAPY_SDR_TIMEOUT] * 3

            with pytest.raises(ss.lib.sources.base.ReceiveStreamError, match='TIMEOUT'):
                ctrl.acquire()

            assert len(device.calls_named('activateStream')) == 2
            assert device.fault_queue == [fake_soapy.SOAPY_SDR_TIMEOUT]

    def test_no_retries_raises_on_the_first_fault(self, fake_soapy, fake_controller):
        with fake_controller(receive_retries=0) as ctrl:
            device = fake_soapy.devices[0]
            ctrl._arm_spec(soapy_capture(port=(0, 1), gain=(0, 0)))
            device.fault_queue = [fake_soapy.SOAPY_SDR_STREAM_ERROR]

            with pytest.raises(
                ss.lib.sources.base.ReceiveStreamError, match='STREAM_ERROR'
            ):
                ctrl.acquire()

            assert len(device.calls_named('activateStream')) == 1


# %% acquire: raw layout against the generators

LAYOUT_PRESETS = {'scale_only': SCALE_ONLY, 'resample_filter': RESAMPLE_FILTER}


def open_and_acquire(binding, capture, source=SOURCE, **kws):
    with BINDINGS[binding].from_source_spec(source, **kws) as ctrl:
        ctrl._arm_spec(capture)
        return ctrl.acquire()


@pytest.mark.parametrize('preset', list(LAYOUT_PRESETS), ids=list(LAYOUT_PRESETS))
@pytest.mark.parametrize('binding', list(BINDINGS), ids=list(BINDINGS))
def test_acquire_layout_is_lead_data_tail_from_the_generator(
    binding, preset, array_backend, isolated_lookup
):
    source = SOURCE.replace(array_backend=array_backend)
    capture = make_capture(binding, **LAYOUT_PRESETS[preset])
    iq = open_and_acquire(binding, capture, source)
    expected = expected_raw(binding, capture, source)
    assert iq.pre_align.shape == expected.shape
    assert_close(iq.pre_align, expected)


def test_acquire_metadata_defaults(isolated_lookup, subtests):
    capture = make_capture('single_tone', **SCALE_ONLY)
    with TONE.from_source_spec(SOURCE) as ctrl:
        ctrl._arm_spec(capture)
        iq = ctrl.acquire()
        with subtests.test('resampler matches the design for the master clock'):
            expected = ss.lib.compute.design_resampler(
                capture, SOURCE.master_clock_rate
            )
            assert dict(iq.resampler) == dict(expected)
        with subtests.test('complex64 transport is already full scale'):
            assert iq.voltage_scale == 1
        with subtests.test('info'):
            assert iq.info.source_id == ctrl.source_id
            assert iq.info.capture_index == 0
            assert iq.info.signal_trigger is None
        with subtests.test('stages other than pre_align are unset'):
            assert iq.pre_filter is None and iq.aligned is None
        with subtests.test('capture and source specs are attached'):
            assert iq.capture == capture and iq.source_spec == SOURCE


@pytest.mark.namespaces('cupy')
def test_acquire_on_cupy_returns_device_arrays(xp, isolated_lookup):
    source = SOURCE.replace(array_backend='cupy')
    iq = open_and_acquire(
        'single_tone', make_capture('single_tone', **SCALE_ONLY), source
    )
    assert sw.is_cupy_array(iq.pre_align)


# %% acquire: reuse_iq

REUSABLE = make_capture('single_tone', **SCALE_ONLY)
REUSABLE_VARIANT = REUSABLE.replace(analysis_bandwidth=5e6)
NOT_REUSABLE = REUSABLE.replace(frequency_offset=1e5)


def test_reuse_iq_returns_the_same_samples_for_a_compatible_capture(isolated_lookup):
    assert ss.lib.sources.buffers.is_reusable(
        REUSABLE, REUSABLE_VARIANT, SOURCE.master_clock_rate
    )
    with TONE.from_source_spec(SOURCE, reuse_iq=True) as ctrl:
        ctrl._arm_spec(REUSABLE)
        first = ctrl.acquire()
        ctrl._arm_spec(REUSABLE_VARIANT)
        second = ctrl.acquire()

    assert second.pre_align is first.pre_align
    assert second.capture == REUSABLE_VARIANT
    assert first.capture == REUSABLE


def test_reuse_iq_reacquires_for_an_incompatible_capture(isolated_lookup):
    with TONE.from_source_spec(SOURCE, reuse_iq=True) as ctrl:
        ctrl._arm_spec(REUSABLE)
        first = ctrl.acquire()
        ctrl._arm_spec(NOT_REUSABLE)
        second = ctrl.acquire()

    assert not np.shares_memory(first.pre_align, second.pre_align)
    assert_close(second.pre_align, expected_raw('single_tone', NOT_REUSABLE, SOURCE))


def test_reuse_iq_disabled_acquires_fresh_samples(isolated_lookup):
    with TONE.from_source_spec(SOURCE, reuse_iq=False) as ctrl:
        ctrl._arm_spec(REUSABLE)
        first = ctrl.acquire()
        ctrl._arm_spec(REUSABLE_VARIANT)
        second = ctrl.acquire()

    assert not np.shares_memory(first.pre_align, second.pre_align)
    assert_close(
        second.pre_align, expected_raw('single_tone', REUSABLE_VARIANT, SOURCE)
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        'buffers.is_reusable ignores analysis_bandwidth, but get_correction_overlaps '
        'pads the acquisition for the FIR of the capture being acquired, so the reused '
        'pre_align is short of the wider filter overlap of the second capture and '
        'correct_iq fails its size_out assertion (compute/corrections.py:116)'
    ),
)
def test_reused_iq_can_be_corrected_for_a_wider_analysis_filter(isolated_lookup):
    """reuse across analysis bandwidths is the purpose of reuse_iq, so the shared
    acquisition must carry the widest filter overlap of the captures that share it"""
    with TONE.from_source_spec(SOURCE, reuse_iq=True) as ctrl:
        ctrl._arm_spec(REUSABLE)
        ctrl.acquire()
        ctrl._arm_spec(REUSABLE_VARIANT)
        reused = ctrl.acquire()

    corrected = ss.correct_iq(reused)
    size_out = round(REUSABLE_VARIANT.duration * REUSABLE_VARIANT.sample_rate)
    assert corrected.pre_align.shape == (2, size_out)


def test_reuse_iq_clears_the_soapy_start_time(fake_controller):
    """a reused acquisition has no acquisition time of its own"""
    capture = soapy_capture(port=(0, 1), gain=(0, 0))
    with fake_controller(reuse_iq=True) as ctrl:
        ctrl._arm_spec(capture)
        first = ctrl.acquire()
        ctrl._arm_spec(capture.replace(analysis_bandwidth=20e6))
        second = ctrl.acquire()

    assert first.info.start_time is not None
    assert second.info.start_time is None
    assert second.pre_align is first.pre_align


# %% read_iq: buffer reuse across captures, chunked reads and validation


def test_one_port_capture_after_a_two_port_capture_has_one_row(isolated_lookup):
    """the receive buffers alternate between acquisitions, so the third
    acquisition is offered the two-port buffer of the first"""
    two_port = make_capture('sawtooth', **SCALE_ONLY)
    one_port = two_port.replace(port=1, duration=0.5e-3)
    one_port_short = one_port.replace(duration=0.25e-3)
    with BINDINGS['sawtooth'].from_source_spec(SOURCE) as ctrl:
        ctrl._arm_spec(two_port)
        ctrl.acquire()
        ctrl._arm_spec(one_port)
        ctrl.acquire()
        ctrl._arm_spec(one_port_short)
        third = ctrl.acquire()

    expected = expected_raw('sawtooth', one_port_short, SOURCE)
    assert third.pre_align.shape == expected.shape
    assert_close(third.pre_align, expected)


class HalfReads(BINDINGS['sawtooth'].sensor.source_cls):
    """delivers at most half of each read request"""

    read_calls = 0

    def read(self, buffers, offset, count, timeout_sec=None, *, on_overflow='except'):
        self.read_calls += 1
        count = max(1, count // 2)
        return super().read(
            buffers, offset, count, timeout_sec, on_overflow=on_overflow
        )


class OverReports(BINDINGS['sawtooth'].sensor.source_cls):
    """reports two more samples than were requested"""

    def read(self, buffers, offset, count, timeout_sec=None, *, on_overflow='except'):
        received, time_ns = super().read(
            buffers, offset, count, timeout_sec, on_overflow=on_overflow
        )
        return received + 2, time_ns


def test_partial_reads_are_completed_contiguously(monkeypatch, isolated_lookup):
    monkeypatch.setattr(BINDINGS['sawtooth'].sensor, 'source_cls', HalfReads)
    capture = make_capture('sawtooth', **SCALE_ONLY)
    with BINDINGS['sawtooth'].from_source_spec(SOURCE) as ctrl:
        ctrl._arm_spec(capture)
        iq = ctrl.acquire()
        read_calls = ctrl.backend.read_calls

    assert read_calls > 1
    assert_close(iq.pre_align, expected_raw('sawtooth', capture, SOURCE))


def test_overfilled_buffer_is_a_memory_error(monkeypatch, isolated_lookup):
    monkeypatch.setattr(BINDINGS['sawtooth'].sensor, 'source_cls', OverReports)
    with BINDINGS['sawtooth'].from_source_spec(SOURCE) as ctrl:
        ctrl._arm_spec(make_capture('sawtooth', **SCALE_ONLY))
        with pytest.raises(MemoryError):
            ctrl.acquire()


@pytest.mark.parametrize(
    'overlaps',
    [(3, 0), (0, 5), (-2, 0), (0, 0, 0), 4],
    ids=['odd_lead', 'odd_tail', 'negative', 'three_values', 'scalar'],
)
def test_read_iq_rejects_invalid_overlaps(overlaps, armed_tone_controller):
    with pytest.raises(ValueError, match='overlaps'):
        armed_tone_controller.read_iq(overlaps)


def test_acquire_rejects_a_list_of_overlaps(armed_tone_controller):
    with pytest.raises(ValueError, match='tuple or None'):
        armed_tone_controller.acquire(overlaps=[0, 0])


def test_acquire_with_explicit_overlaps_matches_the_default(armed_tone_controller):
    ctrl = armed_tone_controller
    overlaps = ss.lib.compute.get_correction_overlaps(REUSABLE, SOURCE)
    explicit = ctrl.acquire(overlaps=overlaps)
    default = ctrl.acquire()

    assert explicit.info.signal_trigger is None
    assert explicit.pre_align.shape == default.pre_align.shape
    np.testing.assert_array_equal(explicit.pre_align, default.pre_align)


def test_acquire_before_arming_is_an_error(isolated_lookup):
    raises = pytest.raises(AttributeError, match='armed')
    with TONE.from_source_spec(SOURCE) as ctrl, raises:
        ctrl.acquire()
