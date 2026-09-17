"""striqt.sensor.lib.controller: registry (lookup) lifetime; closing a stale
controller must not disturb a live one opened later with an equal source spec"""

from __future__ import annotations

import gc
from threading import Event

import pytest
from soapy_factories import fake_controller, soapy_capture

import striqt.sensor as ss


def _controller_cls(sweep):
    return ss.lib.bindings.get_controller(sweep)


def test_close_is_idempotent(cw_sweep):
    closes = []
    ctrl = _controller_cls(cw_sweep).from_source_spec(cw_sweep.source)
    backend_close = ctrl.backend.close
    ctrl.backend.close = lambda: (closes.append(1), backend_close())
    ctrl.close()
    ctrl.close()
    assert closes == [1]


def test_repeated_close_of_a_stale_controller_keeps_the_live_one_open(cw_sweep):
    cls = _controller_cls(cw_sweep)
    stale = cls.from_source_spec(cw_sweep.source)
    stale.close()
    with cls.from_source_spec(cw_sweep.source) as live:
        assert live.is_open()
        stale.close()
        assert live.is_open()


def test_garbage_collected_stale_controller_keeps_the_live_one_open(cw_sweep):
    cls = _controller_cls(cw_sweep)
    gc.disable()
    try:
        with cls.from_source_spec(cw_sweep.source):
            pass
        # the closed controller is unreachable but survives in a reference
        # cycle with its ReceiveBuffers until the collector runs
        with cls.from_source_spec(cw_sweep.source) as live:
            assert live.is_open()
            gc.collect()
            assert live.is_open()
    finally:
        gc.enable()


def test_failed_setup_entry_is_cleared_for_the_next_open(cw_sweep, monkeypatch):
    cls = _controller_cls(cw_sweep)
    lookup = ss.lib.controller.lookup

    def fail(spec):
        raise RuntimeError('simulated backend failure')

    monkeypatch.setattr(cls.sensor, 'source_cls', fail)
    with pytest.raises(RuntimeError):
        cls.from_source_spec(cw_sweep.source)
    assert isinstance(lookup._obj[cw_sweep.source], Event)
    monkeypatch.undo()

    with cls.from_source_spec(cw_sweep.source) as live:
        assert live.is_open()


# %% read_retries (fake SoapySDR device)


class TestReadRetries:
    def test_overflow_after_the_first_read_retriggers(self, fake_soapy, fake_soapy_ext):
        with fake_controller(fake_soapy_ext, receive_retries=3) as ctrl:
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

    def test_persistent_timeout_exhausts_the_retries(self, fake_soapy, fake_soapy_ext):
        with fake_controller(fake_soapy_ext, receive_retries=1) as ctrl:
            device = fake_soapy.devices[0]
            ctrl._arm_spec(soapy_capture(port=(0, 1), gain=(0, 0)))
            device.fault_queue = [fake_soapy.SOAPY_SDR_TIMEOUT] * 3

            with pytest.raises(ss.lib.sources.base.ReceiveStreamError, match='TIMEOUT'):
                ctrl.acquire()

            assert len(device.calls_named('activateStream')) == 2
            assert device.fault_queue == [fake_soapy.SOAPY_SDR_TIMEOUT]

    def test_no_retries_raises_on_the_first_fault(self, fake_soapy, fake_soapy_ext):
        with fake_controller(fake_soapy_ext, receive_retries=0) as ctrl:
            device = fake_soapy.devices[0]
            ctrl._arm_spec(soapy_capture(port=(0, 1), gain=(0, 0)))
            device.fault_queue = [fake_soapy.SOAPY_SDR_STREAM_ERROR]

            with pytest.raises(
                ss.lib.sources.base.ReceiveStreamError, match='STREAM_ERROR'
            ):
                ctrl.acquire()

            assert len(device.calls_named('activateStream')) == 1
