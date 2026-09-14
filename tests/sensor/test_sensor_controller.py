"""striqt.sensor.lib.controller: registry (lookup) lifetime; closing a stale
controller must not disturb a live one opened later with an equal source spec"""

from __future__ import annotations

import gc
from threading import Event

import pytest

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
