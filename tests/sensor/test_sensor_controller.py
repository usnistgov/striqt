"""striqt.sensor.lib.controller: controller lifetime, the raw acquisition layout, IQ
reuse, the chunked read loop, and acquisition through the fake SoapySDR device
"""

from __future__ import annotations

import gc
import logging
import weakref
from queue import Full
from threading import Event

import numpy as np
import pytest
from numeric_checks import assert_close, rms, to_numpy
from soapy_factories import MCR, soapy_capture
from sweep_strategies import SOURCE
from synthetic_sources import (
    BINDINGS,
    RESAMPLE_FILTER,
    SCALE_ONLY,
    acquire_corrected,
    build_acquired_iq,
    expected_raw,
    filtered,
    fs_sdr,
    generator,
    preset_capture,
)

import striqt.sensor as ss
import striqt.waveform as sw
from striqt.sensor.lib.sources import buffers

TONE = ss.bindings.single_tone


class WaitSignal(Event):
    """an Event that sets `entered` when a thread starts waiting on it"""

    def __init__(self):
        super().__init__()
        self.entered = Event()

    def wait(self, timeout=None):
        self.entered.set()
        return super().wait(timeout)


def test_failed_open_wakes_a_thread_waiting_for_the_id(isolated_lookup, monkeypatch):
    """a thread blocked in `lookup.id` raises when the source fails to open.

    The waiter sees the recorded exception, or the slot that the failed controller
    cleared on close, as the two threads interleave. Both raise.
    """
    from threading import Thread

    id_slot = WaitSignal()
    isolated_lookup._id[SOURCE] = id_slot

    def failing_buffers(controller):
        # the open fails only after the waiter has moved from the controller event
        # to the id event. The registry replaces the id event when the failed
        # controller closes
        assert id_slot.entered.wait(timeout=30)
        raise RuntimeError('no buffers')

    monkeypatch.setattr(ss.lib.sources.buffers, 'ReceiveBuffers', failing_buffers)
    outcome = []

    def wait_for_id():
        try:
            outcome.append(isolated_lookup.id(SOURCE, timeout=30))
        except BaseException as ex:
            outcome.append(ex)

    waiter = Thread(target=wait_for_id)
    waiter.start()
    with pytest.raises(RuntimeError):
        TONE.from_source_spec(SOURCE)
    waiter.join(timeout=30)

    assert not waiter.is_alive(), 'the waiter is still blocked on the id'
    assert isinstance(outcome[0], (ss.lib.util.ThreadInterruptRequest, RuntimeError))


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


def test_reopening_an_equal_spec_closes_the_stale_controller(caplog):
    lookup = ss.lib.controller.lookup
    stale = TONE.from_source_spec(SOURCE)
    # the binding call form is the notebook path through Controller.__init__
    with caplog.at_level(logging.WARNING, logger='striqt.source'):
        live = TONE(**SOURCE.to_dict())
    with live:
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert not stale.is_open(wait=False)
        assert lookup.instance(SOURCE) is live
        assert live.is_open()
        stale.close()
        assert live.is_open()


def test_collecting_an_unreferenced_controller_closes_it_and_frees_its_slot(caplog):
    lookup = ss.lib.controller.lookup
    closes = []
    ctrl = TONE.from_source_spec(SOURCE)
    backend_close = ctrl.backend.close
    ctrl.backend.close = lambda: (closes.append(1), backend_close())
    ref = weakref.ref(ctrl)

    del ctrl
    # the controller survives in a reference cycle with its ReceiveBuffers until
    # the collector runs. The registry must not keep the controller alive
    gc.collect()
    assert ref() is None
    assert closes == [1]
    assert isinstance(lookup._get(SOURCE), Event)

    with (
        caplog.at_level(logging.WARNING, logger='striqt.source'),
        TONE.from_source_spec(SOURCE) as live,
    ):
        assert live.is_open()
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_source_info_of_a_replaced_controller_is_a_connection_error():
    stale = TONE.from_source_spec(SOURCE)
    with TONE.from_source_spec(SOURCE), pytest.raises(ConnectionError):
        _ = stale.source_info


def test_failed_setup_entry_is_cleared_for_the_next_open(monkeypatch, caplog):
    lookup = ss.lib.controller.lookup

    def fail(spec):
        raise RuntimeError('simulated backend failure')

    monkeypatch.setattr(TONE.sensor, 'source_cls', fail)
    with pytest.raises(RuntimeError):
        TONE.from_source_spec(SOURCE)
    assert isinstance(lookup._get(SOURCE), Event)
    monkeypatch.undo()

    with (
        caplog.at_level(logging.WARNING, logger='striqt.source'),
        TONE.from_source_spec(SOURCE) as live,
    ):
        assert live.is_open()
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


# %% read_retries (fake SoapySDR device)


class TestReadRetries:
    @pytest.mark.parametrize('time_sync_at', ['acquire', 'open'])
    def test_overflow_after_the_first_read_retriggers(
        self, time_sync_at, fake_soapy, fake_controller, caplog
    ):
        """the retry re-enables the stream whether or not the trigger syncs the
        clock, stamps the retried read from the new activation, and counts the
        re-trigger on the record start log"""
        with (
            fake_controller(receive_retries=3, time_sync_at=time_sync_at) as ctrl,
            caplog.at_level(logging.DEBUG, logger='striqt.source'),
        ):
            device = fake_soapy.devices[0]
            ctrl._arm_spec(soapy_capture(port=(0, 1), gain=(0, 0)))
            # the first read tolerates overflow; the second (holdoff) read does not
            device.fault_queue = [None, fake_soapy.SOAPY_SDR_OVERFLOW]

            iq = ctrl.acquire()

            assert len(device.calls_named('activateStream')) == 2
            assert len(device.calls_named('deactivateStream')) >= 1
            assert iq.x.shape[0] == 2
            assert iq.x.all()
            # the retried acquisition restarted the holdoff from its own activation
            assert iq.info.start_time.value > device.streams[0].activate_time_ns

        starts = [r for r in caplog.records if hasattr(r, 'strobe_residual_ns')]
        assert [r.retry_count for r in starts] == [0, 1]

    def test_overflow_before_the_first_samples_is_stamped_by_the_samples(
        self, fake_soapy, fake_controller
    ):
        """an overflow on the first read is tolerated, and the record start follows
        the time of the samples that arrive after it, not the time of the overflow"""
        lost = 4096
        with fake_controller(receive_retries=0, time_sync_at='open') as ctrl:
            device = fake_soapy.devices[0]
            device.overflow_loss = lost
            ctrl._arm_spec(soapy_capture(port=(0, 1), gain=(0, 0)))
            device.fault_queue = [fake_soapy.SOAPY_SDR_OVERFLOW]

            iq = ctrl.acquire()
            fs = ctrl.get_resampler()['fs_sdr']
            activated = device.streams[0].activate_time_ns

        assert len(device.calls_named('activateStream')) == 1
        assert iq.info.start_time.value >= activated + round(lost * 1e9 / fs)

    @pytest.mark.parametrize('receive_retries', [0, 1])
    def test_unstamped_samples_after_an_overflow_retrigger(
        self, receive_retries, fake_soapy, fake_controller
    ):
        """a device that stamps the first read alone gives no time for the samples
        after a tolerated overflow, so the read is an error that a retry recovers
        from a new activation"""
        with fake_controller(
            receive_retries=receive_retries, time_sync_at='open'
        ) as ctrl:
            device = fake_soapy.devices[0]
            device.stamp_first_read_only = True
            ctrl._arm_spec(soapy_capture(port=(0, 1), gain=(0, 0)))
            device.fault_queue = [fake_soapy.SOAPY_SDR_OVERFLOW]

            if receive_retries == 0:
                with pytest.raises(
                    ss.lib.sources.base.ReceiveStreamError, match='no time stamp'
                ):
                    ctrl.acquire()
                return

            iq = ctrl.acquire()
            activated = device.streams[0].activate_time_ns

        assert len(device.calls_named('activateStream')) == 2
        assert iq.info.start_time.value >= activated

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
def test_acquire_layout_is_lead_data_tail_from_the_generator(
    preset, array_backend, isolated_lookup
):
    """a layout off by any number of samples changes every sample of a sawtooth"""
    source = SOURCE.replace(array_backend=array_backend)
    capture = preset_capture('sawtooth', **LAYOUT_PRESETS[preset])
    iq = open_and_acquire('sawtooth', capture, source)
    expected = expected_raw('sawtooth', capture, source)
    assert iq.x.shape == expected.shape
    assert_close(iq.x, expected)


def test_acquire_metadata_defaults(isolated_lookup, subtests):
    capture = preset_capture('single_tone', **SCALE_ONLY)
    with TONE.from_source_spec(SOURCE) as ctrl:
        ctrl._arm_spec(capture)
        iq = ctrl.acquire()
        with subtests.test('complex64 transport is already full scale'):
            assert iq.voltage_scale == 1
        with subtests.test('info'):
            assert iq.info.source_id == ctrl.source_id
            assert iq.info.capture_index == 0
            assert iq.info.signal_trigger is None
        with subtests.test('no conditioning is attached'):
            assert iq.signal_trigger is None and not iq.overwrite_x
        with subtests.test('capture and source specs are attached'):
            assert iq.capture == capture and iq.source_spec == SOURCE


# 6250 source samples of the resample_filter preset, of which the lead overlap of
# 3125 is half, so the lead is not a whole number of strobe periods
STROBE = 1e-3


@pytest.mark.parametrize('trigger_strobe', [None, STROBE], ids=['no_strobe', 'strobe'])
@pytest.mark.parametrize('time', [0.0, STROBE], ids=['t0', 't_strobe'])
def test_corrected_output_starts_on_the_strobe_grid(
    trigger_strobe, time, isolated_lookup
):
    """an impulse at source time 0 or one strobe period later is on the same
    corrected sample with and without a strobe: the strobe grid of a virtual source
    is referenced to corrected sample 0"""
    source = SOURCE.replace(trigger_strobe=trigger_strobe)
    capture = preset_capture('dirac_delta', **RESAMPLE_FILTER, time=time)

    stages = acquire_corrected('dirac_delta', capture, source=source)

    assert stages.overlaps[0] % round(STROBE * stages.resampler['fs_sdr']) != 0
    peaks = np.argmax(np.abs(to_numpy(filtered(stages.corrected))), axis=1)
    assert list(peaks) == [round(time * capture.sample_rate)] * 2


@pytest.mark.namespaces('cupy')
def test_acquire_on_cupy_returns_device_arrays(xp, isolated_lookup):
    source = SOURCE.replace(array_backend='cupy')
    iq = open_and_acquire(
        'single_tone', preset_capture('single_tone', **SCALE_ONLY), source
    )
    assert sw.is_cupy_array(iq.x)


# %% acquire: reuse_iq

REUSABLE = preset_capture('single_tone', **SCALE_ONLY)
REUSABLE_VARIANT = REUSABLE.replace(
    adjust_analysis={'power_spectral_density': {'window': 'hann'}}
)
FILTERED_VARIANT = REUSABLE.replace(analysis_bandwidth=5e6)
NOT_REUSABLE = REUSABLE.replace(frequency_offset=1e5)


@pytest.mark.parametrize(
    'reuse_iq, second_capture, shares_buffer',
    [
        (True, REUSABLE_VARIANT, True),
        (True, FILTERED_VARIANT, False),
        (True, NOT_REUSABLE, False),
        (False, REUSABLE_VARIANT, False),
    ],
    ids=['reused', 'new_filter', 'incompatible', 'disabled'],
)
def test_reuse_iq(reuse_iq, second_capture, shares_buffer, isolated_lookup):
    """a reused acquisition is the buffer of the first capture. Otherwise the second
    capture is acquired afresh from the generator of that capture."""
    with TONE.from_source_spec(SOURCE, reuse_iq=reuse_iq) as ctrl:
        ctrl._arm_spec(REUSABLE)
        first = ctrl.acquire()
        ctrl._arm_spec(second_capture)
        second = ctrl.acquire()

    assert (first.capture, second.capture) == (REUSABLE, second_capture)
    if shares_buffer:
        assert second.x is first.x
        assert second.preprocess is not first.preprocess
    else:
        assert not np.shares_memory(first.x, second.x)
        expected = expected_raw('single_tone', second_capture, SOURCE)
        assert_close(second.x, expected)


def test_discard_iq_forces_a_fresh_acquisition(isolated_lookup):
    """`discard_iq` makes the next acquisition of a reusable capture read the source
    again. A signal change that the capture spec does not describe, such as a
    switched noise diode, would otherwise reach the analysis as the cached IQ."""
    with TONE.from_source_spec(SOURCE, reuse_iq=True) as ctrl:
        ctrl._arm_spec(REUSABLE)
        first = ctrl.acquire()
        ctrl.discard_iq()
        ctrl._arm_spec(REUSABLE_VARIANT)
        second = ctrl.acquire()
        ctrl._arm_spec(REUSABLE)
        third = ctrl.acquire()

    assert not np.shares_memory(first.x, second.x)
    assert third.x is second.x, 'reuse resumes after the discard'


def test_reuse_iq_clears_the_soapy_start_time(fake_controller):
    """a reused acquisition has no acquisition time of its own"""
    capture = soapy_capture(port=(0, 1), gain=(0, 0))
    with fake_controller(reuse_iq=True) as ctrl:
        ctrl._arm_spec(capture)
        first = ctrl.acquire()
        ctrl._arm_spec(
            capture.replace(
                adjust_analysis={'power_spectral_density': {'window': 'hann'}}
            )
        )
        second = ctrl.acquire()

    assert first.info.start_time is not None
    assert second.info.start_time is None
    assert second.x is first.x


# %% bursts (fake SoapySDR device)

BURST_CAPTURE = soapy_capture(port=(0, 1), gain=(0, 0), sample_rate=MCR / 2, burst=3)
AcquisitionQueue = ss.lib.controller.AcquisitionQueue
AcquisitionOverrunError = ss.lib.sources.base.AcquisitionOverrunError


def record_ns(ctrl, capture) -> int:
    """return the duration (ns) of one source record of `capture` on `ctrl`"""
    design = ctrl.get_resampler()
    record = ss.lib.compute.source_record_size(capture, design)
    return round(1e9 * record / design['fs_sdr'])


def start_deltas_ns(segments) -> list[int]:
    starts = [m.info.start_time.value for m in segments]
    return [b - a for a, b in zip(starts, starts[1:])]


def burst_indexes(segments) -> list[int]:
    return [m.info.burst_index for m in segments]


class TestBurst:
    @pytest.mark.parametrize('via', ['acquire', 'acquire_burst'])
    def test_members_are_stamped_one_record_apart(
        self, via, fake_soapy, fake_controller
    ):
        """the start time of each segment follows the previous start by one record,
        although the carried-over samples begin one lead overlap before that. The
        stream is enabled once, and the later segments read one record each with no
        transient holdoff."""
        with fake_controller(time_sync_at='open', receive_retries=0) as ctrl:
            ctrl._arm_spec(BURST_CAPTURE)
            if via == 'acquire':
                segments = [ctrl.acquire(burst_index=k) for k in range(3)]
            else:
                segments = ctrl.acquire_burst()
            lead, _ = ss.lib.compute.get_correction_overlaps(
                BURST_CAPTURE, ctrl.source_spec
            )
            record = ss.lib.compute.source_record_size(
                BURST_CAPTURE, ctrl.get_resampler()
            )
            period = record_ns(ctrl, BURST_CAPTURE)
            stashed = ctrl._buffers.carryover_samples
            device = fake_soapy.devices[0]

        assert lead > 0, 'the capture must have a lead overlap to be discriminating'
        assert burst_indexes(segments) == [0, 1, 2]
        assert start_deltas_ns(segments) == [period, period]
        assert len(device.calls_named('activateStream')) == 1
        sizes = [c[1] for c in device.calls_named('readStream')]
        assert sizes[0] == segments[0].x.shape[1]
        assert sizes[-2:] == [record, record], 'segments 1 and 2 read one record each'
        assert stashed is None, 'the carryover is cleared after the last segment'

    def test_members_under_a_dividing_strobe_stay_contiguous(
        self, fake_soapy, fake_controller, caplog
    ):
        """with a strobe that divides the duration, the next edge falls at the end of
        the previous record, so the segments are one record apart with one enable.
        Each segment logs its strobe residual at DEBUG and each stream read its
        return, with no WARNING."""
        strobe = BURST_CAPTURE.duration / 2
        with (
            fake_controller(
                time_sync_at='open', receive_retries=0, trigger_strobe=strobe
            ) as ctrl,
            caplog.at_level(logging.DEBUG, logger='striqt.source'),
        ):
            ctrl._arm_spec(BURST_CAPTURE)
            segments = ctrl.acquire_burst()
            period = record_ns(ctrl, BURST_CAPTURE)
            device = fake_soapy.devices[0]
        assert start_deltas_ns(segments) == [period, period]
        assert len(device.calls_named('activateStream')) == 1
        # the holdoff is rounded to whole source samples, so the record start is on
        # the strobe grid to within one sample period
        sample_ns = 1e9 / ctrl.get_resampler()['fs_sdr']
        residual = segments[0].info.start_time.value % round(strobe * 1e9)
        assert min(residual, round(strobe * 1e9) - residual) < sample_ns

        starts = [r for r in caplog.records if hasattr(r, 'strobe_residual_ns')]
        assert [r.burst_index for r in starts] == [0, 1, 2]
        assert all(r.strobe_residual_ns <= sample_ns for r in starts)
        assert [r.resumed for r in starts] == [False] * 3
        reads = [r for r in caplog.records if hasattr(r, 'stream_ret')]
        assert len(reads) == len(device.calls_named('readStream'))
        assert all(r.stream_has_time for r in reads)
        assert all(r.levelno == logging.DEBUG for r in starts + reads)

    def test_a_strobe_with_a_remainder_is_rejected(self, fake_controller):
        strobe = BURST_CAPTURE.duration / 3.5
        with fake_controller(time_sync_at='open', trigger_strobe=strobe) as ctrl:
            ctrl._arm_spec(BURST_CAPTURE)
            with pytest.raises(ValueError, match='whole number of trigger_strobe'):
                ctrl.acquire_burst()

    def test_no_member_keeps_the_holdoff_pad(self, fake_controller):
        """segment 0 reads the transient and strobe holdoff ahead of the record and
        the later segments read none. Every segment keeps a copy of the record and
        overlaps alone."""
        strobe = BURST_CAPTURE.duration / 2
        with fake_controller(
            time_sync_at='open', receive_retries=0, trigger_strobe=strobe
        ) as ctrl:
            ctrl._arm_spec(BURST_CAPTURE)
            overlaps = ss.lib.compute.get_correction_overlaps(
                BURST_CAPTURE, ctrl.source_spec
            )
            segments = ctrl.acquire_burst()
            padded, bare = (
                ss.lib.sources.buffers.get_read_count(
                    BURST_CAPTURE,
                    ctrl.source_spec,
                    include_holdoff=flag,
                    overlap=sum(overlaps),
                )
                for flag in (True, False)
            )
        assert padded > bare
        assert [m.x.shape[1] for m in segments] == [bare] * 3
        assert all(m.x.flags.c_contiguous for m in segments)

    @pytest.mark.parametrize('on_gap', ['skip', 'except'])
    def test_full_queue_delivers_member_0_only(self, on_gap, fake_controller, caplog):
        """with one buffer and no consumer, segment 0 fills the queue. Under ``'skip'``
        the other 2 segments are dropped with one warning each. Under ``'except'``
        the second segment raises."""
        bursts = ss.specs.BurstOptions(on_gap=on_gap, buffers=1)
        queue = AcquisitionQueue(bursts.buffers)
        with (
            fake_controller(time_sync_at='open', bursts=bursts) as ctrl,
            caplog.at_level(logging.WARNING, logger='striqt.source'),
        ):
            ctrl._arm_spec(BURST_CAPTURE)
            if on_gap == 'skip':
                assert ctrl.acquire_burst(queue=queue) == []
            else:
                with pytest.raises(AcquisitionOverrunError):
                    ctrl.acquire_burst(queue=queue)

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == (2 if on_gap == 'skip' else 0)
        assert queue.qsize() == 1
        assert queue.get_nowait().info.burst_index == 0

    @pytest.mark.parametrize('on_gap', ['skip', 'except'])
    def test_overflow_at_member_1_is_retried(self, on_gap, fake_soapy, fake_controller):
        """an overflow during segment 1 re-triggers the stream. Under ``'skip'``
        segment 1 is read again and segments 1 and 2 are contiguous. Under
        ``'except'`` the burst restarts and the queued segment 0 is replaced."""
        bursts = ss.specs.BurstOptions(on_gap=on_gap, buffers=3)
        queue = AcquisitionQueue(bursts.buffers)
        finalized = []

        def finalize(iq):
            finalized.append(iq.info.burst_index)
            if not finalized[:-1]:
                device.fault_queue = [fake_soapy.SOAPY_SDR_OVERFLOW]
            return iq

        with fake_controller(
            time_sync_at='open', receive_retries=1, bursts=bursts
        ) as ctrl:
            device = fake_soapy.devices[0]
            ctrl._arm_spec(BURST_CAPTURE)
            ctrl.acquire_burst(queue=queue, finalize=finalize)
            period = record_ns(ctrl, BURST_CAPTURE)
            retrigger_ns = device.streams[0].activate_time_ns

        segments = [queue.get_nowait() for _ in range(3)]
        assert queue.qsize() == 0
        assert burst_indexes(segments) == [0, 1, 2]
        assert len(device.calls_named('activateStream')) == 2
        deltas = start_deltas_ns(segments)
        if on_gap == 'skip':
            assert finalized == [0, 1, 2]
            assert segments[1].info.start_time.value >= retrigger_ns
            assert deltas[1] == period
        else:
            assert finalized == [0, 0, 1, 2]
            assert segments[0].info.start_time.value >= retrigger_ns
            assert deltas == [period, period]

    def test_overflow_at_member_1_without_retries_raises(
        self, fake_soapy, fake_controller
    ):
        def fault_after_member_0(iq):
            device.fault_queue = [fake_soapy.SOAPY_SDR_OVERFLOW]
            return iq

        with fake_controller(time_sync_at='open', receive_retries=0) as ctrl:
            device = fake_soapy.devices[0]
            ctrl._arm_spec(BURST_CAPTURE)
            with pytest.raises(OverflowError):
                ctrl.acquire_burst(finalize=fault_after_member_0)


def test_acquisition_queue_tracks_nbytes_and_purges_one_capture():
    def segment(capture_index, count):
        info = ss.specs.AcquisitionInfo(capture_index=capture_index)
        x = np.zeros((1, count), dtype='complex64')
        return build_acquired_iq(x, REUSABLE, info=info)

    queue = AcquisitionQueue(3)
    for capture_index, count in [(0, 10), (1, 20), (0, 30)]:
        queue.put_nowait(segment(capture_index, count))
    assert queue.nbytes == 60 * 8
    with pytest.raises(Full):
        queue.put_nowait(segment(1, 1))

    assert queue.purge(0) == 2
    assert queue.qsize() == 1
    assert queue.nbytes == 20 * 8
    queue.put_nowait(segment(2, 5))
    assert queue.get_nowait().info.capture_index == 1
    assert queue.nbytes == 5 * 8
    queue.put_nowait(object())
    assert queue.nbytes == 5 * 8


# %% read_iq: buffer reuse across captures, chunked reads and validation


def test_one_port_capture_after_a_two_port_capture_has_one_row(isolated_lookup):
    """the receive buffers alternate between acquisitions, and the third acquisition
    is offered the two-port buffer of the first"""
    two_port = preset_capture('sawtooth', **SCALE_ONLY)
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
    assert third.x.shape == expected.shape
    assert_close(third.x, expected)


class HalfReads(BINDINGS['sawtooth'].sensor.source_cls):
    """a source that returns at most half of each read request"""

    read_calls = 0

    def read(self, buffers, offset, count, timeout_sec=None, *, on_overflow='except'):
        self.read_calls += 1
        count = max(1, count // 2)
        return super().read(
            buffers, offset, count, timeout_sec, on_overflow=on_overflow
        )


class OverReports(BINDINGS['sawtooth'].sensor.source_cls):
    """a source that reports 2 more samples than were requested"""

    def read(self, buffers, offset, count, timeout_sec=None, *, on_overflow='except'):
        received, time_ns = super().read(
            buffers, offset, count, timeout_sec, on_overflow=on_overflow
        )
        return received + 2, time_ns


def test_partial_reads_are_completed_contiguously(monkeypatch, isolated_lookup):
    monkeypatch.setattr(BINDINGS['sawtooth'].sensor, 'source_cls', HalfReads)
    capture = preset_capture('sawtooth', **SCALE_ONLY)
    with BINDINGS['sawtooth'].from_source_spec(SOURCE) as ctrl:
        ctrl._arm_spec(capture)
        iq = ctrl.acquire()
        read_calls = ctrl.backend.read_calls

    assert read_calls > 1
    assert_close(iq.x, expected_raw('sawtooth', capture, SOURCE))


def test_overfilled_buffer_is_a_memory_error(monkeypatch, isolated_lookup):
    monkeypatch.setattr(BINDINGS['sawtooth'].sensor, 'source_cls', OverReports)
    with BINDINGS['sawtooth'].from_source_spec(SOURCE) as ctrl:
        ctrl._arm_spec(preset_capture('sawtooth', **SCALE_ONLY))
        with pytest.raises(MemoryError):
            ctrl.acquire()


@pytest.mark.parametrize(
    'overlaps',
    [(2.5, 0), (-2, 0), (0, 0, 0), 4],
    ids=['fractional', 'negative', 'three_values', 'scalar'],
)
def test_read_iq_rejects_invalid_overlaps(overlaps, armed_tone_controller):
    with pytest.raises(ValueError, match='overlaps'):
        armed_tone_controller.read_iq(overlaps)


def test_acquire_rejects_a_list_of_overlaps(armed_tone_controller):
    with pytest.raises(ValueError, match='tuple or None'):
        armed_tone_controller.acquire(overlaps=[0, 0])


def test_acquire_before_arming_is_an_error(isolated_lookup):
    raises = pytest.raises(AttributeError, match='armed')
    with TONE.from_source_spec(SOURCE) as ctrl, raises:
        ctrl.acquire()


def test_acquire_returns_a_contiguous_copy_without_the_holdoff(isolated_lookup):
    """the receive buffer of a first segment is padded for the transient and strobe
    holdoff. The acquisition keeps a copy of the record and overlaps alone."""
    source = SOURCE.replace(trigger_strobe=0.25e-3)
    capture = preset_capture('sawtooth', **SCALE_ONLY)
    overlaps = ss.lib.compute.get_correction_overlaps(capture, source, None)
    count = buffers.get_read_count(
        capture, source, include_holdoff=False, overlap=sum(overlaps)
    )
    ports = len(capture.port)

    iq = open_and_acquire('sawtooth', capture, source)

    assert iq.x.flags.c_contiguous
    assert iq.x.nbytes == ports * count * 8


def test_read_iq_after_a_rearm_drops_the_strobe_holdoff(isolated_lookup):
    """the tail overlap of the first capture leaves the stream off the strobe grid,
    so the re-armed read waits out a nonzero holdoff ahead of the record"""
    source = SOURCE.replace(trigger_strobe=0.25e-3)
    capture = preset_capture('sawtooth', **SCALE_ONLY)
    overlaps = ss.lib.compute.get_correction_overlaps(capture, source, None)
    fs = fs_sdr(capture, source)
    count = buffers.get_read_count(
        capture, source, include_holdoff=False, overlap=sum(overlaps)
    )
    with BINDINGS['sawtooth'].from_source_spec(source) as ctrl:
        ctrl._arm_spec(capture)
        ctrl.read_iq(overlaps)
        ctrl._arm_spec(capture)
        x, start_ns = ctrl.read_iq(overlaps)

    strobe_ns = round(0.25e-3 * 1e9)
    assert start_ns % strobe_ns == 0
    assert start_ns > round(capture.duration * 1e9)
    start_index = round(start_ns * fs / 1e9) - overlaps[0]
    expected = generator(
        'sawtooth', capture, sample_rate=fs, start_index=start_index, count=count
    )
    assert x.flags.c_contiguous
    assert x.shape == (len(capture.port), count)
    assert_close(x, expected)


# %% acquire through the fake_soapy binding


def dBfs(x):
    """return the rms level (dBfs) of each port of `x`"""
    return 20 * np.log10(rms(to_numpy(x), axis=-1))


class TestControllerAcquire:
    def test_start_time_and_sample_count(self, fake_soapy, fake_controller):
        capture = soapy_capture(port=(0, 1), gain=(0, 0), duration=2e-3)
        with fake_controller() as ctrl:
            ctrl._arm_spec(capture)
            iq = ctrl.acquire()

            device = fake_soapy.devices[0]
            stream = device.streams[0]
            holdoff = round(2e-3 * MCR)
            read_sizes = [c[1] for c in device.calls_named('readStream')]
            assert len(read_sizes) == 2
            assert read_sizes[1] == holdoff

            # the timestamp is inside the acquisition window
            window_ns = round(sum(read_sizes) * 1e9 / MCR)
            assert stream.activate_time_ns <= iq.info.start_time.value
            assert iq.info.start_time.value <= stream.activate_time_ns + window_ns

    def test_single_port_capture_streams_both_and_keeps_its_own(
        self, fake_soapy, fake_controller
    ):
        fake_soapy.model.tones[1] = (1.505e9, -60.0)
        with fake_controller() as ctrl:
            ctrl._arm_spec(soapy_capture(port=1, duration=1e-3))
            iq = ctrl.acquire()
            assert fake_soapy.devices[0].streams[0].channels == (0, 1)

        # -60 dBm through 50 dB of gain is -10 dBfs. Port 0 would be noise only.
        # The noise floor of the model biases the level 0.014 dB above the tone, so
        # the bound is that mismatch and not the sample fluctuation
        assert iq.x.shape[0] == 1
        assert dBfs(iq.x)[0] == pytest.approx(-10.0, abs=0.05)

    def test_noise_level_matches_the_model(self, fake_soapy, fake_controller):
        with fake_controller() as ctrl:
            ctrl._arm_spec(soapy_capture(port=(0, 1), gain=(0, -10), duration=1e-3))
            iq = ctrl.acquire()

        expected = [
            10 * np.log10(fake_soapy.model.noise_variance(0, 0, MCR, 1e9)),
            10 * np.log10(fake_soapy.model.noise_variance(1, -10, MCR, 1e9)),
        ]
        # the mean of N exponentially distributed sample powers, at 3 sigma
        sigma = 3 / np.sqrt(iq.x.shape[1])
        assert dBfs(iq.x) == pytest.approx(
            expected, abs=sw.power_analysis.level_tolerance_dB(sigma, power=True)
        )

    def test_stale_first_timestamp_is_a_stream_error(self, fake_soapy, fake_controller):
        with fake_controller(receive_retries=0) as ctrl:
            ctrl._arm_spec(soapy_capture(duration=1e-3))
            device = fake_soapy.devices[0]
            activate = device.activateStream

            def activate_in_the_past(stream, **kws):
                activate(stream, **kws)
                stream.activate_time_ns = 1

            device.activateStream = activate_in_the_past
            raises = pytest.raises(
                ss.lib.sources.base.ReceiveStreamError, match='before last sync'
            )
            with raises:
                ctrl.acquire()
