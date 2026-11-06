"""striqt.sensor.lib.actions.core: the acquire/analyze/sink pipeline, iterate_captures

The sequencing and yield-mode cells run through `ss.iterate_sweep`, which dispatches
a bound `Sweep` to `SweepRunner` and the pipeline. The other cells call
`iterate_captures` directly. The captures are 1-port scale-only tones at distinct
offsets, and a result attributed to the wrong capture disagrees exactly with the
generator oracle of that capture.
"""

from __future__ import annotations

import logging
import sys
import threading
from typing import ClassVar

import numpy as np
import pytest
from conftest import assert_source_released
from numeric_checks import assert_close, cross_backend
from sweep_strategies import SOURCE
from synthetic_sources import (
    BINDINGS,
    IQ_ONLY,
    SPECTROGRAM,
    TONE_OFFSETS,
    FakeTrigger,
    expected_corrected,
    preset_sweep,
    run_in_memory,
    tone_captures,
    tone_sweep,
)

import striqt.sensor as ss
from striqt.sensor.lib import compute, util
from striqt.sensor.lib.actions import core
from striqt.waveform import fourier


def index_values(datasets, name) -> list[int]:
    return [int(ds[name].values[0]) for ds in datasets]


def burst_sweep(offsets, *, burst, analysis=IQ_ONLY, source=SOURCE, **bursts):
    """return a `single_tone` sweep over `tone_captures(offsets)` with `burst`
    segments each and the `bursts` policy of `source` replaced by `bursts`"""
    if bursts:
        source = source.replace(bursts=ss.specs.BurstOptions(**bursts))
    return preset_sweep(
        'single_tone',
        tone_captures(offsets, burst=burst),
        analysis=analysis,
        source=source,
    )


# %% iterate_captures: sequencing


def test_results_follow_the_capture_order(subtests):
    captures = tone_captures(TONE_OFFSETS)
    datasets = run_in_memory(tone_sweep(TONE_OFFSETS))

    assert index_values(datasets, 'capture_index') == list(range(len(captures)))
    assert index_values(datasets, 'sweep_index') == [0] * len(captures)
    for i, (capture, ds) in enumerate(zip(captures, datasets)):
        with subtests.test(capture=i):
            expected = expected_corrected('single_tone', capture)
            assert np.array_equal(ds.iq_waveform.values, expected)


@pytest.mark.namespaces('cupy')
def test_results_follow_the_capture_order_cupy(array_backend, subtests):
    """the cupy and numpy sources evaluate the same generator. Scale-only correction
    leaves the tone samples as the complex exponential of the generator. The
    tolerance is the roundoff of 4 roundings per backend: the phase product, the
    cosine, the sine, and the float32 cast."""
    captures = tone_captures(TONE_OFFSETS)
    source = SOURCE.replace(array_backend=array_backend)
    datasets = run_in_memory(tone_sweep(TONE_OFFSETS, source=source))
    sigma = cross_backend(
        fourier.fft_tolerance_rms(np.complex64, [], n_elementwise=4),
        fourier.fft_tolerance_rms(
            np.complex64, [], n_elementwise=4, array_backend=array_backend
        ),
    )

    assert index_values(datasets, 'capture_index') == list(range(len(captures)))
    for i, (capture, ds) in enumerate(zip(captures, datasets)):
        with subtests.test(capture=i):
            expected = expected_corrected('single_tone', capture)
            assert_close(ds.iq_waveform.values, expected, sigma=sigma)


def test_leading_repeat_multiplies_the_captures():
    n = len(TONE_OFFSETS)
    sweep = tone_sweep(TONE_OFFSETS, loops=(ss.specs.Repeat(count=2),))
    datasets = run_in_memory(sweep)

    assert index_values(datasets, 'capture_index') == list(range(n)) * 2
    assert index_values(datasets, 'sweep_index') == [0] * n + [1] * n


def test_loop_cycles_with_an_incrementing_sweep_index():
    n = len(TONE_OFFSETS)
    datasets = run_in_memory(tone_sweep(TONE_OFFSETS), loop=True, take=2 * n + 1)

    assert index_values(datasets, 'capture_index') == (list(range(n)) * 3)[: 2 * n + 1]
    assert index_values(datasets, 'sweep_index') == [0] * n + [1] * n + [2]


def test_empty_captures_yield_nothing():
    assert run_in_memory(preset_sweep('single_tone', (), analysis=IQ_ONLY)) == []


# %% iterate_captures: yield modes

YIELD_MODES = {
    # (yield_values, always_yield): item types for N delivered acquisitions
    'values': ((True, False), lambda n: [None, None] + ['result'] * n),
    'values_always': ((True, True), lambda n: [None, None] + ['result'] * n),
    'placeholders': ((False, True), lambda n: [None] * (n + 2)),
    'silent': ((False, False), lambda n: []),
}


@pytest.mark.parametrize('burst', [1, 2], ids=['single', 'burst'])
@pytest.mark.parametrize('mode', list(YIELD_MODES), ids=list(YIELD_MODES))
def test_yield_modes(mode, burst):
    """the pipeline runs N + 2 stages for N delivered acquisitions. With
    `yield_values` set, every stage yields the sink result, which is None for the 2
    stages before the first acquisition reaches the sink. With only `always_yield`
    set, every stage yields None. With neither, nothing is yielded."""
    (yield_values, always_yield), expect = YIELD_MODES[mode]
    expected = expect(burst * len(TONE_OFFSETS))
    sweep = burst_sweep(TONE_OFFSETS, burst=burst)
    modes = {'yield_values': yield_values, 'always_yield': always_yield}
    with ss.open_resources(sweep, None) as res:
        items = list(ss.iterate_sweep(res, sink=ss.sinks.NoSink(sweep), **modes))
    kinds = [None if item is None else 'result' for item in items]
    assert kinds == expected
    for item in items:
        if item is not None:
            assert isinstance(item, ss.lib.compute.DelayedDataset)


# %% iterate_captures: extra_data and a shared indexer


class TaggingPeripherals(ss.peripherals.NoPeripherals):
    def acquire(self, capture):
        return {'shared': 'peripheral', 'from_peripheral': 1}


@pytest.mark.parametrize('burst', [1, 3], ids=['single', 'burst'])
def test_extra_data_reaches_every_acquisition_and_wins_over_the_peripherals(burst):
    sweep = burst_sweep(TONE_OFFSETS, burst=burst)
    with ss.open_resources(sweep, None) as res:
        items = core.iterate_captures(
            res,
            sink=ss.sinks.NoSink(sweep),
            peripherals=TaggingPeripherals(sweep),
            extra_data={'shared': 'runner', 'state': True},
        )
        results = [item for item in items if item is not None]

    assert len(results) == burst * len(TONE_OFFSETS)
    for result in results:
        # the peripheral entries join when a sink resolves the pending read
        assert 'from_peripheral' not in result.extra_data
        extra_data = compute.datasets.resolve_extra_data(result)
        assert extra_data['shared'] == 'runner'
        assert extra_data['from_peripheral'] == 1
        assert extra_data['state'] is True


class BlockedPeripherals(TaggingPeripherals):
    """a peripherals double with an `acquire` that returns only after `release` is set"""

    release: ClassVar[threading.Event] = threading.Event()

    def acquire(self, capture):
        if not self.release.wait(timeout=30):
            raise TimeoutError('the peripheral read was never released')
        return super().acquire(capture)


class ReleasingSource(BINDINGS['single_tone'].sensor.source_cls):
    """a source that sets `release` at the start of read number `release_at`"""

    release: ClassVar[threading.Event] = threading.Event()
    release_at: ClassVar[int] = 0
    reads: ClassVar[int] = 0

    def read(self, buffers, offset, count, timeout_sec=None, *, on_overflow='except'):
        type(self).reads += 1
        if self.reads == self.release_at:
            self.release.set()
        return super().read(
            buffers, offset, count, timeout_sec, on_overflow=on_overflow
        )


def test_a_peripheral_read_may_last_the_whole_burst(
    caplog, monkeypatch, isolated_lookup
):
    """the acquisition thread never waits for the peripheral inside a burst. Under a
    1-slot queue, no segment is dropped while a peripheral read blocks until the last
    segment is requested, and the peripheral data joins every segment once the read
    resolves."""
    burst = 4
    release = threading.Event()
    monkeypatch.setattr(BINDINGS['single_tone'].sensor, 'source_cls', ReleasingSource)
    for cls in (ReleasingSource, BlockedPeripherals):
        monkeypatch.setattr(cls, 'release', release)
    monkeypatch.setattr(ReleasingSource, 'release_at', burst)
    monkeypatch.setattr(ReleasingSource, 'reads', 0)
    sweep = burst_sweep(TONE_OFFSETS[:1], burst=burst, buffers=1)
    with (
        caplog.at_level(logging.WARNING, logger='striqt.source'),
        ss.open_resources(sweep, None) as res,
    ):
        items = core.iterate_captures(
            res, sink=ss.sinks.NoSink(sweep), peripherals=BlockedPeripherals(sweep)
        )
        results = [item for item in items if item is not None]

    assert ReleasingSource.reads == burst

    assert [r.extra_coords.burst_index for r in results] == list(range(burst))
    assert [rec for rec in caplog.records if rec.levelno == logging.WARNING] == []
    for result in results:
        assert result.pending_extra_data.peripherals.done()
        assert compute.datasets.resolve_extra_data(result)['from_peripheral'] == 1


# %% iterate_captures: on_analyzed callbacks


class RecordingSink(ss.sinks.NoSink):
    def __init__(self, sweep, events):
        super().__init__(sweep)
        self.events = events

    def append(self, capture_result):
        self.events.append(('sink', capture_result.capture))
        return super().append(capture_result)


def test_on_analyzed_sees_each_capture_in_order_before_the_sink():
    """the callback for capture i runs in the analysis step of i. The sink append of
    i is submitted one step later, concurrently with the callback of i + 1."""
    sweep = tone_sweep(TONE_OFFSETS)
    events = []
    on_analyzed = lambda dd: events.append(('callback', dd.capture))
    with ss.open_resources(sweep, None, on_analyzed=on_analyzed) as res:
        sink = RecordingSink(sweep, events)
        results = [r for r in core.iterate_captures(res, sink=sink) if r is not None]

    captures = [r.capture for r in results]
    assert len(captures) == len(TONE_OFFSETS)
    assert [c for kind, c in events if kind == 'callback'] == captures
    for c in captures:
        assert events.index(('callback', c)) < events.index(('sink', c))


def test_iterate_sweep_replaces_the_callbacks_for_one_run():
    sweep = tone_sweep(TONE_OFFSETS)
    seen = []
    callbacks = ss.lib.resources.SweepCallbacks(
        on_analyzed=lambda dd: seen.append(dd.capture)
    )
    with ss.open_resources(sweep, None) as res:
        sink = ss.sinks.NoSink(sweep)
        list(ss.iterate_sweep(res, sink=sink, callbacks=callbacks))
        assert len(seen) == len(TONE_OFFSETS)
        list(ss.iterate_sweep(res, sink=sink))
    assert len(seen) == len(TONE_OFFSETS)


def fail_hook(*args):
    raise RuntimeError('hook failed')


@pytest.mark.parametrize('hook', ['on_analyzed', 'select_analysis'])
def test_a_hook_exception_ends_the_pipeline(hook):
    sweep = tone_sweep(TONE_OFFSETS)
    with (
        ss.open_resources(sweep, None, **{hook: fail_hook}) as res,
        pytest.raises(RuntimeError, match='hook failed'),
    ):
        list(core.iterate_captures(res, sink=ss.sinks.NoSink(sweep)))


def test_a_shared_indexer_numbers_two_runs_consecutively():
    """one indexer numbers the captures of 2 pipeline runs over the same open
    resources consecutively, the second run after the first. The first run leaves
    the last capture armed, and the second run re-arms the first capture."""
    n = len(TONE_OFFSETS)
    sweep = tone_sweep(TONE_OFFSETS)
    indexer = core.AcquisitionIndexer(2 * n)

    with ss.open_resources(sweep, None) as res:
        sink = ss.sinks.NoSink(sweep)
        first = list(core.iterate_captures(res, sink=sink, indexer=indexer))
        second = list(core.iterate_captures(res, sink=sink, indexer=indexer))

    datasets = [
        ss.lib.compute.from_delayed(item) for item in first + second if item is not None
    ]
    assert index_values(datasets, 'capture_index') == list(range(2 * n))
    assert index_values(datasets, 'sweep_index') == [0] * (2 * n)
    for capture, ds in zip(tone_captures(TONE_OFFSETS) * 2, datasets):
        assert np.array_equal(
            ds.iq_waveform.values, expected_corrected('single_tone', capture)
        )


# %% iterate_captures: captures


@pytest.mark.parametrize(
    'sweep_offsets, given_offsets, as_generator',
    [
        (TONE_OFFSETS, TONE_OFFSETS[::-1], False),
        (TONE_OFFSETS[:1], TONE_OFFSETS[1:], True),
    ],
    ids=['reversed_tuple', 'finite_generator'],
)
def test_captures_runs_the_given_captures_in_the_given_order(
    sweep_offsets, given_offsets, as_generator
):
    """the results follow the given captures, not the sweep, and the run ends after
    the last given capture. The indexer counts over the plan of the sweep: with a
    plan of one capture, every result has `capture_index` 0."""
    captures = tone_captures(given_offsets)
    sweep = tone_sweep(sweep_offsets)
    given = (c for c in captures) if as_generator else captures
    datasets = run_in_memory(sweep, captures=given)

    assert len(datasets) == len(captures)
    expected_indexes = [i % len(sweep_offsets) for i in range(len(captures))]
    assert index_values(datasets, 'capture_index') == expected_indexes
    assert [ds.frequency_offset.item() for ds in datasets] == list(given_offsets)
    for capture, ds in zip(captures, datasets):
        assert np.array_equal(
            ds.iq_waveform.values, expected_corrected('single_tone', capture)
        )


def test_an_endless_capture_generator_ends_when_the_consumer_closes_the_run():
    """the generator repeats the one capture of the sweep until `done` is set. With
    a plan of one capture, `capture_index` stays 0 and `sweep_index` counts the
    repeats. Closing the pipeline generator joins the acquisition thread."""
    take = 3
    (capture,) = tone_captures(TONE_OFFSETS[:1])
    sweep = tone_sweep(TONE_OFFSETS[:1])
    done = threading.Event()

    def repeat():
        while not done.is_set():
            yield capture

    results = []
    with ss.open_resources(sweep, None) as res:
        gen = core.iterate_captures(res, sink=ss.sinks.NoSink(sweep), captures=repeat())
        for item in gen:
            if item is not None:
                results.append(ss.lib.compute.from_delayed(item))
            if len(results) == take:
                break
        done.set()
        gen.close()
        assert acquisition_threads() == []

    assert index_values(results, 'capture_index') == [0] * take
    assert index_values(results, 'sweep_index') == list(range(take))


def test_captures_with_loop_is_rejected():
    sweep = tone_sweep(TONE_OFFSETS)
    with (
        ss.open_resources(sweep, None) as res,
        pytest.raises(ValueError, match='loop'),
    ):
        next(ss.iterate_sweep(res, loop=True, captures=tone_captures(TONE_OFFSETS)))


# %% iterate_captures: failure propagation


class FailingSink(ss.sinks.NoSink):
    def append(self, capture_result):
        raise RuntimeError('simulated sink failure')


def test_sink_exception_propagates_and_closes_the_source(isolated_lookup):
    sweep = tone_sweep(TONE_OFFSETS)
    with (
        pytest.raises(RuntimeError, match='simulated sink failure'),
        ss.open_resources(sweep, None) as res,
    ):
        list(core.iterate_captures(res, sink=FailingSink(sweep)))

    assert not res['source'].is_open(wait=False)
    assert_source_released(isolated_lookup, sweep)


class FailingPeripherals(ss.peripherals.NoPeripherals):
    def acquire(self, capture):
        raise RuntimeError('simulated peripheral failure')


def acquisition_threads() -> list:
    return [t for t in threading.enumerate() if t.name == 'striqt-acquire']


def test_acquisition_thread_exception_reaches_the_caller(isolated_lookup):
    """an exception in the acquisition thread ends the pipeline in the calling
    thread, and the acquisition thread has exited by then"""
    sweep = tone_sweep(TONE_OFFSETS)
    with (
        pytest.raises(RuntimeError, match='simulated peripheral failure'),
        ss.open_resources(sweep, None) as res,
    ):
        list(
            core.iterate_captures(
                res, sink=ss.sinks.NoSink(sweep), peripherals=FailingPeripherals(sweep)
            )
        )
    assert acquisition_threads() == []
    assert_source_released(isolated_lookup, sweep)


class CancellingPeripherals(ss.peripherals.NoPeripherals):
    def acquire(self, capture):
        util.cancel_threads()
        return {}


def test_peripheral_cancel_interrupts_a_worker_thread_sweep():
    """a peripheral that requests cancellation during the first acquisition stops
    the pipeline at the interrupt check after the yield of the first stage. The
    only item yielded is the leading None. Leaving `share_thread_interrupts` clears
    the flag."""
    sweep = tone_sweep(TONE_OFFSETS)
    yielded = []

    def consume(res):
        for item in core.iterate_captures(
            res, sink=ss.sinks.NoSink(sweep), peripherals=CancellingPeripherals(sweep)
        ):
            assert item is None, 'no capture reaches the sink before the interrupt'
            yielded.append(item)

    with util.share_thread_interrupts():
        with ss.open_resources(sweep, None) as res:
            # propagate_thread_interrupts is a no-op in the main thread
            future = util.threadpool.submit(consume, res)
            with pytest.raises(util.ThreadInterruptRequest):
                future.result(timeout=30)
        assert util._cancel_threads.is_set()
    assert not util._cancel_threads.is_set()
    assert yielded == [None]


# %% iterate_captures: on_cached callback


def test_on_cached_fires_once_per_capture_when_the_spectrogram_cache_fills():
    sweep = tone_sweep(TONE_OFFSETS, analysis=SPECTROGRAM)
    calls = []

    def on_cached(resources, cache, capture, result, **kwargs):
        calls.append((resources, cache.name, capture, result))

    with ss.open_resources(sweep, None, on_sink_cached=on_cached) as res:
        sink = ss.sinks.NoSink(sweep)
        results = [r for r in core.iterate_captures(res, sink=sink) if r is not None]

    assert [call[2] for call in calls] == [r.capture for r in results]
    for resources, name, _, result in calls:
        assert resources['sweep_spec'] is sweep
        assert 'spectrogram' in name
        spg, attrs = result
        assert spg.ndim == 3
        assert 'noise_bandwidth' in attrs


# %% iterate_captures: bursts


class RecordingSource(BINDINGS['single_tone'].sensor.source_cls):
    """a source that records each `arm` and `read` call"""

    events: ClassVar[list] = []

    def arm(self, capture):
        self.events.append('arm')
        return super().arm(capture)

    def read(self, buffers, offset, count, timeout_sec=None, *, on_overflow='except'):
        self.events.append('read')
        return super().read(
            buffers, offset, count, timeout_sec, on_overflow=on_overflow
        )


def test_one_arm_per_capture_and_one_read_per_segment(monkeypatch, isolated_lookup):
    """the function source has no transient holdoff, so each segment is one read"""
    monkeypatch.setattr(BINDINGS['single_tone'].sensor, 'source_cls', RecordingSource)
    monkeypatch.setattr(RecordingSource, 'events', [])
    sweep = burst_sweep(TONE_OFFSETS[:2], burst=3)
    datasets = run_in_memory(sweep)

    assert RecordingSource.events == ['arm'] + ['read'] * 3 + ['arm'] + ['read'] * 3
    assert index_values(datasets, 'capture_index') == [0, 0, 0, 1, 1, 1]
    assert index_values(datasets, 'burst_index') == [0, 1, 2, 0, 1, 2]
    assert index_values(datasets, 'sweep_index') == [0] * 6


class CountingTrigger(FakeTrigger):
    calls = 0

    def __call__(self, iq, capture):
        type(self).calls += 1
        return super().__call__(iq, capture)


@pytest.mark.parametrize(
    'trigger_on, calls_per_burst', [('first', 1), ('all', 3)], ids=['first', 'all']
)
def test_trigger_on_selects_the_members_that_run_the_trigger(
    trigger_on, calls_per_burst, monkeypatch
):
    """under ``'first'`` the trigger runs on the first segment and the shifts apply to
    the others. Under ``'all'`` the trigger runs on every segment."""
    trigger = CountingTrigger(0.0)
    monkeypatch.setattr(CountingTrigger, 'calls', 0)
    # the controller reads the package attribute and `analyze` the module global
    analyze_module = sys.modules['striqt.sensor.lib.compute.analyze']
    for module in (ss.lib.compute, analyze_module):
        monkeypatch.setattr(
            module, 'get_trigger_from_spec', lambda setup, analysis=None: trigger
        )
    sweep = burst_sweep(TONE_OFFSETS[:2], burst=3, trigger_on=trigger_on)
    datasets = run_in_memory(sweep)

    assert len(datasets) == 6
    assert CountingTrigger.calls == 2 * calls_per_burst
    for capture, ds in zip(sweep.captures, datasets[::3]):
        assert np.array_equal(
            ds.iq_waveform.values, expected_corrected('single_tone', capture)
        )


# %% AcquisitionIndexer


def test_indexer_advances_once_per_capture():
    """`capture_index` advances after the last segment of a capture, and
    `sweep_start_time` is the start time of segment 0 of capture 0 of each sweep"""
    burst_capture, single = tone_captures(TONE_OFFSETS[:2])
    burst_capture = burst_capture.replace(burst=3)
    indexer = core.AcquisitionIndexer(2)

    def apply(burst_index, capture, start_ns):
        info = ss.specs.SoapyAcquisitionInfo(
            burst_index=burst_index,
            start_time=np.datetime64(start_ns, 'ns'),
            backend_sample_rate=None,
        )
        return indexer.apply(info, capture)

    results = [apply(k, burst_capture, 10 + k) for k in range(3)]
    results.extend([apply(0, single, 20), apply(0, burst_capture, 30)])

    assert [r.capture_index for r in results] == [0, 0, 0, 1, 0]
    assert [r.sweep_index for r in results] == [0, 0, 0, 0, 1]
    assert [r.burst_index for r in results] == [0, 1, 2, 0, 0]
    starts = [int(r.sweep_start_time.astype(int)) for r in results]
    assert starts == [10, 10, 10, 10, 30]
