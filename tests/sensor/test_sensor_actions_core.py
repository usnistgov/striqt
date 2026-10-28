"""striqt.sensor.lib.actions.core: the acquire/analyze/sink pipeline, iterate_captures.

The sequencing and yield-mode cells run through `ss.iterate_sweep`, which dispatches
a bound `Sweep` to `SweepRunner` and the pipeline (`test_sensor_actions_sweep.py` pins
that dispatch). The other cells call `iterate_captures` directly.

The captures are 1-port scale-only tones at distinct offsets. A result attributed to
the wrong capture then disagrees with its generator oracle exactly.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import assert_source_released
from numeric_checks import assert_close, cross_backend
from sweep_strategies import SOURCE
from synthetic_sources import (
    IQ_ONLY,
    SPECTROGRAM,
    expected_corrected,
    preset_sweep,
    run_in_memory,
    tone_captures,
    tone_sweep,
)

import striqt.sensor as ss
from striqt.sensor.lib import util
from striqt.sensor.lib.actions import core
from striqt.waveform import fourier

OFFSETS = (1e6, 2e6, 3e6)


def index_values(datasets, name) -> list[int]:
    return [int(ds[name].values[0]) for ds in datasets]


# %% iterate_captures: sequencing


def test_results_follow_the_capture_order(subtests):
    captures = tone_captures(OFFSETS)
    datasets = run_in_memory(tone_sweep(OFFSETS))

    assert index_values(datasets, 'capture_index') == list(range(len(captures)))
    assert index_values(datasets, 'sweep_index') == [0] * len(captures)
    for i, (capture, ds) in enumerate(zip(captures, datasets)):
        with subtests.test(capture=i):
            expected = expected_corrected('single_tone', capture)
            assert np.array_equal(ds.iq_waveform.values, expected)


@pytest.mark.namespaces('cupy')
def test_results_follow_the_capture_order_cupy(array_backend, subtests):
    """the cupy source draws the same generator as numpy. Scale-only correction
    leaves the tone samples as the generator's complex exponential, which rounds
    once each at the phase product, the cosine, the sine, and the float32 cast on
    each backend."""
    captures = tone_captures(OFFSETS)
    source = SOURCE.replace(array_backend=array_backend)
    datasets = run_in_memory(tone_sweep(OFFSETS, source=source))
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
    n = len(OFFSETS)
    sweep = tone_sweep(OFFSETS, loops=(ss.specs.Repeat(count=2),))
    datasets = run_in_memory(sweep)

    assert index_values(datasets, 'capture_index') == list(range(n)) * 2
    assert index_values(datasets, 'sweep_index') == [0] * n + [1] * n


def test_loop_cycles_with_an_incrementing_sweep_index():
    n = len(OFFSETS)
    datasets = run_in_memory(tone_sweep(OFFSETS), loop=True, take=2 * n + 1)

    assert index_values(datasets, 'capture_index') == (list(range(n)) * 3)[: 2 * n + 1]
    assert index_values(datasets, 'sweep_index') == [0] * n + [1] * n + [2]


def test_empty_captures_yield_nothing():
    assert run_in_memory(preset_sweep('single_tone', (), analysis=IQ_ONLY)) == []


# %% iterate_captures: yield modes

YIELD_MODES = {
    # (yield_values, always_yield): item types, with N = 3 captures
    'values': ((True, False), [None, None, 'result', 'result', 'result']),
    'values_always': ((True, True), [None, None, 'result', 'result', 'result']),
    'placeholders': ((False, True), [None] * 5),
    'silent': ((False, False), []),
}


@pytest.mark.parametrize('mode', list(YIELD_MODES), ids=list(YIELD_MODES))
def test_yield_modes(mode):
    """the pipeline runs N + 2 stages. With `yield_values` set it yields the sink
    result of every stage, which is None for the 2 stages before the first capture
    reaches the sink. With only `always_yield` set it yields one None per stage.
    With neither it yields nothing."""
    (yield_values, always_yield), expected = YIELD_MODES[mode]
    sweep = tone_sweep(OFFSETS)
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


def test_extra_data_reaches_every_capture_and_wins_over_the_peripherals():
    sweep = tone_sweep(OFFSETS)
    with ss.open_resources(sweep, None) as res:
        items = core.iterate_captures(
            res,
            sink=ss.sinks.NoSink(sweep),
            peripherals=TaggingPeripherals(sweep),
            extra_data={'shared': 'runner', 'state': True},
        )
        results = [item for item in items if item is not None]

    assert len(results) == len(OFFSETS)
    for result in results:
        assert result.extra_data['shared'] == 'runner'
        assert result.extra_data['from_peripheral'] == 1
        assert result.extra_data['state'] is True


# %% iterate_captures: on_analyzed callbacks


class RecordingSink(ss.sinks.NoSink):
    def __init__(self, sweep, events):
        super().__init__(sweep)
        self.events = events

    def append(self, capture_result):
        self.events.append(('sink', capture_result.capture))
        return super().append(capture_result)


def test_on_analyzed_sees_each_capture_in_order_before_the_sink():
    """the callback for capture i runs in the analysis step of i; the sink append of
    i is submitted one step later, concurrently with the callback of i + 1"""
    sweep = tone_sweep(OFFSETS)
    events = []
    on_analyzed = lambda dd: events.append(('callback', dd.capture))
    with ss.open_resources(sweep, None, on_analyzed=on_analyzed) as res:
        sink = RecordingSink(sweep, events)
        results = [r for r in core.iterate_captures(res, sink=sink) if r is not None]

    captures = [r.capture for r in results]
    assert len(captures) == len(OFFSETS)
    assert [c for kind, c in events if kind == 'callback'] == captures
    for c in captures:
        assert events.index(('callback', c)) < events.index(('sink', c))


def test_iterate_sweep_replaces_the_callbacks_for_one_run():
    sweep = tone_sweep(OFFSETS)
    seen = []
    callbacks = ss.lib.resources.SweepCallbacks(
        on_analyzed=lambda dd: seen.append(dd.capture)
    )
    with ss.open_resources(sweep, None) as res:
        sink = ss.sinks.NoSink(sweep)
        list(ss.iterate_sweep(res, sink=sink, callbacks=callbacks))
        assert len(seen) == len(OFFSETS)
        list(ss.iterate_sweep(res, sink=sink))
    assert len(seen) == len(OFFSETS)


def test_an_on_analyzed_exception_ends_the_pipeline():
    sweep = tone_sweep(OFFSETS)

    def fail(dd):
        raise RuntimeError('callback failed')

    with (
        ss.open_resources(sweep, None, on_analyzed=fail) as res,
        pytest.raises(RuntimeError, match='callback failed'),
    ):
        list(core.iterate_captures(res, sink=ss.sinks.NoSink(sweep)))


def test_a_shared_indexer_numbers_two_runs_consecutively():
    """one indexer passed to 2 pipeline runs over the same open resources numbers
    the second run's captures after the first run's. The second run also re-arms
    its first capture, which differs from the capture the first run left armed."""
    n = len(OFFSETS)
    sweep = tone_sweep(OFFSETS)
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
    for capture, ds in zip(tone_captures(OFFSETS) * 2, datasets):
        assert np.array_equal(
            ds.iq_waveform.values, expected_corrected('single_tone', capture)
        )


# %% iterate_captures: failure propagation


class FailingSink(ss.sinks.NoSink):
    def append(self, capture_result):
        raise RuntimeError('simulated sink failure')


def test_sink_exception_propagates_and_closes_the_source(isolated_lookup):
    sweep = tone_sweep(OFFSETS)
    with (
        pytest.raises(RuntimeError, match='simulated sink failure'),
        ss.open_resources(sweep, None) as res,
    ):
        list(core.iterate_captures(res, sink=FailingSink(sweep)))

    assert not res['source'].is_open(wait=False)
    assert_source_released(isolated_lookup, sweep)


class CancellingPeripherals(ss.peripherals.NoPeripherals):
    def acquire(self, capture):
        util.cancel_threads()
        return {}


def test_peripheral_cancel_interrupts_a_worker_thread_sweep():
    """a peripheral that requests cancellation during the first acquisition stops
    the pipeline at the interrupt check that follows the first stage's yield. The
    only item yielded is the leading None. Leaving `share_thread_interrupts` clears
    the flag. `propagate_thread_interrupts` is a no-op in the main thread, and the
    sweep is therefore consumed in a worker."""
    sweep = tone_sweep(OFFSETS)
    yielded = []

    def consume(res):
        for item in core.iterate_captures(
            res, sink=ss.sinks.NoSink(sweep), peripherals=CancellingPeripherals(sweep)
        ):
            assert item is None, 'no capture reaches the sink before the interrupt'
            yielded.append(item)

    with util.share_thread_interrupts():
        with ss.open_resources(sweep, None) as res:
            future = util.threadpool.submit(consume, res)
            with pytest.raises(util.ThreadInterruptRequest):
                future.result(timeout=30)
        assert util._cancel_threads.is_set()
    assert not util._cancel_threads.is_set()
    assert yielded == [None]


# %% iterate_captures: on_cached callback


def test_on_cached_fires_once_per_capture_when_the_spectrogram_cache_fills():
    sweep = tone_sweep(OFFSETS, analysis=SPECTROGRAM)
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


def test_no_cache_callback_is_installed_without_on_cached(monkeypatch):
    from striqt.analysis.lib.register import KwArgCache

    installed = []
    original = KwArgCache.set_callback

    def set_callback(self, func, capture=None):
        installed.append(func)
        return original(self, func, capture)

    monkeypatch.setattr(KwArgCache, 'set_callback', set_callback)
    sweep = tone_sweep(OFFSETS, analysis=SPECTROGRAM)
    with ss.open_resources(sweep, None) as res:
        list(core.iterate_captures(res, sink=ss.sinks.NoSink(sweep)))
    assert installed == []
