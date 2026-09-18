"""striqt.sensor.lib.execute: the acquire/analyze/sink pipeline of iterate_sweep.

Sequencing and indexing of the results, repeat and loop expansion, the yield modes,
failure propagation from the sink and peripheral threads, and the calibration cache
log. The captures are 1-port scale-only tones at distinct offsets, so a result
attributed to the wrong capture disagrees with its own generator oracle exactly.
"""

from __future__ import annotations

import logging
import math
import types

import numpy as np
import pytest
from conftest import assert_source_released
from numeric_checks import assert_close, cross_backend_rms
from sweep_strategies import SOURCE
from synthetic_sources import (
    SCALE_ONLY,
    expected_corrected,
    make_capture,
    make_sweep,
    run_in_memory,
)

import striqt.sensor as ss
from striqt.sensor.lib import execute, util

ONE_PORT = {**SCALE_ONLY, 'port': 0}
IQ_ONLY = ss.specs.BundledAnalysis.from_dict({'iq_waveform': {}})
OFFSETS = (1e6, 2e6, 3e6)


def tone_captures(offsets=OFFSETS):
    return tuple(
        make_capture('single_tone', **ONE_PORT, frequency_offset=f, snr=None)
        for f in offsets
    )


def tone_sweep(offsets=OFFSETS, **replace):
    return make_sweep(
        'single_tone', tone_captures(offsets), analysis=IQ_ONLY, **replace
    )


def index_values(datasets, name) -> list[int]:
    return [int(ds[name].values[0]) for ds in datasets]


# %% iterate_sweep: sequencing


def test_results_follow_the_capture_order(subtests):
    captures = tone_captures()
    datasets = run_in_memory(tone_sweep())

    assert index_values(datasets, 'capture_index') == list(range(len(captures)))
    assert index_values(datasets, 'sweep_index') == [0] * len(captures)
    for i, (capture, ds) in enumerate(zip(captures, datasets)):
        with subtests.test(capture=i):
            expected = expected_corrected('single_tone', capture)
            assert np.array_equal(ds.iq_waveform.values, expected)


@pytest.mark.namespaces('cupy')
def test_results_follow_the_capture_order_cupy(array_backend, subtests):
    """the cupy source draws the same generator; scale-only correction leaves the
    tone samples as the generator's complex exponential, which rounds once per
    phase product, cosine, sine and float32 cast on each backend"""
    captures = tone_captures()
    source = SOURCE.replace(array_backend=array_backend)
    datasets = run_in_memory(tone_sweep(source=source))
    sigma = cross_backend_rms(np.complex64, [], n_elementwise=4)

    assert index_values(datasets, 'capture_index') == list(range(len(captures)))
    for i, (capture, ds) in enumerate(zip(captures, datasets)):
        with subtests.test(capture=i):
            expected = expected_corrected('single_tone', capture)
            assert_close(ds.iq_waveform.values, expected, sigma=sigma)


def test_leading_repeat_multiplies_the_captures():
    n = len(OFFSETS)
    sweep = tone_sweep(loops=(ss.specs.Repeat(count=2),))
    datasets = run_in_memory(sweep)

    assert index_values(datasets, 'capture_index') == list(range(n)) * 2
    assert index_values(datasets, 'sweep_index') == [0] * n + [1] * n


def test_loop_cycles_with_an_incrementing_sweep_index():
    n = len(OFFSETS)
    datasets = run_in_memory(tone_sweep(), loop=True, take=2 * n + 1)

    assert index_values(datasets, 'capture_index') == (list(range(n)) * 3)[: 2 * n + 1]
    assert index_values(datasets, 'sweep_index') == [0] * n + [1] * n + [2]


def test_empty_captures_yield_nothing():
    assert run_in_memory(make_sweep('single_tone', (), analysis=IQ_ONLY)) == []


# %% iterate_sweep: yield modes

YIELD_MODES = {
    # (yield_values, always_yield): item types, with N = 3 captures
    'values': ((True, False), [None, None, 'result', 'result', 'result']),
    'values_always': ((True, True), [None, None, 'result', 'result', 'result']),
    'placeholders': ((False, True), [None] * 5),
    'silent': ((False, False), []),
}


@pytest.mark.parametrize('mode', list(YIELD_MODES), ids=list(YIELD_MODES))
def test_yield_modes(mode):
    """current behaviour, not a contract: the pipeline runs N + 2 stages and yields
    the sink result of every stage when yield_values is set (None for the two
    stages before the first capture reaches the sink), one None per stage when
    only always_yield is set, and nothing otherwise. The iterate_sweep docstring
    describes the leading placeholders differently."""
    (yield_values, always_yield), expected = YIELD_MODES[mode]
    sweep = tone_sweep()
    with ss.open_resources(sweep, None) as res:
        items = list(
            ss.iterate_sweep(
                res,
                sink=ss.sinks.NoSink(sweep),
                yield_values=yield_values,
                always_yield=always_yield,
            )
        )
    kinds = [None if item is None else 'result' for item in items]
    assert kinds == expected
    for item in items:
        if item is not None:
            assert isinstance(item, ss.lib.compute.DelayedDataset)


# %% iterate_sweep: failure propagation


class FailingSink(ss.sinks.NoSink):
    def append(self, capture_result):
        raise RuntimeError('simulated sink failure')


def test_sink_exception_propagates_and_closes_the_source(isolated_lookup):
    sweep = tone_sweep()
    with (
        pytest.raises(RuntimeError, match='simulated sink failure'),
        ss.open_resources(sweep, None) as res,
    ):
        list(ss.iterate_sweep(res, sink=FailingSink(sweep)))

    assert not res['source'].is_open(wait=False)
    assert_source_released(isolated_lookup, sweep)


class CancellingPeripherals(ss.peripherals.NoPeripherals):
    def acquire(self, capture):
        util.cancel_threads()
        return {}


def test_peripheral_cancel_interrupts_a_worker_thread_sweep():
    """a peripheral that requests cancellation during the first acquisition stops
    the pipeline at the interrupt check after that stage's yield, so exactly one
    item (the leading None) comes out; the flag is cleared on leaving
    share_thread_interrupts. The check is a no-op in the main thread, so the sweep
    is consumed in a worker."""
    sweep = tone_sweep()
    yielded = []

    def consume(res):
        for item in ss.iterate_sweep(
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


# %% _log_cache_info

AIR = ss.bindings.air7101b
BOLTZMANN_MW = 1.380649e-23 * 1e3
T_REF = 290.0
NOISE_BANDWIDTH = 1e4
PEAK_DBM = -100.0


def calibrated_air_resources(calibration_nc, capture):
    source = AIR.schema.source(calibration=calibration_nc)
    sweep = AIR.sensor.sweep_spec_cls(source=source, captures=(capture,))
    return ss.lib.resources.Resources(
        sweep_spec=sweep,
        source=types.SimpleNamespace(source_id='beef'),
        format_path=None,
    )


def spectrogram_result(peak_dBm, shape=(1, 4, 8)):
    """a linear-power spectrogram whose only feature is one bin at `peak_dBm`"""
    spg = np.full(shape, 1e-30, dtype='float32')
    spg[0, shape[1] // 2, shape[2] // 2] = 10 ** (peak_dBm / 10)
    return spg, {'noise_bandwidth': NOISE_BANDWIDTH}


def test_log_cache_info_reports_the_spectrogram_snr(calibration_nc, caplog):
    """the logged SNR is the spectrogram peak above the calibrated system noise
    k*T*B raised by the noise figure the calibration file holds for the capture's
    port and center frequency (5 dB at port 0, 1 GHz)"""
    capture = ss.specs.SoapyCapture(
        port=0,
        center_frequency=1e9,
        gain=0.0,
        sample_rate=125e6,
        duration=1e-3,
        analysis_bandwidth=40e6,
        host_resample=False,
    )
    noise_dBm = 5.0 + 10 * math.log10(BOLTZMANN_MW * T_REF * NOISE_BANDWIDTH)
    expected_snr = round(PEAK_DBM - noise_dBm)

    cache = types.SimpleNamespace(name='_cached_spectrogram')
    with caplog.at_level(logging.INFO, logger='striqt.analysis'):
        execute._log_cache_info(
            calibrated_air_resources(calibration_nc, capture),
            cache,
            capture,
            spectrogram_result(PEAK_DBM),
        )

    messages = [r.getMessage() for r in caplog.records if 'SNR' in r.getMessage()]
    assert len(messages) == 1
    assert f'{expected_snr} dB max SNR' in messages[0]
    assert 'port: 0' in messages[0]


@pytest.mark.parametrize('cache_name', ['_cached_spectrogram', 'ssb_iq_cache'])
def test_log_cache_info_is_silent_without_calibration(cache_name, caplog):
    capture = ss.specs.SoapyCapture(port=0, center_frequency=1e9, gain=0.0)
    sweep = AIR.sensor.sweep_spec_cls(source=AIR.schema.source(), captures=(capture,))
    resources = ss.lib.resources.Resources(
        sweep_spec=sweep,
        source=types.SimpleNamespace(source_id='beef'),
        format_path=None,
    )
    cache = types.SimpleNamespace(name=cache_name)
    with caplog.at_level(logging.DEBUG, logger='striqt.analysis'):
        execute._log_cache_info(resources, cache, capture, spectrogram_result(PEAK_DBM))
    assert caplog.records == []
