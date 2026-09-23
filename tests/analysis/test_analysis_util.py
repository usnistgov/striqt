"""striqt.analysis.lib.util: the logging adapters, stopwatch, compute lock, blocking
input and small helpers that the analysis, sensor, cli and figures packages call"""

from __future__ import annotations

import contextlib
import io
import logging
import sys
import threading
import time

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

import striqt.analysis as sa
import striqt.waveform as sw
from striqt.analysis.lib import util


def screen_handlers(name):
    adapter = util.get_logger(name)
    return [
        h
        for h in adapter.logger.handlers
        if h is getattr(adapter, '_screen_handler', None)
    ]


# %% StriqtLogger / get_logger
class TestStriqtLogger:
    def test_registered_at_import(self):
        logger = util.get_logger('analysis')
        assert isinstance(logger, util.StriqtLogger)
        assert logger.logger.name == 'striqt.analysis'
        assert logger.extra == util.StriqtLogger.EXTRA_DEFAULTS
        assert sa.util is util

    def test_new_adapter_merges_extra_defaults(self):
        logger = util.StriqtLogger(
            'test-util-suffix', {'capture_index': 3, 'site': 'a'}
        )
        assert util.get_logger('test-util-suffix') is logger
        assert logger.extra['capture_index'] == 3
        assert logger.extra['site'] == 'a'
        assert logger.extra['capture_progress'] == 'control'

    def test_unknown_suffix(self):
        with pytest.raises(KeyError):
            util.get_logger('no-such-logger')


# %% isroundmod
class TestIsroundmod:
    @given(k=st.integers(-1000, 1000), div=st.sampled_from([1, 0.5, 15e3, 1 / 3]))
    def test_integer_multiples(self, k, div):
        assert util.isroundmod(k * div, div)
        assert not util.isroundmod((k + 0.5) * div, div)

    def test_tolerance(self):
        assert util.isroundmod(1 + 1e-7, 1)
        assert not util.isroundmod(1 + 1e-5, 1)
        assert util.isroundmod(1 + 1e-5, 1, atol=1e-4)

    def test_array_input_takes_numpy_path(self):
        values = np.array([2.0, 2.5, 3.0])
        result = util.isroundmod(values, 1)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [True, False, True])


# %% show_messages
class TestShowMessages:
    def test_sets_levels_and_one_handler(self):
        util.show_messages(logging.DEBUG, colors=False, logger_names=('analysis',))
        adapter = util.get_logger('analysis')
        assert adapter.logger.level == logging.DEBUG
        assert len(screen_handlers('analysis')) == 1
        assert adapter._screen_handler.level == logging.DEBUG

        # a second call replaces the handler rather than stacking another
        util.show_messages(logging.INFO, colors=False, logger_names=('analysis',))
        assert len(screen_handlers('analysis')) == 1
        assert adapter.logger.handlers.count(adapter._screen_handler) == 1

    def test_format_shows_capture_progress(self):
        util.show_messages(logging.INFO, colors=False, logger_names=('analysis',))
        fmt = util.get_logger('analysis')._screen_handler.formatter._fmt
        assert '{capture_progress}' in fmt

    def test_none_silences_every_named_logger(self, subtests):
        util.StriqtLogger('test-silenced-logger')
        names = ('analysis', 'test-silenced-logger')
        util.show_messages(None, logger_names=names)
        for name in names:
            with subtests.test(msg=name):
                assert util.get_logger(name).logger.level == logging.CRITICAL

    def test_all_applies_to_every_adapter(self, subtests):
        util.show_messages(logging.WARNING, colors=False)
        for name, adapter in util._logger_adapters.items():
            with subtests.test(msg=name):
                assert adapter.logger.level == logging.WARNING


# %% stopwatch
class TestStopwatch:
    @pytest.fixture
    def records(self, caplog):
        caplog.set_level(util.PERFORMANCE_DETAIL, logger='striqt.analysis')
        # caplog.records is re-created for the call phase, so resolve it lazily
        return lambda: caplog.records

    def test_logs_elapsed_time(self, records):
        with util.stopwatch('napping'):
            time.sleep(0.01)

        (record,) = records()
        assert record.levelno == util.PERFORMANCE_INFO
        assert record.capture_progress == 'control'

    def test_below_threshold_is_demoted(self, records):
        with util.stopwatch('quick', threshold=10):
            pass
        assert records()[-1].levelno == util.PERFORMANCE_DETAIL

    def test_empty_description(self, records):
        with util.stopwatch():
            pass
        assert records()[-1].getMessage().startswith('⏱')

    def test_exception_is_logged_and_propagates(self, records):
        with pytest.raises(ValueError, match='boom'), util.stopwatch('failing'):
            raise ValueError('boom')
        record = records()[-1]
        assert record.levelno == logging.ERROR
        assert 'before exception boom' in record.getMessage()

    def test_other_logger_suffix(self, caplog):
        suffix = 'test-stopwatch-logger'
        util.StriqtLogger(suffix)
        caplog.set_level(util.PERFORMANCE_DETAIL, logger=f'striqt.{suffix}')
        with util.stopwatch('x', logger_suffix=suffix, logger_level=logging.INFO):
            pass
        assert caplog.records[-1].name == f'striqt.{suffix}'
        assert caplog.records[-1].levelno == logging.INFO


# %% compute_lock
def lock_is_free(lock):
    """whether another thread could take `lock` right now"""
    result = []

    def probe():
        acquired = lock.acquire(timeout=0)
        result.append(acquired)
        if acquired:
            lock.release()

    thread = threading.Thread(target=probe)
    thread.start()
    thread.join()
    return result[0]


class TestComputeLock:
    def test_locks_without_array(self):
        assert lock_is_free(util._compute_lock)
        with util.compute_lock():
            assert not lock_is_free(util._compute_lock)
        assert lock_is_free(util._compute_lock)

    def test_numpy_arrays_do_not_lock(self):
        with util.compute_lock(np.zeros(3)):
            assert lock_is_free(util._compute_lock)

    def test_cupy_arrays_lock(self, cupy_available):
        with util.compute_lock(cupy_available.zeros(3)):
            assert not lock_is_free(util._compute_lock)
        assert lock_is_free(util._compute_lock)


# %% hold_logger_outputs / blocking_input
class TestHoldLoggerOutputs:
    def test_follows_redirected_stderr(self):
        util.show_messages(logging.INFO, colors=False, logger_names=('analysis',))
        handler = util.get_logger('analysis')._screen_handler
        original = handler.stream

        buffer = io.StringIO()
        with contextlib.redirect_stderr(buffer), util.hold_logger_outputs():
            assert handler.stream is buffer
            util.get_logger('analysis').info('inside')
        assert handler.stream is original
        assert 'inside' in buffer.getvalue()

    def test_adapters_without_screen_handler_are_skipped(self):
        adapter = util.StriqtLogger('test-no-handler')
        with util.hold_logger_outputs():
            assert not hasattr(adapter, '_screen_handler')


class TestBlockingInput:
    def test_prompt_and_deferred_output(self, monkeypatch, capfd):
        def fake_input():
            print('stdout during input')
            print('stderr during input', file=sys.stderr)
            return 'yes'

        monkeypatch.setattr('builtins.input', fake_input)
        assert util.blocking_input('confirm? ') == 'yes'

        out, err = capfd.readouterr()
        assert out.startswith('confirm? ')
        assert 'stdout during input' in out
        assert 'stderr during input' in err

    def test_without_prompt(self, monkeypatch, capfd):
        monkeypatch.setattr('builtins.input', lambda: 'n')
        assert util.blocking_input() == 'n'
        assert capfd.readouterr() == ('', '')

    def test_output_restored_when_input_fails(self, monkeypatch, capfd):
        def fake_input():
            print('partial')
            raise EOFError

        monkeypatch.setattr('builtins.input', fake_input)
        with pytest.raises(EOFError):
            util.blocking_input()
        assert 'partial' in capfd.readouterr().out
        assert lock_is_free(util._input_lock)


# %% ordered_set_union
class TestOrderedSetUnion:
    def test_duplicates_across_arguments(self):
        union = util.ordered_set_union(['b', 'a'], ('a', 'c'), ['c', 'd'])
        assert union == ['b', 'a', 'c', 'd']
        assert util.ordered_set_union() == []


# %% elementwise_atol


def test_elementwise_atol_without_an_off_peak_depth_is_flat():
    tol = sa.specs.Tolerance(
        units='dB', rtol=0.0, on_peak=sa.specs.ErrorBound(rms=1e-3, peak=2e-3)
    )
    atol = util.elementwise_atol(tol, [-10.0, -60.0, -120.0])
    assert atol.tolist() == [2e-3, 2e-3, 2e-3]


def test_elementwise_atol_grows_with_depth_and_diverges_at_off_peak_dBc():
    """the bound equals `peak` at the peak element, widens below it, and is infinite
    from `off_peak_dBc.peak` down"""
    err = 1e-4
    additive = 5e-4
    tol = sa.specs.Tolerance(
        units='dB',
        rtol=0.0,
        on_peak=sa.specs.ErrorBound(
            rms=1e-3, peak=additive + sw.power_analysis.level_tolerance_dB(err)
        ),
        off_peak_dBc=sa.specs.ErrorBound(
            rms=sw.fourier.rms_tolerance_dBc(err) - 10,
            peak=sw.fourier.rms_tolerance_dBc(err),
        ),
    )
    expected = np.array([-3.0, -43.0, -73.0, -3.0 + tol.off_peak_dBc.peak, -150.0])
    atol = util.elementwise_atol(tol, expected)
    assert atol[0] >= tol.on_peak.peak
    assert atol[0] == pytest.approx(
        additive + sw.power_analysis.off_peak_dB_tolerance(0, err)
    )
    assert np.all(np.diff(atol[:3]) > 0)
    assert np.isinf(atol[3]) and np.isinf(atol[4])
