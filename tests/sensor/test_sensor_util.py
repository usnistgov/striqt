"""striqt.sensor.lib.util: the pipeline offset iterator, thread interrupt sharing,
deferred exception handling, retry, and the logging helpers used by the sweep
execution, calibration, sinks, resources and cli modules"""

from __future__ import annotations

import concurrent.futures
import logging
import threading
import time

import exceptiongroup
import pytest
import yaml
from hypothesis import given
from hypothesis import strategies as st

import striqt.analysis as sa
import striqt.sensor as ss
from striqt.sensor.lib import util

ExceptionGroup = exceptiongroup.ExceptionGroup


@pytest.fixture(autouse=True)
def clear_thread_interrupts():
    util._cancel_threads.clear()
    yield
    util._cancel_threads.clear()


def raised(exc):
    """`exc` with a traceback attached, as a caught exception has"""
    try:
        raise exc
    except BaseException as ex:  # noqa: BLE001
        return ex


def raise_inside(context, exc):
    with context:
        raise exc


def run_in_thread(func):
    """call `func` in a worker thread and return its result or raised exception"""
    outcome = []

    def target():
        try:
            outcome.append(('ok', func()))
        except BaseException as ex:  # noqa: BLE001
            outcome.append(('raised', ex))

    thread = threading.Thread(target=target)
    thread.start()
    thread.join()
    return outcome[0]


# %% zip_offsets
def zip_offsets_reference(seq, shifts, fill):
    n = len(seq) - min(shifts)
    return [
        tuple(seq[k + s] if 0 <= k + s < len(seq) else fill for s in shifts)
        for k in range(n)
    ]


class TestZipOffsets:
    @given(
        seq=st.lists(st.integers(), max_size=8),
        shifts=st.lists(st.integers(-3, 3), min_size=2, max_size=4),
    )
    def test_matches_reference(self, seq, shifts):
        result = list(util.zip_offsets(iter(seq), tuple(shifts), fill=None))
        assert result == zip_offsets_reference(seq, shifts, None)

    def test_pipeline_offsets(self):
        """the sweep pipeline iterates (previous-1, previous, current, next)"""
        result = list(util.zip_offsets(['a', 'b', 'c'], (-2, -1, 0, 1), fill=None))
        assert result == [
            (None, None, 'a', 'b'),
            (None, 'a', 'b', 'c'),
            ('a', 'b', 'c', None),
            ('b', 'c', None, None),
            ('c', None, None, None),
        ]

    def test_single_shift_squeeze(self):
        assert list(util.zip_offsets([1, 2, 3], (1,), fill=0)) == [2, 3]
        assert list(util.zip_offsets([1, 2, 3], (1,), fill=0, squeeze=False)) == [
            (2,),
            (3,),
        ]
        assert list(util.zip_offsets([1, 2], (-1,), fill=0)) == [0, 1, 2]


# %% thread interrupts and the thread pool
class TestThreadInterrupts:
    def test_main_thread_never_raises(self):
        util.cancel_threads()
        assert util.propagate_thread_interrupts() is None

    def test_worker_thread_raises_when_cancelled(self):
        assert run_in_thread(util.propagate_thread_interrupts) == ('ok', None)
        util.cancel_threads()
        status, exc = run_in_thread(util.propagate_thread_interrupts)
        assert status == 'raised'
        assert isinstance(exc, util.ThreadInterruptRequest)

    def test_share_context_clears_on_exit(self):
        with util.share_thread_interrupts():
            util.cancel_threads()
            assert util._cancel_threads.is_set()
        assert not util._cancel_threads.is_set()

        def cancel_then_fail():
            with util.share_thread_interrupts():
                util.cancel_threads()
                raise RuntimeError

        with pytest.raises(RuntimeError):
            cancel_then_fail()
        assert not util._cancel_threads.is_set()

    def test_threadpool_is_a_module_singleton(self):
        assert util.threadpool is util.threadpool

    def test_unknown_attribute(self):
        with pytest.raises(AttributeError, match='no attribute'):
            assert util.no_such_attribute


# %% ExceptionStack / await_and_ignore
class TestExceptionStack:
    def test_single_exception_is_reraised_at_exit(self):
        stack = util.ExceptionStack()

        def defer_one_failure():
            with stack:
                with stack.defer():
                    raise ValueError('one')
                with stack.defer():
                    pass

        with pytest.raises(ValueError, match='one'):
            defer_one_failure()
        assert stack.exceptions == []

        with stack, stack.defer():
            pass

    def test_multiple_exceptions_form_a_group(self):
        stack = util.ExceptionStack('label')
        with stack.defer():
            raise ValueError('one')
        with stack.defer():
            raise TypeError('two')

        with pytest.raises(ExceptionGroup) as info:
            stack.handle()
        assert info.value.message == 'label'
        assert [type(e) for e in info.value.exceptions] == [ValueError, TypeError]

    def test_interrupts_yield_to_other_exceptions(self):
        stack = util.ExceptionStack()
        with stack.defer():
            raise util.ThreadInterruptRequest
        with stack.defer():
            raise ValueError('real')
        with pytest.raises(ValueError, match='real'):
            stack.handle()

    def test_keyboard_interrupt_preferred(self):
        stack = util.ExceptionStack()
        with stack.defer():
            raise util.ThreadInterruptRequest
        with stack.defer():
            raise KeyboardInterrupt
        with pytest.raises(KeyboardInterrupt):
            stack.handle()

    def test_lone_thread_interrupt(self):
        stack = util.ExceptionStack()
        with stack.defer():
            raise util.ThreadInterruptRequest
        with pytest.raises(util.ThreadInterruptRequest):
            stack.handle()

    def test_cancel_on_except(self):
        # the deferred exceptions are cleared because ExceptionStack.__del__ calls
        # handle(), which would re-raise them when the stack is garbage collected
        stack = util.ExceptionStack(cancel_on_except=True)
        with stack.defer():
            raise ValueError
        assert util._cancel_threads.is_set()
        stack.exceptions.clear()

        stack = util.ExceptionStack()
        util._cancel_threads.clear()
        with stack.defer():
            raise ValueError
        assert not util._cancel_threads.is_set()
        stack.exceptions.clear()


class TestAwaitAndIgnore:
    @pytest.fixture
    def pool(self):
        with concurrent.futures.ThreadPoolExecutor(4) as pool:
            yield pool

    @staticmethod
    def fail(exc, delay=0):
        time.sleep(delay)
        raise exc

    def test_all_succeed(self, pool):
        futures = [pool.submit(lambda i=i: i) for i in range(3)]
        assert util.await_and_ignore(futures) is None

    def test_single_failure(self, pool):
        futures = [pool.submit(lambda: 1), pool.submit(self.fail, ValueError('x'))]
        with pytest.raises(ValueError, match='x'):
            util.await_and_ignore(futures)

    def test_awaits_every_future_and_groups_failures(self, pool):
        started = threading.Event()

        def slow_success():
            started.wait()
            time.sleep(0.05)
            return 'late'

        first_failure = pool.submit(self.fail, ValueError('first'))
        late_success = pool.submit(slow_success)
        second_failure = pool.submit(self.fail, TypeError('second'), 0.02)
        futures = [first_failure, late_success, second_failure]
        started.set()
        with pytest.raises(ExceptionGroup) as info:
            util.await_and_ignore(futures, 'arm sensor')

        assert info.value.message == 'arm sensor'
        assert {type(e) for e in info.value.exceptions} == {ValueError, TypeError}
        assert all(fut.done() for fut in futures)
        assert late_success.result() == 'late'


# %% DebugOnException
class TestDebugOnException:
    def test_prints_traceback_and_propagates(self, capsys):
        with pytest.raises(ValueError, match='boom'):
            raise_inside(util.DebugOnException(), ValueError('boom'))
        out = capsys.readouterr().out
        assert 'ValueError' in out and 'boom' in out
        assert not util._handling_tracebacks

    def test_verbose(self, capsys):
        def fail_with_local():
            local_detail = 'unique-local-value'  # noqa: F841
            raise ValueError('boom')

        with pytest.raises(ValueError), util.DebugOnException(verbose=True):
            fail_with_local()
        out = capsys.readouterr().out
        assert 'boom' in out
        assert 'unique-local-value' in out

    def test_exception_group_lists_members(self, capsys):
        group = ExceptionGroup(
            'threads', [raised(ValueError('one')), raised(TypeError('two'))]
        )
        with pytest.raises(ExceptionGroup):
            raise_inside(util.DebugOnException(), group)
        out = capsys.readouterr().out
        assert 'one' in out and 'two' in out

    def test_silent_cases(self, capsys):
        handler = util.DebugOnException()
        handler.run(None, None, None)
        with pytest.raises(KeyboardInterrupt):
            raise_inside(handler, KeyboardInterrupt())
        assert capsys.readouterr().out == ''

    def test_repeated_triplet_printed_once(self, capsys):
        handler = util.DebugOnException()
        try:
            raise ValueError('once')
        except ValueError as ex:
            triplet = (type(ex), ex, ex.__traceback__)
        handler.run(*triplet)
        first = capsys.readouterr().out
        handler.run(*triplet)
        assert 'once' in first
        assert capsys.readouterr().out == ''


# %% retry
class Flaky:
    __name__ = 'flaky'

    def __init__(self, failures, exc=ConnectionError):
        self.failures = failures
        self.exc = exc
        self.calls = []

    def __call__(self, *args, **kws):
        self.calls.append((args, kws))
        if len(self.calls) <= self.failures:
            raise self.exc(f'failure {len(self.calls)}')
        return 'ok'


class TestRetry:
    def test_succeeds_after_failures(self):
        flaky = Flaky(2)
        wrapped = util.retry(ConnectionError, tries=3)(flaky)
        assert wrapped(1, k=2) == 'ok'
        assert flaky.calls == [((1,), {'k': 2})] * 3
        assert wrapped.__name__ == 'flaky'
        assert wrapped.__wrapped__ is flaky

    def test_raises_after_last_try(self):
        flaky = Flaky(5)
        wrapped = util.retry((ConnectionError, TimeoutError), tries=3)(flaky)
        with pytest.raises(ConnectionError, match='failure 3'):
            wrapped()
        assert len(flaky.calls) == 3

    def test_other_exceptions_propagate_immediately(self):
        flaky = Flaky(5, exc=KeyError)
        wrapped = util.retry(ConnectionError, tries=3)(flaky)
        with pytest.raises(KeyError):
            wrapped()
        assert len(flaky.calls) == 1

    def test_exception_func_and_logger(self, caplog):
        flaky = Flaky(2)
        hook_calls = []
        records_at_hook = []
        logger = logging.getLogger('test-retry')

        def retry_records():
            return [r for r in caplog.records if r.name == 'test-retry']

        def hook(*a, **k):
            hook_calls.append((a, k))
            records_at_hook.append(len(retry_records()))

        wrapped = util.retry(
            ConnectionError, tries=4, exception_func=hook, logger=logger
        )(flaky)

        with caplog.at_level(logging.INFO, logger='test-retry'):
            assert wrapped('a', b=1) == 'ok'

        assert hook_calls == [(('a',), {'b': 1})] * 2
        assert [r.levelno for r in retry_records()] == [logging.INFO]
        assert records_at_hook == [1, 1]

    def test_delay_and_backoff(self, monkeypatch):
        sleeps = []
        monkeypatch.setattr(util.time, 'sleep', sleeps.append)
        flaky = Flaky(3)
        util.retry(ConnectionError, tries=4, delay=0.1, backoff=2)(flaky)()
        assert sleeps == pytest.approx([0.1, 0.2, 0.4])

    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason='util.py:323-325 run exception_func and time.sleep after the final '
        'failed try as well, before the for-else re-raises',
    )
    def test_exhausted_tries_sleep_only_between_tries(self, monkeypatch):
        sleeps = []
        monkeypatch.setattr(util.time, 'sleep', sleeps.append)
        tries = 4
        wrapped = util.retry(ConnectionError, tries=tries, delay=0.1)(Flaky(tries))
        with pytest.raises(ConnectionError):
            wrapped()
        assert len(sleeps) == tries - 1


# %% logging helpers
LOGGER_NAMES = ('sweep', 'source', 'analysis', 'sink', 'periph')


class TestLogVerbosity:
    @pytest.mark.parametrize(
        'verbose, level',
        [
            (-1, logging.WARNING),
            (0, logging.INFO),
            (1, sa.util.PERFORMANCE_INFO),
            (2, sa.util.PERFORMANCE_DETAIL),
            (3, logging.DEBUG),
        ],
    )
    def test_levels(self, verbose, level, subtests):
        util.log_verbosity(verbose)
        for name in LOGGER_NAMES:
            with subtests.test(msg=name):
                assert sa.util.get_logger(name).logger.level == level


class TestLogCaptureContext:
    def test_sets_progress_and_restores(self):
        logger = sa.util.get_logger('sink')
        before = logger.extra
        with util.log_capture_context('sink', capture_index=2, capture_count=5):
            assert logger.extra['capture_index'] == 2
            assert logger.extra['capture_count'] == 5
            assert logger.extra['capture_progress'] == '3/5'
        assert logger.extra is before

    def test_restores_after_exception(self):
        logger = sa.util.get_logger('sink')
        before = logger.extra
        context = util.log_capture_context('sink', capture_index=1, capture_count=2)
        with pytest.raises(RuntimeError):
            raise_inside(context, RuntimeError())
        assert logger.extra is before

    def test_count_defaults_to_the_current_extra(self):
        logger = sa.util.get_logger('sink')
        count = logger.extra['capture_count']
        with util.log_capture_context('sink', capture_index=0):
            assert logger.extra['capture_progress'] == f'1/{count}'

        fresh = sa.util.StriqtLogger('test-fresh-suffix')
        try:
            with util.log_capture_context('test-fresh-suffix', capture_index=0):
                assert fresh.extra['capture_progress'] == '1/unknown'
        finally:
            del sa.util._logger_adapters['test-fresh-suffix']

    def test_count_carries_from_outer_context(self):
        logger = sa.util.get_logger('sweep')
        with util.log_capture_context('sweep', capture_index=0, capture_count=9):
            with util.log_capture_context('sweep', capture_index=4):
                assert logger.extra['capture_progress'] == '5/9'
            assert logger.extra['capture_progress'] == '1/9'


class TestLogToFile:
    @pytest.fixture
    def striqt_logger(self):
        return logging.getLogger('striqt')

    def test_writes_yaml_records(self, tmp_path, striqt_logger):
        path = tmp_path / 'logs' / 'sweep.json'
        util.log_to_file(path, 'info')
        handler = striqt_logger._striqt_handler
        assert handler in striqt_logger.handlers
        assert handler.level == logging.INFO

        # records from the striqt.* adapters propagate up to the file handler;
        # the JSON formatter merges a dict argument into the record
        sink = sa.util.get_logger('sink')
        sink.info('started', {'radio': 'ab12'})
        sink.debug('filtered out')
        try:
            raise RuntimeError('acquisition failed')
        except RuntimeError:
            sink.error('stream error')
        striqt_logger.removeHandler(handler)
        handler.close()

        records = yaml.safe_load(path.read_text())
        assert [r['message'] for r in records] == ['started', 'stream error']
        assert records[0]['level'] == 'INFO'
        assert records[0]['radio'] == 'ab12'
        assert records[0]['source_file'].endswith('test_sensor_util.py')
        assert records[0]['elapsed_seconds'] >= 0
        assert 'RuntimeError: acquisition failed' == records[1]['exception']
        assert any('raise RuntimeError' in line for line in records[1]['traceback'])

    def test_repeat_call_replaces_the_handler(self, tmp_path, striqt_logger):
        util.log_to_file(tmp_path / 'a.json', 'debug')
        first = striqt_logger._striqt_handler
        util.log_to_file(tmp_path / 'b.json', 'warning')
        assert first not in striqt_logger.handlers
        assert striqt_logger._striqt_handler in striqt_logger.handlers
        assert striqt_logger._striqt_handler.level == logging.WARNING
        first.close()


def test_public_reexports(subtests):
    for name in ('zip_offsets', 'retry', 'ExceptionStack', 'log_verbosity'):
        with subtests.test(msg=name):
            assert getattr(ss.util, name) is getattr(util, name)
