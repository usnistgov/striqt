from __future__ import annotations as __

import collections
import contextlib
import datetime
import functools
import importlib
import itertools
import logging
import queue
import sys
import threading
import time
import typing
from pathlib import Path

from striqt.analysis.lib.util import (
    PERFORMANCE_DETAIL,
    PERFORMANCE_INFO,
    _StriqtLogger,
    blocking_input,
    get_logger,
    show_messages,
    stopwatch,
    configure_cupy,
    cp,
)
from striqt.waveform.util import lazy_import, lru_cache

_LOG_LEVEL_NAMES = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARN,
    'error': logging.ERROR,
    'critical': logging.CRITICAL,
}


class ConcurrentException(Exception):
    """Raised on concurrency errors in `labbench.concurrently`"""

    thread_exceptions: list[BaseException] = []


class ThreadEndedByMaster(Exception):
    """Raised in a thread to indicate the owning thread requested termination"""


if typing.TYPE_CHECKING:
    import typing_extensions

    _P = typing_extensions.ParamSpec('_P')
    _R = typing_extensions.TypeVar('_R', covariant=True)
    _T = typing.TypeVar('_T')
else:
    _P = typing.TypeVar('_P')
    _R = typing.TypeVar('_R')


_StriqtLogger('sweep')
_StriqtLogger('source')
_StriqtLogger('sink')
_StriqtLogger('periph')

_concurrency_count = 0
_handling_tracebacks = False

stop_request_event = threading.Event()


_Tfunc = typing.Callable[..., typing.Any]


_cancel_threads = threading.Event()
_imports_ready = collections.defaultdict(threading.Event)


@contextlib.contextmanager
def share_thread_interrupts():
    try:
        yield
    finally:
        print('reset thread cancelation', file=sys.stderr)
        _cancel_threads.clear()


def cancel_threads():
    print('request thread cancel', file=sys.stderr)
    _cancel_threads.set()


def check_thread_interrupts():
    if threading.current_thread() == threading.main_thread():
        return

    if _cancel_threads.is_set():
        print('raising on thread exception', file=sys.stderr)
        raise ThreadEndedByMaster()


def safe_import(name):
    """wait in child threads until called by the parent with the same name"""

    if threading.current_thread() == threading.main_thread():
        try:
            mod = importlib.import_module(name)
            _imports_ready[name].set()
        except BaseException:
            cancel_threads()
            raise
    else:
        check_thread_interrupts()
        while True:
            if _imports_ready[name].wait(0.5):
                break
            else:
                check_thread_interrupts()
        mod = sys.modules[name]
    return mod


def retry(
    excs: type[BaseException] | typing.Iterable[type[BaseException]],
    tries: int,
    *,
    delay: float = 0,
    backoff: float = 0,
    exception_func=lambda *args, **kws: None,
    logger: logging.Logger | logging.LoggerAdapter | None = None,
) -> typing.Callable[[_Tfunc], _Tfunc]:
    """calls to the decorated function are repeated, suppressing specified exception(s), until a
    maximum number of retries has been attempted.

    If the function raises the exception the specified number of times, the underlying exception is raised.
    Otherwise, return the result of the function call.

    Example:

        The following retries the telnet connection 5 times on ConnectionRefusedError::

            import telnetlib


            # Retry a telnet connection 5 times if the telnet library raises ConnectionRefusedError
            @retry(ConnectionRefusedError, tries=5)
            def open(host, port):
                t = telnetlib.Telnet()
                t.open(host, port, 5)
                return t


    Arguments:
        exception_or_exceptions: Exception (sub)class (or tuple of exception classes) to watch for
        tries: number of times to try before giving up
        delay: initial delay between retries in seconds
        backoff: backoff to multiply to the delay for each retry
        exception_func: function to call on exception before the next retry
        logger: if specified, a log info message is emitted on the first retry
    """

    if isinstance(excs, type) and not issubclass(excs, BaseException):
        excs = tuple(excs)
    else:
        excs = BaseException

    def decorator(f):
        @functools.wraps(f)
        def do_retry(*args, **kwargs):
            notified = False
            active_delay = delay
            for _ in range(tries):
                try:
                    ret = f(*args, **kwargs)
                except excs as e:
                    if not notified and logger is not None:
                        etype = type(e).__qualname__
                        msg = f"caught '{etype}' on first call to '{f.__name__}' - repeating the call {tries - 1} more times or until no exception is raised"

                        logger.info(msg)

                        notified = True
                    ex = e
                    exception_func(*args, **kwargs)
                    time.sleep(active_delay)
                    active_delay = active_delay * backoff
                else:
                    break
            else:
                raise ex  # type: ignore

            return ret

        return do_retry

    return decorator


def _flatten_exceptions(exception: BaseException) -> list[BaseException]:
    result = []
    if isinstance(exception, ConcurrentException):
        for ex in exception.thread_exceptions:
            result += _flatten_exceptions(ex)
    else:
        result = [exception]

    return result


def concurrently_with_fg(calls: dict[str, Call] = {}) -> dict[typing.Any, typing.Any]:
    """runs the first call in the current thread while the rest run in the background"""
    from concurrent.futures import ThreadPoolExecutor

    # split to foreground and backround
    pairs = iter(calls.items())
    if len(calls) > 0:
        fg = dict([next(pairs)])
        bg = dict(pairs)
    else:
        fg = {}
        bg = {}

    executor = ThreadPoolExecutor()
    exc_list = []
    result = {}

    with executor:
        bg_future = executor.submit(concurrently, bg)

        try:
            result = sequentially(fg)
        except ThreadEndedByMaster:
            pass
        except BaseException as ex:
            exc_list.extend(_flatten_exceptions(ex))

        try:
            result.update(bg_future.result())
        except ThreadEndedByMaster:
            pass
        except BaseException as ex:
            exc_list.extend(_flatten_exceptions(ex))

    if len(exc_list) == 0:
        pass
    elif len(exc_list) == 1:
        raise exc_list[0]
    else:
        ex = ConcurrentException('multiple exceptions raised')
        ex.thread_exceptions = exc_list
        raise ex

    return result


def zip_offsets(
    seq: typing.Iterable[_T],
    shifts: tuple[int, ...] | list[int],
    fill: typing.Any,
    *,
    squeeze=True,
) -> typing.Generator[tuple[_T, ...]]:
    """a generator that yields from `seq` at multiple index shifts.

    Shifts that would yield an invalid index (i.e., before or beyond the end of `seq`) are
    replaced with `fill`.

    Args:
        seq: iterable of values to shift
        shifts: a sequence of integers indicating the index shifts to apply
        fill: the value to yield before or after `seq` yields values

    Yields:
        tuples of length `shifts` composed of delayed/advanced values from `seq`
    """
    lo = min(shifts)

    if lo < 0:
        # prepend as necessary to allow [None] values
        seq = itertools.chain(-lo * [fill], seq)
    else:
        lo = 0

    iters = []

    for it, shift in zip(itertools.tee(seq, len(shifts)), shifts):
        sl = itertools.islice(it, shift - lo, None)
        iters.append(sl)

    if squeeze and len(shifts) == 1:
        return iters[0]
    else:
        return itertools.zip_longest(*iters, fillvalue=fill)  # type: ignore


class _JSONFormatter(logging.Formatter):
    _last = []

    def __init__(self):
        super().__init__(style='{')
        self.t0 = time.time()

    @staticmethod
    def json_serialize_dates(obj):
        """JSON serializer for objects not serializable by default json code"""

        if isinstance(obj, (datetime.datetime, datetime.date)):
            return obj.isoformat()
        raise TypeError(f'Type {type(obj).__qualname__} not serializable')

    def format(self, record: logging.LogRecord):
        """Return a YAML string for each logger record"""
        import json

        if isinstance(record.args, dict):
            kwargs = record.args
        else:
            kwargs = {}

        msg = dict(
            message=record.msg,
            time=datetime.datetime.fromtimestamp(record.created),
            elapsed_seconds=record.created - self.t0,
            level=record.levelname,
            object=getattr(record, 'object', None),
            object_log_name=getattr(record, 'owned_name', None),
            source_file=record.pathname,
            source_line=record.lineno,
            process=record.process,
            thread=record.threadName,
            **kwargs,
        )

        if record.threadName != 'MainThread':
            msg['thread'] = record.threadName

        etype, einst, exc_tb = sys.exc_info()
        if etype is not None:
            import traceback

            msg['exception'] = traceback.format_exception_only(etype, einst)[0].rstrip()
            msg['traceback'] = ''.join(traceback.format_tb(exc_tb)).splitlines()

        self._last.append((record, msg))

        return json.dumps(msg, indent=True, default=self.json_serialize_dates)


def log_to_file(log_path: str | Path, level_name: str):
    import logging.handlers

    class _RotatingJSONFileHandler(logging.handlers.RotatingFileHandler):
        def __init__(self, path, *args, **kws):
            path = Path(path)
            if path.exists() and path.stat().st_size > 2:
                self.empty = False
            else:
                self.empty = True

            super().__init__(path, *args, **kws)

            self.stream.write('[\n')

        def emit(self, record: logging.LogRecord):
            super().emit(record)
            self.stream.write(',\n')

        def close(self):
            self.stream.write('\n]')
            super().close()

    Path(log_path).parent.mkdir(exist_ok=True, parents=True)

    logger = logging.getLogger('labbench')
    formatter = _JSONFormatter()
    handler = _RotatingJSONFileHandler(
        log_path, maxBytes=50_000_000, backupCount=5, encoding='utf8'
    )

    handler.setFormatter(formatter)
    handler.setLevel(logging.DEBUG)

    if hasattr(handler, '_striqt_handler'):
        logger.removeHandler(handler._striqt_handler)  # type: ignore

    logger.setLevel(_LOG_LEVEL_NAMES[level_name])
    logger.addHandler(handler)
    logger._striqt_handler = handler  # type: ignore


def _immediate_print_multi_tracebacks(tracebacks):
    if _handling_tracebacks:
        return

    import traceback

    for tb in tracebacks:
        try:
            traceback.print_exception(*tb)
        except BaseException:
            sys.stderr.write('\nthread error (fixme to print message)')
            sys.stderr.write('\n')


class Call(typing.Generic[_P, _R]):
    """Wrap a function to apply arguments for threaded calls to `concurrently`.
    This can be passed in directly by a user in order to provide arguments;
    otherwise, it will automatically be wrapped inside `concurrently` to
    keep track of some call metadata during execution.
    """

    args: list
    kws: dict
    func: typing.Callable
    exc_info: tuple | None = None

    if typing.TYPE_CHECKING:

        def __init__(
            self, func: typing.Callable[_P, _R], *args: _P.args, **kws: _P.kwargs
        ): ...
    else:

        def __init__(self, func, *args, **kws):
            if isinstance(func, Call):
                self.func = func.func
            elif not callable(func):
                raise ValueError('`func` argument is not callable')
            else:
                self.func = func
            self.args = args
            self.kws = kws
            self.queue = None

    def __call__(self) -> _R | None:
        try:
            self.result = self.func(*self.args, **self.kws)
        except BaseException as ex:
            self.result = None
            self.exc_info = sys.exc_info()
        else:
            self.exc_info = None

        if self.queue is not None:
            self.queue.put(self)
        else:
            return self.result

    def set_queue(self, queue):
        """Set the queue object used to communicate between threads"""
        self.queue = queue


def concurrently(
    calls: dict[str, Call], traceback_delay: bool = True, keep_nones: bool = True
) -> dict[str, typing.Any]:
    """see labbench.util for docs"""
    global _concurrency_count

    logger = get_logger('sweep')

    def traceback_skip(exc_tuple, count):
        """Skip the first `count` traceback entries in
        an exception.
        """
        tb = exc_tuple[2]
        for i in range(count):
            if tb is not None and tb.tb_next is not None:
                tb = tb.tb_next
        return exc_tuple[:2] + (tb,)

    stop_request_event.clear()

    results = {}
    threads = {}
    finished = queue.Queue()
    t0 = time.perf_counter()

    for name, call in calls.items():
        call.set_queue(finished)
        threads[name] = threading.Thread(target=call, name=name)
        threads[name].start()
        _concurrency_count += 1

    # As each thread ends, collect the return value and any exceptions
    tracebacks = []
    parent_exception = None
    exceptions = []
    name_lookup = dict(zip(calls.values(), calls.keys()))

    while len(threads) > 0:
        try:
            called = finished.get(timeout=0.25)
        except queue.Empty:
            if time.perf_counter() - t0 > 60 * 15:
                names = ','.join(threads.keys())
                logger.debug(f'threads {names!r} are still running')
                t0 = time.perf_counter()
            continue
        except BaseException as e:
            parent_exception = e
            stop_request_event.set()
            continue

        name = name_lookup[called]

        # Below only happens when called is not none
        if parent_exception is not None:
            names = tuple(threads.keys())
            exc_name = parent_exception.__class__.__name__
            logger.error(f'raising {exc_name} in after child threads {names!r} return')

        # if there was an exception that wasn't us ending the thread,
        # maybe show messages
        if called.exc_info is not None:
            tb = traceback_skip(called.exc_info, 1)

            if called.exc_info[0] is not ThreadEndedByMaster:
                # exception_count += 1
                tracebacks.append(tb)
                exceptions.extend(_flatten_exceptions(called.exc_info[1]))

            if not traceback_delay:
                _immediate_print_multi_tracebacks([tb])

        else:
            if keep_nones or called.result is not None:
                results[name] = called.result

        # Remove this thread from the dictionary of running threads
        del threads[name]
        _concurrency_count -= 1

    # Clear the stop request, if there are no other threads that
    # still need to exit
    if _concurrency_count == 0 and stop_request_event.is_set():
        stop_request_event.clear()

    # Raise exceptions as necessary
    if parent_exception is not None:
        for h in logger.logger.handlers:
            h.flush()

        if len(tracebacks) > 1:
            _immediate_print_multi_tracebacks(tracebacks)

        raise parent_exception

    elif len(tracebacks) > 0:
        # exception(s) raised
        for h in logger.logger.handlers:
            h.flush()
        if len(tracebacks) == 1:
            # assert len(exceptions) == 1
            _immediate_print_multi_tracebacks([tracebacks])
            raise exceptions[0]
        else:
            _immediate_print_multi_tracebacks(tracebacks)
            ex = ConcurrentException(f'{len(tracebacks)} call(s) raised exceptions')
            ex.thread_exceptions = exceptions
            raise ex

    return results


def sequentially(calls: dict[str, Call], keep_nones: bool = True) -> dict:
    """see labbench.util for docs"""
    results = {}

    # Run each callable
    for name, wrapper in calls.items():
        ret = wrapper()
        if wrapper.exc_info is not None:
            raise wrapper.exc_info[1]
        if ret is not None or keep_nones:
            results[name] = ret

    return results


class DebugOnException:
    def __init__(self, enable: bool = False, verbose: bool = False):
        self.enable = enable
        self.verbose = verbose
        self.prev = None
        self.lock = threading.RLock()

    def __enter__(self):
        global _handling_tracebacks
        _handling_tracebacks = True
        return self

    def __exit__(self, *args):
        self.run(*args)
        global _handling_tracebacks
        _handling_tracebacks = False

    def run(self, etype, exc, tb):
        triplet = (etype, exc, tb)

        with self.lock:
            if triplet == (None, None, None):
                return
            elif self.prev == triplet:
                return
            elif isinstance(exc, KeyboardInterrupt):
                return

            from . import tracebacks

            if not hasattr(sys, 'last_value'):
                sys.last_value = exc
            if not hasattr(sys, 'last_traceback'):
                sys.last_traceback = tb

            if self.verbose:
                handler = tracebacks.VerboseTB(call_pdb=self.enable, include_vars=True)
            else:
                handler = tracebacks.FormattedTB(call_pdb=self.enable)

            if isinstance(exc, ConcurrentException):
                handler.call_pdb = False
                for th_exc in exc.thread_exceptions:
                    try:
                        handler(type(th_exc), th_exc, th_exc.__traceback__)
                    except:
                        import traceback

                        traceback.print_exception(etype, exc, tb)
                handler.call_pdb = self.enable

            try:
                handler(etype, exc, tb)
            except:
                import traceback

                traceback.print_exception(etype, exc, tb)
            self.prev = triplet


def log_verbosity(verbose: int = 0):
    names = ('sweep', 'source', 'analysis', 'sink', 'periph')
    if verbose == 0:
        show_messages(logging.INFO, logger_names=names)
    elif verbose == 1:
        show_messages(PERFORMANCE_INFO, logger_names=names)
    elif verbose == 2:
        show_messages(PERFORMANCE_DETAIL, logger_names=names)
    else:
        show_messages(logging.DEBUG, logger_names=names)


@contextlib.contextmanager
def log_capture_context(name_suffix, /, capture_index=0, capture_count=None):
    extra = {'capture_index': capture_index}
    logger = get_logger(name_suffix)

    assert isinstance(logger.extra, dict)

    if capture_count is not None:
        logger.extra['capture_count'] = capture_count

    if capture_count is None:
        capture_count = logger.extra.get('capture_count', 'unknown')
        extra['capture_count'] = capture_count  # type: ignore

    extra['capture_progress'] = f'{capture_index + 1}/{capture_count}'  # type: ignore

    start_extra = logger.extra
    logger.extra = start_extra | extra
    yield
    logger.extra = start_extra


show_messages(logging.INFO)
