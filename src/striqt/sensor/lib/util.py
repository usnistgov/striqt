from __future__ import annotations as __

import contextlib
import datetime
import functools
import itertools
import logging
import sys
import threading
import time
import typing
from pathlib import Path

import striqt.analysis as sa

from striqt.waveform.util import lazy_import

if typing.TYPE_CHECKING:
    import typing_extensions
    import exceptiongroup
    import concurrent.futures

    _P = typing_extensions.ParamSpec('_P')
    _R = typing_extensions.TypeVar('_R', covariant=True)
    _T = typing.TypeVar('_T')
    _Tfunc = typing.Callable[..., typing.Any]

else:
    exceptiongroup = sa.util.lazy_import('exceptiongroup')
    _P = typing.TypeVar('_P')
    _R = typing.TypeVar('_R')


# %% Miscellaneous


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


# %% Concurrency
threadpool: 'concurrent.futures.ThreadPoolExecutor'
_cancel_threads = threading.Event()


class ThreadInterruptRequest(Exception):
    """Raised in a thread to indicate the owning thread requested termination"""


@contextlib.contextmanager
def share_thread_interrupts():
    try:
        yield
    finally:
        _cancel_threads.clear()


def cancel_threads():
    _cancel_threads.set()


def propagate_thread_interrupts():
    if threading.current_thread() == threading.main_thread():
        return

    if _cancel_threads.is_set():
        raise ThreadInterruptRequest()


def __getattr__(name):
    if name == 'threadpool':
        import concurrent.futures

        thread_pool = globals()[name] = concurrent.futures.ThreadPoolExecutor()
        return thread_pool
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


class ExceptionStack:
    """Creates a context manager object that accumulates exceptions.

    Any exception raised within a `defer()` context is stashed to be raised later.
    An exception (possibly an ExceptionGroup) is raised when the ExceptionDeferral
    context exits, or on a call to `handle()`.
    """

    def __init__(self, group_label: str | None = None, cancel_on_except: bool = False):
        if group_label is None:
            self.group_label = 'exceptions raised by multiple threads'
        else:
            self.group_label = group_label

        self.exceptions: list[BaseException] = []
        self.cancel = cancel_on_except

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.handle()

    def __del__(self):
        self.handle()

    @contextlib.contextmanager
    def defer(self):
        try:
            yield
        except BaseException as ex:
            if self.cancel:
                cancel_threads()
            self.exceptions.append(ex)

    def handle(self):
        """raise an exception based on any deferred exceptions.

        The raised exception type follows these rules:
        - No exceptions: no exception is raised
        - Exactly one exception: raise that exception
        - More than one exception: raise an ExceptionGroup

        A ThreadInterruptRequest will only be raised if it was the only
        deferred exception. ThreadInterruptRequest is never included in an
        ExceptionGroup.
        """
        exc_list = self.exceptions
        if len(exc_list) == 0:
            return

        INT_TYPES = (ThreadInterruptRequest, KeyboardInterrupt)

        ints = [exc for exc in exc_list if isinstance(exc, INT_TYPES)]
        non_ints = [exc for exc in exc_list if not isinstance(exc, INT_TYPES)]

        self.exceptions = []

        if len(non_ints) == 1:
            raise non_ints[0]
        elif len(non_ints) > 1:
            raise exceptiongroup.ExceptionGroup(self.group_label, non_ints)  # type: ignore
        else:
            for int in ints:
                # prefer keyboardinterrupts
                if isinstance(int, KeyboardInterrupt):
                    raise int
            else:
                raise ints[0]


def await_and_ignore(
    futures: 'typing.Iterable[concurrent.futures.Future]', except_msg: str | None = None
):
    exc = ExceptionStack(except_msg)
    try:
        for fut in futures:
            with exc:
                fut.result()
    finally:
        exc.handle()


# %% traceback and exception handling
_handling_tracebacks = False


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

            if not hasattr(sys, 'last_value'):
                sys.last_value = exc
            if not hasattr(sys, 'last_traceback'):
                sys.last_traceback = tb

            from . import tracebacks

            if self.verbose:
                handler = tracebacks.VerboseTB(call_pdb=self.enable, include_vars=True)
            else:
                handler = tracebacks.FormattedTB(call_pdb=self.enable)

            if isinstance(exc, exceptiongroup.ExceptionGroup):
                print(
                    '▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀'
                )
                print('Exception in owning thread\n')
                handler(etype, exc, tb)
                for n, sub_exc in enumerate(exc.exceptions):
                    print(
                        '▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀'
                    )
                    print(
                        f'Exception {n + 1}/{len(exc.exceptions)} in spawned threads\n'
                    )
                    handler(type(sub_exc), sub_exc, sub_exc.__traceback__)
            else:
                handler(etype, exc, tb)

            self.prev = triplet


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


# %% Logging
_LOG_LEVEL_NAMES = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARN,
    'error': logging.ERROR,
    'critical': logging.CRITICAL,
}


sa.util._StriqtLogger('sweep')
sa.util._StriqtLogger('source')
sa.util._StriqtLogger('sink')
sa.util._StriqtLogger('periph')
sa.util.show_messages(logging.INFO)


def log_verbosity(verbose: int = 0):
    names = ('sweep', 'source', 'analysis', 'sink', 'periph')
    if verbose == 0:
        sa.util.show_messages(logging.INFO, logger_names=names)
    elif verbose == 1:
        sa.util.show_messages(sa.util.PERFORMANCE_INFO, logger_names=names)
    elif verbose == 2:
        sa.util.show_messages(sa.util.PERFORMANCE_DETAIL, logger_names=names)
    else:
        sa.util.show_messages(logging.DEBUG, logger_names=names)


@contextlib.contextmanager
def log_capture_context(name_suffix, /, capture_index=0, capture_count=None):
    extra = {'capture_index': capture_index}
    logger = sa.util.get_logger(name_suffix)

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
