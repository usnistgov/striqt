from __future__ import annotations

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
_StriqtLogger('ext')

_concurrency_count = 0
_handling_tracebacks = False

stop_request_event = threading.Event()


_Tfunc = typing.Callable[..., typing.Any]


# import_locks = collections.defaultdict(threading.Lock)
# cupy_ready = threading.Event()

# def _blocking_import(name):
#     with import_locks[name]:
#         return importlib.import_module(name)

# def blocking_imports(xarray=False, analysis=False, cupy=False):
#     if cupy:
#         _blocking_import('cupy')
#         _blocking_import('cupyx')
#         _blocking_import('cupyx.scipy')
#         if analysis:
#             _blocking_import('numba.cuda')

#         cupy_ready.set()

#     _blocking_import('scipy')
#     _blocking_import('numpy')

#     if xarray:
#         _blocking_import('xarray')

#     if analysis:
#         _blocking_import('numba')

#     if cupy and analysis:
#         _blocking_import('numba.cuda')


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
        except BaseException as ex:
            exc_list.extend(_flatten_exceptions(ex))

        try:
            result.update(bg_future.result())
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
            assert len(exceptions) == 1
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
                    handler(type(th_exc), th_exc, th_exc.__traceback__)
                handler.call_pdb = self.enable

            handler(etype, exc, tb)
            self.prev = triplet


def exit_context(ctx: typing.ContextManager | None, exc_info=None):
    if ctx is not None:
        if exc_info is None:
            exc_info = sys.exc_info()
        ctx.__exit__(*exc_info)  # type: ignore


def log_verbosity(verbose: int = 0):
    names = ('sweep', 'source', 'analysis', 'sink', 'ext')
    if verbose == 0:
        show_messages(logging.INFO, logger_names=names)
    elif verbose == 1:
        show_messages(PERFORMANCE_INFO, logger_names=names)
    elif verbose == 2:
        show_messages(PERFORMANCE_DETAIL, logger_names=names)
    else:
        show_messages(logging.DEBUG, logger_names=names)


# def _extract_traceback(
#     exc_type: type[BaseException],
#     exc_value: BaseException,
#     traceback,
#     *,
#     show_locals: bool = False,
#     locals_hide_dunder: bool = True,
#     locals_hide_sunder: bool = False,
#     _visited_exceptions: typing.Optional[set[BaseException]] = None,
# ):
#     """labbench implemented ConcurrentException as a container for
#     exceptions that occur in multiple threads.

#     A similar feature was added to python 3.11, ExceptionGroup.
#     rich.traceback supports displaying this, so we can
#     extract the exceptions in the same way here.
#     """

#     print('extract')

#     import inspect
#     import os
#     from itertools import islice

#     from rich import pretty
#     from rich.traceback import (
#         LOCALS_MAX_LENGTH,
#         LOCALS_MAX_STRING,
#         Frame,
#         Stack,
#         Trace,
#         Traceback,
#         _SyntaxError,
#         walk_tb,  # type: ignore
#     )

#     stacks: list[Stack] = []
#     is_cause = False

#     from rich import _IMPORT_CWD

#     notes: list[str] = getattr(exc_value, '__notes__', None) or []

#     grouped_exceptions: set[BaseException] = (
#         set() if _visited_exceptions is None else _visited_exceptions
#     )

#     def safe_str(_object: typing.Any) -> str:
#         """Don't allow exceptions from __str__ to propagate."""
#         try:
#             return str(_object)
#         except Exception:
#             return '<exception str() failed>'

#     while True:
#         stack = Stack(
#             exc_type=safe_str(exc_type.__name__),
#             exc_value=safe_str(exc_value),
#             is_cause=is_cause,
#             notes=notes,
#         )

#         if sys.version_info >= (3, 11):
#             if isinstance(exc_value, (BaseExceptionGroup, ExceptionGroup)):
#                 stack.is_group = True
#                 for exception in exc_value.exceptions:
#                     if exception in grouped_exceptions:
#                         continue
#                     grouped_exceptions.add(exception)
#                     stack.exceptions.append(
#                         _extract_traceback(
#                             type(exception),
#                             exception,
#                             exception.__traceback__,
#                             show_locals=show_locals,
#                             locals_hide_dunder=locals_hide_dunder,
#                             locals_hide_sunder=locals_hide_sunder,
#                             _visited_exceptions=grouped_exceptions,
#                         )
#                     )

#         if isinstance(exc_value, ConcurrentException):
#             stack.is_group = True
#             for exception in exc_value.thread_exceptions:
#                 if exception in grouped_exceptions:
#                     continue
#                 grouped_exceptions.add(exception)
#                 stack.exceptions.append(
#                     _extract_traceback(
#                         type(exception),
#                         exception,
#                         exception.__traceback__,
#                         show_locals=show_locals,
#                         locals_hide_dunder=locals_hide_dunder,
#                         locals_hide_sunder=locals_hide_sunder,
#                         _visited_exceptions=grouped_exceptions,
#                     )
#                 )

#         if isinstance(exc_value, SyntaxError):
#             stack.syntax_error = _SyntaxError(
#                 offset=exc_value.offset or 0,
#                 filename=exc_value.filename or '?',
#                 lineno=exc_value.lineno or 0,
#                 line=exc_value.text or '',
#                 msg=exc_value.msg,
#                 notes=notes,
#             )

#         stacks.append(stack)
#         append = stack.frames.append

#         def get_locals(
#             iter_locals: typing.Iterable[tuple[str, object]],
#         ) -> typing.Iterable[tuple[str, object]]:
#             """Extract locals from an iterator of key pairs."""
#             if not (locals_hide_dunder or locals_hide_sunder):
#                 yield from iter_locals
#                 return
#             for key, value in iter_locals:
#                 if locals_hide_dunder and key.startswith('__'):
#                     continue
#                 if locals_hide_sunder and key.startswith('_'):
#                     continue
#                 yield key, value

#         for frame_summary, line_no in walk_tb(traceback):
#             filename = frame_summary.f_code.co_filename

#             last_instruction: typing.Optional[tuple[tuple[int, int], tuple[int, int]]]
#             last_instruction = None
#             if sys.version_info >= (3, 11):
#                 instruction_index = frame_summary.f_lasti // 2
#                 instruction_position = next(
#                     islice(
#                         frame_summary.f_code.co_positions(),
#                         instruction_index,
#                         instruction_index + 1,
#                     )
#                 )
#                 (
#                     start_line,
#                     end_line,
#                     start_column,
#                     end_column,
#                 ) = instruction_position
#                 if (
#                     start_line is not None
#                     and end_line is not None
#                     and start_column is not None
#                     and end_column is not None
#                 ):
#                     last_instruction = (
#                         (start_line, start_column),
#                         (end_line, end_column),
#                     )

#             if filename and not filename.startswith('<'):
#                 if not os.path.isabs(filename):
#                     filename = os.path.join(_IMPORT_CWD, filename)
#             if frame_summary.f_locals.get('_rich_traceback_omit', False):
#                 continue

#             frame = Frame(
#                 filename=filename or '?',
#                 lineno=line_no,
#                 name=frame_summary.f_code.co_name,
#                 locals=(
#                     {
#                         key: pretty.traverse(
#                             value,
#                             max_length=LOCALS_MAX_LENGTH,
#                             max_string=LOCALS_MAX_STRING,
#                         )
#                         for key, value in get_locals(frame_summary.f_locals.items())
#                         if not (inspect.isfunction(value) or inspect.isclass(value))
#                     }
#                     if show_locals
#                     else None
#                 ),
#                 last_instruction=last_instruction,
#             )
#             append(frame)
#             if frame_summary.f_locals.get('_rich_traceback_guard', False):
#                 del stack.frames[:]

#         if not grouped_exceptions:
#             cause = getattr(exc_value, '__cause__', None)
#             if cause is not None and cause is not exc_value:
#                 exc_type = cause.__class__
#                 exc_value = cause
#                 # __traceback__ can be None, e.g. for exceptions raised by the
#                 # 'multiprocessing' module
#                 traceback = cause.__traceback__
#                 is_cause = True
#                 continue

#             cause = exc_value.__context__
#             if cause is not None and not getattr(
#                 exc_value, '__suppress_context__', False
#             ):
#                 exc_type = cause.__class__
#                 exc_value = cause
#                 traceback = cause.__traceback__
#                 is_cause = False
#                 continue
#         # No cover, code is reached but coverage doesn't recognize it.
#         break  # pragma: no cover

#     trace = Trace(stacks=stacks)

#     return trace


# def print_rich_exception():
#     from rich.console import Console
#     from rich.traceback import Traceback

#     console = Console()

#     exc_type, exc_value, tb = sys.exc_info()

#     if exc_type is None:
#         return
#     if exc_value is None:
#         return

#     trace = _extract_traceback(exc_type, exc_value, tb, show_locals=False)

#     traceback = Traceback(
#         trace,
#         width=None,
#         show_locals=False,
#         word_wrap=True,
#         suppress=[
#             'concurrent',
#             'rich',
#             'textual',
#             'labbench',
#             'zarr',
#             'xarray',
#             'pandas',
#         ],
#     )

#     console.print(traceback)


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
