from __future__ import annotations

import contextlib
import datetime
import functools
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

    thread_exceptions = []


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

_StriqtLogger('controller')
_StriqtLogger('source')
_StriqtLogger('sink')

_concurrency_count = 0

stop_request_event = threading.Event()


_Tfunc = typing.Callable[..., typing.Any]


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


def concurrently_with_fg(
    calls: dict[str, typing.Callable] = {}, flatten: bool = True
) -> dict[typing.Any, typing.Any]:
    """runs foreground() in the current thread, and util.concurrently(**background) in another thread"""
    from concurrent.futures import ThreadPoolExecutor

    # split to foreground and backround
    pairs = iter(calls.items())
    if len(calls) > 0:
        fg = dict([next(pairs)])
        bg = dict(pairs)
    else:
        fg = {}
        bg = {}

    fg = typing.cast(dict[str, typing.Callable], fg)

    executor = ThreadPoolExecutor()
    exc_list = []
    result = {}

    with executor:
        bg_future = executor.submit(concurrently, **bg, flatten=flatten)

        try:
            result = sequentially(**fg, flatten=flatten)
        except BaseException as ex:
            if isinstance(ex, ConcurrentException):
                exc_list.extend(ex.thread_exceptions)
            else:
                exc_list.append(ex)

        try:
            result.update(bg_future.result())
        except BaseException as ex:
            if isinstance(ex, ConcurrentException):
                exc_list.extend(ex.thread_exceptions)
            else:
                exc_list.append(ex)

    if len(exc_list) == 0:
        pass
    elif len(exc_list) == 1:
        raise exc_list[0]
    else:
        ex = ConcurrentException('multiple exceptions raised')
        ex.thread_exceptions = exc_list
        raise ex

    return result


@contextlib.contextmanager
def concurrently_enter_with_fg(
    contexts: dict[str, typing.ContextManager] = {}, flatten=True
):
    cm = contextlib.ExitStack()
    calls = {name: Call(cm.enter_context, ctx) for name, ctx in contexts.items()}
    with cm:
        try:
            concurrently_with_fg(calls)
            yield cm
        except:
            cm.close()
            raise


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


class Call(typing.Generic[_P, _R]):
    """Wrap a function to apply arguments for threaded calls to `concurrently`.
    This can be passed in directly by a user in order to provide arguments;
    otherwise, it will automatically be wrapped inside `concurrently` to
    keep track of some call metadata during execution.
    """

    args: list
    kws: dict
    func: typing.Callable
    name: str

    if typing.TYPE_CHECKING:

        def __init__(
            self, func: typing.Callable[_P, _R], *args: _P.args, **kws: _P.kwargs
        ): ...
    else:

        def __init__(self, func, *args, **kws):
            if not callable(func):
                raise ValueError('`func` argument is not callable')
            self.func = func
            self.name = self.func.__name__
            self.args = args
            self.kws = kws
            self.queue = None

    def rename(self, name):
        self.name = name
        return self

    def __repr__(self):
        args = ','.join(
            [repr(v) for v in self.args]
            + [(k + '=' + repr(v)) for k, v in self.kws.items()]
        )
        if hasattr(self.func, '__qualname__'):
            name = self.func.__module__ + '.' + self.func.__qualname__
        else:
            name = self.name
        return f'Call({name},{args})'

    def __call__(self) -> _R | None:
        try:
            self.result = self.func(*self.args, **self.kws)
        except BaseException:
            self.result = None
            self.traceback = sys.exc_info()
        else:
            self.exc_info = None

        if self.queue is not None:
            self.queue.put(self)
        else:
            return self.result

    def set_queue(self, queue):
        """Set the queue object used to communicate between threads"""
        self.queue = queue

    @classmethod
    def wrap_list_to_dict(
        cls, name_func_pairs: typing.Iterable[tuple[str, Call | typing.Callable]]
    ) -> dict[str, Call]:
        """adjusts naming and wraps callables with Call"""
        ret = {}
        # First, generate the list of callables
        for name, func in name_func_pairs:
            try:
                if name is None:
                    if isinstance(func, Call):
                        name = func.name
                    elif hasattr(func, '__name__'):
                        name = func.__name__  # type: ignore
                    else:
                        raise TypeError(f'could not find name of {func}')

                if not isinstance(func, Call):
                    func = cls(func)  # type: ignore

                func.name = name

                if name in ret:
                    msg = f'another callable is already named {name!r} - pass as a keyword argument to specify a different name'
                    raise KeyError(msg)

                ret[name] = func
            except BaseException:
                raise

        return ret


class MultipleContexts:
    """Handle opening multiple contexts in a single `with` block. This is
    a threadsafe implementation that accepts a handler function that may
    implement any desired any desired type of concurrency in entering
    each context.

    The handler is responsible for sequencing the calls that enter each
    context. In the event of an exception, `MultipleContexts` calls
    the __exit__ condition of each context that has already
    been entered.

    In the current implementation, __exit__ calls are made sequentially
    (not through call_handler), in the reversed order that each context
    __enter__ was called.
    """

    def __init__(
        self,
        call_handler: typing.Callable,
        params: dict,
        objs: list,
    ):
        """
            call_handler: one of `sequentially_call` or `concurrently_call`
            params: a dictionary of operating parameters (see `concurrently`)
            objs: a list of contexts to be entered and dict-like objects to return

        Returns:

            context object for use in a `with` statement

        """

        # enter = self.enter
        # def wrapped_enter(name, context):
        #     return enter(name, context)
        # wrapped_enter.__name__ = 'MultipleContexts_enter_' + hex(id(self)+id(call_handler))

        def name(o):
            return

        self.abort = False
        self._entered = {}
        self.__name__ = '__enter__'

        # make up names for the __enter__ objects
        self.objs = [(f'enter_{type(o).__name__}_{hex(id(o))}', o) for _, o in objs]

        self.params = params
        self.call_handler = call_handler
        self.exc = {}

    def enter(self, name: str, context: typing.ContextManager):
        """
        enter!
        """
        if not self.abort:
            # proceed only if there have been no exceptions
            try:
                context.__enter__()  # start of a context entry thread
            except BaseException:
                self.abort = True
                self.exc[name] = sys.exc_info()
                raise
            else:
                self._entered[name] = context

    def __enter__(self):
        calls = [(name, Call(self.enter, name, obj)) for name, obj in self.objs]
        name = self.params['name']

        try:
            with stopwatch(f'enter {name!r} context', 'controller', 0.5, logging.DEBUG):
                self.call_handler(self.params, calls)
        except BaseException as e:
            try:
                self.__exit__(None, None, None)  # exit any open contexts before raise
            finally:
                raise e

    def __exit__(self, *exc):
        logger = get_logger('controller')
        name = self.params['name']
        with stopwatch(f'exit {name} context', 'controller', 0.5, logging.DEBUG):
            for name in tuple(self._entered.keys())[::-1]:
                context = self._entered[name]

                if name in self.exc:
                    continue

                try:
                    context.__exit__(None, None, None)
                except BaseException:
                    import traceback

                    exc = sys.exc_info()
                    traceback.print_exc()

                    # don't overwrite the original exception, if there was one
                    self.exc.setdefault(name, exc)

            contexts = dict(self.objs)
            for name, exc in self.exc.items():
                if name in contexts and name not in self._entered:
                    try:
                        contexts[name].__exit__(None, None, None)
                    except BaseException as e:
                        if e is not self.exc[name][1]:
                            msg = f'{name}.__exit__ raised {e} in cleanup attempt after another exception in {name}.__enter__'

                            logger.warning(msg)

        if len(self.exc) == 1:
            exc_info = list(self.exc.values())[0]
            raise exc_info[1]
        elif len(self.exc) > 1:
            ex = ConcurrentException(
                f'exceptions raised in {len(self.exc)} contexts are printed inline'
            )
            ex.thread_exceptions = list(self.exc)
            raise ex

        if exc[1] is not None:
            # sys.exc_info() may have been
            # changed by one of the exit methods
            # so provide explicit exception info
            for h in logger.logger.handlers:
                h.flush()
            raise exc[1]


def isdictducktype(cls):
    return hasattr(cls, 'keys') and hasattr(cls, 'pop')


def _select_enter_or_call(
    candidate_objs: typing.Sequence[
        tuple[str | None, typing.ContextManager | typing.Callable]
    ],
) -> typing.Literal['context', 'callable'] | None:
    """ensure candidates are either (1) all context managers
    or (2) all callables. Decide what type of operation to proceed with.
    """

    if len(candidate_objs) == 0:
        return None

    which = 'both'

    for k, obj in candidate_objs:
        is_callable = callable(obj)  # and not hasattr(obj, '__enter__')
        is_cm = hasattr(obj, '__enter__')

        if not is_callable and not is_cm:
            msg = 'each argument must be a callable and/or a context manager, '

            if k is None:
                msg += f'but given {obj!r}'
            else:
                msg += f'but given {k}={obj!r}'

            raise TypeError(msg)

        elif not is_callable:
            if which == 'callable':
                raise ValueError('received both callables and context managers')
            else:
                which = 'context'
        elif not is_cm:
            if which == 'context':
                raise ValueError('received both callables and context managers')
            else:
                which = 'callable'

    if which == 'both':
        raise TypeError(
            'all objects supported both calling and context management - not sure which to run'
        )

    # Enforce uniqueness in the (callable or context manager) object
    check_objs = [c[1] for c in candidate_objs]
    if len(set(check_objs)) != len(check_objs):
        raise ValueError('each callable and context manager must be unique')

    return which


def enter_or_call(
    flexible_caller: typing.Callable,
    objs: typing.Iterable[typing.ContextManager | typing.Callable],
    kws: dict[str, typing.Any],
):
    """Extract value traits from the keyword arguments flags, decide whether
    `objs` and `kws` should be treated as context managers or callables,
    and then either enter the contexts or call the callables.
    """

    objs = list(objs)

    # Treat keyword arguments passed as callables should be left as callables;
    # otherwise, override the parameter
    params = dict(
        catch=False,
        nones=False,
        traceback_delay=False,
        flatten=True,
        name=None,
        which='auto',
    )

    def merge_inputs(dicts: list, candidates: list):
        """merges nested returns and check for data key conflicts"""
        ret = {}
        for name, d in dicts:
            common = set(ret.keys()).difference(d.keys())
            if len(common) > 0:
                which = ', '.join(common)
                msg = f'attempting to merge results and dict arguments, but the key names ({which}) conflict in nested calls'
                raise KeyError(msg)
            ret.update(d)

        conflicts = set(ret.keys()).intersection([n for (n, obj) in candidates])
        if len(conflicts) > 0:
            raise KeyError('keys of conflict in nested return dictionary keys with ')

        return ret

    def merge_results(inputs, result):
        for k, v in dict(result).items():
            if isdictducktype(v.__class__):
                conflicts = set(v.keys()).intersection(start_keys)
                if len(conflicts) > 0:
                    conflicts = ','.join(conflicts)
                    raise KeyError(
                        f'conflicts in keys ({conflicts}) when merging return dictionaries'
                    )
                inputs.update(result.pop(k))

    # Pull parameters from the passed keywords
    for name in params.keys():
        if name in kws and not callable(kws[name]):
            params[name] = kws.pop(name)

    if params['name'] is None:
        raise ValueError(f'use Call to assign a name to {flexible_caller!r}')

    # Combine the position and keyword arguments, and assign labels
    allobjs = list(objs) + list(kws.values())
    names = (len(objs) * [None]) + list(kws.keys())

    candidates = list(zip(names, allobjs))
    del allobjs, names

    dicts = []
    for i, (_, obj) in enumerate(candidates):
        # pass through dictionary objects from nested calls
        if isdictducktype(obj.__class__):
            dicts.append(candidates.pop(i))

    if params['which'] == 'auto':
        which = _select_enter_or_call(candidates)
    else:
        which = params['which']

    if which is None:
        return {}
    elif which == 'context':
        if len(dicts) > 0:
            raise ValueError(
                f'unexpected return value dictionary argument for context management {dicts}'
            )
        return MultipleContexts(flexible_caller, params, candidates)
    else:
        ret = merge_inputs(dicts, candidates)
        result = flexible_caller(params, candidates)

        start_keys = set(ret.keys()).union(result.keys())
        if params['flatten']:
            merge_results(ret, result)
        ret.update(result)
        return ret


def concurrently_call(params: dict, name_func_pairs: list) -> dict:
    global _concurrency_count

    logger = get_logger('controller')

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

    catch = params['catch']
    traceback_delay = params['traceback_delay']

    # Setup calls then funcs
    # Set up mappings between wrappers, threads, and the function to call
    wrappers = Call.wrap_list_to_dict(name_func_pairs)
    threads = {
        name: threading.Thread(target=w, name=name) for name, w in wrappers.items()
    }

    # Start threads with calls to each function
    finished = queue.Queue()
    for name, thread in list(threads.items()):
        wrappers[name].set_queue(finished)
        thread.start()
        _concurrency_count += 1

    # As each thread ends, collect the return value and any exceptions
    tracebacks = []
    parent_exception = None
    last_exception = None

    t0 = time.perf_counter()

    while len(threads) > 0:
        try:
            called = finished.get(timeout=0.25)
        except queue.Empty:
            if time.perf_counter() - t0 > 60 * 15:
                names = ','.join(list(threads.keys()))
                logger.debug(f'{names} threads are still running')
                t0 = time.perf_counter()
            continue
        except BaseException as e:
            parent_exception = e
            stop_request_event.set()
            called = None

        if called is None:
            continue

        # Below only happens when called is not none
        if parent_exception is not None:
            names = ', '.join(list(threads.keys()))
            logger.error(
                f'raising {parent_exception.__class__.__name__} in main thread after child threads {names} return'
            )

        # if there was an exception that wasn't us ending the thread,
        # show messages
        if called.exc_info is not None:
            tb = traceback_skip(called.exc_info, 1)

            if called.exc_info[0] is not ThreadEndedByMaster:
                #                exception_count += 1
                tracebacks.append(tb)
                last_exception = called.exc_info[1]

            if not traceback_delay:
                import traceback

                try:
                    traceback.print_exception(*tb)
                except BaseException as e:
                    sys.stderr.write(
                        '\nthread exception, but failed to print exception'
                    )
                    sys.stderr.write(str(e))
                    sys.stderr.write('\n')
        else:
            if params['nones'] or called.result is not None:
                results[called.name] = called.result

        # Remove this thread from the dictionary of running threads
        del threads[called.name]
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
            import traceback

            for tb in tracebacks:
                try:
                    traceback.print_exception(*tb)
                except BaseException:
                    sys.stderr.write('\nthread error (fixme to print message)')
                    sys.stderr.write('\n')

        raise parent_exception

    elif len(tracebacks) > 0 and not catch:
        import traceback

        # exception(s) raised
        for h in logger.logger.handlers:
            h.flush()
        if len(tracebacks) == 1:
            assert last_exception is not None
            raise last_exception
        else:
            for tb in tracebacks:
                try:
                    traceback.print_exception(*tb)
                except BaseException:
                    sys.stderr.write('\nthread error (fixme to print message)')
                    sys.stderr.write('\n')

            ex = ConcurrentException(f'{len(tracebacks)} call(s) raised exceptions')
            ex.thread_exceptions = tracebacks
            raise ex

    return results


@typing.overload
def concurrently(
    *objs: typing.Callable, flatten: bool = False, **kws: typing.Callable
) -> dict[str, typing.Any]: ...


@typing.overload
def concurrently(
    *objs: typing.ContextManager, flatten: bool = False, **kws: typing.ContextManager
) -> dict[str, typing.ContextManager]: ...


def concurrently(
    *objs: typing.ContextManager | typing.Callable,
    flatten: bool = False,
    **kws: typing.Callable | typing.ContextManager,
) -> typing.Any:
    """see labbench.util for docs"""

    return enter_or_call(concurrently_call, objs, dict(kws, flatten=flatten))


def sequentially_call(params: dict, name_func_pairs: list) -> dict:
    """see labbench.util for docs"""
    results = {}

    wrappers = Call.wrap_list_to_dict(name_func_pairs)

    # Run each callable
    for name, wrapper in wrappers.items():
        ret = wrapper()
        if wrapper.exc_info is not None:
            raise wrapper.exc_info[1]
        if ret is not None or params['nones']:
            results[name] = ret

    return results


@typing.overload
def sequentially(
    *objs: typing.Callable, flatten: bool = False, **kws: typing.Callable
) -> dict[str, typing.Any]: ...


@typing.overload
def sequentially(
    *objs: typing.ContextManager, flatten: bool = False, **kws: typing.ContextManager
) -> dict[str, typing.ContextManager]: ...


def sequentially(
    *objs: typing.ContextManager | typing.Callable,
    flatten: bool = False,
    **kws: typing.Callable | typing.ContextManager,
) -> typing.Any:
    """see labbench.sequentially for docs"""

    if kws.get('catch', False):
        raise ValueError('catch=True is not supported by sequentially')

    return enter_or_call(sequentially_call, objs, dict(kws, flatten=flatten))


class DebugOnException:
    def __init__(
        self,
        enable: bool = False,
    ):
        self.enable = enable

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.run(*args)

    def run(self, etype, exc, tb):
        if (etype, exc, tb) == (None, None, None):
            return

        if self.enable:
            print(exc)
            from IPython.core import ultratb

            if not hasattr(sys, 'last_value'):
                sys.last_value = exc
            if not hasattr(sys, 'last_traceback'):
                sys.last_traceback = tb
            debugger = ultratb.FormattedTB(mode='Plain', call_pdb=True)
            debugger(etype, exc, tb)


def exit_context(ctx: typing.ContextManager | None, exc_info=None):
    if ctx is not None:
        if exc_info is None:
            exc_info = sys.exc_info()
        ctx.__exit__(*exc_info)  # type: ignore


def log_verbosity(verbose: int = 0):
    if verbose == 0:
        show_messages(logging.INFO)
    elif verbose == 1:
        show_messages(PERFORMANCE_INFO)
    elif verbose == 2:
        show_messages(PERFORMANCE_DETAIL)
    else:
        show_messages(logging.DEBUG)


def _extract_traceback(
    exc_type: type[BaseException],
    exc_value: BaseException,
    traceback,
    *,
    show_locals: bool = False,
    locals_hide_dunder: bool = True,
    locals_hide_sunder: bool = False,
    _visited_exceptions: typing.Optional[set[BaseException]] = None,
):
    """labbench implemented ConcurrentException as a container for
    exceptions that occur in multiple threads.

    A similar feature was added to python 3.11, ExceptionGroup.
    rich.traceback supports displaying this, so we can
    extract the exceptions in the same way here.

    """

    import inspect
    import os
    from itertools import islice

    from rich import pretty
    from rich.traceback import (
        LOCALS_MAX_LENGTH,
        LOCALS_MAX_STRING,
        Frame,
        Stack,
        Trace,
        Traceback,
        _SyntaxError,
        walk_tb,  # type: ignore
    )

    stacks: list[Stack] = []
    is_cause = False

    from rich import _IMPORT_CWD

    notes: list[str] = getattr(exc_value, '__notes__', None) or []

    grouped_exceptions: set[BaseException] = (
        set() if _visited_exceptions is None else _visited_exceptions
    )

    def safe_str(_object: typing.Any) -> str:
        """Don't allow exceptions from __str__ to propagate."""
        try:
            return str(_object)
        except Exception:
            return '<exception str() failed>'

    while True:
        stack = Stack(
            exc_type=safe_str(exc_type.__name__),
            exc_value=safe_str(exc_value),
            is_cause=is_cause,
            notes=notes,
        )

        if sys.version_info >= (3, 11):
            if isinstance(exc_value, (BaseExceptionGroup, ExceptionGroup)):
                stack.is_group = True
                for exception in exc_value.exceptions:
                    if exception in grouped_exceptions:
                        continue
                    grouped_exceptions.add(exception)
                    stack.exceptions.append(
                        _extract_traceback(
                            type(exception),
                            exception,
                            exception.__traceback__,
                            show_locals=show_locals,
                            locals_hide_dunder=locals_hide_dunder,
                            locals_hide_sunder=locals_hide_sunder,
                            _visited_exceptions=grouped_exceptions,
                        )
                    )

        if isinstance(exc_value, ConcurrentException):
            stack.is_group = True
            for exception in exc_value.thread_exceptions:
                if exception in grouped_exceptions:
                    continue
                grouped_exceptions.add(exception)
                stack.exceptions.append(
                    _extract_traceback(
                        type(exception),
                        exception,
                        exception.__traceback__,
                        show_locals=show_locals,
                        locals_hide_dunder=locals_hide_dunder,
                        locals_hide_sunder=locals_hide_sunder,
                        _visited_exceptions=grouped_exceptions,
                    )
                )

        if isinstance(exc_value, SyntaxError):
            stack.syntax_error = _SyntaxError(
                offset=exc_value.offset or 0,
                filename=exc_value.filename or '?',
                lineno=exc_value.lineno or 0,
                line=exc_value.text or '',
                msg=exc_value.msg,
                notes=notes,
            )

        stacks.append(stack)
        append = stack.frames.append

        def get_locals(
            iter_locals: typing.Iterable[tuple[str, object]],
        ) -> typing.Iterable[tuple[str, object]]:
            """Extract locals from an iterator of key pairs."""
            if not (locals_hide_dunder or locals_hide_sunder):
                yield from iter_locals
                return
            for key, value in iter_locals:
                if locals_hide_dunder and key.startswith('__'):
                    continue
                if locals_hide_sunder and key.startswith('_'):
                    continue
                yield key, value

        for frame_summary, line_no in walk_tb(traceback):
            filename = frame_summary.f_code.co_filename

            last_instruction: typing.Optional[tuple[tuple[int, int], tuple[int, int]]]
            last_instruction = None
            if sys.version_info >= (3, 11):
                instruction_index = frame_summary.f_lasti // 2
                instruction_position = next(
                    islice(
                        frame_summary.f_code.co_positions(),
                        instruction_index,
                        instruction_index + 1,
                    )
                )
                (
                    start_line,
                    end_line,
                    start_column,
                    end_column,
                ) = instruction_position
                if (
                    start_line is not None
                    and end_line is not None
                    and start_column is not None
                    and end_column is not None
                ):
                    last_instruction = (
                        (start_line, start_column),
                        (end_line, end_column),
                    )

            if filename and not filename.startswith('<'):
                if not os.path.isabs(filename):
                    filename = os.path.join(_IMPORT_CWD, filename)
            if frame_summary.f_locals.get('_rich_traceback_omit', False):
                continue

            frame = Frame(
                filename=filename or '?',
                lineno=line_no,
                name=frame_summary.f_code.co_name,
                locals=(
                    {
                        key: pretty.traverse(
                            value,
                            max_length=LOCALS_MAX_LENGTH,
                            max_string=LOCALS_MAX_STRING,
                        )
                        for key, value in get_locals(frame_summary.f_locals.items())
                        if not (inspect.isfunction(value) or inspect.isclass(value))
                    }
                    if show_locals
                    else None
                ),
                last_instruction=last_instruction,
            )
            append(frame)
            if frame_summary.f_locals.get('_rich_traceback_guard', False):
                del stack.frames[:]

        if not grouped_exceptions:
            cause = getattr(exc_value, '__cause__', None)
            if cause is not None and cause is not exc_value:
                exc_type = cause.__class__
                exc_value = cause
                # __traceback__ can be None, e.g. for exceptions raised by the
                # 'multiprocessing' module
                traceback = cause.__traceback__
                is_cause = True
                continue

            cause = exc_value.__context__
            if cause is not None and not getattr(
                exc_value, '__suppress_context__', False
            ):
                exc_type = cause.__class__
                exc_value = cause
                traceback = cause.__traceback__
                is_cause = False
                continue
        # No cover, code is reached but coverage doesn't recognize it.
        break  # pragma: no cover

    trace = Trace(stacks=stacks)

    return trace


def print_rich_exception():
    from rich.console import Console
    from rich.traceback import Traceback

    console = Console()

    exc_type, exc_value, tb = sys.exc_info()

    if exc_type is None:
        return
    if exc_value is None:
        return

    trace = _extract_traceback(exc_type, exc_value, tb, show_locals=False)

    traceback = Traceback(
        trace,
        width=None,
        show_locals=False,
        word_wrap=True,
        suppress=[
            'concurrent',
            'rich',
            'textual',
            'labbench',
            'zarr',
            'xarray',
            'pandas',
        ],
    )

    console.print(traceback)


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
