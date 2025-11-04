from __future__ import annotations
import contextlib
import datetime
import functools
import itertools
import logging
import queue
from pathlib import Path
import time
import threading
import typing
import sys

from striqt.analysis.lib.util import (
    stopwatch,
    log_capture_context,
    get_logger,
    _StriqtLogger,
    PERFORMANCE_DETAIL,
    PERFORMANCE_INFO,
    show_messages,
    isroundmod,
)


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
    _R = typing_extensions.TypeVar('_R')
    _T = typing.TypeVar('_T')

_StriqtLogger('controller')
_StriqtLogger('source')
_StriqtLogger('sink')


stop_request_event = threading.Event()


def lazy_import(module_name: str):
    """postponed import of the module with the specified name.

    The import is not performed until the module is accessed in the code. This
    reduces the total time to import the module by waiting to import submodules
    until they are used.
    """

    # see https://docs.python.org/3/library/importlib.html#implementing-lazy-imports
    try:
        ret = sys.modules[module_name]
        return ret
    except KeyError:
        pass

    import importlib.util

    spec = importlib.util.find_spec(module_name)
    if spec is None:
        raise ImportError(f'no module found named "{module_name}"')
    spec.loader = importlib.util.LazyLoader(spec.loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_Tfunc = typing.Callable[..., typing.Any]


def retry(
    exception_or_exceptions: typing.Union[
        type[BaseException], typing.Iterable[type[BaseException]]
    ],
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

    def decorator(f):
        @functools.wraps(f)
        def do_retry(*args, **kwargs):
            notified = False
            active_delay = delay
            for retry in range(tries):
                try:
                    ret = f(*args, **kwargs)
                except exception_or_exceptions as e:
                    if not notified and logger is not None:
                        etype = type(e).__qualname__
                        msg = (
                            f"caught '{etype}' on first call to '{f.__name__}' - repeating the call "
                            f'{tries - 1} more times or until no exception is raised'
                        )

                        logger.info(msg)

                        notified = True
                    ex = e
                    exception_func(*args, **kwargs)
                    time.sleep(active_delay)
                    active_delay = active_delay * backoff
                else:
                    break
            else:
                raise ex

            return ret

        return do_retry

    return decorator


@functools.wraps(functools.lru_cache)
def lru_cache(
    maxsize: int | None = 128, typed: bool = False
) -> typing.Callable[[typing.Callable[_P, _R]], typing.Callable[_P, _R]]:
    # presuming that the API is designed to accept only hashable types, set
    # the type hint to match the wrapped function
    return functools.lru_cache(maxsize, typed)


@lru_cache()
def set_cuda_mem_limit(fraction=0.75):
    try:
        import cupy
    except ModuleNotFoundError:
        return

    # Alternative: select an absolute amount of memory
    #
    # import psutil
    # available = psutil.virtual_memory().available

    cupy.get_default_memory_pool().set_limit(fraction=fraction)


def concurrently_with_fg(
    calls: dict[str, callable] = {}, flatten=True
) -> tuple[typing.Any, typing.Any]:
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
        return itertools.zip_longest(*iters, fillvalue=fill)


@functools.cache
def configure_cupy():
    # Tune cupy to perform large FFTs and allocations more quickly
    # for 1-D transforms.
    #
    # Reference:
    # https://docs.cupy.dev/en/v12.1.0/reference/fft.html#code-compatibility-features

    import cupy

    # the FFT plan sets up large caches that don't seem to help performance
    cupy.fft.config.get_plan_cache().set_size(0)

    # double-buffered streams hold on to their buffers; skip the overhead
    # of an allocator
    cupy.cuda.set_pinned_memory_allocator(None)

    # reduce memory consumption of 1-D transforms
    cupy.fft.config.enable_nd_planning = False


class _JSONFormatter(logging.Formatter):
    _last = []

    def __init__(self):
        super().__init__(style='{')
        self.t0 = time.time()
        self.first = True

    @staticmethod
    def json_serialize_dates(obj):
        """JSON serializer for objects not serializable by default json code"""

        if isinstance(obj, (datetime.datetime, datetime.date)):
            return obj.isoformat()
        raise TypeError(f'Type {type(obj).__qualname__} not serializable')

    def format(self, rec: logging.LogRecord):
        """Return a YAML string for each logger record"""
        import json

        if isinstance(rec.args, dict):
            kwargs = rec.args
        else:
            kwargs = {}

        msg = dict(
            message=rec.msg,
            time=datetime.datetime.fromtimestamp(rec.created),
            elapsed_seconds=rec.created - self.t0,
            level=rec.levelname,
            object=getattr(rec, 'object', None),
            object_log_name=getattr(rec, 'owned_name', None),
            source_file=rec.pathname,
            source_line=rec.lineno,
            process=rec.process,
            thread=rec.threadName,
            **kwargs,
        )

        if rec.threadName != 'MainThread':
            msg['thread'] = rec.threadName

        etype, einst, exc_tb = sys.exc_info()
        if etype is not None:
            import traceback

            msg['exception'] = traceback.format_exception_only(etype, einst)[0].rstrip()
            msg['traceback'] = ''.join(traceback.format_tb(exc_tb)).splitlines()

        self._last.append((rec, msg))

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

            self.terminator = ''

            super().__init__(path, *args, **kws)

            self.stream.write('[\n')

        def emit(self, rec):
            super().emit(rec)
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
        logger.removeHandler(handler._striqt_handler)

    logger.setLevel(_LOG_LEVEL_NAMES[level_name])
    logger.addHandler(handler)
    logger._striqt_handler = handler


class Call:
    """Wrap a function to apply arguments for threaded calls to `concurrently`.
    This can be passed in directly by a user in order to provide arguments;
    otherwise, it will automatically be wrapped inside `concurrently` to
    keep track of some call metadata during execution.
    """

    def __init__(self, func: typing.Callable, *args, **kws):
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

    def __call__(self):
        try:
            self.result = self.func(*self.args, **self.kws)
        except BaseException:
            self.result = None
            self.traceback = sys.exc_info()
        else:
            self.traceback = None

        if self.queue is not None:
            self.queue.put(self)
        else:
            return self.result

    def set_queue(self, queue):
        """Set the queue object used to communicate between threads"""
        self.queue = queue

    @classmethod
    def wrap_list_to_dict(
        cls, name_func_pairs: dict[str, typing.Callable]
    ) -> dict[str, Call]:
        """adjusts naming and wraps callables with Call"""
        ret = {}
        # First, generate the list of callables
        for name, func in name_func_pairs:
            try:
                if name is None:
                    if hasattr(func, 'name'):
                        name = func.name
                    elif hasattr(func, '__name__'):
                        name = func.__name__
                    else:
                        raise TypeError(f'could not find name of {func}')

                if not isinstance(func, cls):
                    func = cls(func)

                func.name = name

                if name in ret:
                    msg = (
                        f'another callable is already named {name!r} - '
                        'pass as a keyword argument to specify a different name'
                    )
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
        call_handler: typing.Callable[[dict, list, dict], dict],
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

    def enter(self, name: str, context: object):
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

        try:
            with stopwatch(
                f'entry into context for {self.params["name"]}',
                0.5,
                logger_level='debug',
            ):
                self.call_handler(self.params, calls)
        except BaseException as e:
            try:
                self.__exit__(None, None, None)  # exit any open contexts before raise
            finally:
                raise e

    def __exit__(self, *exc):
        logger = get_logger('controller')
        with stopwatch(
            f'{self.params["name"]} - context exit', 0.5, logger_level='debug'
        ):
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
                            msg = (
                                f'{name}.__exit__ raised {e} in cleanup attempt after another '
                                f'exception in {name}.__enter__'
                            )

                            logger.warning(msg)

        if len(self.exc) == 1:
            exc_info = list(self.exc.values())[0]
            raise exc_info[1]
        elif len(self.exc) > 1:
            ex = ConcurrentException(
                f'exceptions raised in {len(self.exc)} contexts are printed inline'
            )
            ex.thread_exceptions = self.exc
            raise ex
        if exc != (None, None, None):
            # sys.exc_info() may have been
            # changed by one of the exit methods
            # so provide explicit exception info
            for h in logger.logger.handlers:
                h.flush()

            raise exc[1]


RUNNERS = {
    (False, False): None,
    (False, True): 'context',
    (True, False): 'callable',
    (True, True): 'both',
}


def isdictducktype(cls):
    return hasattr(cls, 'keys') and hasattr(cls, 'pop')


class _ContextManagerType(typing.Protocol):
    def __enter__(self):
        pass

    def __exit__(self, /, type, value, traceback):
        pass


def _select_enter_or_call(
    candidate_objs: typing.Iterable[_ContextManagerType | typing.Callable],
) -> typing.Literal['context'] | typing.Literal['call'] | None:
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
    candidate_objs = [c[1] for c in candidate_objs]
    if len(set(candidate_objs)) != len(candidate_objs):
        raise ValueError('each callable and context manager must be unique')

    return which


def enter_or_call(
    flexible_caller: typing.Callable,
    objs: typing.Iterable[_ContextManagerType | typing.Callable],
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
        import hashlib
        import inspect

        # come up with a gobbledigook name that is at least unique
        frame = inspect.currentframe().f_back.f_back
        params['name'] = (
            f'<{frame.f_code.co_filename}:{frame.f_code.co_firstlineno} call 0x{hashlib.md5().hexdigest()}>'
        )

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
    global concurrency_count

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
        concurrency_count += 1

    # As each thread ends, collect the return value and any exceptions
    tracebacks = []
    parent_exception = None

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
        if called.traceback is not None:
            tb = traceback_skip(called.traceback, 1)

            if called.traceback[0] is not ThreadEndedByMaster:
                #                exception_count += 1
                tracebacks.append(tb)
                last_exception = called.traceback[1]

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
        concurrency_count -= 1

    # Clear the stop request, if there are no other threads that
    # still need to exit
    if concurrency_count == 0 and stop_request_event.is_set():
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


def concurrently(*objs, **kws):
    """see labbench.util for docs"""

    return enter_or_call(concurrently_call, objs, kws)


def sequentially_call(params: dict, name_func_pairs: list) -> dict:
    """see labbench.util for docs"""
    results = {}

    wrappers = Call.wrap_list_to_dict(name_func_pairs)

    # Run each callable
    for name, wrapper in wrappers.items():
        ret = wrapper()
        if wrapper.traceback is not None:
            raise wrapper.traceback[1]
        if ret is not None or params['nones']:
            results[name] = ret

    return results


def sequentially(*objs, **kws):
    """see labbench.sequentially for docs"""

    if kws.get('catch', False):
        raise ValueError('catch=True is not supported by sequentially')

    return enter_or_call(sequentially_call, objs, kws)
