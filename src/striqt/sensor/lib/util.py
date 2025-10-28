from __future__ import annotations
import contextlib
import datetime
import functools
import itertools
import logging
import logging.handlers
from pathlib import Path
import time
import typing
import typing_extensions
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

TGen = type[typing.Any]

if typing.TYPE_CHECKING:
    _P = typing_extensions.ParamSpec('_P')
    _R = typing_extensions.TypeVar('_R')

_StriqtLogger('controller')
_StriqtLogger('source')
_StriqtLogger('sink')


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
        BaseException, typing.Iterable[BaseException]
    ],
    tries: int,
    *,
    delay: float = 0,
    backoff: float = 0,
    exception_func=lambda *args, **kws: None,
    logger: logging.Logger | logging.LoggerAdapter | None = None,
) -> callable[[_Tfunc], _Tfunc]:
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
    """runs foreground() in the current thread, and lb.concurrently(**background) in another thread"""
    from concurrent.futures import ThreadPoolExecutor
    import labbench as lb

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
        bg_future = executor.submit(lb.concurrently, **bg, flatten=flatten)

        try:
            result = lb.sequentially(**fg, flatten=flatten)
        except BaseException as ex:
            if isinstance(ex, lb.util.ConcurrentException):
                exc_list.extend(ex.thread_exceptions)
            else:
                exc_list.append(ex)

        try:
            result.update(bg_future.result())
        except BaseException as ex:
            if isinstance(ex, lb.util.ConcurrentException):
                exc_list.extend(ex.thread_exceptions)
            else:
                exc_list.append(ex)

    if len(exc_list) == 0:
        pass
    elif len(exc_list) == 1:
        raise exc_list[0]
    else:
        ex = lb.util.ConcurrentException('multiple exceptions raised')
        ex.thread_exceptions = exc_list
        raise ex

    return result


@contextlib.contextmanager
def concurrently_enter_with_fg(
    contexts: dict[str, typing.ContextManager] = {}, flatten=True
):
    import labbench as lb

    cm = contextlib.ExitStack()
    calls = {name: lb.Call(cm.enter_context, ctx) for name, ctx in contexts.items()}
    with cm:
        try:
            concurrently_with_fg(calls)
            yield cm
        except:
            cm.close()
            raise


def zip_offsets(
    seq: typing.Iterable[TGen],
    shifts: tuple[int, ...] | list[int],
    fill: typing.Any,
    *,
    squeeze=True,
) -> typing.Generator[tuple[TGen, ...]]:
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


def log_to_file(log_path: str | Path, log_level: int):
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

    logger.setLevel(log_level)
    logger.addHandler(handler)
    logger._striqt_handler = handler
