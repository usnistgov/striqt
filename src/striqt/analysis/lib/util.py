from __future__ import annotations
import array_api_compat
import contextlib
import functools
import importlib
import importlib.util
import logging
import time
import sys
import threading
import typing
import iqwaveform.type_stubs
import typing_extensions


_logger_adapters = {}


class _StriqtLogger(logging.LoggerAdapter):
    EXTRA_DEFAULTS = {
        'capture_index': 0,
        'capture_progress': 'initializing',
        'capture_count': 'unknown',
        'capture': None,
    }

    def __init__(self, name_suffix, extra={}):
        _logger = logging.getLogger('striqt').getChild(name_suffix)
        super().__init__(_logger, self.EXTRA_DEFAULTS | extra)
        _logger_adapters[name_suffix] = self


def get_logger(name_suffix) -> _StriqtLogger:
    return _logger_adapters[name_suffix]


@contextlib.contextmanager
def log_capture_context(
    name_suffix, /, capture_index, capture, capture_count='unknown'
):
    extra = locals()
    extra['capture_progress'] = f'{capture_index + 1}/{capture_count}'
    logger = get_logger(name_suffix)
    start_extra = logger.extra
    logger.extra = start_extra | extra
    yield
    logger.extra = start_extra


_StriqtLogger('analysis')


def show_messages(
    level: int,
    colors: bool | None = None,
):
    """filters logging messages displayed to the console by importance

    Arguments:
        minimum_level: logging level threshold for display (or None to disable)
        colors: whether to colorize the message output, or None to select automatically

    Returns:
        None
    """

    for logger in _logger_adapters.values():
        logger.setLevel(logging.DEBUG)

        # clear any stale handlers
        if hasattr(logger, '_screen_handler'):
            logger.logger.removeHandler(logger._screen_handler)

        if level is None:
            return

        logger._screen_handler = logging.StreamHandler()
        logger._screen_handler.setLevel(level)

        if colors or (colors is None and sys.stderr.isatty()):
            log_fmt = (
                '\x1b[32m{asctime}\x1b[0m \x1b[1;30m{name:>15s}\x1b[0m '
                '\x1b[34mcapture {capture_progress} \x1b[0m {message}'
            )
        else:
            log_fmt = '{levelname:^7s} {asctime} • {capture_progress}: {message}'
        formatter = logging.Formatter(log_fmt, style='{', datefmt='%X')
        # formatter.default_msec_format = '%s.%03d'

        logger._screen_handler.setFormatter(formatter)
        logger.logger.addHandler(logger._screen_handler)


def lazy_import(module_name: str, package=None):
    """postponed import of the module with the specified name.

    The import is not performed until the module is accessed in the code. This
    reduces the total time to import labbench by waiting to import the module
    until it is used.
    """

    # see https://docs.python.org/3/library/importlib.html#implementing-lazy-imports
    try:
        ret = sys.modules[module_name]
        return ret
    except KeyError:
        pass

    spec = importlib.util.find_spec(module_name, package=package)
    if spec is None:
        raise ImportError(f'no module found named "{module_name}"')
    spec.loader = importlib.util.LazyLoader(spec.loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@contextlib.contextmanager
def stopwatch(
    desc: str = '',
    logger_suffix: str = 'analysis',
    threshold: float = 0,
    logger_level: int = logging.INFO,
):
    """Time a block of code using a with statement like this:

    >>> with stopwatch('sleep statement'):
    >>>     time.sleep(2)
    sleep statement time elapsed 1.999s.

    Arguments:
        desc: text for display that describes the event being timed
        logger_level: the name of the child logger to use
        threshold: if the duration is smaller than this, demote logger level

    Returns:
        context manager
    """
    t0 = time.perf_counter()
    logger = get_logger(logger_suffix)

    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0

        if elapsed < threshold:
            logger_level = logger_level - 10

        msg = str(desc) + ' ' if len(desc) else ''
        msg += f'{elapsed:0.3f} s elapsed'

        exc_info = sys.exc_info()
        if exc_info != (None, None, None):
            msg += f' before exception {exc_info[1]}'
            logger_level = logging.ERROR

        extra = {'stopwatch_name': desc, 'stopwatch_time': elapsed}
        logger.log(logger_level, msg.strip().lstrip(), logger.extra | extra)


if typing.TYPE_CHECKING:
    _P = typing_extensions.ParamSpec('_P')
    _R = typing_extensions.TypeVar('_R')
    import iqwaveform
    import labbench as lb
else:
    iqwaveform = lazy_import('iqwaveform')
    lb = lazy_import('labbench')


@functools.wraps(functools.lru_cache)
def lru_cache(
    maxsize: int | None = 128, typed: bool = False
) -> typing.Callable[[typing.Callable[_P, _R]], typing.Callable[_P, _R]]:
    # presuming that the API is designed to accept only hashable types, set
    # the type hint to match the wrapped function
    return functools.lru_cache(maxsize, typed)


def pinned_array_as_cupy(x, stream=None):
    import cupy as cp

    out = cp.empty_like(x)
    out.data.copy_from_host_async(x.ctypes.data, x.data.nbytes, stream=stream)
    return out


def except_on_low_memory(threshold_bytes=500_000_000):
    try:
        import cupy as cp
    except ModuleNotFoundError:
        return
    import psutil

    if psutil.virtual_memory().available >= threshold_bytes:
        return

    raise MemoryError('too little memory to proceed')


def free_cupy_mempool():
    try:
        import cupy as cp
    except ModuleNotFoundError:
        pass
    else:
        mempool = cp.get_default_memory_pool()
        if mempool is not None:
            mempool.free_all_blocks()


_compute_lock = threading.RLock()


@contextlib.contextmanager
def compute_lock(array=None):
    is_cupy = array_api_compat.is_cupy_array(array)
    get_lock = array is None or is_cupy
    if get_lock:
        _compute_lock.acquire()
    yield
    if get_lock:
        _compute_lock.release()


def sync_if_cupy(x: 'iqwaveform.type_stubs.ArrayType'):
    if iqwaveform.util.is_cupy_array(x):
        import cupy

        stream = cupy.cuda.get_current_stream()
        with stopwatch('cuda synchronize', threshold=10e-3):
            stream.synchronize()


@functools.cache
def configure_cupy():
    import cupy

    # the FFT plan sets up large caches that don't help us
    cupy.fft.config.get_plan_cache().set_size(0)
    cupy.cuda.set_pinned_memory_allocator(None)
