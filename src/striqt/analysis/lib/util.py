from __future__ import annotations as __

import contextlib
import importlib
import importlib.util
import io
import logging
import math
import sys
import threading
import time
import typing

from striqt.waveform.util import (
    configure_cupy,
    except_on_low_memory,
    free_cupy_mempool,
    is_cupy_array,
    lru_cache,
    pinned_array_as_cupy,
    cp,
)

if typing.TYPE_CHECKING:
    from striqt.waveform._typing import ArrayType


_compute_lock = threading.RLock()
_logger_adapters = {}


# additional logging levels
PERFORMANCE_INFO = 15
PERFORMANCE_DETAIL = 12


class _StriqtLogger(logging.LoggerAdapter):
    EXTRA_DEFAULTS = {
        'capture_index': 0,
        'capture_progress': 'startup',
        'capture_count': 'unknown',
        'capture': None,
    }

    def __init__(self, name_suffix, extra={}):
        _logger = logging.getLogger(name_suffix)
        super().__init__(_logger, self.EXTRA_DEFAULTS | extra)
        _logger_adapters[name_suffix] = self


def get_logger(name_suffix) -> _StriqtLogger:
    return _logger_adapters[name_suffix]


def isroundmod(value: float, div, atol=1e-6) -> bool:
    ratio = value / div
    try:
        return abs(math.remainder(ratio, 1)) <= atol
    except TypeError:
        import numpy as np

        return np.abs(np.rint(ratio) - ratio) <= atol


_StriqtLogger('analysis')


def show_messages(
    level: int | None,
    colors: bool | None = None,
    logger_names: tuple[str, ...] = ('analysis',),
):
    """filters logging messages displayed to the console by importance

    Arguments:
        minimum_level: logging level threshold for display (or None to disable)
        colors: whether to colorize the message output, or None to select automatically

    Returns:
        None
    """

    for name in logger_names:
        logger = _logger_adapters[name]

        # clear any stale handlers
        if hasattr(logger, '_screen_handler'):
            logger.logger.removeHandler(logger._screen_handler)

        if level is None:
            logger.setLevel(logging.CRITICAL)
            logger.logger.setLevel(logging.CRITICAL)
            return

        logger.setLevel(level)
        logger.logger.setLevel(level)

        logger._screen_handler = logging.StreamHandler()
        logger._screen_handler.setLevel(level)

        if colors or (colors is None and sys.stderr.isatty()):
            log_fmt = '\x1b[32m{asctime}\x1b[0m \x1b[1;30m{name:>8s}\x1b[0m \x1b[34m{capture_progress} \x1b[0m {message}'
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
    else:
        assert spec.loader is not None

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
    logger_level: int = PERFORMANCE_INFO,
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
    assert isinstance(logger.extra, dict)

    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0

        if elapsed < threshold:
            logger_level = logger_level - 10

        msg = str(desc) + ' ' if len(desc) else ''
        msg += f'⏱ {elapsed:0.3f} s'

        exc_info = sys.exc_info()
        if exc_info != (None, None, None):
            msg += f' before exception {exc_info[1]}'
            logger_level = logging.ERROR

        extra = {'stopwatch_name': desc, 'stopwatch_time': elapsed}
        logger.log(logger_level, msg.strip().lstrip(), logger.extra | extra)


@contextlib.contextmanager
def compute_lock(array=None):
    is_cupy = is_cupy_array(array)
    get_lock = array is None or is_cupy
    if get_lock:
        _compute_lock.acquire()
    yield
    if get_lock:
        _compute_lock.release()


_input_lock = threading.RLock()


@contextmanager
def hold_logger_outputs(level=logging.DEBUG):
    """
    A context manager that captures log outputs and releases them upon exit.

    Args:
        target_logger: The logger instance to capture logs from.
        level: The minimum log level to capture.
    """

    handlers = {}

    for name, adapter in _logger_adapters.items():
        handlers[name] = MemoryHandler(capacity=1000)
        handlers[name].setLevel(level)

        # Temporarily add the handler to the logger
        adapter.addHandler(handlers[name])
        original_level = adapter.level
        adapter.setLevel(level) # Ensure the logger captures messages at the specified level

    try:
        yield
    finally:
        for name, adapter in _logger_adapters.items():
            adapter.removeHandler(handlers[name])
            adapter.setLevel(original_level)

            for record in handlers[name].buffer:
                print(handlers[name].format(record), file=sys.stderr)

         handlers[name].close()


def blocking_input(prompt: str, /) -> str:
    # 1. Create a string buffer to hold the output
    stderr = io.StringIO()
    stdout = io.StringIO()

    # 2. Use redirect_stderr to point sys.stderr to our buffer
    try:
        with _input_lock, hold_logger_outputs(), contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(stdout):
            sys.__stdout__.write(prompt)
            sys.__stdout__.flush()
            response = input()
    finally:
        output = stderr.getvalue()
        if output:
            sys.stderr.write(output)
            sys.stderr.flush()

        output = stdout.getvalue()
        if output:
            sys.stdout.write(output)
            sys.stdout.flush()

    return response
