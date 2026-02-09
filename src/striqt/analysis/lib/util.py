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
    lazy_import,
    pinned_array_as_cupy,
    cp,
)

if typing.TYPE_CHECKING:
    from striqt.waveform._typing import ArrayType


_compute_lock = threading.RLock()
_logger_adapters = {}


# additional logging levels
from logging import WARNING, INFO, DEBUG

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
    logger_names: tuple[str, ...] | typing.Literal['all'] = 'all',
):
    """filters logging messages displayed to the console by importance

    Arguments:
        minimum_level: logging level threshold for display (or None to disable)
        colors: whether to colorize the message output, or None to select automatically

    Returns:
        None
    """

    if logger_names == 'all':
        logger_names = tuple(_logger_adapters.keys())

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
            logger_level = PERFORMANCE_DETAIL

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


@contextlib.contextmanager
def hold_logger_outputs():
    """apply redirected streams to log outputs, and restore on exit.

    This is needed because the loggers hold their own references to the original
    stderr/stdout, so they are not updated on use of contextlib.redirect_ functions.
    """

    streams = {}

    for name, adapter in _logger_adapters.items():
        if not hasattr(adapter, '_screen_handler'):
            continue
        streams[name] = adapter._screen_handler.stream
        adapter._screen_handler.stream = sys.stderr

    try:
        yield
    finally:
        for name in streams.keys():
            _logger_adapters[name]._screen_handler.stream = streams[name]


_input_lock = threading.RLock()


def blocking_input(prompt: str | None = None, /) -> str:
    """wraps a call to builtin input() to avoid threading issues.

    Specifically:
    - A lock protects access to sys.stdin in case of calls from multiple threads
    - Log outputs, sys.stderr, and sys.stdout are buffered until entry is complete
    """

    stderr = io.StringIO()
    stdout = io.StringIO()

    try:
        with (
            _input_lock,
            contextlib.redirect_stderr(stderr),
            contextlib.redirect_stdout(stdout),
            hold_logger_outputs(),
        ):
            assert isinstance(sys.__stdout__, io.TextIOWrapper)
            if prompt is not None:
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
