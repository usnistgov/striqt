from __future__ import annotations as __

import contextlib
import io
import itertools
import logging
import math
import sys
import threading
import time
from collections.abc import Generator, Iterable
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import striqt.waveform as sw
from striqt.waveform.lib.util import (
    array_api_compat as array_api_compat,
    lazy_import as lazy_import,
    lru_cache as lru_cache,
    np as np,
    pd as pd,
    qualified_name as qualified_name,
    xr as xr,
)

if TYPE_CHECKING:
    from striqt.waveform.lib.typing import ArrayLike

    from .. import specs

    cached_property = property
else:
    from functools import cached_property as cached_property

_compute_lock = threading.RLock()
_logger_adapters = {}


# additional logging levels
from logging import INFO as INFO

PERFORMANCE_INFO = 15
PERFORMANCE_DETAIL = 12


class StriqtLogger(logging.LoggerAdapter):
    _screen_handler: logging.StreamHandler

    EXTRA_DEFAULTS: ClassVar[dict[str, Any]] = {
        'capture_index': 0,
        'capture_progress': 'control',
        'capture_count': 'unknown',
        'capture': None,
    }

    def __init__(self, name_suffix: str, extra: dict[str, Any] | None = None) -> None:
        _logger = logging.getLogger(f'striqt.{name_suffix}')
        super().__init__(_logger, self.EXTRA_DEFAULTS | (extra or {}))
        _logger_adapters[name_suffix] = self


def get_logger(name_suffix) -> StriqtLogger:
    return _logger_adapters[name_suffix]


def isroundmod(value: float, div, atol=1e-6) -> bool:
    ratio = value / div
    try:
        if not math.isfinite(ratio):
            return False
        return abs(math.remainder(ratio, 1)) <= atol
    except TypeError:
        import numpy as np

        return np.abs(np.rint(ratio) - ratio) <= atol


StriqtLogger('analysis')


def show_messages(
    level: int | None,
    colors: bool | None = None,
    logger_names: tuple[str, ...] | Literal['all'] = 'all',
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
            continue

        logger.setLevel(level)
        logger.logger.setLevel(level)

        logger._screen_handler = logging.StreamHandler()
        logger._screen_handler.setLevel(level)

        if colors or (colors is None and sys.stderr.isatty()):
            log_fmt = '\x1b[32m{asctime}\x1b[0m \x1b[1;30m{name:>15s}\x1b[0m \x1b[34m{capture_progress} \x1b[0m {message}'
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
) -> Generator[None, None, None]:
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
    is_cupy = sw.is_cupy_array(array)
    get_lock = array is None or is_cupy
    if get_lock:
        _compute_lock.acquire()
    try:
        yield
    finally:
        if get_lock:
            _compute_lock.release()


@contextlib.contextmanager
def hold_logger_outputs():
    """point the screen handler of each logger at the current `sys.stderr` until exit.

    Yields:
        None, once, for the body of the ``with`` block
    """

    streams = {}

    for name, adapter in _logger_adapters.items():
        if not hasattr(adapter, '_screen_handler'):
            continue
        streams[name] = adapter._screen_handler.stream
        # `contextlib.redirect_stderr` does not update the stream reference of a
        # handler
        adapter._screen_handler.stream = sys.stderr

    try:
        yield
    finally:
        for name, stream in streams.items():
            _logger_adapters[name]._screen_handler.stream = stream


_input_lock = threading.RLock()


def blocking_input(prompt: str | None = None, /) -> str:
    """read one line from stdin under a lock, with other output deferred.

    Output written to `sys.stderr`, `sys.stdout`, and the screen log handlers while
    the read is in progress is buffered and written after the read completes. One
    thread reads at a time.

    Args:
        prompt: text written to the original stdout before the read, or None for
            no prompt

    Returns:
        the line read, without the trailing newline

    Raises:
        EOFError: when stdin is closed
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


def ordered_set_union(*args: Iterable[Any]) -> list[Any]:
    """perform a set union on the supplied iterables.

    Maintains ordering in the same way as dict/OrderedDict.
    """
    return list(dict.fromkeys(itertools.chain.from_iterable(args)))


def elementwise_atol(tolerance: specs.Tolerance, expected_dB: ArrayLike) -> np.ndarray:
    """return the per-element absolute tolerance, in dB, on a dB-valued output.

    The tolerance of an element grows with the depth of the element below the peak
    of `expected_dB`. At the peak the tolerance is `tolerance.on_peak.peak`. Below
    the peak the tolerance follows `sw.power_analysis.off_peak_dB_tolerance` and
    reaches +inf at a depth of `off_peak_dBc.peak`. Without `off_peak_dBc`, the
    tolerance is `on_peak.peak` at every element.

    Args:
        tolerance: the budget; `on_peak.peak` and `off_peak_dBc` are read
        expected_dB: the expected output values, in dB; the nanmax is the peak

    Returns:
        a float64 numpy array of the shape of `expected_dB`, in dB: no element is
        below `tolerance.on_peak.peak`, elements at or beyond a depth of
        `off_peak_dBc.peak` are +inf, and elements are nan where `expected_dB` is
        nan
    """
    import numpy as np

    expected_dB = np.asarray(expected_dB, dtype='float64')
    if tolerance.off_peak_dBc is None:
        return np.full(expected_dB.shape, tolerance.on_peak.peak)
    # the amplitude error at the peak, as a fraction of the peak amplitude
    err = 10 ** (tolerance.off_peak_dBc.peak / 20)
    additive = tolerance.on_peak.peak - 20 * math.log10(1 + err)
    depth = np.nanmax(expected_dB) - expected_dB
    return np.asarray(additive + sw.power_analysis.off_peak_dB_tolerance(depth, err))
