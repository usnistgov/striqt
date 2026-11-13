"""forward the records of the ``striqt`` loggers to the page.

The handler is attached to the parent ``striqt`` logger, which receives every record
that a child logger (``striqt.sweep``, ``striqt.source``, ...) emits at or above the
level of that child.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import traceback
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine, Generator

PARENT_LOGGER = 'striqt'

RECORD_FIELDS = (
    'capture_progress',
    'capture_index',
    'capture_count',
    'stopwatch_name',
    'stopwatch_time',
)
"""the `extra` fields copied from a record into the message when the record has them"""


def record_message(record: logging.LogRecord) -> dict[str, Any]:
    """return the ``log_record`` message for `record`.

    Returns:
        a dict with ``type`` (``'log_record'``), ``time`` (unix s), ``level``,
        ``levelno``, ``logger`` (the name without the ``striqt.`` prefix), ``message``,
        each field of `RECORD_FIELDS` that the record carries, and ``exception`` (the
        formatted traceback) when the record has exception info
    """
    name = record.name
    if name.startswith(PARENT_LOGGER + '.'):
        name = name[len(PARENT_LOGGER) + 1 :]
    message: dict[str, Any] = {
        'type': 'log_record',
        'time': record.created,
        'level': record.levelname,
        'levelno': record.levelno,
        'logger': name,
        'message': record.getMessage(),
    }
    for field in RECORD_FIELDS:
        if hasattr(record, field):
            message[field] = getattr(record, field)
    if record.exc_info:
        message['exception'] = ''.join(traceback.format_exception(*record.exc_info))
    return message


class WebSocketLogHandler(logging.Handler):
    """a logging handler that schedules `send` with the message of each record.

    Construction stores `send` and `loop`; nothing is attached to a logger. `emit`
    runs in the logging thread and schedules the coroutine on `loop`, so `send` runs
    in the server thread.
    """

    def __init__(
        self,
        send: Callable[[dict[str, Any]], Coroutine[Any, Any, None]],
        loop: asyncio.AbstractEventLoop,
        level: int = logging.INFO,
    ) -> None:
        super().__init__(level)
        self.send = send
        self.loop = loop

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record_message(record)
        except Exception:
            self.handleError(record)
            return
        if self.loop.is_closed():
            return
        asyncio.run_coroutine_threadsafe(self.send(message), self.loop)


@contextlib.contextmanager
def forward_records(
    send: Callable[[dict[str, Any]], Coroutine[Any, Any, None]],
    loop: asyncio.AbstractEventLoop,
    level: int = logging.INFO,
) -> Generator[WebSocketLogHandler, None, None]:
    """attach a `WebSocketLogHandler` to the ``striqt`` logger for the duration of the block.

    The handler is removed on exit, including after an exception.

    Yields:
        the attached handler; `setLevel` on the handler changes the records sent
    """
    handler = WebSocketLogHandler(send, loop, level)
    logger = logging.getLogger(PARENT_LOGGER)
    logger.addHandler(handler)
    try:
        yield handler
    finally:
        logger.removeHandler(handler)
