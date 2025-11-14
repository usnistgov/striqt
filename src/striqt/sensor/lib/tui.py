from __future__ import annotations

import logging
import math
import sys
import typing
import warnings

import textual
from rich.segment import Segments
from rich.text import Text
from textual.app import App, ComposeResult
from textual.color import Gradient
from textual.containers import (
    Center,
    CenterMiddle,
    HorizontalGroup,
    VerticalGroup,
    VerticalScroll,
)
from textual.screen import Screen
from textual.widgets import Button, DataTable, Footer, Label, ProgressBar, Static
from textual.worker import Worker

from . import connections, frontend, sweeps, util

if typing.TYPE_CHECKING:
    import psutil
else:
    psutil = util.lazy_import('psutil')


__all__ = ['SweepHUDApp']


class SystemMonitorWidget(Static):
    """A widget to display memory usage."""

    def on_mount(self) -> None:
        """Called when the widget is mounted."""
        self.set_interval(1, self.update_stats)  # Update every 1 second

    def update_stats(self) -> None:
        """Updates the memory usage statistics."""
        memory = psutil.virtual_memory()
        memory_free_gb = memory.free / 1e9
        memory_total_gb = memory.total / 1e9

        self.update(f' {memory_free_gb:0.1f}/{memory_total_gb:0.1f} GB free')


def any_are_running(workers: typing.Iterable[Worker]) -> bool:
    worker: Worker

    for worker in workers:
        if worker.is_running:
            util.get_logger('controller').logger.error(f'still running: {str(worker)}')
            return True
    else:
        return False


def get_log_append(prior_text_size, record):
    if prior_text_size != 0:
        prepend = '\n'
    else:
        prepend = ''

    if 'stopwatch_name' in record.args:
        stopwatch_name = record.args['stopwatch_name']
        stopwatch_time = record.args['stopwatch_time']
        return prepend + f'⏱ {stopwatch_time * 1000:0.0f} ms {stopwatch_name}'
    else:
        return prepend + record.getMessage()


class LogNotifier(logging.Handler):
    def __init__(
        self, callback: typing.Callable, logger_names: list[str], level=logging.INFO
    ):
        self.callback = callback

        super().__init__(level)

        for name in logger_names:
            logger = logging.getLogger(name)
            logger.setLevel(level)
            logger.addHandler(self)

    def emit(self, record):
        self.callback(get_log_append(0, record))


class SplitLogHandler(logging.Handler):
    def __init__(self, app: SweepHUDApp, logger_names=[], verbose=False):
        level = logging.DEBUG if verbose else logging.INFO
        self.app = app

        super().__init__(level)

        for name in logger_names:
            logger = util.get_logger(name)
            logger.logger.setLevel(level)
            logger.logger.addHandler(self)

    def emit(self, record):
        self.app.call_from_thread(self.app._update_sweep_status, record)


class QuitScreen(Screen):
    """Screen with a dialog to quit."""

    CSS = """
    QuitScreen {
        align: center middle;
    }

    #quit-buttons {
        grid-size: 2;
        grid-gutter: 2 2;
        grid-rows: 1fr 3;
        padding: 0 1;
        width: 60;
        height: 11;
        border: thick $background 80%;
        background: $surface;
    }

    #question {
        column-span: 2;
        height: 1fr;
        width: 1fr;
        content-align: center middle;
    }

    """

    def compose(self) -> ComposeResult:
        with VerticalGroup(id='quit-buttons'), CenterMiddle():
            yield Label('Stop the sweep and quit?', id='question')
            with HorizontalGroup(), CenterMiddle():
                yield Button('No', variant='primary', id='cancel')
                yield Button('Yes', variant='error', id='quit')

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.app.pop_screen()

        if event.button.id == 'quit':
            self.app.exit()


class VerticalScrollDataTable(DataTable):
    def action_scroll_end(self):
        return super().action_scroll_bottom()

    def action_scroll_home(self):
        return super().action_scroll_top()


class SweepHUDApp(App):
    _CAPTURE_DISPLAY_LIMIT = 100

    BINDINGS = [
        ('q', 'request_quit', 'Quit'),
    ]

    CSS = """
    Screen {
        align: center top;
    }

    #status-group {
        background: blue 50%;
        border: wide white;
    }
    """

    def __init__(self, cli_kws: dict):
        self.resources = None
        self.cli_kws = cli_kws
        super().__init__()
        self.title = 'Sensor sweep'
        self.handler = SplitLogHandler(
            self,
            ('source', 'analysis', 'sink'),
            verbose=cli_kws['verbose'],
        )
        self.row_keys: dict[int, str] = {}
        self.column_keys: dict[str, str] = {}
        self._original_stderr = sys.stderr

    ### Underlying actions
    @textual.work(exclusive=True, thread=True)
    def do_sweep(self):
        gen = frontend.iter_sweep_cli(
            self.resources,
            remote=self.cli_kws.get('remote', None),
            verbose=self.cli_kws['verbose'],
        )
        for _ in gen:
            self.refresh()

        self.call_from_thread(self._show_done)

    @textual.work(exclusive=True, thread=True)
    def do_startup(self):
        self.resources = frontend.init_sweep_cli(**self.cli_kws)
        self.display_fields = sweeps.varied_capture_fields(self.resources.sweep_spec)
        self.call_from_thread(self._show_ready)
        self.call_from_thread(self.do_sweep)

    ### textual.app.App protocol
    def compose(self) -> ComposeResult:
        with VerticalScroll():
            yield VerticalScrollDataTable(zebra_stripes=True, cursor_type='row')
        with Center(), HorizontalGroup(id='status-group'):
            yield ProgressBar(show_percentage=False, id='progress')
            yield SystemMonitorWidget()
        yield Footer()

    def action_request_quit(self) -> None:
        """Action to display the quit dialog."""
        if any_are_running(self.workers):
            self.push_screen(QuitScreen(name='Confirm'))
        else:
            self.exit()

    ### Utility
    def _notify_warnings(self, warn_message, *args, **kws):
        """A custom function to handle warnings."""
        self.notify(warn_message)
        # You can add any custom logic here, like logging, sending notifications, etc.

    def _show_done(self):
        progress = self.query_one('ProgressBar')

        gradient = Gradient.from_colors(
            '#881177',
            '#aa3355',
            '#cc6666',
            '#ee9944',
            '#eedd00',
            '#99dd55',
            '#44dd88',
            '#22ccbb',
            '#00bbcc',
            '#0099cc',
            '#3366bb',
            '#663399',
        )

        progress.gradient = gradient

    def _show_ready(self):
        def add_column(name: str, width_frac: float):
            return table.add_column(
                Text(name, style='content-align-vertical: bottom'),
                width=round(free_width * width_frac),
            )

        table = self.query_one(VerticalScrollDataTable)
        captures = self.resources['sweep_spec'].looped_captures
        index_size = math.ceil(math.log10(len(captures))) + 2
        free_width = self.size.width - index_size

        self.column_keys = {
            'capture': add_column('', 0.27),
            'striqt.source': add_column('Source', 0.18),
            'striqt.analysis': add_column('Analysis', 0.32),
            'striqt.sink': add_column('Sink', 0.23),
        }

        table.loading = False

        progress = self.query_one(ProgressBar)
        progress.total = len(captures)

        self.sub_title = 'Running'

    def on_mount(self) -> None:
        table = self.query_one(VerticalScrollDataTable)
        table.loading = True

        self._handler = LogNotifier(
            self.notify,
            logger_names=[
                'striqt.controller',
            ],
        )

        # Set a custom showwarning function to intercept warnings
        warnings.showwarning = self._notify_warnings
        self.do_startup()

        self.sub_title = 'Initializing'

    def _update_sweep_status(self, record: logging.LogRecord):
        from striqt.analysis.lib.dataarrays import describe_capture

        if self.resources is None:
            # warmup captures
            return

        captures = self.resources['sweep_spec'].looped_captures

        logger = util.get_logger(record.name.rsplit('.', 1)[1])
        extra = logger.extra
        if isinstance(record.args, dict):
            extra = extra | record.args

        capture_index = extra['capture_index']
        if not 0 <= capture_index < len(captures):
            return

        capture = captures[capture_index]

        table = self.query_one(VerticalScrollDataTable)
        progress = self.query_one(ProgressBar)

        if capture_index not in self.row_keys:
            if len(self.display_fields) > 0:
                desc = describe_capture(
                    capture, constrain=tuple(self.display_fields), join='\n'
                )
                desc = desc.replace('=', ': ')
            else:
                desc = ' '
            row = [desc, Text(''), '', '']
            label = Text(str(capture_index), style='#B0FC38 bold')
            row_key = self.row_keys[capture_index] = table.add_row(
                *row, height=None, label=label
            )
            progress.update(advance=1)
            if table.cursor_row == capture_index - 1:
                table.move_cursor(row=capture_index)

        else:
            row_key = self.row_keys[capture_index]

        col_key = self.column_keys[record.name]
        text = table.get_cell(row_key, col_key)
        msg = text + get_log_append(len(text), record)

        table.update_cell(row_key, col_key, msg)

        if len(self.row_keys) > self._CAPTURE_DISPLAY_LIMIT:
            capture_index = next(iter(self.row_keys.keys()))
            row_key = self.row_keys.pop(capture_index)
            table.remove_row(row_key)

    def _fatal_error(self) -> None:
        """Exits the app after an unhandled exception."""
        from rich.traceback import Traceback

        self.bell()
        trace = frontend._extract_traceback(*sys.exc_info(), show_locals=False)
        traceback = Traceback(
            trace,
            width=None,
            show_locals=False,
            suppress=['rich', 'labbench', 'zarr', 'xarray', 'pandas'],
        )

        segments = Segments(self.console.render(traceback, self.console.options))
        self._exit_renderables.append(segments)
        self._close_messages_no_wait()
