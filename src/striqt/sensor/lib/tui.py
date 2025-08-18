import logging
import math
import typing
from collections import Counter
import time

import rich, textual
from textual.app import App, ComposeResult
from textual.color import Gradient
from textual.widgets import Button, DataTable, Label, Header, Footer, ProgressBar
from textual.containers import (
    Center,
    CenterMiddle,
    VerticalGroup,
    HorizontalGroup,
    ScrollableContainer,
)
from textual.worker import get_current_worker, Worker
from textual.screen import Screen
from rich.text import Text
from rich.segment import Segments
import labbench as lb
import xarray as xr
import warnings

from . import frontend, util


def nonunique_capture_fields(captures):
    values_list = (c.todict().values() for c in captures)
    counts = [len(Counter(v)) for v in zip(*values_list)]
    fields = captures[0].todict().keys()
    return [f for f, c in zip(fields, counts) if c > 1]


def any_are_running(workers: typing.Iterable[Worker]) -> bool:
    worker: Worker

    for worker in workers:
        if worker.is_running:
            lb.logger.error(f'still running: {str(worker)}')
            return True
    else:
        return False


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
        content-align: center middle;
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


class LogNotifier(logging.Handler):
    def __init__(self, callback: callable, logger_names: list[str], level=logging.INFO):
        self.callback = callback

        super().__init__(level)

        for name in logger_names:
            logger = logging.getLogger(name)
            logger.setLevel(level)
            logger.addHandler(self)

    def emit(self, record):
        self.callback(get_log_append(0, record))


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


class SweepHUDApp(App):
    BINDINGS = [
        ('q', 'request_quit', 'Quit'),
        ('b', 'request_scroll', 'Skip to bottom'),
    ]

    def __init__(self, cli_kws: dict):
        self.cli_objs = None
        self.cli_kws = cli_kws
        super().__init__()
        self.handler = SplitLogHandler(
            self,
            'source',
            'analysis',
            'sink',
            verbose=cli_kws['verbose'],
        )
        self.row_keys: dict[int, str] = {}
        self.column_keys: dict[str, str] = {}

    def compose(self) -> ComposeResult:
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

        yield Header()
        with Center():
            yield ProgressBar(
                show_percentage=False, show_eta=True, gradient=gradient, id='progress'
            )
        yield DataTable(zebra_stripes=True, cursor_type='row')
        yield Footer()

    @textual.work(exclusive=True, thread=True)
    def start_sweep(self):
        gen = frontend.iter_sweep_cli(
            self.cli_objs, remote=self.cli_kws.get('remote', None)
        )
        for _ in gen:
            pass

    def _notify_warnings(self, warn_message, *args, **kws):
        """A custom function to handle warnings."""
        self.notify(warn_message)
        # You can add any custom logic here, like logging, sending notifications, etc.

    @textual.work(exclusive=True, thread=True)
    def initialize(self):
        time.sleep(2)
        self.cli_objs = frontend.init_sweep_cli(**self.cli_kws)
        self.display_fields = nonunique_capture_fields(
            self.cli_objs.sweep_spec.captures
        )
        self.call_from_thread(self._setup_table)
        self.call_from_thread(self.start_sweep)

    def _setup_table(self):
        def col_header(name):
            return Text(name, style='content-align-vertical: bottom')

        table = self.query_one(DataTable)
        index_size = math.ceil(math.log10(len(self.cli_objs.sweep_spec.captures))) + 2
        free_width = self.size.width - index_size

        self.column_keys = {
            'capture': table.add_column('', width=int(free_width * 0.27)),
            'striqt.source': table.add_column(
                col_header('Source'), width=round(free_width * 0.18)
            ),
            'striqt.analysis': table.add_column(
                col_header('Analysis'), width=int(free_width * 0.32)
            ),
            'striqt.sink': table.add_column(
                col_header('Sink'), width=round(free_width * 0.23)
            ),
        }

        table.loading = False

        progress = self.query_one(ProgressBar)
        progress.total = len(self.cli_objs.sweep_spec.captures)

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.loading = True

        self._handler = LogNotifier(
            self.notify,
            logger_names=[
                'striqt.controller',
            ],
        )

        # Set a custom showwarning function to intercept warnings
        warnings.showwarning = self._notify_warnings
        self.initialize()

    def action_request_quit(self) -> None:
        """Action to display the quit dialog."""
        if any_are_running(self.workers):
            self.push_screen(QuitScreen(name='Confirm'))
        else:
            self.exit()

    def action_request_scroll(self) -> None:
        """Action to display the quit dialog."""
        table = self.query_one('DataTable')
        table.action_scroll_bottom()

    def update_table(self, record: logging.LogRecord):
        from striqt.analysis.lib.dataarrays import describe_capture

        captures = self.cli_objs.sweep_spec.captures

        logger = util.get_logger(record.name.rsplit('.', 1)[1])
        extra = logger.extra
        if isinstance(record.args, dict):
            extra = extra | record.args

        capture_index = extra['capture_index']
        if not 0 <= capture_index < len(captures):
            return

        capture = captures[capture_index]

        table = self.query_one(DataTable)
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

    def _fatal_error(self) -> None:
        """Exits the app after an unhandled exception."""
        from rich.traceback import Traceback

        self.bell()
        traceback = Traceback(show_locals=False, width=None, suppress=[rich, lb, xr])
        self._exit_renderables.append(
            Segments(self.console.render(traceback, self.console.options))
        )
        self._close_messages_no_wait()


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
        self.app.call_from_thread(self.app.update_table, record)
