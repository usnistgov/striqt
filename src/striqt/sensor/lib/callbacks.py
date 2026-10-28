"""the hooks a front end registers with `open_resources` for one run of an action."""

from __future__ import annotations as __

import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .typing import AnalyzedCallback, SourceOpenCallback


@dataclasses.dataclass(frozen=True)
class SweepCallbacks:
    """the callbacks of one run, invoked by `open_resources` and `iterate_captures`.

    Each field is a tuple of callables invoked in order. `on_source_opened` receives
    ``(sweep, source_id)`` from a worker thread once the source is open and before
    the first capture. `on_analyzed` receives the `DelayedDataset` of each capture in
    the analysis thread, after the analysis of that capture and before the sink
    receives the dataset. An `on_analyzed` callback must return quickly. An exception
    from any callback ends the run.
    """

    on_source_opened: tuple[SourceOpenCallback, ...] = ()
    on_analyzed: tuple[AnalyzedCallback, ...] = ()
