"""the runner contract and the capture pipeline that every action runs.

The acquisition of the next capture and the sink write of the previous one run in
`util.threadpool` while the analysis of the current capture runs in the calling
thread. Exceptions from the 3 stages of one step are collected in an
`ExceptionStack` and raised together at the end of that step.
"""

from __future__ import annotations as __

import contextlib
import itertools
from collections.abc import Generator, Mapping
from typing import Any, Protocol

import striqt.analysis as sa
from striqt.analysis.lib.util import pd, xr

from ... import specs
from .. import compute, util
from ..resources import AnyResources, Resources
from ..sinks import SinkBase
from ..typing import PC, PS, SA, SC, SP, SS, TA, Unpack


class ActionRunner(Protocol[TA, SS, SP, SC, SA, PS, PC]):
    """the role of the object that runs one action spec: derive the `Sweep`, open the
    sink, and iterate the captures."""

    spec: TA
    """the action spec given at construction"""

    def __init__(self, spec: TA) -> None:
        """store `spec`; nothing is opened or derived."""
        ...

    @property
    def sweep(self) -> specs.Sweep[SS, SP, SC, SA]:
        """the `Sweep` derived from `spec`, for which the source, peripherals, and sink
        are opened.

        Raises:
            msgspec.ValidationError: if a field of `spec` is rejected
        """
        ...

    def open_sink(self, format_path: specs.paths.PathFormatter | None) -> SinkBase[SC]:
        """return the sink for `sweep`, not yet entered.

        A ``sweep.sink.path`` of None selects a sink that writes nothing.

        Args:
            format_path: the formatter of the ``{field}`` templates in the sink path,
                or None to use the path as written
        """
        ...

    def iterate(
        self,
        resources: Resources[SS, SP, SC, SA, PS, PC],
        *,
        always_yield: bool = False,
        yield_values: bool = True,
        loop: bool = False,
    ) -> Generator[xr.Dataset | compute.DelayedDataset | None]:
        """run the action over `resources`.

        An implementation runs `iterate_captures` over `resources`.

        Args:
            resources: the open resources
            always_yield: with `yield_values` False, True yields None at every step
                and False yields nothing
            yield_values: True yields the sink result of every step
            loop: True cycles through the captures from the first after the last

        Yields:
            the return value of ``sink.append`` for each capture, or None at the
            steps before the first sink write; with `yield_values` False, None at
            every step or nothing as `always_yield` selects. The iteration never
            ends when `loop` is True.
        """
        ...


def iterate_sweep(
    resources: Resources[SS, SP, SC, SA, PS, PC],
    *,
    always_yield: bool = False,
    yield_values: bool = True,
    loop: bool = False,
    **replace: Unpack[AnyResources[SS, SP, SC, SA, PS, PC]],
) -> Generator[xr.Dataset | compute.DelayedDataset | None]:
    """run the `iterate` of ``resources['runner']`` over `resources`.

    Args:
        resources: the open resources of `open_resources`
        always_yield: with `yield_values` False, True yields None at every step and
            False yields nothing
        yield_values: True yields the sink result of every step
        loop: True cycles through the captures from the first after the last
        **replace: entries that replace those of `resources` for this run

    Yields:
        the return value of ``sink.append`` for each capture, or None at the steps
        before the first sink write; with `yield_values` False, None at every step
        or nothing as `always_yield` selects. The iteration never ends when `loop`
        is True.
    """
    runner = resources['runner']
    return runner.iterate(
        Resources(resources, **replace),
        always_yield=always_yield,
        yield_values=yield_values,
        loop=loop,
    )


def iterate_captures(
    resources: Resources[SS, SP, SC, SA, PS, PC],
    *,
    always_yield: bool = False,
    yield_values: bool = True,
    loop: bool = False,
    extra_data: Mapping[str, Any] | None = None,
    indexer: AcquisitionIndexer | None = None,
    **replace: Unpack[AnyResources[SS, SP, SC, SA, PS, PC]],
) -> Generator[xr.Dataset | compute.DelayedDataset | None]:
    """run the captures of ``resources['sweep_spec']`` through the acquire, analyze, and
    sink pipeline.

    Acquisition, analysis, and the sink write of 3 consecutive captures run
    concurrently. The N captures of the sweep take N + 2 steps, and each step ends
    with one yield:

    ```
    0. `(Acquire 0)` ➔ yield `None`
    1. `Concurrent (Acquire 1, Analyze 0)` ➔ yield `None`
    2. `Concurrent (Acquire 2, Analyze 1, Sink 0)` ➔ yield `(Result 0)`
    (...)
    N. `Concurrent (Analyze N-1, Sink N-2)` ➔ yield `(Result N-2)`
    N+1. `(Sink N-1)` ➔ yield `(Result N-1)`
    ```

    `Result i` is the return value of ``sink.append`` for capture `i`. With
    `yield_values` False and `always_yield` True, every step yields None. With both
    False, the generator yields nothing. The attrs of each result are the fields of
    ``resources['action_spec']``.

    The captures are the loop expansion of the sweep for the open source. A `Repeat`
    first in `loops` runs the expanded captures `count` times. One `indexer` shared
    by several calls numbers the captures of all of those calls consecutively.

    The `on_analyzed` and `on_sink_cached` hooks of ``resources['callbacks']`` run in
    the calling thread. An exception from a hook ends the pipeline.

    Args:
        resources: the open resources of `open_resources`
        always_yield: with `yield_values` False, True yields None at every step and
            False yields nothing
        yield_values: True yields the sink result of every step
        loop: True cycles through the captures from the first after the last
        extra_data: entries merged into the `extra_data` of every acquisition after
            the data of the peripherals; a shared key takes the value from here
        indexer: the counter that stamps `capture_index` and `sweep_index` on each
            acquisition, or None for a counter over the captures of this sweep
        **replace: entries that replace those of `resources` for this run

    Yields:
        None at steps 0 and 1 and then `Result i` at step ``i + 2``; or None at every
        step; or nothing, as `yield_values` and `always_yield` select. The iteration
        ends after step N + 1, or never when `loop` is True.
    """

    resources = Resources(resources, **replace)

    callbacks = resources.get('callbacks')
    on_analyzed = None if callbacks is None else callbacks.on_analyzed
    on_sink_cached = None if callbacks is None else callbacks.on_sink_cached

    if on_sink_cached is None:
        cache_callback = None
    else:

        def cache_callback(cache: Any, capture: SC, result: Any, **kws: Any) -> None:
            assert on_sink_cached is not None
            on_sink_cached(resources, cache, capture, result, **kws)

    spec = resources['sweep_spec']

    compute_opts = compute.EvaluationOptions(
        sweep_spec=spec,
        registry=sa.registry,
        extra_attrs=resources['action_spec'].to_dict(True, allow_tuple_keys=False),
        cache_callback=cache_callback,
        as_xarray='delayed',
        block_each=True,  # data corruption has been observed otherwise
    )

    iq = None
    analysis = None
    source_id = resources['source'].source_id
    captures = specs.sequencing.repeat_captures(spec, source_id=source_id)
    if indexer is None:
        # `sweep_index` advances once per pass of a leading Repeat
        plan = specs.sequencing.loop_captures(spec, source_id=source_id)
        indexer = AcquisitionIndexer(len(plan))

    if loop:
        capture_iter = itertools.cycle(captures)
        count = float('inf')
    else:
        capture_iter = captures
        count = len(captures)

    if count == 0:
        return

    # iterate across (previous-1, previous, current, next) captures to support concurrency
    offset_captures = util.zip_offsets(capture_iter, (-2, -1, 0, 1), fill=None)
    exc = util.ExceptionStack('threaded failure')

    for i, (_, _, this, next_) in enumerate(offset_captures):
        with _log_progress_contexts(i, count), exc:
            if this is None:
                acquire = None  # last 2 iterations
            else:
                acquire = util.threadpool.submit(
                    _acquire_both, resources, this, next_, indexer, extra_data
                )

            if analysis is None:
                sink = None  # first 2 iterations
            else:
                sink = util.threadpool.submit(resources['sink'].append, analysis)

            with exc.defer():
                if iq is None:
                    analysis = None  # first and last iterations
                else:
                    # cupy/cuda analysis must run in the foreground thread
                    analysis = compute.analyze(iq, compute_opts)
                    assert isinstance(analysis, compute.DelayedDataset)
                    if on_analyzed is not None:
                        on_analyzed(analysis)

            with exc.defer():
                if acquire is not None:
                    iq = acquire.result()
                    assert isinstance(iq, specs.AcquiredIQ)
                else:
                    iq = None

            with exc.defer():
                if sink is not None:
                    sink = sink.result()

        if yield_values:
            yield sink
        elif always_yield:
            yield None

        util.propagate_thread_interrupts()


@contextlib.contextmanager
def _log_progress_contexts(index, count):
    """set the log context information for reporting progress"""

    contexts = (
        util.log_capture_context('source', capture_index=index, capture_count=count),
        util.log_capture_context(
            'analysis',
            capture_index=index - 1,
            capture_count=count,
        ),
        util.log_capture_context(
            'sink',
            capture_index=index - 2,
            capture_count=count,
        ),
    )

    cm = contextlib.ExitStack()

    with cm:
        try:
            for ctx in contexts:
                cm.enter_context(ctx)
            yield cm
        except:
            cm.close()
            raise


class AcquisitionIndexer:
    """the counter that stamps `capture_index` and `sweep_index` on the acquisitions
    of one sweep.

    `capture_index` wraps to 0 at `captures_per_sweep`, and `sweep_index` then
    advances. `apply` is not thread-safe and must run once per acquisition, in order.
    """

    def __init__(self, captures_per_sweep: int):
        """start both indexes at 0.

        Args:
            captures_per_sweep: the capture count at which `capture_index` wraps to 0
                and `sweep_index` advances
        """
        self.capture_count = captures_per_sweep
        self.sweep_index: int = 0
        self.capture_index: int = 0
        self.sweep_start_time: pd.Timestamp | None = None

    def apply(self, info: specs.AcquisitionInfo) -> specs.AcquisitionInfo:
        """return a copy of `info` with `sweep_index` and `capture_index` set.

        The counter advances by one capture. On a `SoapyAcquisitionInfo`,
        `sweep_start_time` is set to the `start_time` of the first capture of the
        current sweep.
        """
        result = info.replace(
            sweep_index=self.sweep_index, capture_index=self.capture_index
        )

        if isinstance(info, specs.SoapyAcquisitionInfo):
            if self.capture_index == 0:
                self.sweep_start_time = info.start_time

            result = result.replace(sweep_start_time=self.sweep_start_time)

        self.capture_index += 1

        if self.capture_index == self.capture_count:
            self.capture_index = 0
            self.sweep_index += 1

        return result


@sa.util.stopwatch('acquire', 'sweep', threshold=0.25)
def _acquire_both(
    res: Resources[Any, Any, SC, SA, PS, PC],
    this: SC,
    next_: SC | None,
    indexer: AcquisitionIndexer,
    extra_data: Mapping[str, Any] | None = None,
) -> specs.AcquiredIQ:
    """arm and acquire from the source and peripherals.

    The `extra_data` of the result merges the entries of the acquisition, of the
    peripherals, and of `extra_data`; a later source overrides an earlier one on a
    shared key.
    """

    assert this is not None

    def arm_both(c: SC | None):
        if c is None:
            return

        source = util.threadpool.submit(res['source']._arm_spec, c)
        peripherals = util.threadpool.submit(res['peripherals'].arm, c)
        util.await_and_ignore([source, peripherals], 'arm sensor')

    # the previous call armed `this` as its `next_`; re-arm only for the first capture
    # of a run, or when another run left a different capture armed
    try:
        armed = res['source'].capture_spec
    except AttributeError:
        armed = None
    if armed != this:
        arm_both(this)

    iq = util.threadpool.submit(res['source'].acquire)
    ext_data = util.threadpool.submit(res['peripherals'].acquire, this)

    with util.ExceptionStack('failed to acquire data') as exc:
        with exc.defer():
            iq = iq.result()
        with exc.defer():
            ext_data = ext_data.result()

    if not isinstance(ext_data, dict):
        raise TypeError(f'peripheral acquire() returned {type(ext_data)!r}, not dict')
    if not isinstance(iq, specs.AcquiredIQ):
        raise TypeError(f'source acquire() returned {type(iq)!r}, not AcquiredIQ')

    arm_both(next_)

    iq.info = indexer.apply(iq.info)
    iq.extra_data = iq.extra_data | ext_data
    if extra_data is not None:
        iq.extra_data = iq.extra_data | dict(extra_data)
    return iq
