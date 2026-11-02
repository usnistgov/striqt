"""the runner contract and the capture pipeline that every action runs.

A dedicated thread acquires the captures and feeds a bounded `AcquisitionQueue`.
The calling thread takes one acquisition per step, analyzes the acquisition taken
in the previous step, and submits the sink write of the analysis before that to
`util.threadpool`. Exceptions from the 3 stages of one step are collected in an
`ExceptionStack` and raised together at the end of that step.
"""

from __future__ import annotations as __

import contextlib
import itertools
import queue as queue_lib
import threading
from collections.abc import Generator, Iterable, Mapping
from typing import TYPE_CHECKING, Any, Protocol

import striqt.analysis as sa
from striqt.analysis.lib.util import pd, xr

from ... import specs
from .. import compute, util
from ..compute import corrections
from ..controller import AcquisitionQueue
from ..resources import AnyResources, Resources
from ..sinks import SinkBase
from ..typing import PC, PS, SA, SC, SP, SS, TA, Unpack

if TYPE_CHECKING:
    from ..typing import Array


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
            the return value of ``sink.append`` for each delivered acquisition, or
            None at the steps before the first sink write; with `yield_values`
            False, None at every step or nothing as `always_yield` selects. The
            iteration never ends when `loop` is True.
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
        the return value of ``sink.append`` for each delivered acquisition, or None
        at the steps before the first sink write; with `yield_values` False, None at
        every step or nothing as `always_yield` selects. The iteration never ends
        when `loop` is True.
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

    An acquisition thread arms each capture and acquires the `burst` members of the
    capture into a queue of ``source.bursts.buffers`` acquisitions. The calling
    thread takes one acquisition per step, and the N delivered acquisitions take
    N + 2 steps. Each step ends with one yield:

    ```
    0. `(Take 0)` ➔ yield `None`
    1. `Concurrent (Take 1, Analyze 0)` ➔ yield `None`
    2. `Concurrent (Take 2, Analyze 1, Sink 0)` ➔ yield `(Result 0)`
    (...)
    N. `Concurrent (Analyze N-1, Sink N-2)` ➔ yield `(Result N-2)`
    N+1. `(Sink N-1)` ➔ yield `(Result N-1)`
    ```

    `Result i` is the return value of ``sink.append`` for acquisition `i`. With
    `yield_values` False and `always_yield` True, every step yields None. With both
    False, the generator yields nothing. The attrs of each result are the fields of
    ``resources['action_spec']``.

    A burst member acquired while the queue is full is dropped under
    ``source.bursts.on_gap='skip'`` and raises `AcquisitionOverrunError` under
    ``'except'``. Under ``bursts.trigger_on='first'``, the signal trigger runs on
    the first delivered member of each burst and the shifts apply to the rest.

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

    Raises:
        AcquisitionOverrunError: if a burst member finds the queue full under
            ``source.bursts.on_gap='except'``
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
    sink = resources['sink']

    compute_opts = compute.EvaluationOptions(
        sweep_spec=spec,
        registry=sa.registry,
        extra_attrs=resources['action_spec'].to_dict(True, allow_tuple_keys=False),
        cache_callback=cache_callback,
        as_xarray='delayed',
        block_each=True,  # data corruption has been observed otherwise
    )

    source_id = resources['source'].source_id
    captures = specs.sequencing.repeat_captures(spec, source_id=source_id)
    if indexer is None:
        # `sweep_index` advances once per pass of a leading Repeat
        plan = specs.sequencing.loop_captures(spec, source_id=source_id)
        indexer = AcquisitionIndexer(len(plan))

    if len(captures) == 0:
        return

    capture_iter: Iterable[SC]
    count: float
    if loop:
        capture_iter = itertools.cycle(captures)
        count = float('inf')
    else:
        capture_iter = captures
        count = specs.sequencing.count_acquisitions(spec, source_id)

    queue = AcquisitionQueue(spec.source.bursts.buffers)
    stop = threading.Event()
    worker = threading.Thread(
        target=acquisition_worker,
        args=(resources, capture_iter, queue, indexer, extra_data, stop),
        kwargs={'acquisition_count': count},
        name='striqt-acquire',
        daemon=True,
    )

    def sink_append(analysis: compute.DelayedDataset, index: int) -> Any:
        with util.log_capture_context('sink', capture_index=index, capture_count=count):
            return sink.append(analysis)

    shifts = _BurstShifts(spec.source.bursts.trigger_on)
    exc = util.ExceptionStack('threaded failure')
    iq: specs.AcquiredIQ | None = None
    analysis: compute.DelayedDataset | None = None
    ended = False

    worker.start()
    try:
        for i in itertools.count():
            with exc:
                if analysis is None:
                    sink_future = None  # first 2 iterations
                else:
                    sink_future = util.threadpool.submit(sink_append, analysis, i - 2)

                with exc.defer():
                    if iq is None:
                        analysis = None  # first and last iterations
                    else:
                        # cupy/cuda analysis must run in the foreground thread
                        with util.log_capture_context(
                            'analysis', capture_index=i - 1, capture_count=count
                        ):
                            analysis = compute.analyze(iq, compute_opts)
                            assert isinstance(analysis, compute.DelayedDataset)
                            if on_analyzed is not None:
                                on_analyzed(analysis)

                with exc.defer():
                    if ended:
                        iq = None
                    else:
                        iq = _take(queue)
                        if iq is None:
                            ended = True
                        else:
                            shifts.apply(iq)

                sink_result = None
                with exc.defer():
                    if sink_future is not None:
                        sink_result = sink_future.result()

            if yield_values:
                yield sink_result
            elif always_yield:
                yield None

            util.propagate_thread_interrupts()

            if ended and iq is None and analysis is None:
                break
    finally:
        stop.set()
        while worker.is_alive():
            _drain(queue)
            worker.join(timeout=0.05)


class _End:
    """the item that the acquisition thread puts after the last acquisition"""


END = _End()


def _take(queue: AcquisitionQueue) -> specs.AcquiredIQ | None:
    """return the next acquisition, or None at the end of the acquisitions.

    Raises:
        BaseException: the exception that ended the acquisition thread
    """
    item = queue.get()
    if isinstance(item, _End):
        return None
    if isinstance(item, BaseException):
        raise item
    assert isinstance(item, specs.AcquiredIQ)
    return item


def _drain(queue: AcquisitionQueue) -> None:
    while True:
        try:
            queue.get_nowait()
        except queue_lib.Empty:
            return


class _BurstShifts:
    """the alignment shifts shared by the members of the burst under analysis.

    Under ``trigger_on='first'``, `apply` runs the trigger on the first delivered
    member of a burst with `burst` above 1 and sets `trigger_shifts` on every
    member of that burst. Under ``'all'`` nothing is set.
    """

    def __init__(self, trigger_on: str):
        self.trigger_on = trigger_on
        self.key: tuple[int, int] | None = None
        self.shifts: Array | None = None

    def apply(self, iq: specs.AcquiredIQ) -> None:
        if self.trigger_on != 'first':
            return
        if not isinstance(iq.capture, specs.SensorCapture) or iq.capture.burst == 1:
            return
        if corrections._trigger_of(iq) is None:
            return
        key = (iq.info.sweep_index, iq.info.capture_index)
        if key != self.key or iq.info.burst_index == 0:
            self.key = key
            self.shifts = corrections.trigger_shifts(iq)
        iq.trigger_shifts = self.shifts


def acquisition_worker(
    resources: Resources[SS, SP, SC, SA, PS, PC],
    captures: Iterable[SC],
    queue: AcquisitionQueue,
    indexer: AcquisitionIndexer,
    extra_data: Mapping[str, Any] | None,
    stop: threading.Event,
    *,
    acquisition_count: float = float('inf'),
) -> None:
    """arm and acquire `captures` in order into `queue`.

    Each capture is armed on the source and the peripherals, and the `burst` members
    are acquired with `Controller.acquire_burst`. The peripherals acquire once per
    capture, concurrently with member 0, and the result is merged into the
    `extra_data` of every member before the entries of `extra_data`; a later source
    wins on a shared key. `indexer` stamps each member. After the last capture the
    source is disarmed and `END` is put. An exception ends the acquisitions and is
    put in place of `END`.

    Args:
        resources: the open resources
        captures: the captures to acquire, finite or endless
        queue: the queue that receives the members, `END`, or the exception
        indexer: the counter applied to the `info` of each member
        extra_data: entries merged into every member, or None
        stop: set by the consumer to end the acquisitions after the current capture
        acquisition_count: the total number of acquisitions in the progress fields
            of the ``source`` logger, or ``inf`` for endless captures
    """
    ordinal = 0
    try:
        for capture in captures:
            if stop.is_set():
                break
            util.propagate_thread_interrupts()
            ordinal = _acquire_capture(
                resources,
                capture,
                queue,
                indexer,
                extra_data,
                ordinal,
                acquisition_count,
            )
        resources['source'].disarm()
        queue.put(END)
    except BaseException as ex:
        queue.put(ex)


def _acquire_capture(
    resources: Resources[SS, SP, SC, SA, PS, PC],
    capture: SC,
    queue: AcquisitionQueue,
    indexer: AcquisitionIndexer,
    extra_data: Mapping[str, Any] | None,
    ordinal: int,
    acquisition_count: float,
) -> int:
    """arm `capture` and acquire its burst into `queue`.

    Returns:
        `ordinal` advanced by the number of members that passed through the
        finalizer
    """
    source = resources['source']
    peripherals = resources['peripherals']

    def log_context() -> contextlib.AbstractContextManager[None]:
        return util.log_capture_context(
            'source', capture_index=ordinal, capture_count=acquisition_count
        )

    with contextlib.ExitStack() as stack:
        stack.enter_context(log_context())
        arm = [
            util.threadpool.submit(source._arm_spec, capture),
            util.threadpool.submit(peripherals.arm, capture),
        ]
        util.await_and_ignore(arm, 'arm sensor')
        periph = util.threadpool.submit(peripherals.acquire, capture)

        def finalize(iq: specs.AcquiredIQ) -> specs.AcquiredIQ:
            nonlocal ordinal
            ext_data = periph.result()
            if not isinstance(ext_data, dict):
                raise TypeError(
                    f'peripheral acquire() returned {type(ext_data)!r}, not dict'
                )
            iq.info = indexer.apply(iq.info, capture)
            iq.extra_data = iq.extra_data | ext_data
            if extra_data is not None:
                iq.extra_data = iq.extra_data | dict(extra_data)
            ordinal += 1
            stack.close()
            stack.enter_context(log_context())
            return iq

        source.acquire_burst(queue=queue, finalize=finalize)

    return ordinal


class AcquisitionIndexer:
    """the counter that stamps `capture_index` and `sweep_index` on the acquisitions
    of one sweep.

    `capture_index` advances after the last burst member of a capture and wraps to
    0 at `captures_per_sweep`, where `sweep_index` advances. `apply` is not
    thread-safe and must run once per acquisition, in order.
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

    def apply(
        self, info: specs.AcquisitionInfo, capture: specs.SensorCapture
    ) -> specs.AcquisitionInfo:
        """return a copy of `info` with `sweep_index` and `capture_index` set.

        The counter advances by one capture when ``info.burst_index`` is the last
        member of `capture`. On a `SoapyAcquisitionInfo`, `sweep_start_time` is set
        to the `start_time` of the first member of the first capture of the current
        sweep.
        """
        result = info.replace(
            sweep_index=self.sweep_index, capture_index=self.capture_index
        )

        if isinstance(info, specs.SoapyAcquisitionInfo):
            if self.capture_index == 0 and info.burst_index == 0:
                self.sweep_start_time = info.start_time

            result = result.replace(sweep_start_time=self.sweep_start_time)

        if info.burst_index != capture.burst - 1:
            return result

        self.capture_index += 1

        if self.capture_index == self.capture_count:
            self.capture_index = 0
            self.sweep_index += 1

        return result
