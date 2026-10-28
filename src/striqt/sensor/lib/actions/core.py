"""the runner contract and the capture pipeline that every action runs.

The acquisition of the next capture and the sink write of the previous one run in
`util.threadpool` while the analysis of the current capture runs in the calling
thread. Exceptions from the 3 stages of one step are collected in an
`ExceptionStack` and raised together at the end of that step.
"""

from __future__ import annotations as __

import contextlib
import itertools
from collections.abc import Mapping
from typing import Any, Generator, Protocol, TYPE_CHECKING

import striqt.analysis as sa
from striqt.analysis.lib.util import pd, xr

from .. import compute, util
from ... import specs
from ..resources import Resources, AnyResources
from ..sinks import SinkBase
from ..typing import PC, PS, SA, SC, SP, SS, TA, Unpack


class ActionRunner(Protocol[TA, SS, SP, SC, SA, PS, PC]):
    """the contract between an action spec and the code that runs the spec."""

    spec: TA
    """the action spec given at construction"""

    def __init__(self, spec: TA) -> None:
        """store `spec`; nothing is opened or derived."""
        ...

    @property
    def sweep(self) -> specs.Sweep[SS, SP, SC, SA]:
        """the `Sweep` derived from `spec`.

        The source, peripherals, and sink are opened for this `Sweep`.

        Raises:
            msgspec.ValidationError: if a field of `spec` is rejected; the error path
                names the field
        """
        ...

    def open_sink(self, format_path: specs.paths.PathFormatter | None) -> SinkBase[SC]:
        """return the sink for `sweep`.

        The caller enters the returned sink. A ``sweep.sink.path`` of None selects a
        sink that writes nothing.

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
            always_yield: if True, yield `None` on steps that produce no result
            yield_values: if False, yield `None` instead of the sink result
            loop: if True, the action repeats from the beginning after the last capture

        Yields:
            what `sink.append` returned for each capture, or None as `always_yield`
            and `yield_values` select
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
    """run the action in `resources` using `iterate_captures`.

    Args:
        resources: dictionary of open resources returned by `open_resources`
        always_yield: if True, yield `None` on steps that produce no result
        yield_values: if False, yield `None` instead of the sink result
        loop: if True, the action repeats from the beginning after the last capture
        **replace: resources that replace the entries of `resources` for this run

    Yields:
        the return value from each append to the sink, or None
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
    **replace: 'Unpack[AnyResources[SS, SP, SC, SA, PS, PC]]',
) -> Generator['xr.Dataset|compute.DelayedDataset|None']:
    """step through the captures of a sweep in the acquire, analyze, and sink pipeline.

    Acquisition, analysis, and the sink write of 3 consecutive captures run in
    parallel. The N captures of the sweep take N + 2 steps, and each step ends with
    one yield:

    ```
    0. `(Acquire 0)` ➔ yield `None`
    1. `Concurrent (Acquire 1, Analyze 0)` ➔ yield `None`
    2. `Concurrent (Acquire 2, Analyze 1, Sink 0)` ➔ yield `(Result 0)`
    (...)
    N. `Concurrent (Analyze N-1, Sink N-2)` ➔ yield `(Result N-2)`
    N+1. `(Sink N-1)` ➔ yield `(Result N-1)`
    ```

    `Result i` is what `sink.append` returned for capture `i`. With `yield_values`
    False, every step yields `None` when `always_yield` is True and the generator
    yields nothing when `always_yield` is False. The attrs of each result are the
    fields of ``resources['action_spec']``. The attrs differ from the fields of
    ``resources['sweep_spec']`` when the runner derived the sweep from the spec.

    The `on_analyzed` and `on_cached` hooks of ``resources['callbacks']`` run in the
    calling thread. An exception from a hook ends the pipeline like an analysis error.

    Args:
        resources: dictionary of open resources returned by `open_resources`
        always_yield: with `yield_values` False, yield `None` at every step
        yield_values: if False, yield nothing, or `None` per step with `always_yield`
        loop: if True, the sweep repeats from the beginning after the last capture
        extra_data: merged into the `extra_data` of every acquisition after the
            peripherals' data, and overrides the peripherals' data on a shared key
        indexer: the counter that stamps `capture_index` and `sweep_index` on each
            acquisition. One object passed to several calls numbers the captures of
            all of those calls consecutively. None creates a counter over the
            captures of this sweep.

    Yields:
        what `sink.append` returned for each capture, or None, as the steps above
        describe
    """

    resources = Resources(resources, **replace)

    callbacks = resources.get('callbacks')
    on_analyzed = None if callbacks is None else callbacks.on_analyzed
    on_cached = None if callbacks is None else callbacks.on_cached

    if on_cached is None:
        cache_callback = None
    else:

        def cache_callback(cache: Any, capture: SC, result: Any, **kws: Any) -> None:
            assert on_cached is not None
            on_cached(resources, cache, capture, result, **kws)

    spec = resources['sweep_spec']

    compute_opts = compute.EvaluationOptions(
        sweep_spec=spec,
        registry=sa.registry,
        extra_attrs=resources['action_spec'].to_dict(True, allow_tuple_keys=False),
        correction=True,
        cache_callback=cache_callback,
        as_xarray='delayed',
        block_each=True,  # data corruption has been observed otherwise
    )

    iq = None
    analysis = None
    captures = specs.sequencing.loop_captures(
        spec, source_id=resources['source'].source_id
    )
    if indexer is None:
        indexer = AcquisitionIndexer(len(captures))

    if len(spec.loops) > 0 and isinstance(spec.loops[0], specs.Repeat):
        captures = spec.loops[0].count * captures

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
    """the counter that numbers the acquisitions of one logical sweep.

    `capture_index` wraps to 0 at `captures_per_sweep`, and `sweep_index` then
    advances. One instance passed to several `iterate_captures` calls numbers their
    captures as one sweep. `apply` is not thread-safe and must run once per
    acquisition, in order.
    """

    def __init__(self, captures_per_sweep: int):
        """
        Args:
            captures_per_sweep: the capture count at which `capture_index` wraps to 0
                and `sweep_index` advances
        """
        self.capture_count = captures_per_sweep
        self.sweep_index: int = 0
        self.capture_index: int = 0
        self.sweep_start_time: 'pd.Timestamp|None' = None

    def apply(self, info: specs.AcquisitionInfo) -> specs.AcquisitionInfo:
        """return a copy of `info` stamped with the current indexes, and advance.

        On a `SoapyAcquisitionInfo`, `sweep_start_time` is set to the `start_time`
        of the first capture of the current sweep.
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

    The dict returned by the peripherals, then `extra_data`, are merged into the
    `extra_data` of the returned acquisition struct in that order.
    """

    assert this is not None

    def arm_both(c: SC | None):
        if c is None:
            return

        source = util.threadpool.submit(res['source']._arm_spec, c)
        peripherals = util.threadpool.submit(res['peripherals'].arm, c)
        util.await_and_ignore([source, peripherals], 'arm sensor')

    # the previous call armed `this` as its `next_`, except for the first capture of
    # a call or when `this` differs from what a previous pipeline run left armed
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
