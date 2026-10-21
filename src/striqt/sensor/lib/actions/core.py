"""the action runner base, the `iterate_sweep` dispatcher, and the acquire/analyze/sink pipeline"""

from __future__ import annotations as __

import contextlib
import math
import itertools
from collections.abc import Mapping
from typing import Any, Generator, Protocol, TYPE_CHECKING

import striqt.waveform as sw
import striqt.analysis as sa
from striqt.analysis.lib.util import pd, xr

from .. import compute, util
from ... import specs
from ..resources import Resources, AnyResources
from ..sinks import SinkBase
from ..compute.cal_lookups import lookup_system_noise_power
from ..typing import PC, PS, SC, SP, SS, TA, Unpack


class ActionRunner(Protocol[TA, SS, SP, SC, PS, PC]):
    """the contract between a decoded document and the code that opens and runs it.

    A runner class is registered for one document type with `registry.runner` and
    constructed from the decoded document. `open_resources` opens the source,
    peripherals, and sink for `sweep`, using `sink_cls` unless
    ``sweep.extensions.sink`` names another class, and `iterate_sweep` calls
    `iterate` over the open resources. Each runner implements its own constructor
    and `iterate`; `iterate_captures` is the pipeline they run captures through.
    """

    spec: TA
    sweep: specs.Sweep[SS, SP, SC]
    sink_cls: type[SinkBase[SC]]

    def __init__(self, spec: TA) -> None: ...

    def iterate(
        self,
        resources: Resources[SS, SP, SC, PS, PC],
        *,
        always_yield: bool = False,
        yield_values: bool = True,
        loop: bool = False,
    ) -> Generator[xr.Dataset | compute.DelayedDataset | None]:
        """run the action over `resources`, yielding what the sink's `append` returns.

        Args:
            resources: the open resources returned by `open_resources`
            always_yield: if True, yield `None` on steps that produce no result
            yield_values: if False, yield `None` instead of the sink result
            loop: if True, the action repeats from the beginning after the last capture
        """
        ...


def iterate_sweep(
    resources: Resources[SS, SP, SC, PS, PC],
    *,
    always_yield: bool = False,
    yield_values: bool = True,
    loop: bool = False,
    **replace: Unpack[AnyResources[SS, SP, SC, PS, PC]],
) -> Generator[xr.Dataset | compute.DelayedDataset | None]:
    """run the action of ``resources['runner']`` over `resources`.

    The runner was built by `open_resources` from the decoded document. A plain
    `Sweep` runs `iterate_captures`, whose docstring describes the yields.

    Args:
        resources: dictionary of open resources returned by `open_resources`
        always_yield: if True, yield `None` on steps that produce no result
        yield_values: if False, yield `None` instead of the sink result
        loop: if True, the action repeats from the beginning after the last capture

    Returns:
        the iterator returned by the runner's `iterate`
    """
    runner = resources['runner']
    return runner.iterate(
        Resources(resources, **replace),
        always_yield=always_yield,
        yield_values=yield_values,
        loop=loop,
    )


def iterate_captures(
    resources: Resources[SS, SP, SC, PS, PC],
    *,
    always_yield: bool = False,
    yield_values: bool = True,
    loop: bool = False,
    extra_data: Mapping[str, Any] | None = None,
    indexer: AcquisitionIndexer | None = None,
    **replace: 'Unpack[AnyResources[SS, SP, SC, PS, PC]]',
) -> Generator['xr.Dataset|compute.DelayedDataset|None']:
    """step through the captures of a sweep in the acquire, analyze, and sink pipeline.

    Data acquisition, analysis, and sink operations each run in parallel in
    separate threads. Normally, the iterator yields a result for each
    of the N captures after the sink has appended it:

    ```
    0. `(Acquire 0)`
    1. `Concurrent (Acquire 1, Analyze 0)`.
    2. `Concurrent (Acquire 2, Analyze 1, Sink 0)` ➔ yield (`Result 0)`.
    (...)
    N. `Concurrent (Analyze N-1, Sink N-2)` ➔ yield `(Result N-2)`.
    N+1. `(Sink N-2)` ➔ yield `(Result N-1)`.
    ```

    With the `always_yield` argument, the iterator yields `None` to support
    status information in the 2 steps that do not yield analysis results:

    ```
    0. `(Acquire 0)` ➔ yield `None`
    1. `Concurrent (Acquire 1, Analyze 0)` ➔ yield `(Analysis 0)`.
    2. `Concurrent (Acquire 2, Analyze 1, Sink 0)` ➔ yield `(Analysis 1)`.
    (...)
    N. `Concurrent (Analyze N-1, Sink N-2)` ➔ yield `(Analysis N-1)`.
    N+1. `(Sink N-2)` ➔ yield `None`
    ```

    The type of each yielded result depends on the return value of `sink.append`.
    To minimize memory usage, the yield values can also be explicitly set to
    `None` with `yield_values`. The attrs of each result are the fields of
    ``resources['action_spec']``, the decoded document, which differ from those of
    ``resources['sweep_spec']`` when a runner derived the sweep.

    Args:
        resources: dictionary of open resources returned by open_resources
        always_yield: if True, yield `None` on steps that produce no analysis
        yield_values: if False, yield `None` instead of the sink result
        loop: if True, the sweep repeats from the beginning after the last capture
        extra_data: merged into `AcquiredIQ.extra_data` of every capture after the
            peripherals' data, whose keys it overrides
        indexer: the counter that stamps `capture_index` and `sweep_index` on each
            acquisition; pass one object to several calls to number their captures
            consecutively. A new counter over this sweep's captures when None.

    Returns:
        An iterator of analyzed data or None
    """

    resources = Resources(resources, **replace)

    def log(*args, **kws):
        return _log_cache_info(resources, *args, **kws)

    spec = resources['sweep_spec']

    compute_opts = compute.EvaluationOptions(
        sweep_spec=spec,
        registry=sa.registry,
        extra_attrs=resources['action_spec'].to_dict(True, allow_tuple_keys=False),
        correction=True,
        cache_callback=log,
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
    """number the acquisitions of one logical sweep.

    `iterate_captures` applies one of these to every acquisition it makes. `apply`
    stamps `capture_index` and `sweep_index` on the acquisition info and wraps
    `capture_index` at `captures_per_sweep` into the next `sweep_index`. On a
    `SoapyAcquisitionInfo` it also copies the first capture's `start_time` of each
    sweep to `sweep_start_time`. The only state is the position in the sweep. One
    instance passed to several `iterate_captures` calls numbers their captures as
    one sweep. `iterate_captures` creates an instance over its own captures when none
    is passed. `apply` is not thread-safe and must run once per acquisition, in order.
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
        """return a copy of `info` stamped with the current indexes, and advance."""
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
    res: Resources[Any, Any, SC, PS, PC],
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


def _log_cache_info(
    resources: Resources[SS, SP, SC, PS, PC],
    cache,
    capture: SC,
    result,
    *_,
    **__,
):
    cal = resources['sweep_spec'].source.calibration
    if cal is None or 'spectrogram' not in cache.name:
        return

    spg, attrs = result

    xp = sw.array_namespace(spg)

    if isinstance(capture, specs.SoapyCapture):
        info_fields = ('center_frequency', 'port', 'gain')
    else:
        info_fields = ('port',)

    # converted to dB below
    peaks = spg.max(axis=tuple(range(1, spg.ndim)))
    noise = lookup_system_noise_power(
        cal,
        specs.SoapyCapture.from_spec(capture),
        master_clock_rate=resources['sweep_spec'].source.master_clock_rate,
        format_path=resources['format_path'],
        B=attrs['noise_bandwidth'],
        xp=xp,
    )

    logger = sa.util.get_logger('analysis')
    capture_splits = specs.captures.split_capture_ports(capture)

    for c, snr in zip(capture_splits, sw.powtodB(peaks) - noise):
        if sw.is_cupy_array(snr.data):
            snr = float(snr.data.get())
        else:
            snr = float(snr.values)
        if not abs(snr) < math.inf:
            continue

        snr_desc = f'{round(snr)} dB max SNR'
        capture_desc = specs.sequencing.describe_capture(
            c,
            fields=info_fields,
            source_id=resources['source'].source_id,
            adjust_spec=resources['sweep_spec'].adjust_captures,
        )
        logger.info(f'spectrogram ▮ {snr_desc:<14} ▮ {capture_desc}')
