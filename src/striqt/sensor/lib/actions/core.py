"""the action runner base, the `iterate_sweep` dispatcher, and the acquire/analyze/sink pipeline"""

from __future__ import annotations as __

import contextlib
import math
import itertools
from typing import Any, Generator, Generic, TYPE_CHECKING

import striqt.waveform as sw
import striqt.analysis as sa
from striqt.analysis.lib.util import pd, xr

from .. import compute, util
from ..compute import gpu
from ... import specs
from ..resources import Resources, AnyResources
from ..calibration import lookup_system_noise_power
from ..typing import PC, PS, SC, SP, SS, Unpack
from .registry import registry

if TYPE_CHECKING:
    from ..bindings import Sensor


class ActionRunner(Generic[SS, SP, SC, PS, PC]):
    """base class of the actions that `iterate_sweep` runs over open resources.

    A subclass is registered for one action spec type with `registry.runner`.
    `iterate_sweep` instantiates the runner registered for the type of
    ``resources['sweep_spec']`` or the nearest of its bases. `warmup_spec` is
    consulted by `prepare_compute` before the resources open. The `adapt_sensor` and
    `adapt_schema` hooks run once at bind time. Each receives the `Sensor` or
    `Schema` a binding is built from and returns the version the action requires.
    The base implementations return their argument unchanged.
    """

    resources: Resources[SS, SP, SC, PS, PC]

    @classmethod
    def adapt_sensor(cls, sensor: Sensor[SS, SP, SC]) -> Sensor[SS, SP, SC]:
        """return the `Sensor` to bind for this action, derived from `sensor`.

        A subclass replaces the implementation classes that the action requires,
        such as the sink or peripherals.
        """
        return sensor

    @classmethod
    def adapt_schema(
        cls, schema: specs.Schema[SS, SP, SC, PS, PC]
    ) -> specs.Schema[SS, SP, SC, PS, PC]:
        """return the `Schema` to bind for this action, derived from `schema`.

        A subclass replaces the spec types that the action requires, such as a
        capture type with an added field.
        """
        return schema

    @classmethod
    def warmup_spec(cls, spec: specs.Sweep[Any, Any, Any]) -> specs.Warmup | None:
        """return the `Warmup` to run before `spec`, or None when none is needed.

        None when `spec.options.skip_warmup` is set or the sweep does not use the
        GPU (`compute.gpu.sweep_touches_gpu`).
        """
        if spec.options.skip_warmup or not gpu.sweep_touches_gpu(spec):
            return None

        from . import warmup

        return warmup.derive_warmup_spec(spec)

    def __init__(
        self,
        resources: Resources[SS, SP, SC, PS, PC],
        **replace: Unpack[AnyResources[SS, SP, SC, PS, PC]],
    ):
        self.resources = Resources(resources, **replace)

    def iterate(
        self,
        *,
        always_yield: bool = False,
        yield_values: bool = True,
        loop: bool = False,
    ) -> Generator[xr.Dataset | compute.DelayedDataset | None]:
        """run the action and yield what the sink's `append` returns for each capture.

        Args:
            always_yield: if True, yield `None` on steps that produce no result
            yield_values: if False, yield `None` instead of the sink result
            loop: if True, the action repeats from the beginning after the last capture
        """
        raise NotImplementedError

    def __iter__(self) -> Generator[xr.Dataset | compute.DelayedDataset | None]:
        return self.iterate()


def iterate_sweep(
    resources: Resources[SS, SP, SC, PS, PC],
    *,
    always_yield: bool = False,
    yield_values: bool = True,
    loop: bool = False,
    **replace: Unpack[AnyResources[SS, SP, SC, PS, PC]],
) -> Generator[xr.Dataset | compute.DelayedDataset | None]:
    """run the action registered for the type of ``resources['sweep_spec']``.

    The `ActionRunner` subclass registered for the spec type, or the nearest of its
    bases, is instantiated with `resources` updated by `replace` and iterated. A
    plain `Sweep` runs `iterate_captures`, whose docstring describes the yields.

    Args:
        resources: dictionary of open resources returned by `open_resources`
        always_yield: if True, yield `None` on steps that produce no result
        yield_values: if False, yield `None` instead of the sink result
        loop: if True, the action repeats from the beginning after the last capture

    Returns:
        the iterator returned by the runner's `iterate`

    Raises:
        TypeError: if no runner is registered for the spec type or any of its bases
    """
    runner_cls = registry.lookup(resources['sweep_spec'])
    runner = runner_cls(resources, **replace)
    return runner.iterate(
        always_yield=always_yield, yield_values=yield_values, loop=loop
    )


def iterate_captures(
    resources: Resources[SS, SP, SC, PS, PC],
    *,
    always_yield: bool = False,
    yield_values: bool = True,
    loop: bool = False,
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
    `None` with `yield_values`.

    Args:
        resources: dictionary of open resources returned by open_resources
        always_yield: if True, yield `None` on steps that produce no analysis
        yield_values: if False, yield `None` instead of the sink result
        loop: if True, the sweep repeats from the beginning after the last capture

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
        extra_attrs=spec.to_dict(True, allow_tuple_keys=False),
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
    indexer = _AcquisitionIndexer(len(captures))

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
                    _acquire_both, resources, this, next_, indexer
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


class _AcquisitionIndexer:
    def __init__(self, captures_per_sweep: int):
        self.capture_count = captures_per_sweep
        self.sweep_index: int = 0
        self.capture_index: int = 0
        self.sweep_start_time: 'pd.Timestamp|None' = None

    def apply(self, info: specs.AcquisitionInfo) -> specs.AcquisitionInfo:
        """return a copy of `info` with updated indexing variables"""
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
    indexer: _AcquisitionIndexer,
) -> specs.AcquiredIQ:
    """arm and acquire from the source and peripherals.

    Any acquired data returned from the peripherals is merged
    into the `extra_data` of the returned acquisition struct.
    """

    assert this is not None

    def arm_both(c: SC | None):
        if c is None:
            return

        source = util.threadpool.submit(res['source']._arm_spec, c)
        peripherals = util.threadpool.submit(res['peripherals'].arm, c)
        util.await_and_ignore([source, peripherals], 'arm sensor')

    try:
        res['source'].capture_spec
    except AttributeError:
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
