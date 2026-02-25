"""implement sensor anlysis sweeps as parallel (acquisition, analysis, sink)"""

from __future__ import annotations as __

import contextlib
import dataclasses
import math
import itertools
import typing

import striqt.waveform as sw
import striqt.analysis as sa

from . import compute, sources, util
from .. import specs
from .resources import Resources, AnyResources
from .calibration import lookup_system_noise_power
from .sources import _PS, _PC
from ..specs import _TC, _TP, _TS

if typing.TYPE_CHECKING:
    from typing_extensions import Unpack
    import xarray as xr
    import pandas as pd
else:
    xr = util.lazy_import('xarray')


def iterate_sweep(
    resources: Resources[_TS, _TP, _TC, _PS, _PC],
    *,
    always_yield: bool = False,
    yield_values: bool = True,
    loop: bool = False,
    **replace: 'Unpack[AnyResources[_TS, _TP, _TC, _PS, _PC]]',
) -> typing.Generator['xr.Dataset|compute.DelayedDataset|None']:
    """an iterator that steps through the execution of a sensor sweep.

    Data acquisition, analysis, and sink operations each run in parallel in
    separate threads. Normally, the iterator yields a result for each
    of the N captures after it is handled by the sink:

    ```
    0. `(Acquire 0)`
    1. `Concurrent (Acquire 1, Analyze 0)`.
    2. `Concurrent (Acquire 2, Analyze 1, Sink 0)` ➔ yield (`Result 0)`.
    (...)
    N. `Concurrent (Analyze N-1, Sink N-2)` ➔ yield `(Result N-2)`.
    N+1. `(Sink N-2)` ➔ yield `(Result N-1)`.
    ```

    With the `always_yield` argument, the iterator yields `None` to support
    status information in the 2 steps that do not yield anlaysis results:

    ```
    0. `(Acquire 0)` ➔ yield `None`
    1. `Concurrent (Acquire 1, Analyze 0)` ➔ yield `(Analysis 0)`.
    2. `Concurrent (Acquire 2, Analyze 1, Sink 0)` ➔ yield `(Analysis 1)`.
    (...)
    N. `Concurrent (Analyze N-1, Sink N-2)` ➔ yield `(Analysis N-1)`.
    N+1. `(Sink N-2)` ➔ yield `None`
    ```

    The type of each yielded result depends on the return value of `sink.append`.
    To to minimize memory usage, the yield values can also be explicitly set to
    `None` with `yield_values`.

    Args:
        resources: dictionary of open resources returned by open_resources
        always_yield: if True, yield `None` on steps that produce no analysis
        always_values: if False, yield will return `None` instead of the sink result
        loop: if True, the sweep will repeat at the beginning after last capture

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
        extra_attrs=compute.build_dataset_attrs(spec),
        correction=True,
        cache_callback=log,
        as_xarray='delayed',
        block_each=False,
    )

    iq = None
    analysis = None
    captures = specs.helpers.loop_captures(spec, source_id=resources['source'].id)
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
                    # cupy/cuda wants this in the foreground
                    analysis = compute.analyze(iq, compute_opts)
                    assert isinstance(analysis, compute.DelayedDataset)

            with exc.defer():
                if acquire is not None:
                    iq = acquire.result()
                    assert isinstance(iq, sources.AcquiredIQ)
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
    res: Resources[typing.Any, typing.Any, _TC, _PS, _PC],
    this: _TC,
    next_: _TC | None,
    indexer: _AcquisitionIndexer,
) -> sources.AcquiredIQ:
    """arm and acquire from the source and peripherals.

    Any acquired data returned from the peripherals is merged
    into the `extra_data` of the returned acquisition struct.
    """

    assert this is not None

    def arm_both(c: _TC | None):
        if c is None:
            return

        source = util.threadpool.submit(res['source'].arm_spec, c)
        peripherals = util.threadpool.submit(res['peripherals'].arm, c)
        util.await_and_ignore([source, peripherals], 'arm sensor')

    try:
        res['source'].capture_spec
    except AttributeError:
        arm_both(this)

    analysis = specs.helpers.adjust_analysis(
        res['sweep_spec'].analysis, this.adjust_analysis
    )

    iq = util.threadpool.submit(
        res['source'].acquire,
        analysis=analysis,
        correction=False,
        alias_func=res['alias_func'],
    )
    ext_data = util.threadpool.submit(res['peripherals'].acquire, this)

    with util.ExceptionStack('failed to acquire data') as exc:
        with exc.defer():
            iq = iq.result()
        with exc.defer():
            ext_data = ext_data.result()

    if not isinstance(ext_data, dict):
        raise TypeError(f'peripheral acquire() returned {type(ext_data)!r}, not dict')
    if not isinstance(iq, sources.AcquiredIQ):
        raise TypeError(f'source acquire() returned {type(iq)!r}, not AcquiredIQ')

    arm_both(next_)

    iq.info = indexer.apply(iq.info)
    iq.extra_data = iq.extra_data | ext_data
    return iq


def _log_cache_info(
    resources: Resources[_TS, _TP, _TC, _PS, _PC], cache, capture: _TC, result, *_, **__
):
    cal = resources['sweep_spec'].source.calibration
    if cal is None or 'spectrogram' not in cache.name:
        return

    spg, attrs = result

    xp = sw.util.array_namespace(spg)

    if isinstance(capture, specs.SoapyCapture):
        info_fields = ('center_frequency', 'port', 'gain')
    else:
        info_fields = ('port',)

    desc_kws = {
        'fields': info_fields,
        'source_id': resources['source'].id,
        'adjust_spec': resources['sweep_spec'].adjust_captures,
    }

    # convert to dB after this function
    peaks = spg.max(axis=tuple(range(1, spg.ndim)))
    noise = lookup_system_noise_power(
        cal,
        specs.SoapyCapture.from_spec(capture),
        master_clock_rate=resources['sweep_spec'].source.master_clock_rate,
        alias_func=resources['alias_func'],
        B=attrs['noise_bandwidth'],
        xp=xp,
    )

    logger = sa.util.get_logger('analysis')
    capture_splits = specs.helpers.split_capture_ports(capture)

    for c, snr in zip(capture_splits, sw.powtodB(peaks) - noise):
        if sa.util.is_cupy_array(snr.data):
            snr = float(snr.data.get())
        else:
            snr = float(snr.values)
        if not abs(snr) < math.inf:
            continue

        snr_desc = f'{round(snr)} dB max SNR'
        capture_desc = specs.helpers.describe_capture(c, **desc_kws)
        logger.info(f'spectrogram ▮ {snr_desc:<14} ▮ {capture_desc}')
