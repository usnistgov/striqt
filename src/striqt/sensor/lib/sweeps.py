"""implementation of performant acquisition and analysis sequencing for a series of captures"""

from __future__ import annotations

import contextlib
import dataclasses
import itertools
import typing
from collections import Counter

import striqt.waveform as iqwaveform
from striqt.analysis import measurements, describe_capture, registry

from . import datasets, peripherals, sinks, sources, specs, util
from .resources import Resources, AnyResources
from .calibration import lookup_system_noise_power
from .specs import _TC, _TP, _TS

if typing.TYPE_CHECKING:
    from typing_extensions import Unpack
    import xarray as xr
    from .. import bindings
else:
    xr = util.lazy_import('xarray')


def varied_capture_fields(sweep: specs.Sweep) -> list[str]:
    """generate a list of capture fields with at least 2 values in the specified sweep"""
    inner_values = (c.todict().values() for c in sweep.captures)
    inner_counts = [len(Counter(v)) for v in zip(*inner_values)]
    fields = sweep.captures[0].todict().keys()
    inner_counts = dict(zip(fields, inner_counts))
    outer_counts = {loop.field: len(loop.get_points()) for loop in sweep.loops}
    totals = {
        field: max(inner_counts[field], outer_counts.get(field, 0)) for field in fields
    }
    return [f for f, c in totals.items() if c > 1]


def sweep_touches_gpu(sweep: specs.Sweep) -> bool:
    """returns True if the sweep would benefit from the GPU"""

    if sweep.source.calibration is not None:
        return True

    analysis_dict = sweep.analysis.todict()
    if tuple(analysis_dict.keys()) != (measurements.iq_waveform.__name__,):
        # everything except iq_clipping requires a warmup
        return True

    # check the inner loop (explicit) values
    for capture in sweep.captures:
        if capture.host_resample or capture.analysis_bandwidth is not None:
            return True

    # check any values specified in outer loops
    for loop in sweep.loops:
        if loop.field == 'host_resample' and True in loop.get_points():
            return True
        elif loop.field == 'analysis_bandwidth' and not all(loop.get_points()):
            return True

    return False


def design_warmup(
    sweep: specs.Sweep[_TS, _TP, _TC], skip: tuple[_TC, ...] = ()
) -> specs.Sweep[specs.NoSource, specs.NoPeripherals, specs.ResampledCapture]:
    """returns a Sweep object for a NullRadio consisting of capture combinations from
    `sweep`.

    This is meant to trigger expensive python imports and warm up JIT caches
    in order to avoid analysis slowdowns during sweeps.
    """

    from .bindings import mock_binding
    from ..bindings import warmup

    # check loops to select an easy case for warmup
    updates = {}
    ports = []
    for loop in sweep.loops:
        if loop.field in ('duration', 'sample_rate', 'backend_sample_rate'):
            updates[loop.field] = min(loop.get_points())
        if loop.field == 'host_resample':
            updates[loop.field] = any(loop.get_points())
        if loop.field == 'port':
            ports.extend(loop.get_points())

    captures = [c.replace(**updates) for c in sweep.captures]
    by_size = {round(c.sample_rate * c.duration): c for c in captures}

    num_rx_ports = 0
    for port in ports + [c.port for c in captures]:
        if port is None:
            continue
        if isinstance(port, tuple):
            n = max(port)
        else:
            n = port
        if n > num_rx_ports:
            num_rx_ports = n

    if len(captures) > 1:
        captures = [by_size[min(by_size.keys())]]

    b = mock_binding(sweep.__bindings__, 'warmup', register=False)

    source = warmup.schema.source(
        num_rx_ports=num_rx_ports,
        base_clock_rate=sweep.source.base_clock_rate,
        calibration=None,
        warmup_sweep=False,
        gapless_rearm=sweep.source.gapless_rearm,
        periodic_trigger=None,
        channel_sync_source=sweep.source.channel_sync_source,
        uncalibrated_peak_detect=sweep.source.uncalibrated_peak_detect,
    )

    return b.sweep_spec(
        source=source,
        captures=tuple(captures),
        loops=(),
        analysis=sweep.analysis,
        sink=sweep.sink,
    )


def prepare_compute(input_spec: specs.Sweep):
    util.expensive_imports(cupy=input_spec.source.array_backend == 'cupy')

    if not input_spec.source.warmup_sweep:
        return

    from .. import bindings

    spec = design_warmup(input_spec)

    if len(spec.captures) == 0:
        return

    resources = Resources(
        sweep_spec=spec,
        source=bindings.warmup.source(spec.source, analysis=spec.analysis),
        peripherals=bindings.warmup.peripherals(spec),
        sink=sinks.NoSink(spec),
        calibration=None,
        alias_func=None,
    )

    with resources['source']:
        sweep = SweepIterator(resources, always_yield=True, calibration=None)

        for _ in sweep:
            pass


@contextlib.contextmanager
def log_progress_contexts(index, count):
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


class SweepIterator:
    spec: specs.Sweep

    def __init__(
        self,
        resources: Resources,
        *,
        always_yield=False,
        loop=False,
        **extra_resources: 'Unpack[AnyResources]',
    ):
        self.resources = Resources(resources, **extra_resources)

        self.source = self.resources['source']
        self._peripherals = self.resources['peripherals']
        self._sink = self.resources['sink']
        self.spec = self.resources['sweep_spec']

        self._always_yield = always_yield
        self._loop = loop

        self._opts = datasets.EvaluationOptions(
            sweep_spec=self.spec,
            registry=registry,
            extra_attrs=datasets.build_dataset_attrs(self.spec),
            correction=True,
            cache_callback=self.show_cache_info,
            as_xarray='delayed',
            block_each=False,
        )

    def show_cache_info(self, cache, capture: specs.ResampledCapture, result, *_, **__):
        cal = self.spec.source.calibration
        if cal is None or 'spectrogram' not in cache.name:
            return

        spg, attrs = result

        xp = iqwaveform.util.array_namespace(spg)

        # conversion to dB is left for after this function, but display
        # log messages in dB
        peaks = spg.max(axis=tuple(range(1, spg.ndim)))

        noise = lookup_system_noise_power(
            cal,
            specs.SoapyCapture.fromspec(capture),
            base_clock_rate=self.spec.source.base_clock_rate,
            alias_func=self.resources['alias_func'],
            B=attrs['noise_bandwidth'],
            xp=xp,
        )

        snr = iqwaveform.powtodB(peaks) - noise

        snr_desc = ','.join(f'{p:+02.0f}' for p in snr)
        if 'nan' not in snr_desc.lower():
            logger = util.get_logger('analysis')
            logger.info(f'({snr_desc}) dB SNR spectrogram peak')

    def __iter__(self) -> typing.Generator['xr.Dataset | bytes | None']:
        iq = None
        result = None

        captures = self.spec.loop_captures()

        if self._loop:
            capture_iter = itertools.cycle(captures)
            count = float('inf')
        else:
            capture_iter = captures
            count = len(captures)

        if count == 0:
            return

        # iterate across (previous-1, previous, current, next) captures to support concurrency
        offset_captures = util.zip_offsets(capture_iter, (-2, -1, 0, 1), fill=None)

        for i, (_, _, this, next_) in enumerate(offset_captures):
            assert isinstance(this, specs.ResampledCapture)
            assert isinstance(next_, specs.ResampledCapture)

            calls = {}

            with log_progress_contexts(i, count):
                if iq is None:
                    # first and last iterations
                    pass
                else:
                    # first item so that concurrently_with_fg runs it in the foreground
                    calls['analyze'] = util.Call(datasets.analyze, iq, self._opts)

                if this is None:
                    # last 2 iterations
                    pass
                else:
                    calls['acquire'] = util.Call(self._acquire, this, next_)

                if result is None:
                    # first 2 iterations
                    pass
                else:
                    calls['sink'] = util.Call(self._sink.append, result)

                ret = util.concurrently_with_fg(calls)

                if 'analyze' in ret:
                    result = ret['analyze']
                    assert isinstance(result, datasets.DelayedDataset)

                if 'acquire' in ret:
                    iq = ret['acquire']
                    assert isinstance(iq, sources.AcquiredIQ)
                else:
                    iq = None

                if 'sink' in ret:
                    yield ret['sink']
                elif self._always_yield:
                    yield None

    @util.stopwatch('acquire', 'sweep', threshold=0.25)
    def _acquire(
        self,
        this: specs.ResampledCapture,
        next_: specs.ResampledCapture|None,
    ) -> sources.AcquiredIQ:
        """arm and acquire from the source and peripherals"""

        assert this is not None

        try:
            self.source.capture_spec
        except AttributeError:
            # this is the first capture
            util.concurrently({
                'source': util.Call(self.source.arm, this),
                'peripherals': util.Call(self._peripherals.arm, this)
            })

        results = util.concurrently({
            'source': util.Call(
                self.source.acquire,
                this,
                next_,
                correction=False,
                alias_func=self.resources['alias_func']
            ),
            'peripherals': util.Call(
                peripherals.acquire_arm, self._peripherals, this, next_
            )
        })

        iq, ext_data = results.values()
        assert isinstance(iq, sources.AcquiredIQ)

        return dataclasses.replace(
            iq, extra_data=iq.extra_data|ext_data
        )


def iter_sweep(
    resources: Resources,
    *,
    always_yield=False,
    loop=False,
    **extra_resources: 'Unpack[Resources]',
) -> SweepIterator:
    """iterate through sweep captures on the source, yielding a dataset for each.

    Normally, for performance reasons, the first iteration consists of
    `(capture 1) ➔ concurrent(capture 2, analysis 1) ➔ (yield analysis 1)`.
    The `always_yield` argument is provided to allow synchronization of hardware between capture 1 and capture 2:
    `(capture 1) ➔ yield None ➔ concurrent(capture 2, analysis 1) ➔ (yield analysis 1)`.
    Added checks are needed to handle `None` before recording data.

    Args:
        source: the IQ source
        sweep: the specification that configures the sweep
        calibration: if specified, the calibration data used to scale the output from full-scale to physical power
        always_yield: if `True`, yield `None` before the second capture
        quiet: if True, log at the debug level, and show 'info' level log messages or higher only to the screen

    Returns:
        An iterator of analyzed data
    """

    return SweepIterator(**locals())


def iter_raw_iq(
    source: 'sources.SourceBase',
    sweep: specs.Sweep,
) -> typing.Generator[sources.AcquiredIQ]:
    """iterate through the sweep and yield the raw IQ vector for each.

    Normally, for performance reasons, the first iteration consists of
    `(capture 1) ➔ concurrent(capture 2, analysis 1) ➔ (yield analysis 1)`.
    The `always_yield` argument is provided to allow synchronization of hardware between capture 1 and capture 2:
    `(capture 1) ➔ yield None ➔ concurrent(capture 2, analysis 1) ➔ (yield analysis 1)`.
    Added checks are needed to handle `None` before recording data.

    Args:
        source: the device that runs the sweep
        sweep: the specification that configures the sweep
        quiet: if True, log at the debug level, and show 'info' level log messages or higher only to the screen

    Returns:
        An iterator of analyzed data
    """

    if len(sweep.captures) == 0:
        return

    capture_prev = None

    captures = sweep.loop_captures()

    # iterate across (previous, current, next) captures to support concurrency
    offset_captures = util.zip_offsets(captures, (0, 1), fill=None)

    for i, (capture_this, capture_next) in enumerate(offset_captures):
        desc = describe_capture(
            capture_this, capture_prev, index=i, count=len(captures)
        )

        with util.stopwatch(desc, 'source', threshold=0.5):
            # extra iteration at the end for the last analysis
            yield source.acquire(
                capture_this,
                next=capture_next,
                correction=False,
            )
