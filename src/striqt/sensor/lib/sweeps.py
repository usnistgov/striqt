"""implementation of performant acquisition and analysis sequencing for a series of captures"""

from __future__ import annotations

import contextlib
import itertools
import typing
from collections import Counter

import striqt.waveform as iqwaveform
from striqt.analysis import measurements, describe_capture, registry

from . import datasets, sinks, sources, specs, util
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

    # captures that have unique sampling parameters, which are those
    # specified in structs.WaveformCapture
    wcaptures = [specs.ResampledCapture.fromspec(c) for c in sweep.loop_captures()]
    unique_map = dict(zip(wcaptures, sweep.loop_captures()))
    skip_wcaptures = {specs.ResampledCapture.fromspec(c) for c in skip}
    captures = [unique_map[c] for c in unique_map.keys() if c not in skip_wcaptures]

    num_rx_ports = 0
    for c in captures:
        if c.port is None:
            continue
        if isinstance(c.port, tuple):
            n = max(c.port)
        else:
            n = c.port
        if n > num_rx_ports:
            num_rx_ports = n

    if len(captures) > 1:
        captures = captures[:1]

    b = mock_binding(sweep.__bindings__, 'warmup')

    source=warmup.schema.source(
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
        sink=sweep.sink
    )


def run_warmup(input_spec: specs.Sweep):
    if not input_spec.source.warmup_sweep:
        return

    from .. import bindings

    if input_spec.source.array_backend == 'cupy':
        iqwaveform.set_max_cupy_fft_chunk(input_spec.source.cupy_max_fft_chunk_size)

    spec = design_warmup(input_spec)

    if len(spec.captures) == 0:
        return

    resources = Resources(
        sweep_spec=spec,
        source=bindings.warmup.source(spec.source, analysis=spec.analysis),
        peripherals=bindings.warmup.peripherals(spec),
        sink=sinks.NoSink(spec),
        calibration=None,
    )

    with resources['source']:
        sweep = SweepIterator(resources, always_yield=True, calibration=None)

        for _ in sweep:
            pass


def _iq_is_reusable(
    c1: specs.ResampledCapture | None, c2: specs.ResampledCapture, base_clock_rate
):
    """return True if c2 is compatible with the raw and uncalibrated IQ acquired for c1"""

    if c1 is None or c2 is None:
        return False

    fsb1 = sources.design_capture_resampler(base_clock_rate, c1)['fs_sdr']
    fsb2 = sources.design_capture_resampler(base_clock_rate, c2)['fs_sdr']

    if fsb1 != fsb2:
        # the realized backend sample rates need to be the same
        return False

    downstream_kws = {
        'host_resample': False,
        'start_time': None,
        'backend_sample_rate': None,
    }

    c1_compare = c1.replace(**downstream_kws)
    c2_compare = c2.replace(
        # ignore parameters that only affect downstream processing
        analysis_bandwidth=c1.analysis_bandwidth,
        sample_rate=c1.sample_rate,
        **downstream_kws,
    )

    return c1_compare == c2_compare


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
        resources = Resources(resources, **extra_resources)

        self.source = resources['source']
        self._peripherals = resources['peripherals']
        self._sink = resources['sink']
        self.spec = resources['sweep_spec']
        self.cal = resources['calibration']

        self._always_yield = always_yield
        self._loop = loop

        self._analysis_opts = datasets.EvaluationOptions(
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
            base_clock_rate=capture.backend_sample_rate,
            B=attrs['noise_bandwidth'],
            xp=xp,
        )

        snr = iqwaveform.powtodB(peaks) - noise

        snr_desc = ','.join(f'{p:+02.0f}' for p in snr)
        util.get_logger('analysis').info(f'({snr_desc}) dB SNR spectrogram peak')

    def __iter__(self) -> typing.Generator['xr.Dataset | bytes | None']:
        iq = None
        this_ext_data = {}
        prior_ext_data = {}
        canalyze = None
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

        for i, (csink, _, cacquire, cnext) in enumerate(offset_captures):
            calls = {}

            with log_progress_contexts(i, count):
                if iq is None:
                    # no pending data in the first iteration
                    pass
                else:
                    assert canalyze is not None

                    calls['analyze'] = util.Call(
                        datasets.analyze_capture,
                        iq,
                        self.source,
                        canalyze,
                        self._analysis_opts,
                        calibration_data=self.cal
                    )

                if cacquire is None:
                    # this happens at the end during the last post-analysis and intakes
                    pass
                else:
                    calls['acquire'] = util.Call(
                        self._acquire, iq, canalyze, cacquire, cnext
                    )

                if result is None:
                    # for the first two iterations, there is no data to save
                    pass
                else:
                    assert csink is not None
                    result.extra_data.update(prior_ext_data)
                    calls['intake'] = util.Call(
                        self._intake,
                        results=result,
                        capture=csink,
                    )

                ret = util.concurrently_with_fg(calls)

                result = typing.cast(
                    typing.Union[datasets.DelayedDataset, None],
                    ret.get('analyze', None),
                )

                if 'acquire' in ret:
                    iq = ret['acquire']['source']
                    canalyze = iq.capture
                    prior_ext_data = this_ext_data
                    this_ext_data = ret['acquire'].get('peripherals', {}) or {}
                else:
                    iq = None
                    canalyze = None

                if 'intake' in ret:
                    yield ret['intake']
                elif self._always_yield:
                    yield None

    @util.stopwatch('', 'source', logger_level=util.PERFORMANCE_INFO)
    def _acquire(
        self,
        iq_prev: sources.AcquiredIQ | None,
        capture_prev,
        capture_this,
        capture_next,
    ):
        if self.spec.info.reuse_iq and not isinstance(self.spec.source, specs.NoSource):
            reuse_this = _iq_is_reusable(
                capture_prev, capture_this, self.source.setup_spec.base_clock_rate
            )
            reuse_next = _iq_is_reusable(
                capture_this, capture_next, self.source.setup_spec.base_clock_rate
            )
        else:
            reuse_this = reuse_next = False

        if capture_prev is None:
            self._arm(capture_this)

        # acquire from the radio and any peripherals
        calls = {}
        if reuse_this and iq_prev is not None:
            ret_iq = sources.AcquiredIQ(
                raw=iq_prev.raw,
                aligned=iq_prev.aligned,
                capture=capture_this,
                info=iq_prev.info.replace(start_time=None),
                extra_data=iq_prev.extra_data,
            )
            calls['source'] = util.Call(lambda: ret_iq)
        else:
            calls['source'] = util.Call(
                self.source.acquire,
                capture_this,
                correction=False,
            )

        if self._peripherals is not None:
            calls['peripherals'] = util.Call(self._peripherals_acquire, capture_next)

        result = util.concurrently(calls)

        if capture_next is not None and not reuse_next:
            self._arm(capture_next)

        return result

    def _arm(self, capture):
        calls = {}

        calls['source'] = util.Call(self.source.arm, capture)
        if self._peripherals is not None:
            calls['peripherals'] = util.Call(self._peripherals.arm, capture)

        return util.concurrently(calls)

    def _peripherals_acquire(self, capture):
        if capture is None:
            return {}

        data = self._peripherals.acquire(capture)

        if data is None:
            data = {}
        else:
            try:
                data = dict(data)
            except TypeError:
                raise TypeError(
                    f'{self._peripherals.acquire!r} must return a dict or None, not {type(data)!r}'
                )

        if self.spec.source.calibration is None:
            return data

        system_noise = lookup_system_noise_power(
            self.spec.source.calibration,
            capture,
            self.source.setup_spec.base_clock_rate,
        )

        return dict(data, sensor_system_noise=system_noise)

    @util.stopwatch('', 'sink', threshold=10e-3, logger_level=util.PERFORMANCE_INFO)
    def _intake(
        self,
        results: datasets.DelayedDataset,
        capture: specs.ResampledCapture,
    ) -> 'xr.Dataset | None':
        if not isinstance(results, datasets.DelayedDataset):
            raise ValueError(
                f'expected DelayedAnalysisResult type for data, not {type(results)}'
            )

        ds = datasets.from_delayed(results)

        if self._sink is None:
            return ds
        else:
            self._sink.append(ds, capture)


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

        with util.stopwatch(desc, 'source', logger_level=util.PERFORMANCE_INFO):
            # extra iteration at the end for the last analysis
            yield source.acquire(
                capture_this,
                next_capture=capture_next,
                correction=False,
            )
