#!/usr/bin/env python3
"""Profile the analysis of one ADA-style 3750 MHz capture on 1 and 2 CUDA streams.

The capture, source, and analysis reproduce the aggregate-directivity-acquisition
fragments ``denver-3750`` (capture), ``7201b`` (source), and ``cellular`` (analysis):
107.52 MS/s resampled from the 125 MHz master clock, 2 ports, 100 MHz analysis
bandwidth, 0.1 s, 12 measurements, and the ``cellular_5g_pss_sync`` trigger. The
input is a buffer of ones at the source rate, with the lead and tail overlaps of the
production acquisition. Nothing under ``src/`` is modified. Every variant runs the
measurements through the live registry on a fresh `Preprocessor` per repetition.

Variants:

- ``serial``: `striqt.analysis.analyze_by_spec` with ``block_each=True``, the call that
  `iterate_captures` makes in production. Each measurement is copied to the host, and
  the device synchronized, before the next measurement launches.
- ``deferred``: the same call with ``block_each=False``. One stream, with the host
  copies after the last launch.
- ``streams``: 2 `cupy.cuda.Stream` in one thread. The measurements registered
  ``prefer_filtered=False`` launch on stream U behind the resampler. The rest launch on
  stream F behind the analysis FIR. Events order the FIR after the resampler and the
  aligned stages after the trigger.
- ``threads``: the ``streams`` graph with one Python thread per stream.

Usage:
    pixi run -e cupy python tests/profiling/cuda_streams.py
    pixi run -e cupy python tests/profiling/cuda_streams.py --no-trigger --repeat 10
    pixi run -e cupy python tests/profiling/cuda_streams.py --non-blocking
    pixi run -e test39 python tests/profiling/cuda_streams.py --backend numpy --duration 0.02
    nsys profile -t cuda,nvtx pixi run -e cupy python tests/profiling/cuda_streams.py --variants streams --repeat 2

Reading the output:
    ``wall`` is the host time between device synchronizations around one evaluation,
    and ``x dur`` is that time divided by the capture duration. A pipeline is
    acquisition bound only when ``x dur`` is below 1. ``gpu U`` and ``gpu F`` are the
    event-timed spans of the unfiltered and filtered streams. A sum of the two above
    ``wall`` means the streams overlapped on the device. ``pool`` is the high-water
    mark of the cupy memory pool sampled between phases. ``mismatch`` names the
    measurements that differ from ``serial`` beyond ``allclose``. A mismatch on
    ``streams`` without one on ``deferred`` is a cross-stream read of a memoized stage.
    The breakdown table is the `stopwatch` time of each measurement in the ``serial``
    variant. The time of a conditioning stage is counted in the first measurement
    that selects the stage, and the stages are also listed indented. Streams are
    blocking by default and synchronize with the legacy default stream, on which the
    numba CUDA kernel of the cyclic autocorrelation launches. ``--non-blocking``
    removes that ordering.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import logging
import math
import platform
import statistics
import threading
import time
from collections.abc import Generator
from fractions import Fraction
from typing import Any, cast

import numpy as np
from typing_extensions import Self

import striqt.analysis as sa
import striqt.sensor as ss
import striqt.waveform as sw

cp: Any = None

VARIANTS = ('serial', 'deferred', 'streams', 'threads')

# fragments/captures/denver-3750.yaml
CAPTURE = {
    'center_frequency': 3750e6,
    'analysis_bandwidth': 100e6,
    'sample_rate': 107.52e6,
    'port': (0, 1),
    'gain': -10,
    'duration': 0.1,
    'adjust_analysis': {
        'frequency_offset': -40.08e6,
        'guard_bandwidths': (875e3, 575e3),
        'frame_slots': 'dddsuudddddddsuudddd',
        'special_symbols': 'ddddddfffffffu',
    },
}

# fragments/analysis/cellular.yaml, with the YAML anchors expanded
_SPECTROGRAM = {
    'frequency_resolution': 15000,
    'fractional_overlap': Fraction(13, 28),
    'window_fill': Fraction(15, 28),
    'integration_bandwidth': 360e3,
    'window': ('kaiser', 11.884),
    'lo_bandstop': 120e3,
}
_SCS = 30000
_POWER_DETECTORS = ('rms', 'peak')
_SYMBOL_PERIOD = Fraction(1, 28000)
ANALYSIS = {
    'spectrogram': _SPECTROGRAM,
    'cellular_resource_power_histogram': {
        'subcarrier_spacing': _SCS,
        'window': _SPECTROGRAM['window'],
        'power_low': -150,
        'power_high': 0,
        'power_resolution': 0.1,
        'average_rbs': True,
        'average_slots': False,
        'cyclic_prefix': 'normal',
        'lo_bandstop': _SPECTROGRAM['lo_bandstop'],
        'frame_slots': 'dddsuudddddddsuudddd',
        'special_symbols': 'ddddddfffffffu',
        'guard_bandwidths': (875e3, 875e3),
    },
    'cellular_5g_pss_correlation': {
        'subcarrier_spacing': _SCS,
        'delay': 0,
        'discovery_periodicity': 20e-3,
        'frequency_offset': 0,
        'symbol_indexes': 'c',
    },
    'cellular_5g_pss_sync': {
        'subcarrier_spacing': _SCS,
        'delay': 0,
        'symbol_indexes': 'c',
        'discovery_periodicity': 20e-3,
        'per_port': True,
        'max_beams': 4,
    },
    'cellular_5g_ssb_spectrogram': {
        'frequency_offset': 0,
        'subcarrier_spacing': _SCS,
        'lo_bandstop': _SPECTROGRAM['lo_bandstop'],
        'window': _SPECTROGRAM['window'],
    },
    'power_spectral_density': dict(
        _SPECTROGRAM, time_statistic=('mean', 0.5, 0.9, 0.95, 0.99, 'max')
    ),
    'spectrogram_ratio_histogram': dict(
        _SPECTROGRAM, power_low=-50, power_high=50, power_resolution=0.1
    ),
    'spectrogram_histogram': dict(
        _SPECTROGRAM, power_low=-150, power_high=0, power_resolution=0.1
    ),
    'cellular_cyclic_autocorrelation': {
        'subcarrier_spacings': (_SCS,),
        'frame_slots': 'dddsuudddddddsuudddd',
        'frame_range': (0, 1),
    },
    'cyclic_channel_power': {
        'cyclic_period': 0.01,
        'cyclic_statistics': ('min', 'mean', 'max'),
        'detector_period': _SYMBOL_PERIOD,
        'power_detectors': _POWER_DETECTORS,
    },
    'channel_power_histogram': {
        'detector_period': _SYMBOL_PERIOD,
        'power_detectors': _POWER_DETECTORS,
        'power_low': -100,
        'power_high': 5,
        'power_resolution': 0.1,
    },
    'channel_power_time_series': {
        'detector_period': 0.01,
        'power_detectors': _POWER_DETECTORS,
    },
}


# %% Specs and the input buffer
@dataclasses.dataclass(frozen=True)
class Setup:
    capture: ss.specs.SoapyCapture
    """the ADA 3750 MHz capture, with `duration` from the command line"""
    source: ss.specs.Source
    """the air7201b source spec, with `signal_trigger` and `array_backend` set"""
    analysis: sa.specs.AnalysisGroup
    """the cellular analysis after `adjust_analysis` of the capture"""
    trigger: sa.Trigger | None
    """the trigger that `attach_corrections` installs, or None"""


def build_setup(duration: float, backend: str, trigger: bool) -> Setup:
    binding = ss.bindings.air7201b
    capture = binding.schema.capture(**dict(CAPTURE, duration=duration))
    source = binding.schema.source(
        signal_trigger='cellular_5g_pss_sync' if trigger else None,
        trigger_strobe=20e-3,
        array_backend=backend,
    )
    analysis = ss.specs.BundledAnalysis.from_dict(ANALYSIS)
    analysis = ss.specs.sequencing.adjust_analysis(analysis, capture.adjust_analysis)
    trig = ss.lib.compute.get_trigger_from_spec(source, analysis)
    return Setup(capture, source, analysis, trig)


def source_sample_count(setup: Setup) -> int:
    """return the length of the acquisition buffer at the source rate.

    The count is the lead overlap, the record of `duration` through the exact
    resampler ratio, and the tail overlap, as the controller acquires them.
    """
    capture, source = setup.capture, setup.source
    design = ss.lib.compute.design_resampler(capture, source.master_clock_rate)
    samples_out = round(capture.duration * capture.sample_rate)
    exact = Fraction(samples_out * design['nfft'], design['nfft_out'])
    record = math.floor(exact + Fraction(1, 2))
    lead, tail = ss.lib.compute.get_correction_overlaps(capture, source, setup.analysis)
    return lead + record + tail


def build_raw(setup: Setup, xp: Any) -> ss.specs.AcquiredIQ:
    """return the acquisition of ones that each repetition conditions."""
    count = source_sample_count(setup)
    port = setup.capture.port
    ports = len(port) if isinstance(port, tuple) else 1
    x = xp.ones((ports, count), dtype=xp.complex64)
    return ss.specs.AcquiredIQ(
        x=x,
        capture=setup.capture,
        info=ss.specs.AcquisitionInfo(),
        extra_data={},
        source_spec=setup.source,
        # the air7201b source without a calibration file
        power_units='FS',
        resampler=ss.lib.compute.design_resampler(
            setup.capture, setup.source.master_clock_rate
        ),
    )


def partition(analysis: sa.specs.AnalysisGroup) -> tuple[list[str], list[str]]:
    """return the (unfiltered, filtered) measurement names of `analysis`, in field order."""
    unfiltered: list[str] = []
    filtered: list[str] = []
    for name, kws in analysis.to_dict().items():
        if not kws:
            continue
        info = sa.registry[type(getattr(analysis, name))]
        (filtered if info.prefer_filtered else unfiltered).append(name)
    return unfiltered, filtered


# %% Stream and event stand-ins for the numpy backend
class NullEvent:
    def record(self, stream: Any = None) -> None:
        pass

    def synchronize(self) -> None:
        pass


class NullStream:
    """a stand-in for `cupy.cuda.Stream` on the numpy backend"""

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        pass

    def record(self, event: Any = None) -> Any:
        return event if event is not None else NullEvent()

    def wait_event(self, event: Any) -> None:
        pass

    def synchronize(self) -> None:
        pass


def new_stream(non_blocking: bool) -> Any:
    if cp is None:
        return NullStream()
    return cp.cuda.Stream(non_blocking=non_blocking)


def new_event() -> Any:
    if cp is None:
        return NullEvent()
    return cp.cuda.Event(block=False, disable_timing=False)


def elapsed_s(start: Any, end: Any) -> float | None:
    if cp is None:
        return None
    return cp.cuda.get_elapsed_time(start, end) / 1e3


def device_synchronize() -> None:
    if cp is not None:
        cp.cuda.Device().synchronize()


@contextlib.contextmanager
def nvtx_range(name: str) -> Generator[None, None, None]:
    nvtx: Any = getattr(cp.cuda, 'nvtx', None) if cp is not None else None
    pushed = False
    if nvtx is not None:
        try:
            nvtx.RangePush(name)
            pushed = True
        except (RuntimeError, AttributeError):
            pushed = False
    try:
        yield
    finally:
        if pushed:
            nvtx.RangePop()


class PoolSampler:
    """the high-water mark of the cupy memory pool across `sample` calls"""

    def __init__(self) -> None:
        self.peak_bytes = 0

    def sample(self) -> None:
        if cp is None:
            return
        used = cp.get_default_memory_pool().used_bytes()
        self.peak_bytes = max(self.peak_bytes, used)


# %% The variants
@dataclasses.dataclass
class Timing:
    wall_s: float
    gpu_u_s: float | None = None
    gpu_f_s: float | None = None
    peak_pool_bytes: int = 0


def launch(iq: ss.specs.AcquiredIQ, setup: Setup, names: list[str]) -> dict[str, Any]:
    """launch the measurements `names` on the current stream and return the delayed results."""
    spec_dict = setup.analysis.to_dict()
    delayed = {}
    for name in names:
        info = sa.registry[type(getattr(setup.analysis, name))]
        with nvtx_range(name):
            delayed[name] = info.func(
                iq=iq, capture=setup.capture, as_xarray='delayed', **spec_dict[name]
            )
    return delayed


def materialize(delayed: dict[str, Any]) -> dict[str, np.ndarray]:
    return {name: d.compute().result for name, d in delayed.items()}


def run_serial(
    iq: ss.specs.AcquiredIQ, setup: Setup, block_each: bool, pool: PoolSampler
) -> tuple[dict[str, np.ndarray], Timing]:
    options = sa.EvaluationOptions(
        as_xarray='delayed', registry=sa.registry, block_each=block_each
    )
    start, end = new_event(), new_event()
    t0 = time.perf_counter()
    start.record()
    results = sa.analyze_by_spec(iq, setup.analysis, setup.capture, options)
    pool.sample()
    end.record()
    device_synchronize()
    wall = time.perf_counter() - t0
    arrays = {name: cast(Any, d).result for name, d in results.items()}
    return arrays, Timing(wall, gpu_u_s=elapsed_s(start, end))


def run_streams(
    iq: ss.specs.AcquiredIQ, setup: Setup, non_blocking: bool, pool: PoolSampler
) -> tuple[dict[str, np.ndarray], Timing]:
    unfiltered, filtered = partition(setup.analysis)
    aligned = setup.trigger is not None
    stream_u, stream_f = new_stream(non_blocking), new_stream(non_blocking)
    ev = {k: new_event() for k in ('u0', 'u1', 'f0', 'f1', 'resampled', 'shifts')}
    filtered_unaligned = [
        n
        for n in filtered
        if not sa.registry[type(getattr(setup.analysis, n))].prefer_aligned
    ]
    filtered_aligned = [n for n in filtered if n not in filtered_unaligned]

    t0 = time.perf_counter()
    delayed: dict[str, Any] = {}
    with stream_u, nvtx_range('U: resample'):
        ev['u0'].record(stream_u)
        iq.preprocess(prefer_filtered=False, prefer_aligned=False)
        ev['resampled'].record(stream_u)
    with stream_f:
        ev['f0'].record(stream_f)
        stream_f.wait_event(ev['resampled'])
        with nvtx_range('F: fir'):
            iq.preprocess(prefer_filtered=True, prefer_aligned=False)
        delayed.update(launch(iq, setup, filtered_unaligned))
    pool.sample()
    with stream_u:
        if aligned:
            with nvtx_range('U: trigger'):
                iq.preprocess(prefer_filtered=False, prefer_aligned=True)
            ev['shifts'].record(stream_u)
        delayed.update(launch(iq, setup, unfiltered))
        ev['u1'].record(stream_u)
    pool.sample()
    with stream_f:
        if aligned:
            stream_f.wait_event(ev['shifts'])
            with nvtx_range('F: align'):
                iq.preprocess(prefer_filtered=True, prefer_aligned=True)
        delayed.update(launch(iq, setup, filtered_aligned))
        ev['f1'].record(stream_f)
    pool.sample()
    stream_u.synchronize()
    stream_f.synchronize()
    arrays = materialize(delayed)
    device_synchronize()
    wall = time.perf_counter() - t0
    timing = Timing(
        wall,
        gpu_u_s=elapsed_s(ev['u0'], ev['u1']),
        gpu_f_s=elapsed_s(ev['f0'], ev['f1']),
    )
    return arrays, timing


def run_threads(
    iq: ss.specs.AcquiredIQ, setup: Setup, non_blocking: bool, pool: PoolSampler
) -> tuple[dict[str, np.ndarray], Timing]:
    unfiltered, filtered = partition(setup.analysis)
    aligned = setup.trigger is not None
    stream_u, stream_f = new_stream(non_blocking), new_stream(non_blocking)
    ev = {k: new_event() for k in ('u0', 'u1', 'f0', 'f1', 'resampled', 'shifts')}
    resampled_done, shifts_done = threading.Event(), threading.Event()
    filtered_unaligned = [
        n
        for n in filtered
        if not sa.registry[type(getattr(setup.analysis, n))].prefer_aligned
    ]
    filtered_aligned = [n for n in filtered if n not in filtered_unaligned]
    delayed_u: dict[str, Any] = {}
    delayed_f: dict[str, Any] = {}
    errors: list[BaseException] = []

    def worker_u() -> None:
        try:
            with stream_u:
                ev['u0'].record(stream_u)
                with nvtx_range('U: resample'):
                    iq.preprocess(prefer_filtered=False, prefer_aligned=False)
                ev['resampled'].record(stream_u)
                resampled_done.set()
                if aligned:
                    with nvtx_range('U: trigger'):
                        iq.preprocess(prefer_filtered=False, prefer_aligned=True)
                    ev['shifts'].record(stream_u)
                shifts_done.set()
                delayed_u.update(launch(iq, setup, unfiltered))
                ev['u1'].record(stream_u)
                pool.sample()
                stream_u.synchronize()
        except BaseException as exc:
            errors.append(exc)
            resampled_done.set()
            shifts_done.set()

    def worker_f() -> None:
        try:
            with stream_f:
                ev['f0'].record(stream_f)
                resampled_done.wait()
                if errors:
                    return
                stream_f.wait_event(ev['resampled'])
                with nvtx_range('F: fir'):
                    iq.preprocess(prefer_filtered=True, prefer_aligned=False)
                delayed_f.update(launch(iq, setup, filtered_unaligned))
                shifts_done.wait()
                if errors:
                    return
                if aligned:
                    stream_f.wait_event(ev['shifts'])
                    with nvtx_range('F: align'):
                        iq.preprocess(prefer_filtered=True, prefer_aligned=True)
                delayed_f.update(launch(iq, setup, filtered_aligned))
                ev['f1'].record(stream_f)
                pool.sample()
                stream_f.synchronize()
        except BaseException as exc:
            errors.append(exc)

    t0 = time.perf_counter()
    threads = [threading.Thread(target=worker_u), threading.Thread(target=worker_f)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise errors[0]
    arrays = materialize({**delayed_u, **delayed_f})
    device_synchronize()
    wall = time.perf_counter() - t0
    timing = Timing(
        wall,
        gpu_u_s=elapsed_s(ev['u0'], ev['u1']),
        gpu_f_s=elapsed_s(ev['f0'], ev['f1']),
    )
    return arrays, timing


def evaluate(
    variant: str, raw: ss.specs.AcquiredIQ, setup: Setup, non_blocking: bool
) -> tuple[dict[str, np.ndarray], Timing]:
    """run one repetition of `variant` on a fresh conditioning of `raw`."""
    raw.x[:] = 1
    device_synchronize()
    iq = ss.lib.compute.attach_corrections(
        raw, signal_trigger=setup.trigger, overwrite_x=True
    )
    pool = PoolSampler()
    with sa.registry.cache_context(setup.capture, None), nvtx_range(variant):
        if variant == 'serial':
            arrays, timing = run_serial(iq, setup, True, pool)
        elif variant == 'deferred':
            arrays, timing = run_serial(iq, setup, False, pool)
        elif variant == 'streams':
            arrays, timing = run_streams(iq, setup, non_blocking, pool)
        elif variant == 'threads':
            arrays, timing = run_threads(iq, setup, non_blocking, pool)
        else:
            raise ValueError(f'unknown variant {variant!r}')
    pool.sample()
    sw.arrays.free_cupy_mempool()
    timing.peak_pool_bytes = pool.peak_bytes
    return arrays, timing


# %% Reporting
class StopwatchCollector(logging.Handler):
    """a handler that sums the `stopwatch` time of each named block"""

    def __init__(self) -> None:
        super().__init__(level=sa.util.PERFORMANCE_DETAIL)
        self.times: dict[str, float] = {}
        self.enabled = False

    def emit(self, record: logging.LogRecord) -> None:
        # `stopwatch` passes the timing dict as the message argument of the record
        fields: dict[str, Any] = (
            record.args if isinstance(record.args, dict) else vars(record)
        )
        name = fields.get('stopwatch_name')
        if not self.enabled or not isinstance(name, str):
            return
        self.times[name] = self.times.get(name, 0.0) + float(fields['stopwatch_time'])


def mismatches(
    arrays: dict[str, np.ndarray], reference: dict[str, np.ndarray]
) -> list[str]:
    bad = []
    for name, ref in reference.items():
        got = arrays.get(name)
        if got is None or got.shape != ref.shape:
            bad.append(f'{name}(shape)')
            continue
        if not np.allclose(got, ref, rtol=1e-5, atol=0, equal_nan=True):
            diff = np.nanmax(np.abs(got.astype(np.float64) - ref.astype(np.float64)))
            bad.append(f'{name}({diff:.3g})')
    return bad


def fmt_s(value: float | None) -> str:
    return '      -' if value is None else f'{value:7.3f}'


def report(
    rows: dict[str, list[Timing]],
    bad: dict[str, list[str]],
    duration: float,
) -> None:
    print(
        f'\n{"variant":<10} {"wall min":>8} {"wall med":>8} {"ratio":>6} {"x dur":>6} '
        f'{"gpu U":>7} {"gpu F":>7} {"pool MB":>8}  mismatch'
    )
    base = min(t.wall_s for t in rows['serial']) if 'serial' in rows else None
    for variant, timings in rows.items():
        walls = [t.wall_s for t in timings]
        best = timings[walls.index(min(walls))]
        ratio = f'{min(walls) / base:6.2f}' if base else '     -'
        pool = f'{best.peak_pool_bytes / 2**20:8.0f}' if cp is not None else '       -'
        print(
            f'{variant:<10} {min(walls):8.3f} {statistics.median(walls):8.3f} {ratio} '
            f'{min(walls) / duration:6.1f} {fmt_s(best.gpu_u_s)} {fmt_s(best.gpu_f_s)} '
            f'{pool}  {", ".join(bad.get(variant, [])) or "-"}'
        )


def report_breakdown(
    times: dict[str, float], repeat: int, measurements: list[str]
) -> None:
    """print the `stopwatch` time of each block, with the share among the measurements.

    A conditioning stage (``resampling filter``, ``analysis filter``) is nested in
    the measurement that first selected the stage, and is listed indented with no
    share.
    """
    total = sum(times.get(name, 0.0) for name in measurements)
    print(f'\nserial breakdown, mean of {repeat} repetitions')
    print(f'{"block":<36} {"s":>7} {"share":>6}')
    for name, t in sorted(times.items(), key=lambda kv: -kv[1]):
        if name in measurements:
            print(f'{name:<36} {t / repeat:7.3f} {t / total:6.1%}')
        else:
            print(f'  {name:<34} {t / repeat:7.3f} {"-":>6}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--variants', nargs='+', choices=VARIANTS, default=VARIANTS)
    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--duration', type=float, default=CAPTURE['duration'])
    parser.add_argument('--backend', choices=('cupy', 'numpy'), default='cupy')
    parser.add_argument(
        '--no-trigger',
        action='store_true',
        help='drop the cellular_5g_pss_sync trigger',
    )
    parser.add_argument(
        '--non-blocking',
        action='store_true',
        help='streams that do not synchronize with the legacy default stream',
    )
    args = parser.parse_args()

    global cp
    if args.backend == 'cupy':
        # numba.cuda must be imported before cupy (numba issue 6131)
        import numba.cuda  # noqa: F401, I001
        import cupy  # ty: ignore[unresolved-import]

        cp = cupy
        sw.arrays.configure_cupy()
        xp: Any = cp
        device = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
    else:
        xp = np
        device = 'none'
    print(
        f'{platform.machine()} numpy {np.__version__} '
        f'cupy {getattr(cp, "__version__", "-")} device: {device}'
    )

    setup = build_setup(args.duration, args.backend, not args.no_trigger)
    unfiltered, filtered = partition(setup.analysis)
    print(
        f'capture: {args.duration} s at {setup.capture.sample_rate / 1e6:g} MS/s, '
        f'trigger: {setup.trigger is not None}, '
        f'streams non_blocking: {args.non_blocking}'
    )
    print(f'  stream U (unfiltered): {", ".join(unfiltered)}')
    print(f'  stream F (filtered):   {", ".join(filtered)}')

    raw = build_raw(setup, xp)
    print(f'  buffer: {raw.x.shape} complex64, {raw.x.nbytes / 2**20:.0f} MB')

    collector = StopwatchCollector()
    analysis_logger = sa.util.get_logger('analysis').logger
    analysis_logger.setLevel(min(analysis_logger.level, sa.util.PERFORMANCE_DETAIL))
    analysis_logger.addHandler(collector)

    rows: dict[str, list[Timing]] = {}
    bad: dict[str, list[str]] = {}
    reference: dict[str, np.ndarray] | None = None
    for variant in args.variants:
        for _ in range(args.warmup):
            evaluate(variant, raw, setup, args.non_blocking)
        collector.enabled = variant == 'serial'
        timings = []
        for _ in range(args.repeat):
            arrays, timing = evaluate(variant, raw, setup, args.non_blocking)
            timings.append(timing)
        collector.enabled = False
        rows[variant] = timings
        if reference is None:
            reference = arrays
        else:
            bad[variant] = mismatches(arrays, reference)
        print(f'{variant}: {min(t.wall_s for t in timings):.3f} s')

    report(rows, bad, args.duration)
    if collector.times:
        report_breakdown(collector.times, args.repeat, unfiltered + filtered)


if __name__ == '__main__':
    main()
