#!/usr/bin/env python3
"""Run the sweep pipeline end to end on a synthetic source with the per-measurement
device synchronization on and off, time both, and compare every result bit for bit.

The sweep is the production path: `open_resources` and `iterate_sweep` over the
``single_tone`` function-generator source with the acquisition thread, the analysis in
the foreground thread, and the sink in the thread pool, through `NoSink`. The captures
are the ADA 3750 MHz cellular capture (``cuda_streams.CAPTURE``) with the ``cellular``
analysis (``cuda_streams.ANALYSIS``), each capture carrying a tone at its own frequency,
so a result read from a stale or half-copied receive buffer differs from the result of
the right capture. ``--buffers`` and ``--burst`` exercise the 2-deep pinned receive
buffer pair and the burst re-read that `cast_iq` synchronizes against.

Variants:

- ``block-each``: `actions.core.BLOCK_EACH` 1, the production setting. Each
  measurement result is copied to the host as soon as it returns.
- ``deferred``: `BLOCK_EACH` 0. The copies run at the end of the capture.

Usage:
    pixi run -e cupy python tests/profiling/pipeline_sync.py
    pixi run -e cupy python tests/profiling/pipeline_sync.py --buffers 2 --burst 4 --stress 20
    pixi run -e cupy python tests/profiling/pipeline_sync.py --no-trigger --captures 8
    pixi run -e test39 python tests/profiling/pipeline_sync.py --backend numpy --captures 2 --duration 0.02
    nsys profile -t cuda,nvtx,osrt -o pipeline pixi run -e cupy python tests/profiling/pipeline_sync.py --captures 4 --stress 1

Run with its own cache directory beside a test session or another striqt script
(``STRIQT_CACHE_DIR=$(mktemp -d)``): two processes sharing the window shelf corrupt it on
macOS Python 3.9.

Reading the output:
    One row per sweep run: the variant, the wall time per capture of the whole
    pipeline (the time of `iterate_sweep` over all captures divided by the capture
    count, after one untimed warmup run), ``x dur`` (that time over the capture
    duration; the pipeline keeps up with acquisition below 1), the growth of the cupy
    memory pool over the run in MB (the pool is never freed, so the growth is its
    high-water mark), the resident set of the process
    after the run and a garbage collection (``RSS MB``; a rise from run to run is a
    host-side leak, and ``--keep-open`` runs every repetition on one set of open
    resources to tell a leak per `open_resources` from one per capture; the synthetic
    source then continues its sample index across runs, so their results are not
    compared), and ``mismatch``: the data variables
    of any capture that differ from the first ``block-each`` run, with the capture
    index and the largest difference, or ``-``. The kernels are the same in both
    variants, so any mismatch is a race, and a mismatch between two ``block-each``
    runs is nondeterminism in the pipeline itself. ``--stress`` sets how many
    ``deferred`` runs are compared. The breakdown table is the `stopwatch` time of
    each measurement per capture in each variant: on ``block-each`` a measurement's
    time ends at its device copy and is its device time; on ``deferred`` it ends at
    the launch and is the host time of the launch. The exit status is 1 on any
    mismatch.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import gc
import logging
import platform
import resource
import statistics
import sys
import time
from typing import Any, Literal, cast

import numpy as np
from cuda_streams import ANALYSIS, CAPTURE, StopwatchCollector, nvtx_range

import striqt.analysis as sa
import striqt.sensor as ss
from striqt.sensor.lib.actions import core

cp: Any = None

MASTER_CLOCK_RATE = 125e6
TONE_OFFSET_START = -30e6
TONE_OFFSET_STEP = 7e6
VARIANTS = ('block-each', 'deferred')


# %% The sweep
def capture_fields(duration: float) -> dict[str, Any]:
    """return the fields of the ADA capture that a `SingleToneCapture` accepts."""
    fields = {k: v for k, v in CAPTURE.items() if k not in ('center_frequency', 'gain')}
    fields['duration'] = duration
    return fields


def build_sweep(
    captures: int,
    duration: float,
    backend: Literal['numpy', 'cupy'],
    trigger: bool,
    buffers: int,
    burst: int,
) -> Any:
    binding = ss.bindings.single_tone
    source = ss.specs.FunctionSource(
        master_clock_rate=MASTER_CLOCK_RATE,
        num_rx_ports=2,
        array_backend=backend,
        signal_trigger='cellular_5g_pss_sync' if trigger else None,
        trigger_strobe=20e-3 if trigger else None,
        bursts=ss.specs.BurstOptions(buffers=buffers),
    )
    fields = capture_fields(duration)
    specs = tuple(
        binding.schema.capture(
            **fields,
            frequency_offset=TONE_OFFSET_START + TONE_OFFSET_STEP * i,
            snr=None,
            burst=burst,
        )
        for i in range(captures)
    )
    analysis = ss.specs.BundledAnalysis.from_dict(ANALYSIS)
    sweep_cls = cast(Any, binding.sensor.sweep_spec_cls)
    return sweep_cls(
        source=source,
        captures=specs,
        loops=(),
        analysis=analysis,
        sink=ss.specs.ZarrSink(path=None),
    )


# %% One run
@dataclasses.dataclass
class Run:
    variant: str
    """the variant name"""
    wall_s: float
    """the time of `iterate_sweep` over every capture"""
    capture_count: int
    """the datasets delivered"""
    pool_growth_bytes: int
    """the growth of the cupy memory pool over the run"""
    rss_bytes: int
    """the resident set of the process after the run and a garbage collection"""
    stopwatch: dict[str, float]
    """the summed `stopwatch` time of each measurement over the run"""
    mismatches: list[str] = dataclasses.field(default_factory=list)
    """the variables that differ from the reference, with capture index and difference"""


def pool_bytes() -> int:
    if cp is None:
        return 0
    return int(cp.get_default_memory_pool().total_bytes())


def device_synchronize() -> None:
    if cp is not None:
        cp.cuda.Device().synchronize()


def rss_bytes() -> int:
    """return the resident set of the process (in bytes).

    Linux reads the current value from ``/proc/self/statm``; elsewhere the value is
    the high-water mark that `resource.getrusage` reports.
    """
    try:
        with open('/proc/self/statm') as fd:
            pages = int(fd.read().split()[1])
        return pages * resource.getpagesize()
    except (OSError, IndexError, ValueError):
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # ru_maxrss is in kilobytes on Linux and in bytes on macOS
        return usage if sys.platform == 'darwin' else usage * 1024


def run_sweep(
    sweep: Any,
    variant: str,
    collector: StopwatchCollector,
    resources: Any = None,
) -> tuple[Run, list]:
    """run `sweep` once under `variant` and return the timing and the datasets.

    `resources` are open resources to run on; None opens and closes a set for the run.
    """
    core.BLOCK_EACH = 1 if variant == 'block-each' else 0
    collector.times.clear()
    pool_before = pool_bytes()
    datasets = []
    if resources is None:
        opened: Any = ss.open_resources(sweep, None)
    else:
        opened = contextlib.nullcontext(resources)
    with opened as res:
        device_synchronize()
        collector.enabled = True
        t0 = time.perf_counter()
        with nvtx_range(variant):
            results = ss.iterate_sweep(res, sink=ss.sinks.NoSink(sweep))
            for dd in results:
                if isinstance(dd, ss.lib.compute.DelayedDataset):
                    datasets.append(ss.lib.compute.from_delayed(dd))
                elif dd is not None:
                    datasets.append(dd)
            device_synchronize()
        wall = time.perf_counter() - t0
        collector.enabled = False
    gc.collect()
    run = Run(
        variant=variant,
        wall_s=wall,
        capture_count=len(datasets),
        pool_growth_bytes=pool_bytes() - pool_before,
        rss_bytes=rss_bytes(),
        stopwatch=dict(collector.times),
    )
    return run, datasets


def compare(reference: list, datasets: list) -> list[str]:
    """return one entry per data variable of a capture that differs from `reference`."""
    bad = []
    if len(datasets) != len(reference):
        return [f'{len(datasets)} datasets, reference {len(reference)}']
    for index, (ref, got) in enumerate(zip(reference, datasets)):
        for name in ref.data_vars:
            a = np.asarray(ref[name].values)
            b = np.asarray(got[name].values)
            if a.shape != b.shape:
                bad.append(f'{index}:{name}(shape)')
            elif not np.array_equal(a, b, equal_nan=True):
                diff = np.nanmax(
                    np.abs(a.astype(np.complex128) - b.astype(np.complex128))
                )
                bad.append(f'{index}:{name}({diff:.3g})')
    return bad


# %% Reporting
def print_header() -> None:
    print(
        f'\n{"variant":<11} {"run":>4} {"ms/capture":>11} {"x dur":>6} {"pool MB":>8} '
        f'{"RSS MB":>7}  mismatch'
    )


def print_run(run: Run, index: int, duration: float) -> None:
    per_capture = run.wall_s / max(run.capture_count, 1)
    bad = ', '.join(run.mismatches[:6])
    if len(run.mismatches) > 6:
        bad += f', ... ({len(run.mismatches)} total)'
    print(
        f'{run.variant:<11} {index:4d} {per_capture * 1e3:11.1f} {per_capture / duration:6.2f} '
        f'{run.pool_growth_bytes / 2**20:8.0f} {run.rss_bytes / 2**20:7.0f}  {bad or "-"}',
        flush=True,
    )


def print_breakdown(runs: dict[str, list[Run]], names: list[str]) -> None:
    print('\nstopwatch per capture (ms), mean over the timed runs')
    print(f'{"measurement":<36} ' + ' '.join(f'{v:>11}' for v in runs))
    for name in names:
        cells = []
        for variant_runs in runs.values():
            per = [
                r.stopwatch.get(name, 0.0) / max(r.capture_count, 1)
                for r in variant_runs
            ]
            cells.append(f'{statistics.mean(per) * 1e3:11.2f}' if per else f'{"-":>11}')
        print(f'{name:<36} ' + ' '.join(cells))


def quiet_sweep_loggers() -> None:
    for name in ('sweep', 'source', 'sink', 'periph'):
        sa.util.get_logger(name).logger.setLevel(logging.WARNING)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--backend', choices=('cupy', 'numpy'), default='cupy')
    parser.add_argument('--captures', type=int, default=4)
    parser.add_argument('--duration', type=float, default=CAPTURE['duration'])
    parser.add_argument('--buffers', type=int, default=1)
    parser.add_argument('--burst', type=int, default=1)
    parser.add_argument('--no-trigger', action='store_true')
    parser.add_argument('--repeat', type=int, default=3, help='timed block-each runs')
    parser.add_argument('--stress', type=int, default=10, help='deferred runs compared')
    parser.add_argument(
        '--variants', nargs='+', choices=VARIANTS, default=list(VARIANTS)
    )
    parser.add_argument(
        '--keep-open',
        action='store_true',
        help='open the resources once for every run instead of once per run',
    )
    args = parser.parse_args()

    global cp
    if args.backend == 'cupy':
        import cuda_streams
        import cupy  # ty: ignore[unresolved-import]

        cp = cupy
        cuda_streams.cp = cupy
        device = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
    else:
        device = 'none'
    print(
        f'{platform.machine()} numpy {np.__version__} '
        f'cupy {getattr(cp, "__version__", "-")} device: {device}'
    )

    trigger = not args.no_trigger
    backend = cast("Literal['numpy', 'cupy']", args.backend)
    sample_rate = float(cast(float, CAPTURE['sample_rate']))
    sweep = build_sweep(
        args.captures, args.duration, backend, trigger, args.buffers, args.burst
    )
    print(
        f'{args.captures} captures of {args.duration} s at '
        f'{sample_rate / 1e6:g} MS/s, 2 ports, trigger {trigger}, '
        f'buffers {args.buffers}, burst {args.burst}, '
        f'STRIQT_BLOCK_EACH default {core.BLOCK_EACH}'
    )

    quiet_sweep_loggers()
    collector = StopwatchCollector()
    analysis_logger = sa.util.get_logger('analysis').logger
    analysis_logger.setLevel(min(analysis_logger.level, sa.util.PERFORMANCE_DETAIL))
    analysis_logger.addHandler(collector)

    # the warmup run compiles the kernels, designs the windows, and fills the caches
    run_sweep(sweep, 'block-each', collector)

    print_header()
    runs: dict[str, list[Run]] = {v: [] for v in args.variants}
    reference: list | None = None
    failed = False
    schedule = []
    if 'block-each' in args.variants:
        schedule += [('block-each', i) for i in range(args.repeat)]
    if 'deferred' in args.variants:
        schedule += [('deferred', i) for i in range(args.stress)]
    shared: Any = ss.open_resources(sweep, None) if args.keep_open else None
    with shared if shared is not None else contextlib.nullcontext() as resources:
        for variant, index in schedule:
            run, datasets = run_sweep(sweep, variant, collector, resources)
            if reference is None:
                reference = datasets
            elif args.keep_open:
                # the synthetic source continues its sample index across runs on one
                # open source, so the results of later runs differ by design
                run.mismatches = ['not compared']
            else:
                run.mismatches = compare(reference, datasets)
                failed |= bool(run.mismatches)
            runs[variant].append(run)
            print_run(run, index, args.duration)

    names = [name for name, kws in ANALYSIS.items() if kws]
    print_breakdown({v: r for v, r in runs.items() if r}, names)
    if failed:
        print('\nFAIL: results differ between runs')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
