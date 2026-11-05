#!/usr/bin/env python3
"""Profile the analysis of one ADA-style 3750 MHz capture, measurement by measurement.

The capture, source, and analysis are the fixtures of `profiling_tools` (2 ports, 0.1 s
at 107.52 MS/s, 12 measurements, the ``cellular_5g_pss_sync`` trigger). The input is a
buffer of ones at the source rate, with the lead and tail overlaps of the production
acquisition. Each repetition conditions the input afresh and runs the measurements
through the live registry with `striqt.analysis.analyze_by_spec` under
``block_each=True``, the call that `iterate_captures` makes in production: each
measurement is copied to the host, and the device synchronized, before the next
measurement launches.

Usage:
    pixi run -e cupy python tests/profiling/cuda_streams.py
    pixi run -e cupy python tests/profiling/cuda_streams.py --no-trigger --repeat 10
    pixi run -e test39 python tests/profiling/cuda_streams.py --backend numpy --duration 0.02
    nsys profile -t cuda,nvtx pixi run -e cupy python tests/profiling/cuda_streams.py --repeat 2

Reading the output:
    ``wall min`` and ``wall med`` are the host time between device synchronizations
    around one evaluation, over ``--repeat`` repetitions after ``--warmup``. ``x dur``
    is the minimum divided by the capture duration; the analysis keeps up with
    acquisition only below 1. ``pool MB`` is the high-water mark of the cupy memory
    pool during one further evaluation above the bytes in use before it, with the
    input buffer already allocated. The pool is kept between repetitions, as
    production keeps the pool under ``block_each``. The breakdown table is the `stopwatch`
    time of each measurement, averaged over the timed repetitions. The time of a
    conditioning stage is counted in the first measurement that selects the stage,
    and the stages are also listed indented.
"""

from __future__ import annotations

import argparse
import statistics
from types import ModuleType
from typing import Any

from profiling_tools import (
    CAPTURE,
    Setup,
    attach_stopwatch_collector,
    build_raw,
    build_setup,
    device_synchronize,
    measurement_names,
    nvtx_range,
    peak_pool_bytes,
    select_backend,
    time_call,
)

import striqt.analysis as sa
import striqt.sensor as ss


def evaluate(raw: ss.specs.AcquiredIQ, setup: Setup) -> dict[str, Any]:
    """return the results of every measurement of `setup`, by name, on a fresh
    conditioning of `raw`.

    The evaluation runs under ``block_each=True``.
    """
    raw.x[:] = 1
    device_synchronize()
    iq = ss.lib.compute.attach_corrections(
        raw, signal_trigger=setup.trigger, overwrite_x=True
    )
    options = sa.EvaluationOptions(
        as_xarray='delayed', registry=sa.registry, block_each=True
    )
    with sa.registry.cache_context(setup.capture, None), nvtx_range('analysis'):
        results = sa.analyze_by_spec(iq, setup.analysis, setup.capture, options)
    device_synchronize()
    return dict(results)


def report(
    walls: list[float], peak_bytes: int, duration: float, xp: ModuleType
) -> None:
    pool = f'{peak_bytes / 2**20:8.0f}' if xp.__name__ == 'cupy' else '       -'
    print(f'\n{"wall min":>8} {"wall med":>8} {"x dur":>6} {"pool MB":>8}')
    print(
        f'{min(walls):8.3f} {statistics.median(walls):8.3f} '
        f'{min(walls) / duration:6.1f} {pool}'
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
    print(f'\nbreakdown, mean of {repeat} repetitions')
    print(f'{"block":<36} {"s":>7} {"share":>6}')
    for name, t in sorted(times.items(), key=lambda kv: -kv[1]):
        if name in measurements:
            print(f'{name:<36} {t / repeat:7.3f} {t / total:6.1%}')
        else:
            print(f'  {name:<34} {t / repeat:7.3f} {"-":>6}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--duration', type=float, default=CAPTURE['duration'])
    parser.add_argument('--backend', choices=('cupy', 'numpy'), default='cupy')
    parser.add_argument(
        '--no-trigger',
        action='store_true',
        help='drop the cellular_5g_pss_sync trigger',
    )
    args = parser.parse_args()

    xp, _ = select_backend(args.backend)
    setup = build_setup(args.duration, args.backend, not args.no_trigger)
    names = measurement_names(setup.analysis)
    print(
        f'capture: {args.duration} s at {setup.capture.sample_rate / 1e6:g} MS/s, '
        f'trigger: {setup.trigger is not None}'
    )
    print(f'  measurements: {", ".join(names)}')

    raw = build_raw(setup, xp)
    print(f'  buffer: {raw.x.shape} complex64, {raw.x.nbytes / 2**20:.0f} MB')

    collector = attach_stopwatch_collector()
    for _ in range(args.warmup):
        evaluate(raw, setup)
    peak = peak_pool_bytes(lambda: evaluate(raw, setup))

    collector.enabled = True
    walls = []
    for _ in range(args.repeat):
        wall, _, _ = time_call(lambda: evaluate(raw, setup))
        walls.append(wall)
        print(f'evaluation: {wall:.3f} s', flush=True)
    collector.enabled = False

    report(walls, peak, args.duration, xp)
    if collector.times:
        report_breakdown(collector.times, args.repeat, names)


if __name__ == '__main__':
    main()
