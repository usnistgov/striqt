#!/usr/bin/env python3
"""Profile the binned channel power detector at the production shape, against the
tolerance model and the memory bandwidth.

`channel_power_time_series` calls `sw.iq_to_bin_power` once per detector over a view of
the waveform as ``(port, bin, samples)``. On cupy the library reduces the squared
magnitude in one `cupy.ReductionKernel` pass; on numpy it runs `envtopow` into a float32
buffer of the size of the waveform and one reduction over the samples axis (``mean`` for
``rms``, ``max`` for ``peak``). At a 10 ms detector period the reduction has 20 outputs
of about 1 million samples each.

Rows, one per detector and detector period:

- ``library``: `sw.iq_to_bin_power` as the measurement calls it.
- ``envtopow+reduce``: the 2-stage numpy path, timed on the selected backend.

Usage:
    pixi run -e cupy python tests/profiling/bin_power.py
    pixi run -e cupy python tests/profiling/bin_power.py --periods 0.01 --repeat 20
    pixi run -e test39 python tests/profiling/bin_power.py --backend numpy --duration 0.02

Reading the output:
    One row per variant, detector, and detector period: the host time until the call
    returns (``host ms``), the device time by CUDA events (``dev ms``), the input
    bytes over the device time (``GB/s``; the waveform is read once at least, so the
    bandwidth of the Tegra X2 bounds it near 20 GB/s), and the largest relative
    difference from a float64 numpy evaluation with the tolerance of
    `sw.power_analysis.bin_power_rtol` plus the accumulation term of
    `bin_power_rms` for the mean. A host time equal to the device time means the
    call blocks the host. The exit status is 1 when a variant exceeds the tolerance,
    which the ``library`` row does on numpy 2.0 (arm64) for million-sample bins: the
    float32 mean over one long contiguous axis accumulates above the pairwise model
    of `accum_rms`.
"""

from __future__ import annotations

import argparse
import dataclasses
import statistics
import sys
from collections.abc import Callable
from types import ModuleType
from typing import Any

import numpy as np
from profiling_tools import (
    device_synchronize,
    free_pool,
    select_backend,
    time_call,
    to_numpy,
)

import striqt.waveform as sw
from striqt.analysis import testing

SAMPLE_RATE = 107.52e6
PORTS = 2
DEFAULT_PERIODS = (0.01, 1 / 28000)
DETECTORS = ('rms', 'peak')

Array = Any


# %% Variants
def blocks(iq: Array, count: int) -> Array:
    return sw.arrays.axis_to_blocks(iq, count, axis=1)


def make_variants(
    iq: Array, kind: str, count: int, xp: ModuleType
) -> dict[str, Callable[[], Array]]:
    """return the callables that evaluate the `kind` detector over bins of `count`."""
    reduce = xp.mean if kind == 'rms' else xp.max
    view = blocks(iq, count)

    def library() -> Array:
        return sw.iq_to_bin_power(
            iq, kind=kind, Ts=1 / SAMPLE_RATE, Tbin=count / SAMPLE_RATE, axis=1
        )

    def envtopow_reduce() -> Array:
        return reduce(sw.envtopow(view), axis=2)

    return {'library': library, 'envtopow+reduce': envtopow_reduce}


# %% Reference and tolerance
def reference(iq_np: np.ndarray, kind: str, count: int) -> np.ndarray:
    power = np.abs(blocks(iq_np.astype(np.complex128), count)) ** 2
    return power.mean(axis=2) if kind == 'rms' else power.max(axis=2)


def tolerance(kind: str, count: int) -> float:
    return sw.power_analysis.bin_power_rtol(
        np.float32, kind=kind
    ) + sw.power_analysis.bin_power_rms(np.float32, count, kind=kind)


@dataclasses.dataclass
class Row:
    period: float
    kind: str
    variant: str
    host_s: float
    device_s: float
    rel_err: float
    """the largest relative difference from the reference"""
    tol: float

    @property
    def passed(self) -> bool:
        return self.rel_err <= self.tol


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--backend', choices=('cupy', 'numpy'), default='cupy')
    parser.add_argument('--duration', type=float, default=0.1)
    parser.add_argument(
        '--periods', type=float, nargs='+', default=list(DEFAULT_PERIODS)
    )
    parser.add_argument('--repeat', type=int, default=10)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    xp, _ = select_backend(args.backend)
    size = round(args.duration * SAMPLE_RATE)
    iq_np = np.asarray(
        testing.circular_awgn(
            None, SAMPLE_RATE, ports=PORTS, count=size, seed=args.seed
        ),
        dtype=np.complex64,
    )
    iq = xp.asarray(iq_np)
    in_bytes = iq.nbytes
    print(
        f'waveform {tuple(iq.shape)} complex64 ({in_bytes / 2**20:.0f} MB) at '
        f'{SAMPLE_RATE / 1e6:g} MS/s'
    )

    rows: list[Row] = []
    print(
        f'\n{"period s":>9} {"bins":>5} {"kind":<5} {"variant":<16} {"host ms":>8} '
        f'{"dev ms":>8} {"GB/s":>6} {"rel err":>9} {"tol":>9}  check'
    )
    for period in args.periods:
        count = round(period * SAMPLE_RATE)
        if size % count:
            parser.error(f'period {period} is not a whole number of bins of {size}')
        bins = size // count
        for kind in DETECTORS:
            ref = reference(iq_np, kind, count)
            tol = tolerance(kind, count)
            for name, func in make_variants(iq, kind, count, xp).items():
                for _ in range(args.warmup):
                    func()
                device_synchronize()
                times = [time_call(func)[:2] for _ in range(args.repeat)]
                host = statistics.median(t[0] for t in times)
                dev = statistics.median(t[1] for t in times)
                result = to_numpy(func())
                err = float(np.max(np.abs(result / ref - 1)))
                row = Row(period, kind, name, host, dev, err, tol)
                rows.append(row)
                check = 'ok' if row.passed else 'FAIL'
                print(
                    f'{period:9.5f} {bins:5d} {kind:<5} {name:<16} {host * 1e3:8.2f} '
                    f'{dev * 1e3:8.2f} {in_bytes / dev / 1e9:6.1f} {err:9.2e} '
                    f'{tol:9.2e}  {check}',
                    flush=True,
                )
            free_pool()

    failed = [f'{r.variant}/{r.kind}@{r.period:g}' for r in rows if not r.passed]
    if failed:
        print(f'\nFAIL: {", ".join(failed)}')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
