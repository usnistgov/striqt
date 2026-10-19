#!/usr/bin/env python3
"""Profile ``10*log10(x)`` against ``(10/ln 10)*log(x)`` through numexpr and cupy.fuse.

Runs against stock numpy, numexpr and (when importable) cupy, so it needs no change to
striqt. For each backend, dtype and expression variant it reports ns per element, the
ratio to the ``ten*log10`` form with the same abs/eps shape, and the max error against
a reference evaluated in the widest float the platform offers.

Usage:
    pixi run -e cupy python chores/tests/profile_log_scaling.py [--sizes 18 24]
    pixi run -e test39 python chores/tests/profile_log_scaling.py   # CPU only

Reading the output:
    ratio < 1 means the ``c*log`` form is faster than ``ten*log10``. numexpr rows with
    ``--threads 1`` are the closest proxy for the C library: numexpr calls libm
    directly, while numpy uses its own SIMD ``log`` for float32 and libm for
    ``log10``, so the ``numpy`` rows measure numpy's dispatch. ``real(...)`` applied to
    a real array upcasts through complex128, which is why that row is slow. ``rel ulp``
    is in spacing units of the output dtype over |y| >= 1 dB; ``abs/u_in`` is the
    absolute error below 1 dB in units of the dB error of one input unit roundoff.
    When ``longdouble`` is not wider than float64 (macOS arm64), the float64 rows
    measure disagreement with numpy's ``log10`` rather than with exact math.
"""

from __future__ import annotations

import argparse
import functools
import math
import platform
import time
from typing import Callable

import numexpr as ne
import numpy as np

C10 = 4.342944819032518  # 10/ln 10, correctly rounded in float64

# name: numexpr expression; `ten`, `c`, `eps` are frame scalars of the array dtype
NUMEXPR_VARIANTS = {
    'ten*log10(x)': 'ten*log10(x)',
    'c*log(x)': 'c*log(x)',
    'real(ten*log10(x))': 'real(ten*log10(x))',
    'ten*log10(abs(x))': 'ten*log10(abs(x))',
    'c*log(abs(x))': 'c*log(abs(x))',
    'ten*log10(abs(x)+eps)': 'ten*log10(abs(x)+eps)',
    'c*log(abs(x)+eps)': 'c*log(abs(x)+eps)',
}
FUSED_VARIANTS = ('ten*log10', 'c*log')


def inputs(n: int, dtype: type) -> np.ndarray:
    rng = np.random.RandomState(0)
    span = 5 if dtype is np.float32 else 10
    x = np.exp(rng.uniform(-span * math.log(10), span * math.log(10), n))
    near_one = slice(0, n // 10)
    x[near_one] = 1 + rng.uniform(-1e-3, 1e-3, n // 10)
    return x.astype(dtype)


def reference(x: np.ndarray) -> np.ndarray:
    return (10 * np.log10(x.astype(np.longdouble))).astype(np.float64)


def errors(y: np.ndarray, ref: np.ndarray, dtype: type) -> tuple[float, float]:
    err = np.abs(y.astype(np.float64) - ref)
    big = np.abs(ref) >= 1
    rel_ulp = (err[big] / np.spacing(np.abs(ref[big]).astype(dtype))).max()
    u_in = C10 * np.finfo(dtype).eps / 2
    abs_u_in = err[~big].max() / u_in
    return float(rel_ulp), float(abs_u_in)


def time_cpu(fn: Callable[[], object], n: int, repeat: int) -> float:
    loops = max(1, 2**22 // n)
    best = math.inf
    for _ in range(repeat):
        t0 = time.perf_counter()
        for _ in range(loops):
            fn()
        best = min(best, (time.perf_counter() - t0) / loops)
    return best


def profile_numexpr(x: np.ndarray, ref: np.ndarray, repeat: int, threads: int) -> list:
    ne.set_num_threads(threads or ne.detect_number_of_cores())
    dtype = x.dtype.type
    ten, c, eps = dtype(10), dtype(C10), np.finfo(dtype).tiny
    out = np.empty_like(x)
    names = {'x': x, 'ten': ten, 'c': c, 'eps': eps}
    rows = []
    for name, expr in NUMEXPR_VARIANTS.items():
        fn = functools.partial(
            ne.evaluate, expr, local_dict=names, out=out, casting='unsafe'
        )
        fn()
        rows.append((
            f'numexpr[{threads or "all"}]',
            name,
            time_cpu(fn, x.size, repeat),
            *errors(out, ref, dtype),
        ))
    return rows


def profile_numpy(x: np.ndarray, ref: np.ndarray, repeat: int) -> list:
    dtype = x.dtype.type
    out = np.empty_like(x)
    rows = []
    for name, fn in (
        ('ten*log10(x)', lambda: np.multiply(np.log10(x, out=out), dtype(10), out=out)),
        ('c*log(x)', lambda: np.multiply(np.log(x, out=out), dtype(C10), out=out)),
    ):
        fn()
        rows.append((
            'numpy',
            name,
            time_cpu(fn, x.size, repeat),
            *errors(out, ref, dtype),
        ))
    return rows


def make_fused(cp, use_log: bool, use_abs: bool, use_eps: bool):
    log = cp.log if use_log else cp.log10

    @cp.fuse()
    def kernel(x, out, scale, eps):
        v = cp.abs(x) if use_abs else x
        if use_eps:
            v = v + eps
        out[:] = scale * log(v)
        return out

    return kernel


def profile_cupy(cp, x: np.ndarray, ref: np.ndarray, repeat: int) -> list:
    from cupyx.profiler import benchmark

    dtype = x.dtype.type
    xg = cp.asarray(x)
    out = cp.empty_like(xg)
    rows = []
    for use_abs in (False, True):
        for use_eps in (False, True):
            for use_log in (False, True):
                kernel = make_fused(cp, use_log, use_abs, use_eps)
                scale = dtype(C10) if use_log else dtype(10)
                eps = np.finfo(dtype).tiny if use_eps else dtype(0)
                kernel(xg, out, scale, eps)
                cp.cuda.Device().synchronize()
                t = benchmark(
                    kernel, (xg, out, scale, eps), n_repeat=repeat
                ).gpu_times.mean()
                inner = 'abs(x)' if use_abs else 'x'
                inner = f'{inner}+eps' if use_eps else inner
                name = f'{FUSED_VARIANTS[use_log]}({inner})'
                rows.append((
                    'cupy.fuse',
                    name,
                    float(t),
                    *errors(cp.asnumpy(out), ref, dtype),
                ))
    return rows


def report(rows: list, n: int, dtype: type) -> None:
    print(f'\nn = 2**{int(math.log2(n))}, {np.dtype(dtype).name}')
    print(
        f'{"backend":<14} {"variant":<24} {"ns/elem":>8} {"ratio":>6} {"rel ulp":>8} {"abs/u_in":>9}'
    )
    baselines = {}
    for backend, name, t, *_ in rows:
        if name.startswith('ten*log10('):
            baselines[(backend, name[len('ten*log10') :])] = t
    for backend, name, t, rel_ulp, abs_u_in in rows:
        shape = (
            name.split('(', 1)[1] if name.startswith(('ten*log10(', 'c*log(')) else None
        )
        base = baselines.get((backend, '(' + shape)) if shape else None
        ratio = f'{t / base:6.2f}' if base else '     -'
        print(
            f'{backend:<14} {name:<24} {t / n * 1e9:8.2f} {ratio} {rel_ulp:8.2f} {abs_u_in:9.2f}'
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument(
        '--sizes',
        type=int,
        nargs=2,
        default=(18, 24),
        metavar=('LO', 'HI'),
        help='log2 of the smallest and largest array sizes',
    )
    parser.add_argument('--repeat', type=int, default=7)
    parser.add_argument(
        '--threads',
        type=int,
        nargs='*',
        default=[1, 0],
        help='numexpr thread counts to run (0: all cores)',
    )
    args = parser.parse_args()

    try:
        import cupy as cp

        device = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
    except (ImportError, RuntimeError):
        cp, device = None, 'none'
    print(
        f'{platform.machine()} libc={platform.libc_ver()} numpy {np.__version__} numexpr {ne.__version__} cupy device: {device}'
    )
    print(
        f'longdouble bits: {np.finfo(np.longdouble).bits} (reference is exact only above 64)'
    )

    for k in range(args.sizes[0], args.sizes[1] + 1, 2):
        for dtype in (np.float32, np.float64):
            x = inputs(2**k, dtype)
            ref = reference(x)
            rows = profile_numpy(x, ref, args.repeat)
            for threads in args.threads:
                rows += profile_numexpr(x, ref, args.repeat, threads)
            if cp is not None:
                rows += profile_cupy(cp, x, ref, args.repeat)
            report(rows, 2**k, dtype)


if __name__ == '__main__':
    main()
