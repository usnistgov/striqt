#!/usr/bin/env python3
"""Measure ulp errors of the striqt.waveform.power_analysis conversions per backend.

Run on a machine with a GPU to ground the ulp budgets used by
tests/waveform/test_power_analysis.py. References are computed with `decimal` at 40
digits from the exact float inputs, so the reported errors are those of the
implementation alone (numexpr/libm on CPU, cupy.fuse/CUDA libdevice on GPU).

Usage:
    uv run --extra test --extra gpu python chores/tests/measure_db_accuracy.py [--points 5000]

Columns:
    rel ulp   max error in ulps of the output dtype, over outputs with |y| >= 1 (dB) or
              all outputs (linear); this is the number to compare against a per-function
              ulp budget
    abs/u_in  for dB outputs with |y| < 1 dB, max absolute error divided by
              (scale/ln 10) * u, i.e. in units of the dB error caused by a one-unit-
              roundoff error of the *input*; this is what bounds the error near 0 dB
              where ulps of the output are meaningless
"""

from __future__ import annotations

import argparse
import math
from decimal import Decimal, getcontext
from functools import partial

import numpy as np

getcontext().prec = 40


def _u(dtype):
    return np.finfo(dtype).eps / 2


def _to_numpy(x):
    return x.get() if hasattr(x, 'get') else np.asarray(x)


def _ulps(actual, ref, dtype):
    err = np.abs(actual.astype(np.float64) - ref)
    return err / np.spacing(np.abs(ref).astype(dtype)).astype(np.float64)


def _d(v):
    return Decimal(float(v))


def ref_log10(x, scale):
    return np.array([float(scale * _d(v).log10()) for v in x])


def ref_log10_complex(z, scale):
    return np.array([
        float(scale * (_d(v.real) ** 2 + _d(v.imag) ** 2).sqrt().log10()) for v in z
    ])


def ref_pow10(x):
    return np.array([float(Decimal(10) ** (_d(v) / 10)) for v in x])


def ref_abs2(z):
    return np.array([float(_d(v.real) ** 2 + _d(v.imag) ** 2) for v in z])


def ref_linstat(x, stat):
    lin = [Decimal(10) ** (_d(v) / 10) for v in x]
    total = sum(lin)
    if stat == 'mean':
        total /= len(lin)
    return float(10 * total.log10())


def report_elementwise(name, backends, func, x, ref, dtype, db_scale=None):
    cells = []
    for xp, _ in backends.values():
        y = _to_numpy(func(xp.asarray(x)))
        if np.iscomplexobj(y):
            y = y.real
        y = y.astype(np.float64)
        if db_scale is None:
            cells.append(f'{_ulps(y, ref, dtype).max():8.2f} {"":>9}')
        else:
            big = np.abs(ref) >= 1
            rel = _ulps(y[big], ref[big], dtype).max() if big.any() else float('nan')
            small = ~big
            abs_err = (
                np.abs(y[small] - ref[small]).max() if small.any() else float('nan')
            )
            abs_u = abs_err / (db_scale / math.log(10) * _u(dtype))
            cells.append(f'{rel:8.2f} {abs_u:9.2f}')
    print(f'{name:>28} {np.dtype(dtype).name:>8} ' + ' '.join(cells))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--points', type=int, default=5000)
    args = parser.parse_args()

    from striqt.waveform.lib import power_analysis as pa

    backends = {'numpy': (np, None)}
    try:
        import cupy as cp  # type: ignore

        cp.zeros(1).sum()
        backends['cupy'] = (cp, None)
        name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
        print(f'cupy {cp.__version__} on {name}')
    except Exception as exc:  # noqa: BLE001
        print(f'cupy unavailable ({exc!r}); measuring the CPU backend only')

    rng = np.random.default_rng(0)
    M = args.points
    hdr = ' '.join(f'{b + " rel ulp":>8} {b + " abs/u_in":>9}' for b in backends)
    print(f'\n{"function":>28} {"dtype":>8} {hdr}')

    for dt in (np.float32, np.float64):
        cdt = np.complex64 if dt is np.float32 else np.complex128
        u = _u(dt)

        # log-domain conversions over the ranges used by the cross-backend tests, with
        # extra points within a few ulps of 1 (0 dB), where output ulps are meaningless
        lo, hi = (1e-5, 1e5) if dt is np.float32 else (1e-10, 1e10)
        x = np.exp(rng.uniform(math.log(lo), math.log(hi), M)).astype(dt)
        x[: M // 10] = (1 + rng.uniform(-1e3, 1e3, M // 10) * u).astype(dt)
        z = (x * np.exp(1j * rng.uniform(0, 2 * math.pi, M))).astype(cdt)

        report_elementwise(
            'powtodB real',
            backends,
            partial(pa.powtodB, min_dtype=dt),
            x,
            ref_log10(x, 10),
            dt,
            db_scale=10,
        )
        report_elementwise(
            'powtodB complex',
            backends,
            partial(pa.powtodB, min_dtype=dt),
            z,
            ref_log10_complex(z, 10),
            dt,
            db_scale=10,
        )
        report_elementwise(
            'envtodB real',
            backends,
            partial(pa.envtodB, min_dtype=dt),
            x,
            ref_log10(x, 20),
            dt,
            db_scale=20,
        )
        report_elementwise(
            'envtodB complex',
            backends,
            partial(pa.envtodB, min_dtype=dt),
            z,
            ref_log10_complex(z, 20),
            dt,
            db_scale=20,
        )
        report_elementwise(
            'envtopow real',
            backends,
            partial(pa.envtopow, min_dtype=dt),
            x,
            np.array([float(_d(v) ** 2) for v in x]),
            dt,
        )
        report_elementwise(
            'envtopow complex',
            backends,
            partial(pa.envtopow, min_dtype=dt),
            z,
            ref_abs2(z),
            dt,
        )

        lim = 30 if dt is np.float32 else 100
        xdb = rng.uniform(-lim, lim, M).astype(dt)
        report_elementwise(
            f'dBtopow |x|<={lim} dB',
            backends,
            partial(pa.dBtopow, min_dtype=dt),
            xdb,
            ref_pow10(xdb),
            dt,
        )

    # reductions: float64 only, as in the tests (N <= 50, |x| <= 50 dB)
    print(
        f'\n{"reduction (float64)":>28} {"":>8} '
        + ' '.join(f'{b + " max abs err / u":>18}' for b in backends)
    )
    for stat, func in (('mean', pa.dBlinmean), ('sum', pa.dBlinsum)):
        worst = dict.fromkeys(backends, 0.0)
        for _ in range(200):
            n = rng.integers(2, 51)
            arr = rng.uniform(-50, 50, n)
            ref = ref_linstat(arr, stat)
            for b, (xp, _) in backends.items():
                y = float(_to_numpy(func(xp.asarray(arr), axis=None)))
                worst[b] = max(worst[b], abs(y - ref) / _u(np.float64))
        print(
            f'{"dBlin" + stat:>28} {"":>8} '
            + ' '.join(f'{worst[b]:18.1f}' for b in backends)
        )

    print(
        '\nInterpretation: rel ulp should stay within the per-function ulp budget of the'
        ' tests (library log10/pow/hypot error plus one rounding for the scale factor);'
        ' abs/u_in should stay within a few units for real inputs and within the hypot'
        ' budget for complex inputs. Reductions report absolute dB error in units of u.'
    )


if __name__ == '__main__':
    main()
