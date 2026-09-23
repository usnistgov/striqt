#!/usr/bin/env python3
"""Measure ulp errors of the striqt.waveform.power_analysis conversions per backend.

Run on a machine with a GPU to ground the ulp budgets used by
tests/waveform/test_power_analysis.py and the reduction roundoff model in
striqt.waveform.lib.arrays. References for the elementwise conversions are computed
with `decimal` at 40 digits from the exact float inputs, so the reported errors are
those of the implementation alone (numexpr/libm on CPU, cupy.fuse/CUDA libdevice on
GPU).

Usage:
    pixi run -e cupy python chores/tests/measure_db_accuracy.py [--points 5000]
    pixi run -e test39 python chores/tests/measure_db_accuracy.py   # CPU only

Columns (elementwise conversions):
    rel ulp   max error in ulps of the output dtype, over outputs with |y| >= 1 (dB) or
              all outputs (linear); this is the number to compare against a per-function
              ulp budget
    abs/u_in  for dB outputs with |y| < 1 dB, max absolute error divided by
              (scale/ln 10) * u, i.e. in units of the dB error caused by a one-unit-
              roundoff error of the *input*; this is what bounds the error near 0 dB
              where ulps of the output are meaningless

Columns (binned power mean, float32, the production `iq_to_bin_power` path over
Gaussian noise and a unit tone against a float64 reference):
    max/u        max relative error in units of the float32 unit roundoff u = 2**-24
    rms/u        rms relative error over the rows and draws, same units
    rms model/u  `accum_rms(float32, n)`, the shipped rms model (with its safety
                 factor) for a reduction over a contiguous axis; the worst element
                 over k outputs is peak_factor(k) times it
    n·u model/u  `accum_rtol(float32, n)`, the recursive-summation bound that applies
                 to the strided-axis row
The measurements ground REDUCTION_RUN and REDUCTION_SAFETY in
striqt.waveform.lib.arrays; the printed cupy accelerators tell whether CUB handled the
reduction.
"""

from __future__ import annotations

import argparse
import math
from decimal import Decimal, getcontext
from functools import partial

import numpy as np

try:
    # workaround a library linkage bug
    import numba.cuda  # noqa: F401
except ImportError:
    pass
getcontext().prec = 40

# bin sizes for the float32 power mean; the largest is 4 x 4e6 complex64 = 128 MB
REDUCE_SIZES = (1_000, 10_000, 100_000, 1_000_000, 4_000_000)
STRIDED_SIZE = 1_000_000


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


def _iq_draws(rng, n, draws):
    """(label, iq) pairs of complex64 shape (4, n): circular Gaussian, then a unit
    tone at a frequency that is not bin-centred"""
    t = np.arange(n)
    for _ in range(draws):
        iq = (
            rng.standard_normal((4, n)) + 1j * rng.standard_normal((4, n))
        ) / math.sqrt(2)
        yield 'gaussian', iq.astype(np.complex64)
        f = (rng.integers(1, 5) + 1 / 3) / n
        phase = rng.uniform(0, 2 * math.pi, (4, 1))
        yield 'tone', np.exp(2j * math.pi * f * t + 1j * phase).astype(np.complex64)


def _bin_power_mean_errors(xp, iq, strided):
    """per-row relative error of the float32 binned power mean against a float64
    reference, reducing along a contiguous axis (or a strided one, the naive path)"""
    from striqt.waveform.lib import power_analysis as pa

    ref = (np.abs(iq.astype(np.complex128)) ** 2).mean(axis=1)
    if strided:
        x = xp.asarray(np.ascontiguousarray(iq.T))
        y = pa.iq_to_bin_power(x, Ts=1, Tbin=iq.shape[1], kind='mean', axis=0)[0, :]
    else:
        y = pa.iq_to_bin_power(
            xp.asarray(iq), Ts=1, Tbin=iq.shape[1], kind='mean', axis=1
        )
        y = y[:, 0]
    y = _to_numpy(y)
    assert y.dtype == np.float32, y.dtype
    return np.abs(y.astype(np.float64) - ref) / ref


def report_binned_power_mean(backends, rng, draws=5):
    from striqt.waveform.lib import arrays as arrays_lib

    u = _u(np.float32)
    hdr = ' '.join(f'{b + " max/u":>10} {b + " rms/u":>10}' for b in backends)
    print(
        f'\n{"binned power mean (float32)":>28} {"n":>9} {hdr} {"rms model/u":>11}'
        f' {"n·u model/u":>12}'
    )
    cases = [(n, False) for n in REDUCE_SIZES] + [(STRIDED_SIZE, True)]
    for n, strided in cases:
        worst = {
            (b, label): [0.0, 0.0] for b in backends for label in ('gaussian', 'tone')
        }
        for label, iq in _iq_draws(rng, n, draws):
            for b, (xp, _) in backends.items():
                rel = _bin_power_mean_errors(xp, iq, strided) / u
                cell = worst[b, label]
                cell[0] = max(cell[0], rel.max())
                cell[1] = max(cell[1], math.sqrt(np.mean(rel**2)))
        model = arrays_lib.accum_rms(np.float32, n) / u
        naive = arrays_lib.accum_rtol(np.float32, n) / u
        for label in ('gaussian', 'tone'):
            name = f'{label} strided (naive)' if strided else label
            cells = ' '.join(
                f'{worst[b, label][0]:10.2f} {worst[b, label][1]:10.2f}'
                for b in backends
            )
            print(f'{name:>28} {n:>9} {cells} {model:11.1f} {naive:12.1f}')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--points', type=int, default=5000)
    args = parser.parse_args()

    from striqt.waveform.lib import power_analysis as pa

    print(f'numpy {np.__version__}')
    backends = {'numpy': (np, None)}
    try:
        import cupy as cp  # type: ignore

        cp.zeros(1).sum()
        backends['cupy'] = (cp, None)
        name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
        print(f'cupy {cp.__version__} on {name}')
        try:
            print(
                f'cupy routine accelerators {cp._core.get_routine_accelerators()},'
                f' reduction accelerators {cp._core.get_reduction_accelerators()}'
            )
        except Exception as exc:  # noqa: BLE001
            print(f'cupy accelerators unknown ({exc!r})')
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

    report_binned_power_mean(backends, rng)

    print(
        '\nInterpretation: rel ulp should stay within the per-function ulp budget of the'
        ' tests (library log10/pow/hypot error plus one rounding for the scale factor);'
        ' abs/u_in should stay within a few units for real inputs and within the hypot'
        ' budget for complex inputs. Reductions report absolute dB error in units of u.'
        ' In the binned power mean table, rms/u should stay below rms model/u and max/u'
        ' below a few times it on both backends; if a backend grows faster than sqrt(n),'
        ' lower REDUCTION_RUN rather than raising REDUCTION_SAFETY.'
    )


if __name__ == '__main__':
    main()
