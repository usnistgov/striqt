#!/usr/bin/env python3
"""Measure float32 FFT roundoff constants for the FFT backends used by striqt.

Run on a machine with a GPU to ground the `c` constant in the error model used by
tests/waveform/test_fourier.py:

    rms_rel_error = c * eps * sqrt(log2(N))        (per FFT pass, eps = 2**-24)

Usage:
    uv run --extra test --extra gpu python chores/tests/measure_fft_accuracy.py [--trials 32]

The single-fft tables report the fitted c per backend; the pipeline table reports the
ratio of the measured error to the model with c = C_ASSUMED (the value used by the
tests). Any c or ratio consistently above C_ASSUMED / 1.0 means the constant assumed
by the tests is too optimistic for that backend.
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import scipy.fft

EPS32 = 2.0**-24
SMOOTH_SIZES = [64, 128, 256, 512, 1024, 4096, 16384, 96, 100, 250, 1000]
# sizes with a prime factor >= 128: cuFFT falls back to Bluestein for these
BLUESTEIN_SIZES = [254, 1018, 4094, 2 * 8191]
C_ASSUMED = 2.2


def _rms(x):
    return math.sqrt(np.mean(np.abs(x) ** 2))


def _model_fft(n, c=1.0):
    return c * EPS32 * math.sqrt(math.log2(n))


def _to_numpy(x):
    return x.get() if hasattr(x, 'get') else np.asarray(x)


def _inputs(rng, n, kind):
    if kind == 'noise':
        x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    else:
        x = np.exp(2j * np.pi * rng.uniform(-n / 4, n / 4) * np.arange(n) / n)
    return x.astype(np.complex64)


def measure_fft(backends, sizes, kind, trials, rng):
    """single FFT pass: per-backend c estimate, peak statistics, and cross-backend quadrature check"""
    names = list(backends)
    print(f'\n== single fft, {kind} input, complex64 ==')
    hdr = ' '.join(f'{f"c[{b}]":>10}' for b in names)
    print(f'{"N":>6} {hdr} {"diff/quad":>10} {"pk/rms/√lnM":>12} {"pk/peak/ε":>10}')

    for n in sizes:
        rel = {b: [] for b in names}
        diff = []
        pk_rms, pk_peak = [], []
        for _ in range(trials):
            x = _inputs(rng, n, kind)
            ref = scipy.fft.fft(x.astype(np.complex128))
            outs = {}
            for b, (xp, fft) in backends.items():
                y = _to_numpy(fft(xp.asarray(x))).astype(np.complex128)
                outs[b] = y
                rel[b].append(_rms(y - ref) / _rms(ref))
            if len(names) == 2:
                a, b = (outs[k] for k in names)
                diff.append(_rms(a - b) / _rms(ref))
            err = outs[names[-1]] - ref
            pk_rms.append(np.max(np.abs(err)) / _rms(ref))
            pk_peak.append(np.max(np.abs(err)) / np.max(np.abs(ref)))

        c = {b: np.mean(rel[b]) / _model_fft(n) for b in names}
        if diff:
            quad = math.sqrt(sum(np.mean(rel[b]) ** 2 for b in names))
            quad_ratio = f'{np.mean(diff) / quad:10.2f}'
        else:
            quad_ratio = f'{"n/a":>10}'
        sigma = np.mean(rel[names[-1]])
        m = n * trials
        print(
            f'{n:>6} ' + ' '.join(f'{c[b]:10.2f}' for b in names) + f' {quad_ratio}'
            f' {np.max(pk_rms) / (sigma * math.sqrt(math.log(m))):12.2f}'
            f' {np.max(pk_peak) / EPS32:10.1f}'
        )


def measure_pipelines(backends, trials, rng):
    """striqt fourier functions vs. a float64 reference, compared to the composed model"""
    from striqt.waveform import fourier

    mul = (EPS32 / math.sqrt(3)) ** 2

    def model(nffts, n_mul, c=C_ASSUMED):
        return math.sqrt(n_mul * mul + sum(_model_fft(n, c) ** 2 for n in nffts))

    stft_kws = {'fs': 1e6, 'window': 'hamming', 'nperseg': 64, 'noverlap': 32}
    h = np.array([0.25, 0.5, 0.25], dtype=np.float32)

    def run(xp, x, name):
        if name == 'stft':
            return fourier.stft(x, **stft_kws)[2]
        if name == 'spectrogram':
            return fourier.spectrogram(x, **stft_kws)[2]
        if name == 'resample':
            return fourier.resample(x, x.shape[0] // 2)
        if name == 'oaconvolve':
            return fourier.oaconvolve(
                x.real, xp.asarray(h.astype(x.real.dtype)), mode='same'
            )
        raise KeyError(name)

    names = list(backends)
    print(f'\n== striqt pipelines vs float64 reference, model with c={C_ASSUMED} ==')
    hdr = ' '.join(f'{f"meas/model[{b}]":>16}' for b in names)
    print(f'{"func":>12} {"N":>5} {"kind":>6} {hdr} {"diff/quad":>10}')

    for n in (256, 512):
        models = {
            'stft': model([64], 2),
            'spectrogram': model([64], 3),
            'resample': model([n, n // 2], 2),
            'oaconvolve': model([n, n], 1),
        }
        for kind in ('noise', 'tone'):
            for func, m in models.items():
                rel = {b: [] for b in names}
                diff = []
                for _ in range(trials):
                    x = _inputs(rng, n, kind)
                    ref = _to_numpy(run(np, x.astype(np.complex128), func))
                    ref_rms = _rms(ref)
                    if func == 'oaconvolve':
                        # the 3-tap lowpass attenuates noise by sqrt(0.375); anchor to the input
                        ref_rms = _rms(x.real) * math.sqrt(0.375)
                    outs = {}
                    for b, (xp, _) in backends.items():
                        y = _to_numpy(run(xp, xp.asarray(x), func)).astype(
                            np.complex128
                        )
                        outs[b] = y
                        rel[b].append(_rms(y - ref) / ref_rms)
                    if len(names) == 2:
                        a, b = (outs[k] for k in names)
                        diff.append(_rms(a - b) / ref_rms)
                ratios = ' '.join(f'{np.mean(rel[b]) / m:16.2f}' for b in names)
                if diff:
                    quad = math.sqrt(sum(np.mean(rel[b]) ** 2 for b in names))
                    q = f'{np.mean(diff) / quad:10.2f}'
                else:
                    q = f'{"n/a":>10}'
                print(f'{func:>12} {n:>5} {kind:>6} {ratios} {q}')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--trials', type=int, default=32)
    args = parser.parse_args()

    from striqt.waveform import fourier

    backends = {'numpy': (np, fourier.fft)}
    try:
        import cupy as cp  # type: ignore

        cp.zeros(1).sum()
        backends['cupy'] = (cp, fourier.fft)
        print(
            f'cupy {cp.__version__} on {cp.cuda.runtime.getDeviceProperties(0)["name"].decode()}'
        )
    except Exception as exc:  # noqa: BLE001
        print(f'cupy unavailable ({exc!r}); measuring the CPU backend only')
    print(f'scipy {scipy.__version__}, numpy {np.__version__}')

    rng = np.random.default_rng(0)
    for kind in ('noise', 'tone'):
        measure_fft(backends, SMOOTH_SIZES, kind, args.trials, rng)
        measure_fft(backends, BLUESTEIN_SIZES, kind, args.trials, rng)
    measure_pipelines(backends, args.trials, rng)

    print(
        '\nReading the tables: c is the fitted constant in rms_rel = c*eps*sqrt(log2 N);'
        ' diff/quad should be ~1.0 if backend errors are independent; pk/rms/√lnM ~1-1.5'
        ' for noise; pk/peak/ε is the tone peak error in units of eps.'
    )


if __name__ == '__main__':
    main()
