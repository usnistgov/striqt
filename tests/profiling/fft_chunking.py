#!/usr/bin/env python3
"""Sweep `MAX_CUPY_FFT_SAMPLES` over the resampling stage and report time, pool peak,
and the difference from the unchunked result.

On cupy, `fft` and `ifft` with an ``out=`` buffer run the batched transform in groups
of at most `MAX_CUPY_FFT_SAMPLES` samples along the leading axes. cuFFT allocates a
work area of about the size of each batch for a transform length such as 25000 that is
not a power of 2, so the group size bounds that allocation. The default is 2**23. The
input is the ADA 3750 MHz acquisition of `profiling_tools.build_raw` (2 ports, 0.1 s at
107.52 MS/s from the 125 MHz master clock). Nothing under ``src/`` is modified on disk;
the knob is set through `set_max_cupy_fft_chunk` for each row and restored to the
default at exit.

Columns, one row per value of the knob:

- ``groups``: the number of FFT batches the stacked STFT frames split into.
- ``stft ms`` and ``stft peak``: the forward STFT of the stage alone, device time and
  the pool high-water mark above the bytes in use before the call.
- ``oaresample ms`` and ``oaresample peak``: the whole resampling stage with the
  production arguments, consuming the input in place as the pipeline does.
- ``max |diff|``: the largest absolute difference of the resampled record from the
  first row, and ``equal`` when the arrays are identical.

Usage:
    pixi run -e cupy python tests/profiling/fft_chunking.py
    pixi run -e cupy python tests/profiling/fft_chunking.py --chunks 0 33554432 8388608 --repeat 3
    pixi run -e test39 python tests/profiling/fft_chunking.py --backend numpy --duration 0.02

A chunk of 0 on the command line selects None (no chunking). On numpy the knob has no
effect and every row repeats the first.
"""

from __future__ import annotations

import argparse
import math
from typing import Any

from profiling_tools import (
    build_raw,
    build_setup,
    free_pool,
    oaresample_kws,
    peak_pool_bytes,
    repeat_call,
    select_backend,
)

import striqt.sensor as ss
import striqt.waveform as sw
from striqt.waveform.lib.arrays import grouped_slices_along_axis
from striqt.waveform.lib.fourier import fft as fft_module

DEFAULT_CHUNKS = (0, 2**26, 2**25, 2**24, 2**23, 2**22)


def group_count(shape: tuple[int, ...], chunk: int | None, axis: int) -> int:
    if chunk is None:
        return 1
    slices = grouped_slices_along_axis(shape, chunk, axis=axis)
    return math.prod(len(s) for s in slices)


def sweep(
    raw: ss.specs.AcquiredIQ, chunks: list[int | None], repeat: int, warmup: int
) -> None:
    design: Any = raw.resampler
    xp = sw.array_namespace(raw.x)
    nfft = design['nfft']
    kws = oaresample_kws(raw)

    def stft_only() -> Any:
        return sw.stft(
            raw.x,
            fs=design['fs_sdr'],
            window=design['window'],
            nperseg=nfft,
            noverlap=nfft // 2,
            axis=1,
            truncate=True,
        )[2]

    def resample() -> Any:
        raw.x[:] = 1
        return sw.oaresample(raw.x, overwrite_x=True, **kws)

    frames_shape = tuple(stft_only().shape)
    print(
        f'input: {raw.x.shape} complex64, {raw.x.nbytes / 2**20:.0f} MB; STFT frames '
        f'{frames_shape} ({math.prod(frames_shape) * 8 / 2**20:.0f} MB); '
        f'resampler nfft {nfft} -> {design["nfft_out"]}\n'
    )
    print(
        f'{"chunk samples":>14} {"groups":>6} {"stft ms":>8} {"stft peak":>10} '
        f'{"oaresample ms":>13} {"oaresample peak":>15} {"max |diff|":>11}'
    )

    reference = None
    for chunk in chunks:
        fft_module.set_max_cupy_fft_chunk(chunk)
        groups = group_count(frames_shape, chunk, axis=2)

        free_pool()
        stft_peak = peak_pool_bytes(stft_only)
        _, stft_s, frames = repeat_call(stft_only, repeat, warmup)
        del frames
        raw.x[:] = 1

        free_pool()
        resample_peak = peak_pool_bytes(resample)
        _, resample_s, out = repeat_call(resample, repeat, warmup)

        if reference is None:
            reference = out
            diff = 'equal'
        else:
            delta = float(xp.abs(out - reference).max())
            diff = 'equal' if delta == 0 else f'{delta:.3g}'
        del out

        label = 'None' if chunk is None else str(chunk)
        print(
            f'{label:>14} {groups:>6} {stft_s * 1e3:8.1f} '
            f'{stft_peak / 2**20:7.0f} MB {resample_s * 1e3:13.1f} '
            f'{resample_peak / 2**20:12.0f} MB {diff:>11}',
            flush=True,
        )
        free_pool()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--backend', choices=('cupy', 'numpy'), default='cupy')
    parser.add_argument('--duration', type=float, default=0.1)
    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument(
        '--chunks',
        type=int,
        nargs='+',
        default=list(DEFAULT_CHUNKS),
        help='values of MAX_CUPY_FFT_SAMPLES to sweep; 0 selects None',
    )
    args = parser.parse_args()
    chunks: list[int | None] = [None if c == 0 else c for c in args.chunks]

    xp, _ = select_backend(args.backend)
    print(f'median of {args.repeat} calls after {args.warmup} warmup')

    setup = build_setup(args.duration, args.backend, trigger=False)
    raw = build_raw(setup, xp)
    default_chunk = fft_module.get_max_cupy_fft_chunk()
    try:
        sweep(raw, chunks, args.repeat, args.warmup)
    finally:
        fft_module.set_max_cupy_fft_chunk(default_chunk)


if __name__ == '__main__':
    main()
