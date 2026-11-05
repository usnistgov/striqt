#!/usr/bin/env python3
"""Profile the device memory peak inside the resampling stage and the cost of a
device-to-host copy from a column slice of the resampled record.

The input is the ADA 3750 MHz acquisition of `cuda_streams.build_raw` (2 ports, 0.1 s
at 107.52 MS/s resampled from the 125 MHz master clock, with the production
overlaps). Nothing under ``src/`` is modified. ``pool peak`` is the high-water mark of
the cupy memory pool during one call above the bytes in use before it, sampled by a
memory hook at every allocation; the arrays that exist before the call (the input, the
resampled record) are not counted. ``bytes`` is the size of the array the call returns.

Rows:

- ``oaresample, fresh output`` and ``oaresample, overwrite_x``: `sw.oaresample` with
  the arguments of the resampling stage, with the input kept and with the input
  consumed in place as the pipeline runs it. The peak is the stacked STFT frames, the
  reordered bins buffer, and the output together.
- ``stft, stage block size``: the analysis half of the resampler alone, so the peak
  of the row above splits into the frames and the rest.
- ``D2H contiguous``: `.get()` of the whole resampled record.
- ``D2H column slice``: `.get()` of the record without its lead and tail pads, the
  array that `iq_waveform` returns for a 2-port capture. cupy copies a non-contiguous
  array into a contiguous device temporary before the transfer.
- ``D2H row by row``: the same slice copied one contiguous row at a time into a
  preallocated host array with ``row.get(out=...)``, with no device temporary.

Usage:
    pixi run -e cupy python tests/profiling/resample_peak.py
    pixi run -e cupy python tests/profiling/resample_peak.py --repeat 3
    pixi run -e test39 python tests/profiling/resample_peak.py --backend numpy --duration 0.02

Reading the output:
    ``host`` is the wall time of the call and ``device`` the CUDA event span around
    it (equal to ``host`` on numpy). Rows are the median of ``--repeat`` calls after
    ``--warmup`` calls; the peak is sampled on one further call.
"""

from __future__ import annotations

import argparse
import platform
import statistics
import time
from collections.abc import Callable
from typing import Any, cast

import numpy as np

import striqt.sensor as ss
import striqt.waveform as sw

cp: Any = None


# %% Timing
def device_synchronize() -> None:
    if cp is not None:
        cp.cuda.Device().synchronize()


def time_call(func: Callable[[], Any]) -> tuple[float, float, Any]:
    """return the (host, device) duration (in s) of one call and the value it returns."""
    if cp is None:
        t0 = time.perf_counter()
        ret = func()
        dt = time.perf_counter() - t0
        return dt, dt, ret
    start, end = cp.cuda.Event(), cp.cuda.Event()
    device_synchronize()
    start.record()
    t0 = time.perf_counter()
    ret = func()
    host = time.perf_counter() - t0
    end.record()
    end.synchronize()
    return host, float(cp.cuda.get_elapsed_time(start, end)) / 1e3, ret


def repeat_call(
    func: Callable[[], Any], repeat: int, warmup: int
) -> tuple[float, float, Any]:
    """return the median (host, device) durations (in s) and the last returned value."""
    for _ in range(warmup):
        func()
    hosts, devices = [], []
    ret = None
    for _ in range(repeat):
        host, device, ret = time_call(func)
        hosts.append(host)
        devices.append(device)
    return statistics.median(hosts), statistics.median(devices), ret


def peak_pool_bytes(func: Callable[[], Any]) -> int:
    """return the high-water mark (in bytes) of the cupy pool during `func`, above the
    bytes in use before the call; 0 on numpy."""
    if cp is None:
        func()
        return 0
    pool = cp.get_default_memory_pool()
    base = pool.used_bytes()
    peak = [base]

    class Hook(cp.cuda.MemoryHook):
        name = 'resample-peak'

        def malloc_postprocess(self, **_: Any) -> None:
            peak[0] = max(peak[0], pool.used_bytes())

    with Hook():
        func()
    device_synchronize()
    return peak[0] - base


def print_row(
    name: str, host: float, device: float, nbytes: int, peak: int | None
) -> None:
    peak_text = '-' if peak is None else f'{peak / 2**20:.0f} MB'
    print(
        f'{name:<28} {host * 1e3:9.1f} {device * 1e3:9.1f} {nbytes / 2**20:9.1f}  '
        f'pool peak {peak_text}'
    )


def nbytes_of(x: Any) -> int:
    return int(x.nbytes)


# %% Rows
def profile_resampler(raw: ss.specs.AcquiredIQ, repeat: int, warmup: int) -> Any:
    """print the resampler rows and return the resampled record."""
    design = cast(Any, raw.resampler)
    capture = cast(Any, raw.capture)
    kws: dict[str, Any] = {
        'up': design['nfft_out'],
        'down': design['nfft'],
        'fs': design['fs_sdr'],
        'window': design['window'],
        'axis': 1,
        'frequency_shift': design['lo_offset'],
        'filter_bandwidth': capture.analysis_bandwidth,
        'transition_bandwidth': ss.lib.compute.corrections.FIR_TRANSITION_BW,
    }
    xp = sw.array_namespace(raw.x)
    nfft = design['nfft']
    print(
        f'input: {raw.x.shape} complex64, {raw.x.nbytes / 2**20:.0f} MB; '
        f'resampler nfft {nfft} -> {design["nfft_out"]}'
    )

    def fresh_output() -> Any:
        return sw.oaresample(raw.x, overwrite_x=False, **kws)

    def in_place() -> Any:
        raw.x[:] = 1
        return sw.oaresample(raw.x, overwrite_x=True, **kws)

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

    resampled = None
    for name, func in [
        ('oaresample, fresh output', fresh_output),
        ('oaresample, overwrite_x', in_place),
        ('stft, stage block size', stft_only),
    ]:
        sw.arrays.free_cupy_mempool()
        peak = peak_pool_bytes(func)
        host, device, out = repeat_call(func, repeat, warmup)
        if name.startswith('oaresample') and resampled is None:
            resampled = out
        print_row(name, host, device, nbytes_of(out), peak)
        del out
        raw.x[:] = 1
    sw.arrays.free_cupy_mempool()
    assert resampled is not None
    return xp.asarray(resampled)


def profile_d2h(resampled: Any, capture: Any, repeat: int, warmup: int) -> None:
    ports, width = resampled.shape
    record = round(capture.duration * capture.sample_rate)
    pad = (width - record) // 2
    sliced = resampled[:, pad : pad + record]
    print(
        f'resampled record: {resampled.shape} complex64; '
        f'iq_waveform slice {sliced.shape} ({sliced.nbytes / 2**20:.0f} MB)'
    )

    def to_host(x: Any) -> Callable[[], Any]:
        if cp is None:
            return lambda: np.array(x)
        return x.get

    def row_by_row() -> Any:
        host = np.empty(sliced.shape, dtype=sliced.dtype)
        if cp is None:
            host[:] = sliced
            return host
        for i in range(ports):
            sliced[i].get(out=host[i])
        return host

    for name, func in [
        ('D2H contiguous', to_host(resampled)),
        ('D2H column slice', to_host(sliced)),
        ('D2H row by row', row_by_row),
    ]:
        sw.arrays.free_cupy_mempool()
        peak = peak_pool_bytes(func)
        host, device, out = repeat_call(func, repeat, warmup)
        print_row(name, host, device, nbytes_of(out), peak)
        del out

    if cp is not None:
        reference = sliced.get()
        assert np.array_equal(row_by_row(), reference), 'row-by-row copy differs'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--backend', choices=('cupy', 'numpy'), default='cupy')
    parser.add_argument('--duration', type=float, default=0.1)
    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument('--warmup', type=int, default=1)
    args = parser.parse_args()

    import cuda_streams

    global cp
    if args.backend == 'cupy':
        import cupy  # ty: ignore[unresolved-import]

        cp = cupy
        sw.arrays.configure_cupy()
        xp: Any = cp
        device = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
    else:
        xp = np
        device = 'none'
    cuda_streams.cp = cp
    print(
        f'{platform.machine()} numpy {np.__version__} '
        f'cupy {getattr(cp, "__version__", "-")} device: {device}'
    )
    print(f'median of {args.repeat} calls after {args.warmup} warmup\n')

    setup = cuda_streams.build_setup(args.duration, args.backend, trigger=False)
    raw = cuda_streams.build_raw(setup, xp)
    print(f'{"row":<28} {"host ms":>9} {"device ms":>9} {"MB":>9}')
    resampled = profile_resampler(raw, args.repeat, args.warmup)
    del raw
    profile_d2h(resampled, setup.capture, args.repeat, args.warmup)


if __name__ == '__main__':
    main()
