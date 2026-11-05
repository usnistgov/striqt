#!/usr/bin/env python3
"""Profile the record-sized copies that follow a lost array layout, at the ADA shapes.

Each row times one library call as the current code makes it, on the capture and source
of the aggregate-directivity-acquisition fragments ``denver-3750`` and ``7201b``
(2 ports, 0.1 s at 107.52 MS/s from the 125 MHz master clock, a 2 ms transient
holdoff, and a 20 ms trigger strobe). Nothing under ``src/`` is modified. The same run
before and after a change to the library is the measurement of that change.

Rows:

- ``cast_iq whole + view``: `cast_iq` over the whole receive buffer, then a
  ``[:, span]`` view of the device copy. ``resident`` is the size of the device buffer
  that the view keeps alive.
- ``cast_iq column slice``: `cast_iq` over the ``[:, span]`` column slice of the host
  buffer, the copy that `Controller.read_iq` makes with ``start`` at the end of the
  holdoff. The difference between the 2 rows is the holdoff fraction of the H2D
  bytes.
- ``pinned alloc + free``: one allocation and release of the throwaway port buffer that
  `_alloc_empty_iq` makes on every read under ``stream_all_rx_ports``.
- ``correlate pss`` and ``correlate sss``: `correlate_sync_spans` at 15.36 MS/s, 30 kHz
  subcarrier spacing (SSB case C), 2 ports, and 5 synchronization blocks, for the 3 PSS templates
  and for ``--cells`` of the 1008 SSS templates. ``pool peak`` is the high-water mark
  of the cupy pool during one call above the bytes in use before it, sampled by a
  memory hook at every allocation. The 8 GB Tegra X2 is killed by the kernel at the
  full 1008 cells, so the default is 252; the same count before and after a change
  is the comparison.
- ``histogram flatten`` and ``histogram ravel``: `xp.histogram` over one port of a
  spectrogram at the ADA shape, from a copy and from a view.

Usage:
    pixi run -e cupy python tests/profiling/contiguity.py
    pixi run -e cupy python tests/profiling/contiguity.py --cells 504 --repeat 3
    pixi run -e test39 python tests/profiling/contiguity.py --backend numpy

Reading the output:
    ``host`` is the wall time of the call, and ``device`` the CUDA event span around it
    (equal to ``host`` on numpy). ``bytes`` is the size of the array the call returns,
    and the last column the quantity named in the row description. Rows are the median
    of ``--repeat`` calls after ``--warmup`` calls.
"""

from __future__ import annotations

import argparse
import math
import platform
import statistics
import time
from collections.abc import Callable
from fractions import Fraction
from typing import Any

import numpy as np

import striqt.sensor as ss
import striqt.waveform as sw
from striqt.analysis.measurements import power
from striqt.sensor.lib.sources import buffers

cp: Any = None

# fragments/captures/denver-3750.yaml
PORTS: tuple[int, ...] = (0, 1)
SAMPLE_RATE = 107.52e6
ANALYSIS_BANDWIDTH = 100e6
DURATION = 0.1
CAPTURE = {
    'center_frequency': 3750e6,
    'analysis_bandwidth': ANALYSIS_BANDWIDTH,
    'sample_rate': SAMPLE_RATE,
    'port': PORTS,
    'gain': -10,
    'duration': DURATION,
}
TRIGGER_STROBE = 20e-3

# fragments/analysis/cellular.yaml spectrogram and spectrogram_histogram
FREQUENCY_RESOLUTION = 15000
FRACTIONAL_OVERLAP = Fraction(13, 28)
INTEGRATION_BANDWIDTH = 360e3
HISTOGRAM = {'power_low': -150, 'power_high': 0, 'power_resolution': 0.1}

SYNC_SAMPLE_RATE = 15.36e6
SYNC_SCS = 30e3
SYNC_CASE = 'c'
SYNC_BLOCKS = 5
SSS_CELL_COUNT = 1008
DEFAULT_SSS_CELLS = 252


# %% Timing
def device_synchronize() -> None:
    if cp is not None:
        cp.cuda.Device().synchronize()


def free_pool() -> None:
    """release the unused blocks of the cupy memory pool, so a row starts from an empty
    pool; nothing on numpy."""
    if cp is not None:
        cp.get_default_memory_pool().free_all_blocks()


def time_call(func: Callable[[], Any]) -> tuple[float, float, Any]:
    """return the (host, device) duration (in s) of one call and the value it returns.

    The host time ends when the call returns; the device time is the CUDA event span
    around it, equal to the host time on numpy.
    """
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
        name = 'contiguity-peak'

        def malloc_postprocess(self, **kwargs: Any) -> None:
            peak[0] = max(peak[0], pool.used_bytes())

    with Hook():
        func()
    device_synchronize()
    return peak[0] - base


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


# %% Rows
def print_row(name: str, host: float, device: float, nbytes: int, extra: str) -> None:
    print(
        f'{name:<26} {host * 1e3:9.2f} {device * 1e3:9.2f} {nbytes / 2**20:9.1f}  {extra}'
    )


def ada_setup(backend: str) -> tuple[ss.specs.SoapyCapture, ss.specs.Source]:
    binding = ss.bindings.air7201b
    capture = binding.schema.capture(**CAPTURE)
    source = binding.schema.source(trigger_strobe=TRIGGER_STROBE, array_backend=backend)
    return capture, source


def host_empty(shape: tuple[int, ...], dtype: Any) -> np.ndarray:
    if cp is None:
        return np.empty(shape, dtype=dtype)
    from cupyx import empty_pinned  # ty: ignore[unresolved-import]

    return empty_pinned(shape, dtype=dtype)


def profile_cast_iq(backend: str, repeat: int, warmup: int) -> None:
    capture, source = ada_setup(backend)
    lead, tail = ss.lib.compute.get_correction_overlaps(capture, source, None)
    overlap = lead + tail
    output_count = buffers.get_read_count(capture, source, overlap=overlap)
    buffer_count = buffers.get_read_count(
        capture, source, include_holdoff=True, overlap=overlap
    )
    holdoff = buffer_count - output_count
    ports = len(PORTS)
    span = slice(holdoff, holdoff + output_count)

    buffer = host_empty((ports, buffer_count), np.complex64)
    buffer[:] = 0
    print(
        f'receive buffer: {ports} ports x {buffer_count} samples, '
        f'holdoff {holdoff} samples ({holdoff / buffer_count:.1%})'
    )

    def whole_then_view() -> Any:
        return buffers.cast_iq(source, buffer, buffer_count)[:, span]

    def column_slice() -> Any:
        return buffers.cast_iq(source, buffer[:, span], output_count)

    for name, func in [
        ('cast_iq whole + view', whole_then_view),
        ('cast_iq column slice', column_slice),
    ]:
        host, device, out = repeat_call(func, repeat, warmup)
        base = out if out.base is None else out.base
        print_row(
            name, host, device, out.nbytes, f'resident {base.nbytes / 2**20:.1f} MB'
        )
        del out, base

    extra_count = 2 * buffer_count
    extra_dtype = np.dtype(source.transport_dtype)

    def alloc_and_free() -> None:
        extra = host_empty((extra_count,), extra_dtype)
        del extra

    host, device, _ = repeat_call(alloc_and_free, repeat, warmup)
    print_row(
        'pinned alloc + free',
        host,
        device,
        extra_count * extra_dtype.itemsize,
        f'{extra_count} x {extra_dtype.name}',
    )


def profile_correlator(xp: Any, cells: int, repeat: int, warmup: int) -> None:
    ports = len(PORTS)
    rng = np.random.default_rng(0)

    rows = [
        (
            'correlate pss',
            sw.comms_5gnr.pss_params(
                sample_rate=SYNC_SAMPLE_RATE,
                subcarrier_spacing=SYNC_SCS,
                symbol_indexes=SYNC_CASE,
            ),
            sw.comms_5gnr.pss_5g_nr(SYNC_SAMPLE_RATE, SYNC_SCS, xp=xp),
        ),
        (
            'correlate sss',
            sw.comms_5gnr.sss_params(
                sample_rate=SYNC_SAMPLE_RATE,
                subcarrier_spacing=SYNC_SCS,
                symbol_indexes=SYNC_CASE,
            ),
            sw.comms_5gnr.sss_5g_nr(SYNC_SAMPLE_RATE, SYNC_SCS, xp=xp)[:cells],
        ),
    ]

    for name, params, templates in rows:
        shape = (ports, SYNC_BLOCKS, params.corr_size)
        spans = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        spans = xp.asarray(spans.astype('complex64'))

        def run(
            spans: Any = spans, templates: Any = templates, params: Any = params
        ) -> Any:
            return sw.comms_5gnr.correlate_sync_spans(spans, templates, params=params)

        free_pool()
        try:
            peak = peak_pool_bytes(run)
            host, device, out = repeat_call(run, repeat, warmup)
        except Exception as ex:
            if 'OutOfMemory' not in type(ex).__name__:
                raise
            print(f'{name:<26} {"-":>9} {"-":>9} {"-":>9}  out of device memory')
            free_pool()
            continue
        label = f'{templates.shape[0]} cells'
        if cp is not None:
            label += f', pool peak {peak / 2**20:.0f} MB'
        print_row(name, host, device, out.nbytes, label)
        del out, spans
        free_pool()


def profile_histogram(xp: Any, repeat: int, warmup: int) -> None:
    ports = len(PORTS)
    nperseg = round(SAMPLE_RATE / FREQUENCY_RESOLUTION)
    hop = round(nperseg * (1 - FRACTIONAL_OVERLAP))
    frames = math.floor(DURATION * SAMPLE_RATE / hop)
    bins = round(ANALYSIS_BANDWIDTH / INTEGRATION_BANDWIDTH)
    spg = xp.asarray(
        np.random
        .default_rng(0)
        .uniform(-120, -60, (ports, frames, bins))
        .astype('float32')
    )
    edges = power.make_power_histogram_bin_edges(**HISTOGRAM, xp=xp)
    print(f'spectrogram: {spg.shape} float32')

    def from_flatten() -> Any:
        return xp.histogram(spg[0].flatten(), edges)[0]

    def from_ravel() -> Any:
        return xp.histogram(spg[0].ravel(), edges)[0]

    for name, func in [
        ('histogram flatten', from_flatten),
        ('histogram ravel', from_ravel),
    ]:
        host, device, out = repeat_call(func, repeat, warmup)
        print_row(name, host, device, out.nbytes, f'{edges.size - 1} bins')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--backend', choices=('cupy', 'numpy'), default='cupy')
    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument(
        '--cells',
        type=int,
        default=DEFAULT_SSS_CELLS,
        help=f'number of SSS templates to correlate ({SSS_CELL_COUNT} is the full set)',
    )
    args = parser.parse_args()

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
    print(
        f'{platform.machine()} numpy {np.__version__} '
        f'cupy {getattr(cp, "__version__", "-")} device: {device}'
    )
    print(f'median of {args.repeat} calls after {args.warmup} warmup\n')
    print(f'{"row":<26} {"host ms":>9} {"device ms":>9} {"MB":>9}')

    profile_cast_iq(args.backend, args.repeat, args.warmup)
    profile_correlator(xp, args.cells, args.repeat, args.warmup)
    profile_histogram(xp, args.repeat, args.warmup)


if __name__ == '__main__':
    main()
