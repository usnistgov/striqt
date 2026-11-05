#!/usr/bin/env python3
"""Profile the convolution inside the PSS correlator against block-FFT paths built from
the primitives of `striqt.waveform`.

The input reproduces the operands of `correlate_sync_sequence`: the SSB-band waveform
at 15.36 MS/s cut into 30 kHz synchronization blocks of ``corr_size`` samples, shape
``(port, sync block, sample)``, and the 3 PSS templates of `pss_5g_nr`. The waveform is
complex gaussian noise with the matched copy of one template embedded per port at a
known sample. Each path is checked by the position of the correlation peak, by the
error against a complex128 `scipy.signal.fftconvolve` reference, and by the error
against the ``baseline`` path. Nothing under ``src/`` is modified.

Paths:

- ``baseline``: `sw.oaconvolve` once per cell ID, then the roll and slice that
  `correlate_sync_sequence` applied before the direct-FFT core.
- ``oa:<M>``: overlap-add from the primitives. Blocks of ``M`` samples are copied into
  a zeroed ``(port, sync, block, 2M)`` buffer, transformed in place, multiplied by the
  template spectra for all cell IDs at once, inverse transformed in place, and
  overlap-added with `_unstack_stft_windows` at hop ``M``. The waveform spectrum is
  computed once for all cell IDs. The template spectra are computed before the timed
  region.
- ``direct:<N>``: one zero-padded ``N``-point transform per row, the broadcast
  multiply, and one inverse transform.
- ``os:<M>``: overlap-save. The waveform is front padded with ``L - 1`` zeros, stacked
  with `sliding_window_view` at hop ``M`` into blocks of ``next_fast_len(M + L - 1)``
  samples, transformed, multiplied, inverse transformed, and the first ``L - 1`` samples
  of each block discarded.

Usage:
    pixi run -e test39 python tests/profiling/pss_oaconvolve.py --backend numpy
    pixi run -e cupy python tests/profiling/pss_oaconvolve.py
    pixi run -e cupy python tests/profiling/pss_oaconvolve.py --block-sizes 2048 5120 --repeat 20

Run with its own cache directory beside a test session or another striqt script
(``STRIQT_CACHE_DIR=$(mktemp -d)``): two processes sharing the window shelf corrupt it on
macOS Python 3.9.

Reading the output:
    The base operation table times each stage of the ``oa`` and ``direct`` paths in
    isolation on the real shapes. ``GB/s`` is the bytes the stage reads and writes
    divided by the median time. An FFT makes more than one pass over the buffer, so
    the figure is a lower bound for the FFT stages. The path table reports the min
    and median time of each path, the ratio of the min to the ``baseline`` min, the
    relative rms error against the complex128 reference (``err64``) and against
    ``baseline`` (``errOA``), the tolerance of `sw.fourier.fft_tolerance_rms` for the
    path (``tol64``; ``tolOA`` adds the tolerances of the two paths in quadrature),
    and whether the correlation peak of every port is at the index of the ``baseline``
    peak. A path that misses any check is marked FAIL. The exit status is 1 when a
    path fails or when the reference peak is not at the embedded sample. ``MB`` is the
    peak of the allocations made during one call, above the bytes live at the start
    of the call: the cupy memory pool through a `cupy.cuda.MemoryHook` on cupy, and
    `tracemalloc` on numpy (numpy reports array data through the tracemalloc hooks).
    The footprint table lists the bytes of the intermediates each path allocates by
    design beside the measured peak, the bytes that grow per cell ID, and the cell ID
    count at which the path exceeds ``--memory-budget``. The ``library`` line times
    `correlate_sync_spans` on the same waveform through to the ``(port, cell, sync,
    beam, lag)`` grid, which the path table does not form, with its peak MB.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import platform
import statistics
import sys
import time
import tracemalloc
from collections.abc import Callable
from typing import Any, Literal, cast

import numpy as np
import scipy.fft
import scipy.signal
from typing_extensions import Self

import striqt.waveform as sw
from striqt.waveform.lib.arrays import sliding_window_view
from striqt.waveform.lib.fourier.stft import _unstack_stft_windows

cp: Any = None

# the cellular analysis of the ADA 3750 MHz capture (cuda_streams.ANALYSIS)
SAMPLE_RATE = 15.36e6
SUBCARRIER_SPACING = 30e3
DISCOVERY_PERIODICITY = 20e-3
SYMBOL_INDEXES = 'c'

DEFAULT_BLOCK_SIZES = (1024, 2048, 3072, 5120, 6144, 10240)
DIRECT_SIZES = (31360, 32768)
ITEM_BYTES = 8
NOISE_TO_TEMPLATE_RMS = 0.05
EMBED_START = 5000
EMBED_STRIDE = 1000

Array = Any


# %% Geometry and inputs
@dataclasses.dataclass(frozen=True)
class Geometry:
    ports: int
    """the number of ports, the first axis of the waveform"""
    sync_blocks: int
    """the number of synchronization blocks in the capture"""
    corr_size: int
    """the samples per synchronization block that the correlator convolves"""
    template_size: int
    """the samples per PSS template, including the zeroed cyclic prefix"""
    cell_count: int
    """the number of templates (cell IDs)"""
    offset: int
    """the roll (in samples) that `correlate_sync_sequence` applies before the slice"""

    @property
    def full_size(self) -> int:
        """the length of the full linear convolution of one row"""
        return self.corr_size + self.template_size - 1

    def row_bytes(self, count: int) -> int:
        """the bytes of `count` complex64 values"""
        return count * ITEM_BYTES


@dataclasses.dataclass(frozen=True)
class Inputs:
    x: Array
    """the waveform of shape ``(port, sync, corr_size)`` in the backend namespace"""
    templates: Array
    """the PSS templates of shape ``(cell, template_size)`` in the backend namespace"""
    geometry: Geometry
    expected_peaks: tuple[tuple[int, int, int], ...]
    """the ``(cell, sync, lag)`` index of the embedded peak of each port, rolled"""


def build_inputs(
    ports: int, duration: float, center_frequency: float, xp: Any, seed: int
) -> Inputs:
    params = sw.comms_5gnr.pss_params(
        sample_rate=SAMPLE_RATE,
        subcarrier_spacing=SUBCARRIER_SPACING,
        discovery_periodicity=DISCOVERY_PERIODICITY,
        symbol_indexes=SYMBOL_INDEXES,
        center_frequency=center_frequency,
    )
    templates_np = np.asarray(
        sw.comms_5gnr.pss_5g_nr(SAMPLE_RATE, SUBCARRIER_SPACING), dtype=np.complex64
    )
    cell_count, template_size = templates_np.shape
    sync_blocks = max(1, round(duration / DISCOVERY_PERIODICITY))
    # the roll of correlate_sync_sequence
    offset = round(params.sample_rate / params.subcarrier_spacing) + 2 * int(
        params.min_cp_size
    )
    geometry = Geometry(
        ports=ports,
        sync_blocks=sync_blocks,
        corr_size=params.corr_size,
        template_size=template_size,
        cell_count=cell_count,
        offset=offset,
    )

    rng = np.random.default_rng(seed)
    shape = (ports, sync_blocks, geometry.corr_size)
    template_rms = float(np.sqrt(np.mean(np.abs(templates_np[0]) ** 2)))
    noise_scale = NOISE_TO_TEMPLATE_RMS * template_rms / math.sqrt(2)
    x = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(
        np.complex64
    ) * np.complex64(noise_scale)

    peaks: list[tuple[int, int, int]] = []
    for port in range(ports):
        cell = port % cell_count
        start = EMBED_START + EMBED_STRIDE * port
        # the matched filter of a template is its conjugate reverse: the convolution
        # peaks at start + L - 1 in the full output
        x[port, 0, start : start + template_size] += np.conj(templates_np[cell][::-1])
        peaks.append((cell, 0, start + template_size - 1 - offset))

    return Inputs(
        x=xp.asarray(x),
        templates=xp.asarray(templates_np),
        geometry=geometry,
        expected_peaks=tuple(peaks),
    )


# %% Timing and memory
def device_synchronize() -> None:
    if cp is not None:
        cp.cuda.Device().synchronize()


def time_call(func: Callable[[], Any]) -> float:
    """return the duration (in s) of one call: CUDA events on cupy, wall on numpy."""
    if cp is None:
        t0 = time.perf_counter()
        func()
        return time.perf_counter() - t0
    start, end = cp.cuda.Event(), cp.cuda.Event()
    device_synchronize()
    start.record()
    func()
    end.record()
    end.synchronize()
    return float(cp.cuda.get_elapsed_time(start, end)) / 1e3


class PeakTracker:
    """the peak of the live bytes allocated through the cupy pool during a span"""

    def __init__(self) -> None:
        self.live = 0
        self.peak = 0
        self._hook: Any = None

    def __enter__(self) -> Self:
        self.live = 0
        self.peak = 0
        tracker = self

        class Hook(cp.cuda.MemoryHook):
            name = 'striqt-profile-peak'

            def malloc_postprocess(self, **kwargs: Any) -> None:
                tracker.live += int(kwargs['mem_size'])
                tracker.peak = max(tracker.peak, tracker.live)

            def free_postprocess(self, **kwargs: Any) -> None:
                tracker.live -= int(kwargs['mem_size'])

        self._hook = Hook()
        self._hook.__enter__()
        return self

    def __exit__(self, *exc: object) -> None:
        device_synchronize()
        self._hook.__exit__(*exc)


def peak_bytes(func: Callable[[], Any]) -> int:
    """return the peak bytes allocated in one call above the bytes live at the start."""
    if cp is not None:
        sw.arrays.free_cupy_mempool()
        with PeakTracker() as tracker:
            func()
        sw.arrays.free_cupy_mempool()
        return tracker.peak

    tracemalloc.start()
    try:
        base = tracemalloc.get_traced_memory()[0]
        tracemalloc.reset_peak()
        func()
        return tracemalloc.get_traced_memory()[1] - base
    finally:
        tracemalloc.stop()


# %% Paths
@dataclasses.dataclass
class Path:
    name: str
    """the label in the tables"""
    nffts: tuple[int, ...]
    """the FFT sizes of one pass, for `sw.fourier.fft_tolerance_rms`"""
    n_elementwise: int
    """the elementwise roundings of one pass, for `sw.fourier.fft_tolerance_rms`"""
    prepare: Callable[[Inputs, Any], Any]
    """build the state outside the timed region (the template spectra)"""
    run: Callable[[Inputs, Any, Any], Array]
    """return the correlation of shape ``(port, cell, sync, corr_size)``"""
    footprint: Callable[[Geometry], dict[str, int]]
    """the bytes of each intermediate the path allocates"""
    per_cell: Callable[[Geometry], int]
    """the bytes that grow with each cell ID"""


def finish(y: Array, geometry: Geometry, xp: Any) -> Array:
    """apply the roll and slice of `correlate_sync_sequence` to the full convolution."""
    y = y[..., : geometry.full_size]
    return xp.roll(y, -geometry.offset, axis=-1)[..., : geometry.corr_size]


def template_spectra(inputs: Inputs, nfft: int, xp: Any) -> Array:
    g = inputs.geometry
    buf = xp.zeros((g.cell_count, nfft), dtype=inputs.x.dtype)
    buf[:, : g.template_size] = inputs.templates
    return sw.fft(buf, axis=-1, out=buf, overwrite_x=True)


def baseline_path(g: Geometry) -> Path:
    def prepare(inputs: Inputs, xp: Any) -> None:
        return None

    def run(inputs: Inputs, state: Any, xp: Any) -> Array:
        g = inputs.geometry
        R = xp.empty(
            (g.ports, g.cell_count, g.sync_blocks, g.full_size), dtype='complex64'
        )
        for cell in range(g.cell_count):
            R[:, cell] = sw.oaconvolve(
                inputs.x,
                inputs.templates[cell][xp.newaxis, xp.newaxis, :],
                'full',
                axes=2,
            )
        return finish(R, g, xp)

    block, _, step, _ = scipy_oa_lens(g)

    def footprint(g: Geometry) -> dict[str, int]:
        nblocks = -(-g.corr_size // step)
        rows = g.ports * g.sync_blocks
        return {
            'scipy blocks (x2, per cell call)': 2 * g.row_bytes(rows * nblocks * block),
            'R (full)': g.row_bytes(rows * g.cell_count * g.full_size),
            'roll copy': g.row_bytes(rows * g.cell_count * g.full_size),
        }

    def per_cell(g: Geometry) -> int:
        return 2 * g.row_bytes(g.ports * g.sync_blocks * g.full_size)

    return Path('baseline', (block, block), 2, prepare, run, footprint, per_cell)


def scipy_oa_lens(g: Geometry) -> tuple[int, int, int, int]:
    """return scipy's ``(block, overlap, in1_step, in2_step)`` for a correlator row."""
    from scipy.signal._signaltools import _calc_oa_lens

    block, overlap, in1_step, in2_step = _calc_oa_lens(g.corr_size, g.template_size)
    return int(block), int(overlap), int(in1_step), int(in2_step)


def oa_path(M: int) -> Path:
    N = 2 * M

    def prepare(inputs: Inputs, xp: Any) -> Array:
        return template_spectra(inputs, N, xp)

    def run(inputs: Inputs, H: Array, xp: Any) -> Array:
        g = inputs.geometry
        nblocks = g.corr_size // M
        buf = xp.zeros((g.ports, g.sync_blocks, nblocks, N), dtype=inputs.x.dtype)
        buf[..., :M] = sw.arrays.axis_to_blocks(inputs.x, M, axis=2)
        X = sw.fft(buf, axis=-1, out=buf, overwrite_x=True)
        prod = xp.empty(
            (g.ports, g.cell_count, g.sync_blocks, nblocks, N), dtype=X.dtype
        )
        xp.multiply(
            X[:, xp.newaxis], H[xp.newaxis, :, xp.newaxis, xp.newaxis, :], out=prod
        )
        prod = sw.ifft(prod, axis=-1, out=prod, overwrite_x=True)
        y = _unstack_stft_windows(prod, noverlap=M, nperseg=N, axis=3)
        return finish(y, g, xp)

    def footprint(g: Geometry) -> dict[str, int]:
        nblocks = g.corr_size // M
        rows = g.ports * g.sync_blocks
        return {
            'blocks': g.row_bytes(rows * nblocks * N),
            'product': g.row_bytes(rows * g.cell_count * nblocks * N),
            'overlap-add out': g.row_bytes(rows * g.cell_count * (nblocks + 1) * M),
            'roll copy': g.row_bytes(rows * g.cell_count * g.full_size),
        }

    def per_cell(g: Geometry) -> int:
        nblocks = g.corr_size // M
        rows = g.ports * g.sync_blocks
        return g.row_bytes(rows * (nblocks * N + (nblocks + 1) * M + g.full_size))

    return Path(f'oa:{M}', (N, N), 2, prepare, run, footprint, per_cell)


def direct_path(N: int) -> Path:
    def prepare(inputs: Inputs, xp: Any) -> Array:
        return template_spectra(inputs, N, xp)

    def run(inputs: Inputs, H: Array, xp: Any) -> Array:
        g = inputs.geometry
        buf = xp.zeros((g.ports, g.sync_blocks, N), dtype=inputs.x.dtype)
        buf[..., : g.corr_size] = inputs.x
        X = sw.fft(buf, axis=-1, out=buf, overwrite_x=True)
        prod = xp.empty((g.ports, g.cell_count, g.sync_blocks, N), dtype=X.dtype)
        xp.multiply(X[:, xp.newaxis], H[xp.newaxis, :, xp.newaxis, :], out=prod)
        prod = sw.ifft(prod, axis=-1, out=prod, overwrite_x=True)
        return finish(prod, g, xp)

    def footprint(g: Geometry) -> dict[str, int]:
        rows = g.ports * g.sync_blocks
        return {
            'padded rows': g.row_bytes(rows * N),
            'product': g.row_bytes(rows * g.cell_count * N),
            'roll copy': g.row_bytes(rows * g.cell_count * g.full_size),
        }

    def per_cell(g: Geometry) -> int:
        rows = g.ports * g.sync_blocks
        return g.row_bytes(rows * (N + g.full_size))

    return Path(f'direct:{N}', (N, N), 1, prepare, run, footprint, per_cell)


def os_sizes(g: Geometry, M: int) -> tuple[int, int, int]:
    """return ``(block, block count, padded length)`` of overlap-save at hop `M`."""
    N = int(scipy.fft.next_fast_len(M + g.template_size - 1, real=False))
    nblocks = -(-g.full_size // M)
    return N, nblocks, (nblocks - 1) * M + N


def os_path(g: Geometry, M: int) -> Path:
    def prepare(inputs: Inputs, xp: Any) -> Array:
        N, _, _ = os_sizes(inputs.geometry, M)
        return template_spectra(inputs, N, xp)

    def run(inputs: Inputs, H: Array, xp: Any) -> Array:
        g = inputs.geometry
        N, nblocks, pad_len = os_sizes(g, M)
        L = g.template_size
        xpad = xp.zeros((g.ports, g.sync_blocks, pad_len), dtype=inputs.x.dtype)
        xpad[..., L - 1 : L - 1 + g.corr_size] = inputs.x
        view = sliding_window_view(xpad, N, axis=-1)[..., ::M, :]
        buf = xp.ascontiguousarray(view)
        X = sw.fft(buf, axis=-1, out=buf, overwrite_x=True)
        prod = xp.empty(
            (g.ports, g.cell_count, g.sync_blocks, nblocks, N), dtype=X.dtype
        )
        xp.multiply(
            X[:, xp.newaxis], H[xp.newaxis, :, xp.newaxis, xp.newaxis, :], out=prod
        )
        prod = sw.ifft(prod, axis=-1, out=prod, overwrite_x=True)
        valid = prod[..., L - 1 : L - 1 + M].reshape(prod.shape[:3] + (nblocks * M,))
        return finish(valid, g, xp)

    def footprint(g: Geometry) -> dict[str, int]:
        N, nblocks, pad_len = os_sizes(g, M)
        rows = g.ports * g.sync_blocks
        return {
            'padded input': g.row_bytes(rows * pad_len),
            'blocks': g.row_bytes(rows * nblocks * N),
            'product': g.row_bytes(rows * g.cell_count * nblocks * N),
            'valid copy': g.row_bytes(rows * g.cell_count * nblocks * M),
            'roll copy': g.row_bytes(rows * g.cell_count * g.full_size),
        }

    def per_cell(g: Geometry) -> int:
        N, nblocks, _ = os_sizes(g, M)
        rows = g.ports * g.sync_blocks
        return g.row_bytes(rows * (nblocks * N + nblocks * M + g.full_size))

    N, _, _ = os_sizes(g, M)
    return Path(f'os:{M}', (N, N), 1, prepare, run, footprint, per_cell)


# %% Base operations
@dataclasses.dataclass(frozen=True)
class Operation:
    group: str
    """the path the stage belongs to"""
    name: str
    """the stage"""
    func: Callable[[], Any]
    """one evaluation on prebuilt buffers"""
    bytes_moved: int
    """the bytes read plus written by one evaluation, 0 when not memory bound"""


def oa_operations(inputs: Inputs, M: int, xp: Any) -> list[Operation]:
    g = inputs.geometry
    N = 2 * M
    nblocks = g.corr_size // M
    rows = g.ports * g.sync_blocks
    dtype = inputs.x.dtype
    blocks_shape = (g.ports, g.sync_blocks, nblocks, N)
    prod_shape = (g.ports, g.cell_count, g.sync_blocks, nblocks, N)
    buf = xp.zeros(blocks_shape, dtype=dtype)
    H = template_spectra(inputs, N, xp)
    prod = xp.zeros(prod_shape, dtype=dtype)
    y = _unstack_stft_windows(prod, noverlap=M, nperseg=N, axis=3)
    blocks_bytes = g.row_bytes(rows * nblocks * N)
    prod_bytes = g.row_bytes(rows * g.cell_count * nblocks * N)
    out_bytes = g.row_bytes(rows * g.cell_count * (nblocks + 1) * M)
    full_bytes = g.row_bytes(rows * g.cell_count * g.full_size)
    x_bytes = g.row_bytes(rows * g.corr_size)

    def pad_copy() -> None:
        b = xp.zeros(blocks_shape, dtype=dtype)
        b[..., :M] = sw.arrays.axis_to_blocks(inputs.x, M, axis=2)

    group = f'oa:{M}'
    return [
        Operation(group, 'pad-copy', pad_copy, x_bytes + blocks_bytes),
        Operation(
            group,
            'fft',
            lambda: sw.fft(buf, axis=-1, out=buf, overwrite_x=True),
            2 * blocks_bytes,
        ),
        Operation(group, 'template fft', lambda: template_spectra(inputs, N, xp), 0),
        Operation(
            group,
            'multiply',
            lambda: xp.multiply(
                buf[:, xp.newaxis],
                H[xp.newaxis, :, xp.newaxis, xp.newaxis, :],
                out=prod,
            ),
            blocks_bytes + prod_bytes,
        ),
        Operation(
            group,
            'ifft',
            lambda: sw.ifft(prod, axis=-1, out=prod, overwrite_x=True),
            2 * prod_bytes,
        ),
        Operation(
            group,
            'overlap-add',
            lambda: _unstack_stft_windows(prod, noverlap=M, nperseg=N, axis=3),
            prod_bytes + out_bytes,
        ),
        Operation(group, 'roll+slice', lambda: finish(y, g, xp), 2 * full_bytes),
    ]


def direct_operations(inputs: Inputs, N: int, xp: Any) -> list[Operation]:
    g = inputs.geometry
    rows = g.ports * g.sync_blocks
    dtype = inputs.x.dtype
    buf_shape = (g.ports, g.sync_blocks, N)
    prod_shape = (g.ports, g.cell_count, g.sync_blocks, N)
    buf = xp.zeros(buf_shape, dtype=dtype)
    H = template_spectra(inputs, N, xp)
    prod = xp.zeros(prod_shape, dtype=dtype)
    buf_bytes = g.row_bytes(rows * N)
    prod_bytes = g.row_bytes(rows * g.cell_count * N)
    full_bytes = g.row_bytes(rows * g.cell_count * g.full_size)
    x_bytes = g.row_bytes(rows * g.corr_size)

    def pad_copy() -> None:
        b = xp.zeros(buf_shape, dtype=dtype)
        b[..., : g.corr_size] = inputs.x

    group = f'direct:{N}'
    return [
        Operation(group, 'pad-copy', pad_copy, x_bytes + buf_bytes),
        Operation(
            group,
            'fft',
            lambda: sw.fft(buf, axis=-1, out=buf, overwrite_x=True),
            2 * buf_bytes,
        ),
        Operation(group, 'template fft', lambda: template_spectra(inputs, N, xp), 0),
        Operation(
            group,
            'multiply',
            lambda: xp.multiply(
                buf[:, xp.newaxis], H[xp.newaxis, :, xp.newaxis, :], out=prod
            ),
            buf_bytes + prod_bytes,
        ),
        Operation(
            group,
            'ifft',
            lambda: sw.ifft(prod, axis=-1, out=prod, overwrite_x=True),
            2 * prod_bytes,
        ),
        Operation(group, 'roll+slice', lambda: finish(prod, g, xp), 2 * full_bytes),
    ]


# %% Checks
@dataclasses.dataclass
class Result:
    path: str
    """the path name"""
    times: list[float]
    """the time (in s) of each repetition"""
    peak_bytes: int
    """the peak bytes allocated during one call"""
    err_ref: float
    """relative rms error against the complex128 reference"""
    err_base: float
    """relative rms error against the ``baseline`` path"""
    tol_ref: float
    """the tolerance on `err_ref`, relative to the output rms"""
    tol_base: float
    """the tolerance on `err_base`, relative to the output rms"""
    peaks: tuple[tuple[int, int, int], ...]
    """the ``(cell, sync, lag)`` argmax of each port"""
    peaks_match: bool
    """True when every port peaks at the index of the ``baseline`` peak"""

    @property
    def passed(self) -> bool:
        return (
            self.err_ref <= self.tol_ref
            and self.err_base <= self.tol_base
            and self.peaks_match
        )


def reference(inputs_np: Inputs) -> np.ndarray:
    """return the complex128 correlation of `inputs_np` after the roll and slice."""
    x = inputs_np.x.astype(np.complex128)
    h = inputs_np.templates.astype(np.complex128)
    full = scipy.signal.fftconvolve(
        x[:, np.newaxis], h[np.newaxis, :, np.newaxis, :], mode='full', axes=-1
    )
    return finish(full, inputs_np.geometry, np)


def rel_rms_error(got: np.ndarray, ref: np.ndarray) -> float:
    diff = got.astype(np.complex128) - ref.astype(np.complex128)
    return float(np.sqrt(np.mean(np.abs(diff) ** 2) / np.mean(np.abs(ref) ** 2)))


def argmax_peaks(R: np.ndarray) -> tuple[tuple[int, int, int], ...]:
    mag = np.abs(R)
    peaks = []
    for port in range(mag.shape[0]):
        cell, sync, lag = np.unravel_index(int(mag[port].argmax()), mag[port].shape)
        peaks.append((int(cell), int(sync), int(lag)))
    return tuple(peaks)


def to_numpy(a: Array) -> np.ndarray:
    if cp is not None and isinstance(a, cp.ndarray):
        return cp.asnumpy(a)
    return np.asarray(a)


def evaluate(
    path: Path,
    inputs: Inputs,
    xp: Any,
    ref: np.ndarray,
    base: np.ndarray | None,
    base_tol: float,
    base_peaks: tuple[tuple[int, int, int], ...] | None,
    backend: Literal['numpy', 'cupy'],
    repeat: int,
    warmup: int,
) -> tuple[Result, np.ndarray]:
    state = path.prepare(inputs, xp)
    for _ in range(warmup):
        path.run(inputs, state, xp)
    device_synchronize()
    times = [time_call(lambda: path.run(inputs, state, xp)) for _ in range(repeat)]
    peak = peak_bytes(lambda: path.run(inputs, state, xp))
    R = to_numpy(path.run(inputs, state, xp))
    sw.arrays.free_cupy_mempool()

    tol = sw.fourier.fft_tolerance_rms(
        'complex64', path.nffts, path.n_elementwise, array_backend=backend
    )
    peaks = argmax_peaks(R)
    if base is None:
        err_base, tol_base, match = 0.0, tol, True
    else:
        err_base = rel_rms_error(R, base)
        tol_base = math.hypot(tol, base_tol)
        match = peaks == base_peaks
    result = Result(
        path=path.name,
        times=times,
        peak_bytes=peak,
        err_ref=rel_rms_error(R, ref),
        err_base=err_base,
        tol_ref=tol,
        tol_base=tol_base,
        peaks=peaks,
        peaks_match=match,
    )
    return result, R


# %% Reporting
def fmt_mb(value: int) -> str:
    return f'{value / 2**20:8.1f}'


def report_operations(ops: list[Operation], repeat: int) -> None:
    print(f'\nbase operations, median of {repeat}')
    print(f'{"path":<13} {"stage":<14} {"ms":>8} {"GB/s":>7} {"MB":>8}')
    for op in ops:
        for _ in range(2):
            op.func()
        device_synchronize()
        t = statistics.median(time_call(op.func) for _ in range(repeat))
        peak = peak_bytes(op.func)
        rate = f'{op.bytes_moved / t / 1e9:7.1f}' if op.bytes_moved else '      -'
        print(f'{op.group:<13} {op.name:<14} {t * 1e3:8.3f} {rate} {fmt_mb(peak)}')


def report_paths(results: list[Result], base_min: float) -> None:
    print(
        f'\n{"path":<13} {"min ms":>8} {"med ms":>8} {"ratio":>6} {"MB":>8} '
        f'{"err64":>9} {"tol64":>9} {"errOA":>9} {"tolOA":>9} {"peaks":>5}  check'
    )
    for r in results:
        lo = min(r.times)
        print(
            f'{r.path:<13} {lo * 1e3:8.3f} {statistics.median(r.times) * 1e3:8.3f} '
            f'{lo / base_min:6.2f} {fmt_mb(r.peak_bytes)} {r.err_ref:9.2e} {r.tol_ref:9.2e} '
            f'{r.err_base:9.2e} {r.tol_base:9.2e} {"same" if r.peaks_match else "DIFF":>5}  '
            f'{"ok" if r.passed else "FAIL"}'
        )


def report_library(
    inputs: Inputs, center_frequency: float, repeat: int, warmup: int
) -> None:
    """print the time and peak memory of `correlate_sync_spans` on the waveform."""
    params = sw.comms_5gnr.pss_params(
        sample_rate=SAMPLE_RATE,
        subcarrier_spacing=SUBCARRIER_SPACING,
        discovery_periodicity=DISCOVERY_PERIODICITY,
        symbol_indexes=SYMBOL_INDEXES,
        center_frequency=center_frequency,
    )

    def run() -> Any:
        return sw.comms_5gnr.correlate_sync_spans(
            inputs.x, inputs.templates, params=params
        )

    for _ in range(warmup):
        run()
    device_synchronize()
    times = [time_call(run) for _ in range(repeat)]
    peak = peak_bytes(run)
    grid = run()
    print(
        f'\nlibrary correlate_sync_spans: {min(times) * 1e3:.3f} ms min, '
        f'{statistics.median(times) * 1e3:.3f} ms median, {peak / 2**20:.1f} MB, '
        f'grid {tuple(grid.shape)}'
    )


def report_footprint(
    paths: list[Path], results: dict[str, Result], g: Geometry, budget: int
) -> None:
    print(
        f'\nfootprint by design (MB), with the measured peak, budget {budget / 2**20:.0f} MB'
    )
    print(
        f'{"path":<13} {"designed":>9} {"measured":>9} {"per cell":>9} {"cells at budget":>16}  '
        'intermediates'
    )
    for path in paths:
        parts = path.footprint(g)
        designed = sum(parts.values())
        per_cell = path.per_cell(g)
        detail = ', '.join(f'{k} {v / 2**20:.1f}' for k, v in parts.items())
        print(
            f'{path.name:<13} {designed / 2**20:9.1f} {results[path.name].peak_bytes / 2**20:9.1f} '
            f'{per_cell / 2**20:9.2f} {budget // per_cell:16d}  {detail}'
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--backend', choices=('cupy', 'numpy'), default='cupy')
    parser.add_argument('--duration', type=float, default=0.1)
    parser.add_argument('--ports', type=int, default=2)
    parser.add_argument('--center-frequency', type=float, default=3750e6)
    parser.add_argument(
        '--block-sizes', type=int, nargs='+', default=list(DEFAULT_BLOCK_SIZES)
    )
    parser.add_argument('--repeat', type=int, default=10)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument(
        '--variants',
        nargs='+',
        default=None,
        help='a subset of the path names (baseline, oa:<M>, direct:<N>, os:<M>)',
    )
    parser.add_argument('--memory-budget', type=int, default=2**30, help='bytes')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--skip-operations', action='store_true')
    args = parser.parse_args()
    backend = cast("Literal['numpy', 'cupy']", args.backend)

    global cp
    if backend == 'cupy':
        import cupy  # ty: ignore[unresolved-import]

        cp = cupy
        sw.arrays.configure_cupy()
        xp: Any = cp
        device = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
    else:
        xp = np
        device = 'none'
    print(
        f'{platform.machine()} numpy {np.__version__} scipy {scipy.__version__} '
        f'cupy {getattr(cp, "__version__", "-")} device: {device}'
    )

    inputs = build_inputs(
        args.ports, args.duration, args.center_frequency, xp, args.seed
    )
    inputs_np = dataclasses.replace(
        inputs, x=to_numpy(inputs.x), templates=to_numpy(inputs.templates)
    )
    g = inputs.geometry
    block, overlap, step, _ = scipy_oa_lens(g)
    for M in args.block_sizes:
        if g.corr_size % M or M < g.template_size - 1:
            parser.error(
                f'block size {M} must divide corr_size {g.corr_size} and be at least '
                f'{g.template_size - 1}'
            )
    print(
        f'waveform {tuple(inputs.x.shape)} complex64 ({inputs.x.nbytes / 2**20:.1f} MB), '
        f'templates {tuple(inputs.templates.shape)}, full convolution {g.full_size}, '
        f'roll {g.offset}'
    )
    print(
        f'scipy oaconvolve block {block}, overlap {overlap}, step {step} '
        f'({-(-g.corr_size // step)} blocks per row); next_fast_len {scipy.fft.next_fast_len(g.full_size, real=False)}'
    )
    print(f'embedded peaks (cell, sync, lag) per port: {inputs.expected_peaks}')

    paths = [baseline_path(g)]
    paths += [oa_path(M) for M in args.block_sizes]
    paths += [direct_path(N) for N in DIRECT_SIZES]
    paths += [os_path(g, M) for M in args.block_sizes]
    if args.variants:
        wanted = set(args.variants)
        unknown = wanted - {p.name for p in paths}
        if unknown:
            parser.error(f'unknown variants {sorted(unknown)}')
        paths = [p for p in paths if p.name in wanted or p.name == 'baseline']

    ref = reference(inputs_np)
    ref_peaks = argmax_peaks(ref)
    peaks_embedded = ref_peaks == inputs.expected_peaks
    if not peaks_embedded:
        print(
            f'FAIL: reference peaks {ref_peaks} differ from the embedded '
            f'{inputs.expected_peaks}'
        )

    results: list[Result] = []
    base: np.ndarray | None = None
    base_tol = 0.0
    base_peaks: tuple[tuple[int, int, int], ...] | None = None
    for path in paths:
        result, R = evaluate(
            path,
            inputs,
            xp,
            ref,
            base,
            base_tol,
            base_peaks,
            backend,
            args.repeat,
            args.warmup,
        )
        results.append(result)
        if path.name == 'baseline':
            base, base_tol, base_peaks = R, result.tol_ref, result.peaks
        print(
            f'{path.name}: {min(result.times) * 1e3:.3f} ms {"ok" if result.passed else "FAIL"}'
        )

    base_min = min(results[0].times)
    report_paths(results, base_min)
    report_library(inputs, args.center_frequency, args.repeat, args.warmup)
    report_footprint(paths, {r.path: r for r in results}, g, args.memory_budget)

    if not args.skip_operations:
        ops: list[Operation] = []
        for M in args.block_sizes:
            ops += oa_operations(inputs, M, xp)
        for N in DIRECT_SIZES:
            ops += direct_operations(inputs, N, xp)
        report_operations(ops, args.repeat)

    failed = [r.path for r in results if not r.passed]
    if failed:
        print(f'\nFAIL: {", ".join(failed)}')
    return 0 if peaks_embedded and not failed else 1


if __name__ == '__main__':
    sys.exit(main())
