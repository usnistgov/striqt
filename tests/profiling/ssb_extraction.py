#!/usr/bin/env python3
"""Profile the SSB extraction of the PSS correlator: the whole-record FFT resampler
against overlap-add resampling at several block sizes, each on the whole capture or on
the correlated span of each synchronization block alone.

The input reproduces the operand of `get_5g_ssb_iq` in the ADA 3750 MHz capture: a
``(port, sample)`` waveform at 107.52 MS/s with the SSB centered 40.08 MHz below the
capture center, resampled to 15.36 MS/s for 30 kHz synchronization blocks every 20 ms.
The waveform is complex gaussian noise with one PSS symbol per port embedded in the
first synchronization block and one tone inside the SSB band, so each path is checked
by the correlator peak and by the level and frequency of the tone after recentering.
Nothing under ``src/`` is modified.

Paths:

- ``fft``: `get_5g_ssb_iq` as the measurements call it, one brick-wall FFT per port over
  the whole record. The reference for the ratios and the peak comparison.
- ``fft:span``: the same resampler on the correlated span of each synchronization block
  with a margin at each end, trimmed to ``corr_size`` samples per block. The correlator
  reads only that span, so this path measures the work the whole-record path spends on
  samples the correlator never reads.
- ``oa:<down>``: `oaresample` per port with input blocks of ``down`` samples, as the
  ``oaresample=True`` branch of `get_5g_ssb_iq` calls it. The library sizes the block as
  ``3 * fs_in / subcarrier_spacing / 8``, 1344 here; the larger sizes give a finer grid
  for `frequency_offset`, which must be a multiple of ``frames_per_hop * fs_in / down``.
- ``oa:<down>:span``: the same on the correlated spans.
- ``library``: `get_5g_ssb_spans`, the span-cropped batched FFT path of the
  measurements, placed into the correlator input like the ``span`` paths.

Usage:
    pixi run -e test39 python tests/profiling/ssb_extraction.py --backend numpy
    pixi run -e cupy python tests/profiling/ssb_extraction.py
    pixi run -e cupy python tests/profiling/ssb_extraction.py --window hann --block-sizes 14336 43008

Run with its own cache directory beside a test session or another striqt script
(``STRIQT_CACHE_DIR=$(mktemp -d)``): two processes sharing the window shelf corrupt it on
macOS Python 3.9.

Reading the output:
    The header gives the geometry and the fraction of the resampled output that the
    correlator reads. The path table reports, per path, the frequency grid (Hz) that
    the path imposes on `frequency_offset`, the STFT frames per port, the hop, and the
    lead (all 0 for the FFT paths), the min and median time with the ratio of the min
    to the ``fft`` min, the measured peak MB of allocations during one call (a
    `cupy.cuda.MemoryHook` on cupy, `tracemalloc` on numpy) beside the MB the path
    allocates by design, the tone level difference (dB) and frequency error (output
    bins of the measured segment) against ``fft``, whether the correlator peak of every
    port is at the ``(cell, sync, beam, lag)`` of the ``fft`` peak with the magnitude
    error (%), and the relative rms difference from ``fft`` over the correlated span of
    block 0 and over the spans of the later blocks. The overlap-add output begins with
    ``down - hop`` input samples of partial window sum. ``lead`` is that length in
    output samples, and block 0 differs from the whole-record FFT over it. The
    whole-record FFT wraps the record edges into its first and last samples instead.
    The tone is measured in block 1 (block 0 for a one-block capture). The correlator
    and the tone measurement run on the host over a copy of each output, so the checks
    compare the resampled waveforms alone. A path fails when the peak index differs,
    the magnitude error exceeds 1%, the tone level differs by more than 0.1 dB, or the
    tone frequency is off by more than one bin, and the row then ends with the failed
    checks. Each row prints as soon as the path finishes. ``--device-correlator`` also
    runs `correlate_sync_sequence` on the output in the backend namespace, as the
    measurement does, and fails the path when the peak index or magnitude (1%) differs
    from the host correlator. A CUDA error is sticky, so
    the device is synchronized after each path and the run stops at the first path
    that raises one, naming the path. The exit status is 1 when a path fails or
    crashes, or when the ``fft`` peak is not at the embedded cell in synchronization
    block 0.
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
from fractions import Fraction
from typing import Any, Literal, cast

import numpy as np
from typing_extensions import Self

import striqt.waveform as sw
from striqt.waveform.lib.fourier.filters import _COLA_WINDOW_SIZE_DIVISOR

cp: Any = None

# the cellular analysis of the ADA 3750 MHz capture (cuda_streams.CAPTURE)
FS_OUT = 15.36e6
SUBCARRIER_SPACING = 30e3
DISCOVERY_PERIODICITY = 20e-3
SYMBOL_INDEXES = 'c'
CENTER_FREQUENCY = 3750e6

DEFAULT_BLOCK_SIZES = (1344, 21504, 64512)
ITEM_BYTES = 8
NOISE_TO_TEMPLATE_RMS = 0.05
TONE_TO_TEMPLATE_RMS = 10.0
TONE_FREQUENCY = 5e6
EMBED_BEAM = 2
EMBED_DELAY = 100
SEGMENT_TRIM = 2000
PEAK_MAGNITUDE_TOL = 0.01
TONE_LEVEL_TOL_DB = 0.1

Array = Any


# %% Geometry and inputs
@dataclasses.dataclass(frozen=True)
class Geometry:
    fs_in: float
    """the capture sample rate (in S/s)"""
    fs_out: float
    """the SSB sample rate (in S/s)"""
    ports: int
    """the number of ports, the first axis of the waveform"""
    sync_blocks: int
    """the number of synchronization blocks in the capture"""
    block_in: int
    """the input samples per synchronization block"""
    block_out: int
    """the output samples per synchronization block"""
    corr_size: int
    """the output samples per block that the correlator reads, from the block start"""
    frequency_offset: float
    """the SSB center (in Hz) relative to the capture center"""
    params: sw.comms_5gnr.SyncParams
    """the synchronization parameters at `fs_out`"""

    @property
    def size_in(self) -> int:
        return self.sync_blocks * self.block_in

    @property
    def size_out(self) -> int:
        return self.sync_blocks * self.block_out

    @property
    def ratio(self) -> Fraction:
        """the exact rate ratio ``fs_in / fs_out``"""
        return Fraction(self.fs_in / self.fs_out).limit_denominator(10**6)

    def to_input(self, count: int) -> int:
        """return the input samples that resample to `count` output samples."""
        return int(count * self.ratio)

    def row_bytes(self, count: int) -> int:
        return count * ITEM_BYTES


@dataclasses.dataclass(frozen=True)
class Inputs:
    x: Array
    """the capture waveform of shape ``(port, sample)`` in the backend namespace"""
    templates: Array
    """the PSS templates at `fs_out`, shape ``(cell, sample)``"""
    geometry: Geometry
    embedded_cells: tuple[int, ...]
    """the cell ID of the PSS symbol embedded in block 0 of each port"""


def build_geometry(
    fs_in: float, duration: float, ports: int, frequency_offset: float
) -> Geometry:
    params = sw.comms_5gnr.pss_params(
        sample_rate=FS_OUT,
        subcarrier_spacing=SUBCARRIER_SPACING,
        discovery_periodicity=DISCOVERY_PERIODICITY,
        symbol_indexes=SYMBOL_INDEXES,
        center_frequency=CENTER_FREQUENCY,
    )
    block_out = params.frame_size * params.frames_per_sync
    ratio = Fraction(fs_in / FS_OUT).limit_denominator(10**6)
    block_in = block_out * ratio
    if block_in.denominator != 1:
        raise ValueError(
            f'a {DISCOVERY_PERIODICITY} s block is not a whole number of samples at '
            f'{fs_in} S/s'
        )
    return Geometry(
        fs_in=fs_in,
        fs_out=FS_OUT,
        ports=ports,
        sync_blocks=max(1, round(duration / DISCOVERY_PERIODICITY)),
        block_in=int(block_in),
        block_out=block_out,
        corr_size=params.corr_size,
        frequency_offset=frequency_offset,
        params=params,
    )


def build_inputs(g: Geometry, xp: Any, seed: int) -> Inputs:
    templates_out = np.asarray(
        sw.comms_5gnr.pss_5g_nr(g.fs_out, SUBCARRIER_SPACING), dtype=np.complex64
    )
    # the symbol at the capture rate, modulated to the SSB center
    templates_in = np.asarray(
        sw.comms_5gnr.pss_5g_nr(g.fs_in, SUBCARRIER_SPACING, g.frequency_offset),
        dtype=np.complex64,
    )
    phy = sw.comms_5gnr.get_3gpp_phy(
        1, subcarrier_spacing=SUBCARRIER_SPACING, sample_rate=g.fs_in, generation='5G'
    )
    cp_min = int(phy.cp_sizes.min())
    template_rms = float(np.sqrt(np.mean(np.abs(templates_in[0]) ** 2)))

    rng = np.random.default_rng(seed)
    shape = (g.ports, g.size_in)
    noise_scale = NOISE_TO_TEMPLATE_RMS * template_rms / math.sqrt(2)
    x = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(
        np.complex64
    ) * np.complex64(noise_scale)

    t = np.arange(g.size_in) / g.fs_in
    tone = np.exp(2j * np.pi * (g.frequency_offset + TONE_FREQUENCY) * t)
    x += (TONE_TO_TEMPLATE_RMS * template_rms * tone).astype(np.complex64)

    symbol = g.params.symbol_indexes[EMBED_BEAM]
    slot, symbol_in_slot = divmod(symbol, phy.FFT_PER_SLOT)
    ncp = int(phy.cp_sizes[symbol_in_slot])
    start = (
        slot * int(phy.contiguous_size)
        + int(phy.cp_start_idx[symbol_in_slot])
        + EMBED_DELAY * g.ratio.numerator
    )
    cells = []
    for port in range(g.ports):
        cell = port % templates_in.shape[0]
        body = templates_in[cell][cp_min:]
        wave = np.concatenate([body[-ncp:], body])
        x[port, start : start + wave.size] += wave
        cells.append(cell)

    return Inputs(
        x=xp.asarray(x),
        templates=xp.asarray(templates_out),
        geometry=g,
        embedded_cells=tuple(cells),
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
    grid_hz: float
    """the grid (in Hz) that the path imposes on `frequency_offset`"""
    frames: int
    """the STFT frames per port, 0 for the FFT paths"""
    hop: int
    """the STFT hop (in input samples), 0 for the FFT paths"""
    lead: int
    """the output samples at the record start with a partial window sum"""
    run: Callable[[Inputs, Any], Array]
    """return the SSB waveform of shape ``(port, size_out)``"""
    footprint: Callable[[Geometry], dict[str, int]]
    """the bytes of each intermediate the path allocates"""


def oa_design(g: Geometry, down: int, window: str) -> tuple[int, int, int, float]:
    """return ``(down, up, hop, grid)`` of `oaresample` at block `down`, or raise.

    Raises:
        ValueError: if `up` is not an integer, `down` or `up` is not an even multiple
            of the COLA divisor of `window`, the record is not a whole number of hops,
            or `frequency_offset` is off the grid of the block
    """
    divisor = _COLA_WINDOW_SIZE_DIVISOR[window]
    up = Fraction(down) / g.ratio
    if up.denominator != 1:
        raise ValueError(f'down={down} gives a non-integer up={up}')
    up_int = int(up)
    step = math.lcm(2, divisor)
    for name, size in (('down', down), ('up', up_int)):
        if size % step:
            raise ValueError(
                f'{name}={size} is not a multiple of {step} for {window!r}'
            )
    hop = down // divisor
    if g.block_in % hop:
        raise ValueError(f'a block of {g.block_in} samples is not whole hops of {hop}')
    grid = divisor * g.fs_in / down
    if not sw.arrays.isroundmod(g.frequency_offset, grid):
        raise ValueError(
            f'frequency_offset {g.frequency_offset} is off the {grid} Hz grid of '
            f'down={down}'
        )
    return down, up_int, hop, grid


def span_margin_in(g: Geometry, hops: list[int], downs: list[int]) -> int:
    """return the margin (in input samples) at each end of a correlated span.

    The margin covers one slot and the longest overlap-add block, rounded up to a
    whole number of every hop in `hops` so that each span is whole hops for every path.
    """
    slot_in = g.to_input(
        round(sw.comms_5gnr.slot_period(SUBCARRIER_SPACING) * g.fs_out)
    )
    step = math.lcm(2, *hops) if hops else 2
    need = max([slot_in, *downs])
    return -(-need // step) * step


def place_spans(spans: list[Array], g: Geometry, xp: Any) -> Array:
    """return the correlator input with the span of each block at the block start."""
    out = xp.zeros((g.ports, g.size_out), dtype=spans[0].dtype)
    for block, span in enumerate(spans):
        start = block * g.block_out
        out[:, start : start + g.corr_size] = span
    return out


def span_bounds(g: Geometry, block: int, margin_in: int) -> tuple[int, int, int]:
    """return ``(start, stop, lead)`` of the input span of `block`.

    `lead` is the margin before the block start that the record provides, which is
    shorter than `margin_in` only for block 0.
    """
    block_start = block * g.block_in
    start = max(0, block_start - margin_in)
    stop = block_start + g.to_input(g.corr_size) + margin_in
    return start, stop, block_start - start


def fft_path(g: Geometry) -> Path:
    def run(inputs: Inputs, xp: Any) -> Array:
        return sw.comms_5gnr.get_5g_ssb_iq(
            inputs.x,
            discovery_periodicity=DISCOVERY_PERIODICITY,
            fs_out=g.fs_out,
            fs_in=g.fs_in,
            subcarrier_spacing=SUBCARRIER_SPACING,
            frequency_offset=g.frequency_offset,
            oaresample=False,
        )

    def footprint(g: Geometry) -> dict[str, int]:
        return {
            'fftshift copy (1 port)': g.row_bytes(g.size_in),
            'out': g.row_bytes(g.ports * g.size_out),
        }

    return Path('fft', g.fs_in / g.size_in, 0, 0, 0, run, footprint)


def fft_span_path(g: Geometry, margin_in: int) -> Path:
    def run(inputs: Inputs, xp: Any) -> Array:
        spans = []
        for block in range(g.sync_blocks):
            start, stop, lead = span_bounds(g, block, margin_in)
            seg = inputs.x[:, start:stop]
            size_out = round(seg.shape[1] * g.fs_out / g.fs_in)
            shift = round(seg.shape[1] * g.frequency_offset / g.fs_in)
            y = xp.empty((g.ports, size_out), dtype=seg.dtype)
            for i in range(g.ports):
                y[i] = sw.resample(seg[i], num=size_out, axis=0, shift=shift)
            lead_out = round(lead * g.fs_out / g.fs_in)
            spans.append(y[:, lead_out : lead_out + g.corr_size])
        return place_spans(spans, g, xp)

    def footprint(g: Geometry) -> dict[str, int]:
        span_in = g.to_input(g.corr_size) + 2 * margin_in
        span_out = round(span_in * g.fs_out / g.fs_in)
        return {
            'fftshift copy (1 port, 1 span)': g.row_bytes(span_in),
            'span outputs': g.row_bytes(g.ports * g.sync_blocks * span_out),
            'out': g.row_bytes(g.ports * g.size_out),
        }

    span_in = g.to_input(g.corr_size) + 2 * margin_in
    return Path('fft:span', g.fs_in / span_in, 0, 0, 0, run, footprint)


def library_path(g: Geometry) -> Path:
    def run(inputs: Inputs, xp: Any) -> Array:
        spans = sw.comms_5gnr.get_5g_ssb_spans(
            inputs.x,
            params=g.params,
            fs_in=g.fs_in,
            frequency_offset=g.frequency_offset,
        )
        return place_spans([spans[:, b] for b in range(g.sync_blocks)], g, xp)

    slot_in = g.to_input(
        round(sw.comms_5gnr.slot_period(SUBCARRIER_SPACING) * g.fs_out)
    )
    step = math.lcm(2, g.ratio.numerator)
    margin_in = -(-slot_in // step) * step
    span_in = g.to_input(g.corr_size) + 2 * margin_in

    def footprint(g: Geometry) -> dict[str, int]:
        return {
            'span buffer (resampled in place)': g.row_bytes(
                g.ports * g.sync_blocks * span_in
            ),
            'out': g.row_bytes(g.ports * g.size_out),
        }

    return Path('library', g.fs_in / span_in, 0, 0, 0, run, footprint)


def oa_frames(size_in: int, down: int, hop: int) -> int:
    return (size_in - down) // hop + 1


def oa_path(g: Geometry, down: int, window: str) -> Path:
    down, up, hop, grid = oa_design(g, down, window)

    def run(inputs: Inputs, xp: Any) -> Array:
        out = xp.empty((g.ports, g.size_out), dtype=inputs.x.dtype)
        for i in range(g.ports):
            out[i] = sw.oaresample(
                inputs.x[i],
                fs=g.fs_in,
                up=up,
                down=down,
                axis=0,
                window=window,
                frequency_shift=g.frequency_offset,
            )
        return out

    def footprint(g: Geometry) -> dict[str, int]:
        frames = oa_frames(g.size_in, down, hop)
        return {
            'stacked frames (1 port)': g.row_bytes(frames * down),
            'bin copy (1 port)': g.row_bytes(frames * up),
            'istft out (1 port)': g.row_bytes(g.size_out),
            'out': g.row_bytes(g.ports * g.size_out),
        }

    frames = oa_frames(g.size_in, down, hop)
    lead = (down - hop) * up // down
    return Path(f'oa:{down}', grid, frames, hop, lead, run, footprint)


def oa_span_path(g: Geometry, down: int, window: str, margin_in: int) -> Path:
    down, up, hop, grid = oa_design(g, down, window)

    def run(inputs: Inputs, xp: Any) -> Array:
        spans = []
        for block in range(g.sync_blocks):
            start, stop, lead = span_bounds(g, block, margin_in)
            seg = inputs.x[:, start:stop]
            size_out = seg.shape[1] * up // down
            y = xp.empty((g.ports, size_out), dtype=seg.dtype)
            for i in range(g.ports):
                y[i] = sw.oaresample(
                    seg[i],
                    fs=g.fs_in,
                    up=up,
                    down=down,
                    axis=0,
                    window=window,
                    frequency_shift=g.frequency_offset,
                )
            lead_out = lead * up // down
            spans.append(y[:, lead_out : lead_out + g.corr_size])
        return place_spans(spans, g, xp)

    def footprint(g: Geometry) -> dict[str, int]:
        span_in = g.to_input(g.corr_size) + 2 * margin_in
        frames = oa_frames(span_in, down, hop)
        span_out = span_in * up // down
        return {
            'stacked frames (1 port, 1 span)': g.row_bytes(frames * down),
            'bin copy (1 port, 1 span)': g.row_bytes(frames * up),
            'span outputs': g.row_bytes(g.ports * g.sync_blocks * span_out),
            'out': g.row_bytes(g.ports * g.size_out),
        }

    span_in = g.to_input(g.corr_size) + 2 * margin_in
    frames = oa_frames(span_in, down, hop) * g.sync_blocks
    lead = (down - hop) * up // down
    return Path(f'oa:{down}:span', grid, frames, hop, lead, run, footprint)


# %% Checks
@dataclasses.dataclass
class Result:
    path: str
    """the path name"""
    times: list[float]
    """the time (in s) of each repetition"""
    peak_bytes: int
    """the peak bytes allocated during one call"""
    tone_level_dB: float
    """the tone level (dB) in the measured segment of port 0"""
    tone_bin: int
    """the tone frequency as a bin of the measured segment"""
    peaks: tuple[tuple[int, int, int, int], ...]
    """the ``(cell, sync, beam, lag)`` argmax of each port"""
    peak_magnitudes: tuple[float, ...]
    """the correlator peak magnitude of each port"""
    rel_rms_block0: float
    """the relative rms difference from ``fft`` over the correlated span of block 0"""
    rel_rms_rest: float
    """the relative rms difference from ``fft`` over the spans of the later blocks"""
    tone_level_err_dB: float = 0.0
    tone_bin_err: int = 0
    peaks_match: bool = True
    peak_magnitude_err: float = 0.0
    device_mismatch: str | None = None
    """the device correlator disagreement with the host correlator, or None"""

    def failures(self) -> list[str]:
        """return the names of the failed checks, empty when the path passed."""
        failed = []
        if not self.peaks_match:
            failed.append(f'peak index {self.peaks}')
        if self.peak_magnitude_err > PEAK_MAGNITUDE_TOL:
            failed.append(f'peak magnitude {100 * self.peak_magnitude_err:.2f}%')
        if abs(self.tone_level_err_dB) > TONE_LEVEL_TOL_DB:
            failed.append(f'tone level {self.tone_level_err_dB:+.3f} dB')
        if abs(self.tone_bin_err) > 1:
            failed.append(f'tone bin {self.tone_bin_err:+d}')
        if self.device_mismatch:
            failed.append(f'device correlator {self.device_mismatch}')
        return failed

    @property
    def passed(self) -> bool:
        return not self.failures()


def to_numpy(a: Array) -> np.ndarray:
    if cp is not None and isinstance(a, cp.ndarray):
        return cp.asnumpy(a)
    return np.asarray(a)


def rel_rms(got: np.ndarray, ref: np.ndarray) -> float:
    diff = got.astype(np.complex128) - ref.astype(np.complex128)
    return float(np.sqrt(np.mean(np.abs(diff) ** 2) / np.mean(np.abs(ref) ** 2)))


def correlated_spans(y: np.ndarray, g: Geometry) -> np.ndarray:
    """return the ``(port, sync, corr_size)`` samples that the correlator reads."""
    blocks = y.reshape(g.ports, g.sync_blocks, g.block_out)
    return blocks[..., : g.corr_size]


def tone_block(g: Geometry) -> int:
    return min(1, g.sync_blocks - 1)


def tone_measurement(y: np.ndarray, g: Geometry) -> tuple[float, int]:
    """return the level (dB) and bin of the strongest line in one span of port 0."""
    start = tone_block(g) * g.block_out
    segment = y[0, start + SEGMENT_TRIM : start + g.corr_size - SEGMENT_TRIM]
    segment = segment.astype(np.complex128)
    spectrum = np.abs(np.fft.fft(segment)) / segment.size
    bin_index = int(np.argmax(spectrum))
    return float(20 * np.log10(spectrum[bin_index])), bin_index


def correlator_peaks(
    y: Array, inputs: Inputs
) -> tuple[tuple[tuple[int, int, int, int], ...], tuple[float, ...]]:
    """return the ``(cell, sync, beam, lag)`` argmax and its magnitude per port.

    The correlator runs in the array namespace of `y`; the reduction runs on the host.
    """
    templates = inputs.templates if sw.is_cupy_array(y) else to_numpy(inputs.templates)
    R = sw.comms_5gnr.correlate_sync_sequence(
        y, templates, params=inputs.geometry.params
    )
    device_synchronize()
    mag = np.abs(to_numpy(R))
    peaks = []
    magnitudes = []
    for port in range(mag.shape[0]):
        cell, sync, beam, lag = np.unravel_index(
            int(mag[port].argmax()), mag[port].shape
        )
        peaks.append((int(cell), int(sync), int(beam), int(lag)))
        magnitudes.append(float(mag[port].max()))
    return tuple(peaks), tuple(magnitudes)


def evaluate(
    path: Path,
    inputs: Inputs,
    xp: Any,
    reference: Result | None,
    reference_spans: np.ndarray | None,
    repeat: int,
    warmup: int,
    device_correlator: bool,
) -> tuple[Result, np.ndarray]:
    g = inputs.geometry
    for _ in range(warmup):
        path.run(inputs, xp)
    device_synchronize()
    times = [time_call(lambda: path.run(inputs, xp)) for _ in range(repeat)]
    peak = peak_bytes(lambda: path.run(inputs, xp))
    y = path.run(inputs, xp)
    device_synchronize()
    y_np = to_numpy(y)
    peaks, magnitudes = correlator_peaks(y_np, inputs)
    device_mismatch = None
    if device_correlator:
        device_peaks, device_magnitudes = correlator_peaks(y, inputs)
        magnitude_err = max(
            abs(a - b) / b for a, b in zip(device_magnitudes, magnitudes)
        )
        if device_peaks != peaks:
            device_mismatch = f'peaks {device_peaks} vs host {peaks}'
        elif magnitude_err > PEAK_MAGNITUDE_TOL:
            device_mismatch = f'magnitude {100 * magnitude_err:.2f}%'
    del y
    sw.arrays.free_cupy_mempool()

    spans = correlated_spans(y_np, g)
    level, tone_bin = tone_measurement(y_np, g)
    result = Result(
        path=path.name,
        times=times,
        peak_bytes=peak,
        tone_level_dB=level,
        tone_bin=tone_bin,
        peaks=peaks,
        peak_magnitudes=magnitudes,
        rel_rms_block0=0.0,
        rel_rms_rest=0.0,
        device_mismatch=device_mismatch,
    )
    if reference is not None and reference_spans is not None:
        result.rel_rms_block0 = rel_rms(spans[:, :1], reference_spans[:, :1])
        if g.sync_blocks > 1:
            result.rel_rms_rest = rel_rms(spans[:, 1:], reference_spans[:, 1:])
        result.tone_level_err_dB = level - reference.tone_level_dB
        result.tone_bin_err = tone_bin - reference.tone_bin
        result.peaks_match = peaks == reference.peaks
        result.peak_magnitude_err = max(
            abs(a - b) / b for a, b in zip(magnitudes, reference.peak_magnitudes)
        )
    return result, spans


# %% Reporting
def print_header() -> None:
    print(
        f'\n{"path":<15} {"grid Hz":>8} {"frames":>7} {"hop":>6} {"lead":>5} '
        f'{"min ms":>8} {"med ms":>8} {"ratio":>6} {"MB":>7} {"design":>7} '
        f'{"tone dB":>8} {"bin":>4} {"peak":>5} {"mag %":>6} {"rms b0":>9} '
        f'{"rms b1+":>9}  check'
    )


def print_row(path: Path, r: Result, g: Geometry, base_min: float) -> None:
    lo = min(r.times)
    designed = sum(path.footprint(g).values())
    failures = r.failures()
    check = 'ok' if not failures else 'FAIL: ' + ', '.join(failures)
    print(
        f'{r.path:<15} {path.grid_hz:8.0f} {path.frames:7d} {path.hop:6d} '
        f'{path.lead:5d} {lo * 1e3:8.2f} {statistics.median(r.times) * 1e3:8.2f} '
        f'{lo / base_min:6.2f} {r.peak_bytes / 2**20:7.0f} {designed / 2**20:7.0f} '
        f'{r.tone_level_err_dB:+8.3f} {r.tone_bin_err:4d} '
        f'{"same" if r.peaks_match else "DIFF":>5} {100 * r.peak_magnitude_err:6.2f} '
        f'{r.rel_rms_block0:9.2e} {r.rel_rms_rest:9.2e}  {check}',
        flush=True,
    )


def cuda_error_types() -> tuple[type[BaseException], ...]:
    if cp is None:
        return ()
    return (cp.cuda.runtime.CUDARuntimeError, cp.cuda.driver.CUDADriverError)


def report_footprint(paths: list[Path], g: Geometry) -> None:
    print('\nfootprint by design (MB)')
    for path in paths:
        parts = path.footprint(g)
        detail = ', '.join(f'{k} {v / 2**20:.1f}' for k, v in parts.items())
        print(f'{path.name:<15} {sum(parts.values()) / 2**20:7.1f}  {detail}')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--backend', choices=('cupy', 'numpy'), default='cupy')
    parser.add_argument('--duration', type=float, default=0.1)
    parser.add_argument('--ports', type=int, default=2)
    parser.add_argument('--sample-rate', type=float, default=107.52e6)
    parser.add_argument('--frequency-offset', type=float, default=-40.08e6)
    parser.add_argument(
        '--block-sizes', type=int, nargs='+', default=list(DEFAULT_BLOCK_SIZES)
    )
    parser.add_argument('--window', choices=('blackman', 'hann'), default='blackman')
    parser.add_argument('--repeat', type=int, default=10)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument(
        '--variants',
        nargs='+',
        default=None,
        help='a subset of the path names (fft, fft:span, library, oa:<down>, oa:<down>:span)',
    )
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument(
        '--device-correlator',
        action='store_true',
        help='also run the correlator in the backend namespace and compare',
    )
    args = parser.parse_args()
    backend = cast("Literal['numpy', 'cupy']", args.backend)

    global cp
    if backend == 'cupy':
        # numba.cuda must be imported before cupy (numba issue 6131)
        import numba.cuda  # noqa: F401, I001
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
    if cp is not None:
        free, total = cp.cuda.runtime.memGetInfo()
        print(f'device memory: {free / 2**20:.0f} MB free of {total / 2**20:.0f} MB')

    g = build_geometry(
        args.sample_rate, args.duration, args.ports, args.frequency_offset
    )
    designs = []
    for down in args.block_sizes:
        try:
            designs.append(oa_design(g, down, args.window))
        except ValueError as exc:
            parser.error(str(exc))
    margin_in = span_margin_in(g, [d[2] for d in designs], [d[0] for d in designs])
    inputs = build_inputs(g, xp, args.seed)

    print(
        f'capture {tuple(inputs.x.shape)} complex64 ({inputs.x.nbytes / 2**20:.0f} MB) at '
        f'{g.fs_in / 1e6:g} MS/s, SSB at {g.frequency_offset / 1e6:+g} MHz to '
        f'{g.fs_out / 1e6:g} MS/s (ratio {g.ratio}), {g.sync_blocks} sync blocks of '
        f'{g.block_in} -> {g.block_out} samples'
    )
    print(
        f'correlator reads {g.corr_size} of {g.block_out} output samples per block '
        f'({g.corr_size / g.block_out:.0%}); span margin {margin_in} input samples '
        f'({margin_in / g.fs_in * 1e3:.2f} ms); window {args.window}'
    )
    print(f'embedded PSS cells per port: {inputs.embedded_cells}, beam {EMBED_BEAM}')

    paths = [fft_path(g), fft_span_path(g, margin_in), library_path(g)]
    for down in args.block_sizes:
        paths.extend((
            oa_path(g, down, args.window),
            oa_span_path(g, down, args.window, margin_in),
        ))
    if args.variants:
        wanted = set(args.variants)
        unknown = wanted - {p.name for p in paths}
        if unknown:
            parser.error(f'unknown variants {sorted(unknown)}')
        paths = [p for p in paths if p.name in wanted or p.name == 'fft']

    print_header()
    results: list[Result] = []
    reference: Result | None = None
    reference_spans: np.ndarray | None = None
    embedded_ok = True
    crashed: str | None = None
    for path in paths:
        try:
            result, spans = evaluate(
                path,
                inputs,
                xp,
                reference,
                reference_spans,
                args.repeat,
                args.warmup,
                args.device_correlator,
            )
        except cuda_error_types() as exc:
            crashed = f'{path.name}: {exc}'
            print(f'{path.name:<15} CUDA error, run stopped: {exc}', flush=True)
            break
        results.append(result)
        if reference is None:
            reference, reference_spans = result, spans
            embedded_ok = all(
                peak[0] == cell and peak[1] == 0
                for peak, cell in zip(result.peaks, inputs.embedded_cells)
            )
            if not embedded_ok:
                print(
                    f'FAIL: fft peaks {result.peaks} are not at the embedded cells '
                    f'{inputs.embedded_cells} in sync block 0'
                )
        assert reference is not None
        print_row(path, result, g, min(reference.times))

    report_footprint(paths, g)

    failed = [r.path for r in results if not r.passed]
    if failed:
        print(f'\nFAIL: {", ".join(failed)}')
    if crashed:
        print(f'\nCUDA error in {crashed}')
    return 0 if embedded_ok and not failed and crashed is None else 1


if __name__ == '__main__':
    sys.exit(main())
