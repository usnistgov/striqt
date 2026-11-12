#!/usr/bin/env python3
"""Compare `correlate_sync_sequence` on cupy against numpy, stage by stage, on a
multi-port, multi-block synchronization waveform.

The waveform is complex gaussian noise at 15.36 MS/s over several 20 ms blocks, with
one PSS symbol per port embedded in block 0, so the expected ``(cell, sync, beam, lag)``
peak of each port is known. The stages are the ones of `correlate_sync_sequence`:

- ``correlator``: the whole function, device against host.
- ``oaconvolve strided``: `sw.oaconvolve` on the strided ``(port, sync, sample)`` view
  that the function forms from the waveform, device against host.
- ``oaconvolve contiguous``: the same on a contiguous copy of that view.
- ``tail``: the roll, the per-slot reshape, the sliding window, and the symbol
  selection, applied on the device to the host convolution.
- ``tail/<op>``: each operation of the tail alone, applied on the device to the host
  result of the operations before it. The ``tail`` operations are the ones of the
  correlator before 2026-11, with the sliding window view and the fancy index that
  `correlate_sync_sequence` has since replaced by slices. The ``correlator`` row tests
  the library as it is; the ``tail`` rows characterize the cupy fault.

Usage:
    pixi run -e cupy python chores/tests/check_cupy_correlator.py
    pixi run -e cupy python chores/tests/check_cupy_correlator.py --ports 4 --blocks 3

Reading the output:
    Each stage prints the largest difference between the device and host results,
    relative to the largest host magnitude, for each port, and for the correlator the
    peak index per port on each side with the expected index. A relative difference
    at the float32 roundoff level (1e-6 class) is agreement. The exit status is 1 when
    any stage differs beyond ``--rtol`` or when a peak index differs.
"""

from __future__ import annotations

import argparse
import math
import platform
import sys
from collections.abc import Callable
from typing import Any

import numpy as np

import striqt.waveform as sw
from striqt.waveform.lib.arrays import sliding_window_view

cp: Any = None

FS = 15.36e6
SUBCARRIER_SPACING = 30e3
DISCOVERY_PERIODICITY = 20e-3
SYMBOL_INDEXES = 'c'
CENTER_FREQUENCY = 3750e6
NOISE_TO_TEMPLATE_RMS = 0.05
EMBED_BEAM = 2
EMBED_DELAY = 100

Array = Any


def build_waveform(
    params: sw.comms_5gnr.SyncParams, ports: int, blocks: int, seed: int
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int, int, int]]]:
    """return the waveform, the templates, and the expected peak of each port."""
    templates = np.asarray(
        sw.comms_5gnr.pss_5g_nr(FS, SUBCARRIER_SPACING), dtype=np.complex64
    )
    phy = sw.comms_5gnr.get_3gpp_phy(
        1, subcarrier_spacing=SUBCARRIER_SPACING, sample_rate=FS, generation='5G'
    )
    cp_min = int(phy.cp_sizes.min())
    template_rms = float(np.sqrt(np.mean(np.abs(templates[0]) ** 2)))

    size = params.frame_size * params.frames_per_sync * blocks
    rng = np.random.default_rng(seed)
    scale = NOISE_TO_TEMPLATE_RMS * template_rms / math.sqrt(2)
    x = rng.standard_normal((ports, size)) + 1j * rng.standard_normal((ports, size))
    x = x.astype(np.complex64) * np.complex64(scale)

    symbol = params.symbol_indexes[EMBED_BEAM]
    slot, symbol_in_slot = divmod(symbol, phy.FFT_PER_SLOT)
    ncp = int(phy.cp_sizes[symbol_in_slot])
    start = (
        slot * int(phy.contiguous_size)
        + int(phy.cp_start_idx[symbol_in_slot])
        + EMBED_DELAY
    )
    expected = []
    for port in range(ports):
        cell = port % templates.shape[0]
        body = templates[cell][cp_min:]
        wave = np.concatenate([body[-ncp:], body])
        x[port, start : start + wave.size] += wave
        expected.append((cell, 0, EMBED_BEAM, EMBED_DELAY))
    return x, templates, expected


# %% The stages of correlate_sync_sequence
def strided_blocks(x: Array, params: sw.comms_5gnr.SyncParams) -> Array:
    """return the ``(port, sync, sample)`` view that the correlator convolves."""
    blocks = x.reshape((x.shape[0], -1, params.frame_size))
    return blocks[:, :: params.frames_per_sync, : params.corr_size]


def oaconvolve_stage(iq_bcast: Array, templates: Array, xp: Any) -> Array:
    """return the full convolution ``(port, cell, sync, sample)`` of the correlator."""
    template_bcast = templates[xp.newaxis, :, xp.newaxis, :]
    shape = (
        iq_bcast.shape[0],
        template_bcast.shape[1],
        iq_bcast.shape[1],
        iq_bcast.shape[-1] + template_bcast.shape[-1] - 1,
    )
    R = xp.empty(shape, dtype='complex64')
    for cell_id in range(template_bcast.shape[1]):
        R[:, cell_id] = sw.oaconvolve(
            iq_bcast, template_bcast[:, cell_id], 'full', axes=2
        )
    return R


TailOp = Callable[[Array, sw.comms_5gnr.SyncParams, Any], Array]


def op_roll_slice(R: Array, params: sw.comms_5gnr.SyncParams, xp: Any) -> Array:
    offs = (
        round(params.sample_rate / params.subcarrier_spacing) + 2 * params.min_cp_size
    )
    return xp.roll(R, -offs, axis=-1)[..., : params.corr_size]


def op_reshape_per_slot(R: Array, params: sw.comms_5gnr.SyncParams, xp: Any) -> Array:
    excess_cp = int(params.cp_offsets[params.symbol_indexes[0] % 14])
    return R.reshape(R.shape[:-1] + (params.slot_count, -1))[..., excess_cp:]


def op_reshape_merge(R: Array, params: sw.comms_5gnr.SyncParams, xp: Any) -> Array:
    return R.reshape(R.shape[:-2] + (-1,))


def op_window_select(R: Array, params: sw.comms_5gnr.SyncParams, xp: Any) -> Array:
    start_inds = xp.array(params.symbol_indexes) * params.short_symbol_size
    Rwins = sliding_window_view(R, params.lag_count, axis=-1)
    return Rwins[..., start_inds, :]


def op_zero_cp(R: Array, params: sw.comms_5gnr.SyncParams, xp: Any) -> Array:
    result = R.copy()
    if params.min_cp_size:
        result[..., -params.min_cp_size :] = 0
    return result


TAIL_OPS: list[tuple[str, TailOp]] = [
    ('roll+slice', op_roll_slice),
    ('reshape per slot', op_reshape_per_slot),
    ('reshape merge', op_reshape_merge),
    ('window+select', op_window_select),
    ('zero cp', op_zero_cp),
]


def tail_stage(R: Array, params: sw.comms_5gnr.SyncParams, xp: Any) -> Array:
    """return the ``(port, cell, sync, beam, lag)`` grid from the full convolution."""
    for _, op in TAIL_OPS:
        R = op(R, params, xp)
    return R


# %% Comparison
def to_numpy(a: Array) -> np.ndarray:
    if cp is not None and isinstance(a, cp.ndarray):
        return cp.asnumpy(a)
    return np.asarray(a)


def port_differences(device: Array, host: np.ndarray) -> list[float]:
    """return the max |device - host| of each port relative to the max |host|."""
    d = to_numpy(device).astype(np.complex128)
    h = host.astype(np.complex128)
    scale = float(np.abs(h).max())
    return [float(np.abs(d[p] - h[p]).max() / scale) for p in range(h.shape[0])]


def peaks_of(R: Array) -> list[tuple[int, ...]]:
    mag = np.abs(to_numpy(R))
    return [
        tuple(int(i) for i in np.unravel_index(int(mag[p].argmax()), mag[p].shape))
        for p in range(mag.shape[0])
    ]


def fmt(values: list[float]) -> str:
    return ' '.join(f'{v:9.2e}' for v in values)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--ports', type=int, default=2)
    parser.add_argument('--blocks', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--rtol', type=float, default=1e-4)
    args = parser.parse_args()

    global cp
    import cupy  # ty: ignore[unresolved-import]

    cp = cupy
    sw.arrays.configure_cupy()
    device = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
    print(
        f'{platform.machine()} numpy {np.__version__} cupy {cp.__version__} '
        f'device: {device}'
    )

    params = sw.comms_5gnr.pss_params(
        sample_rate=FS,
        subcarrier_spacing=SUBCARRIER_SPACING,
        discovery_periodicity=DISCOVERY_PERIODICITY,
        symbol_indexes=SYMBOL_INDEXES,
        center_frequency=CENTER_FREQUENCY,
    )
    x, templates, expected = build_waveform(params, args.ports, args.blocks, args.seed)
    x_dev, templates_dev = cp.asarray(x), cp.asarray(templates)
    print(
        f'waveform {x.shape} complex64, {args.blocks} sync blocks, '
        f'expected peaks {expected}'
    )
    failed = False

    # the whole function
    host = sw.comms_5gnr.correlate_sync_sequence(x, templates, params=params)
    dev = sw.comms_5gnr.correlate_sync_sequence(x_dev, templates_dev, params=params)
    cp.cuda.Device().synchronize()
    host_peaks, dev_peaks = peaks_of(host), peaks_of(dev)
    diffs = port_differences(dev, host)
    print(f'\n{"stage":<24} {"rel max diff per port":<{10 * args.ports}}  verdict')
    ok = max(diffs) <= args.rtol and dev_peaks == host_peaks
    failed |= not ok
    print(
        f'{"correlator":<24} {fmt(diffs):<{10 * args.ports}}  {"ok" if ok else "DIFF"}'
    )
    print(f'  host peaks   {host_peaks}')
    print(f'  device peaks {dev_peaks}')
    if host_peaks != expected:
        failed = True
        print(f'  FAIL: host peaks differ from the embedded {expected}')

    # the convolution on the strided view and on a contiguous copy
    iq_host = strided_blocks(x, params)
    iq_dev = strided_blocks(x_dev, params)
    R_host = oaconvolve_stage(iq_host, templates, np)
    for label, iq in (
        ('oaconvolve strided', iq_dev),
        ('oaconvolve contiguous', cp.ascontiguousarray(iq_dev)),
    ):
        R_dev = oaconvolve_stage(iq, templates_dev, cp)
        cp.cuda.Device().synchronize()
        diffs = port_differences(R_dev, R_host)
        ok = max(diffs) <= args.rtol
        failed |= not ok
        print(f'{label:<24} {fmt(diffs):<{10 * args.ports}}  {"ok" if ok else "DIFF"}')
        del R_dev

    # the tail on the host convolution
    tail_host = tail_stage(R_host, params, np)
    tail_dev = tail_stage(cp.asarray(R_host), params, cp)
    cp.cuda.Device().synchronize()
    diffs = port_differences(tail_dev, tail_host)
    ok = max(diffs) <= args.rtol
    failed |= not ok
    print(f'{"tail":<24} {fmt(diffs):<{10 * args.ports}}  {"ok" if ok else "DIFF"}')

    # each operation of the tail on the host result of the operations before it
    h = R_host
    for name, op in TAIL_OPS:
        h_out = op(h, params, np)
        d_out = op(cp.asarray(h), params, cp)
        cp.cuda.Device().synchronize()
        diffs = port_differences(d_out, h_out)
        ok = max(diffs) <= args.rtol
        failed |= not ok
        label = f'tail/{name}'
        print(
            f'{label:<24} {fmt(diffs):<{10 * args.ports}}  {"ok" if ok else "DIFF"}'
            f'  in {h.shape} strides {to_numpy(cp.asarray(h)).strides}'
            f' -> out {tuple(d_out.shape)} strides {d_out.strides}'
        )
        del d_out
        h = h_out

    print(
        f'\nstrided view strides (bytes): {iq_dev.strides}, contiguous: {iq_dev.flags.c_contiguous}'
    )
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
