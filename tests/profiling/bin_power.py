#!/usr/bin/env python3
"""Profile the binned channel power detector at the production shape, stage by stage,
against reductions that read the waveform once.

`channel_power_time_series` takes about a third of the analysis of a 0.1 s, 2-port
capture on the Tegra X2, and its time does not fall when the per-measurement device
copy is deferred, so the host blocks inside it. The measurement calls
`sw.iq_to_bin_power` once per detector: a view of the waveform as
``(port, bin, samples)``, `envtopow` into a float32 buffer of the waveform's size, and
one reduction over the samples axis (``mean`` for ``rms``, ``max`` for ``peak``). At a
10 ms detector period the reduction has 20 outputs of about 1 million samples each.

Variants of one detector over the ``(port, bin, samples)`` view:

- ``library``: `sw.iq_to_bin_power` as the measurement calls it.
- ``envtopow+reduce``: the 2 stages of the library path timed separately.
- ``two-stage``: the power buffer viewed as ``(port, bin, chunks, k)`` and reduced over
  ``k`` then over ``chunks``, which gives the reduction kernel one block per chunk.
- ``matmul`` (``rms`` only): the mean as a matrix product with a vector of ``1/N``,
  which cuBLAS runs at the memory bandwidth.
- ``fused``: one `cupy.ReductionKernel` that squares the complex samples and reduces in
  the same pass, with no float32 buffer.

The measurement table then times `channel_power_time_series` as the pipeline reaches
it, on a conditioned acquisition of the ADA capture: the resampling stage, the
aligned stage the measurement selects on top of the resampled waveform, the
measurement body on that stage, and the registry call with the stages memoized and
from cold. The indented rows split the resampling stage: `sw.oaresample` with the
arguments of the stage, the same without the low-pass response, and the STFT alone
at the block size of the design, which is the analysis half of the resampler. Two
flags reproduce the conditions of the pipeline. ``--trigger`` conditions the
acquisition with the ``cellular_5g_pss_sync`` trigger of the production source. The
aligned stage then runs the trigger measurement and shifts each port to the trigger.
``--overwrite-x`` lets the resampling stage consume the raw waveform in place. Each
fresh conditioning then first refills the raw waveform with ones. The ``refill
raw.x`` row times that refill alone.

Usage:
    pixi run -e cupy python tests/profiling/bin_power.py
    pixi run -e cupy python tests/profiling/bin_power.py --skip-variants
    pixi run -e cupy python tests/profiling/bin_power.py --skip-variants --trigger --overwrite-x
    pixi run -e cupy python tests/profiling/bin_power.py --periods 0.01 --repeat 20
    pixi run -e test39 python tests/profiling/bin_power.py --backend numpy --duration 0.02

Reading the output:
    One row per variant, detector, and detector period: the host time until the call
    returns (``host ms``), the device time by CUDA events (``dev ms``), the input
    bytes over the device time (``GB/s``; the waveform is read once at least, so the
    bandwidth of the Tegra X2 bounds it near 20 GB/s), and the largest relative
    difference from a float64 numpy evaluation with the tolerance of
    `sw.power_analysis.bin_power_rtol` plus the accumulation term of
    `bin_power_rms` for the mean; a stage that stops before the reduction has no
    result to compare and shows ``-``. A host time equal to the device time means the
    call blocks the host. A path that passes the tolerance and runs near the bandwidth
    is the candidate for `iq_to_bin_power`. In the measurement table, a registry
    call far above the body is time in the wrapper or the stage it selects, and a
    stage row names which. The ``aligned stage`` row is the cost of the aligned
    preprocess after the resampling stage: with ``--trigger`` the trigger measurement
    and the port shifts, without the flag a view. With ``--overwrite-x`` the
    ``resample stage`` and ``registry call, cold`` rows include the ``refill raw.x``
    time. The ``aligned stage`` row does not. The exit status is 1 when a variant
    exceeds its tolerance, which the ``library`` row itself does on numpy 2.0 (arm64)
    for million-sample bins: the float32 mean over one long contiguous axis
    accumulates above the pairwise model of `accum_rms`, where the two-stage and
    matmul reductions stay inside it.
"""

from __future__ import annotations

import argparse
import dataclasses
import platform
import statistics
import sys
import time
from collections.abc import Callable
from typing import Any, cast

import numpy as np

import striqt.waveform as sw
from striqt.analysis import testing

cp: Any = None

SAMPLE_RATE = 107.52e6
PORTS = 2
DEFAULT_PERIODS = (0.01, 1 / 28000)
DETECTORS = ('rms', 'peak')
CHUNK = 1024
ITEM_BYTES = 8

Array = Any


# %% Timing
def device_synchronize() -> None:
    if cp is not None:
        cp.cuda.Device().synchronize()


def time_call(func: Callable[[], Any]) -> tuple[float, float]:
    """return the (host, device) duration (in s) of one call.

    The host time ends when the call returns; the device time is the CUDA event span
    around it, equal to the host time on numpy.
    """
    if cp is None:
        t0 = time.perf_counter()
        func()
        dt = time.perf_counter() - t0
        return dt, dt
    start, end = cp.cuda.Event(), cp.cuda.Event()
    device_synchronize()
    start.record()
    t0 = time.perf_counter()
    func()
    host = time.perf_counter() - t0
    end.record()
    end.synchronize()
    return host, float(cp.cuda.get_elapsed_time(start, end)) / 1e3


# %% Variants
def blocks(iq: Array, count: int) -> Array:
    return sw.arrays.axis_to_blocks(iq, count, axis=1)


def chunked(power: Array, k: int) -> Array:
    """return `power` viewed as ``(port, bin, chunks, k)``."""
    ports, bins, n = power.shape
    return power.reshape(ports, bins, n // k, k)


def chunk_size(n: int) -> int:
    k = CHUNK
    while n % k:
        k //= 2
    return k


def make_variants(
    iq: Array, kind: str, count: int, xp: Any
) -> dict[str, Callable[[], Array]]:
    """return the callables that evaluate the `kind` detector over bins of `count`."""
    reduce = xp.mean if kind == 'rms' else xp.max
    k = chunk_size(count)
    view = blocks(iq, count)

    def library() -> Array:
        return sw.iq_to_bin_power(
            iq, kind=kind, Ts=1 / SAMPLE_RATE, Tbin=count / SAMPLE_RATE, axis=1
        )

    def envtopow_only() -> Array:
        return sw.envtopow(view)

    power = sw.envtopow(view)

    def reduce_only() -> Array:
        return reduce(power, axis=2)

    def two_stage() -> Array:
        return reduce(reduce(chunked(power, k), axis=3), axis=2)

    def envtopow_two_stage() -> Array:
        return reduce(reduce(chunked(sw.envtopow(view), k), axis=3), axis=2)

    variants: dict[str, Callable[[], Array]] = {
        'library': library,
        'envtopow only': envtopow_only,
        'reduce only': reduce_only,
        'two-stage reduce only': two_stage,
        'envtopow+two-stage': envtopow_two_stage,
    }

    if kind == 'rms':
        ones = xp.full((count,), 1.0 / count, dtype=power.dtype)

        def matmul() -> Array:
            return (power.reshape(-1, count) @ ones).reshape(power.shape[:2])

        variants['envtopow+matmul'] = lambda: (
            sw.envtopow(view).reshape(-1, count) @ ones
        ).reshape(view.shape[:2])
        variants['matmul reduce only'] = matmul

    if cp is not None:
        if kind == 'rms':
            kernel = cp.ReductionKernel(
                'T x',
                'float32 y',
                'x.real() * x.real() + x.imag() * x.imag()',
                'a + b',
                'y = a / (_in_ind.size() / _out_ind.size())',
                '0',
                'striqt_profile_mean_power',
            )
        else:
            kernel = cp.ReductionKernel(
                'T x',
                'float32 y',
                'x.real() * x.real() + x.imag() * x.imag()',
                'max(a, b)',
                'y = a',
                '0',
                'striqt_profile_peak_power',
            )
        variants['fused'] = lambda: kernel(view, axis=2)

    return variants


# %% Reference and tolerance
def reference(iq_np: np.ndarray, kind: str, count: int) -> np.ndarray:
    power = np.abs(blocks(iq_np.astype(np.complex128), count)) ** 2
    return power.mean(axis=2) if kind == 'rms' else power.max(axis=2)


def tolerance(kind: str, count: int) -> float:
    return sw.power_analysis.bin_power_rtol(
        np.float32, kind=kind
    ) + sw.power_analysis.bin_power_rms(np.float32, count, kind=kind)


def to_numpy(a: Array) -> np.ndarray:
    if cp is not None and isinstance(a, cp.ndarray):
        return cp.asnumpy(a)
    return np.asarray(a)


@dataclasses.dataclass
class Layer:
    label: str
    """the row label of the measurement table"""
    func: Callable[[], Any]
    """the timed call"""
    prepare: Callable[[], None] | None = None
    """untimed setup that runs before each call of `func`; None for no setup"""


def profile_measurement(
    duration: float,
    backend: str,
    xp: Any,
    repeat: int,
    trigger: bool,
    overwrite_x: bool,
) -> None:
    """print the time of the channel power measurement layer by layer.

    Args:
        duration: the capture duration (in s)
        backend: the array backend of the source spec, ``'numpy'`` or ``'cupy'``
        xp: the array namespace of the raw waveform
        repeat: the number of timed calls per row; the table prints the median
        trigger: True to condition the acquisition with the ``cellular_5g_pss_sync``
            trigger of the production source
        overwrite_x: True to let the resampling stage consume the raw waveform in
            place, as the pipeline does

    With `overwrite_x`, each fresh conditioning refills the raw waveform with ones
    before the stage consumes it. The ``refill raw.x`` row times that refill alone.
    The rows that condition a fresh acquisition inside the timed call (``resample
    stage``, ``registry call, cold``) include the refill. The ``aligned stage`` row
    conditions and resamples before the timed call and excludes the refill.
    """
    import cuda_streams

    import striqt.analysis as sa
    import striqt.sensor as ss
    from striqt.analysis.measurements import power

    cuda_streams.cp = cp
    setup = cuda_streams.build_setup(duration, backend, trigger)
    raw = cuda_streams.build_raw(setup, xp)
    name = 'channel_power_time_series'
    spec = getattr(setup.analysis, name)
    info = sa.registry[type(spec)]
    kws = setup.analysis.to_dict()[name]
    capture = setup.capture

    def refill() -> None:
        raw.x[:] = 1

    def conditioned() -> Any:
        if overwrite_x:
            refill()
        return ss.lib.compute.attach_corrections(
            raw, signal_trigger=setup.trigger, overwrite_x=overwrite_x
        )

    # the preprocessor holds its acquisition weakly, so each fresh acquisition stays
    # bound to a local, or to `pending`, for the duration of the call
    def resample_stage() -> Any:
        fresh = conditioned()
        return fresh.preprocess(prefer_filtered=False, prefer_aligned=False)

    def registry_cold() -> Any:
        fresh = conditioned()
        return info.func(iq=fresh, capture=capture, as_xarray=False, **kws)

    pending: list[Any] = []

    def prepare_aligned() -> None:
        fresh = conditioned()
        fresh.preprocess(prefer_filtered=False, prefer_aligned=False)
        pending[:] = [fresh]

    def aligned_stage() -> Any:
        return pending[0].preprocess(prefer_filtered=True, prefer_aligned=True)

    iq = conditioned()
    layers: list[Layer] = []
    if overwrite_x:
        layers.append(Layer('refill raw.x', refill))
    layers.append(Layer('resample stage', resample_stage))
    iq.preprocess(prefer_filtered=False, prefer_aligned=False)
    x = iq.preprocess(prefer_filtered=True, prefer_aligned=True)

    # the resample stage is `sw.oaresample` with the design of the acquisition plus
    # the conjugation and 2 slices; these rows split the resampler itself
    design = cast(Any, raw.resampler)
    oaresample_kws: dict[str, Any] = {
        'up': design['nfft_out'],
        'down': design['nfft'],
        'fs': design['fs_sdr'],
        'window': design['window'],
        'axis': 1,
        'frequency_shift': design['lo_offset'],
        'transition_bandwidth': ss.lib.compute.corrections.FIR_TRANSITION_BW,
    }
    nfft = design['nfft']

    def oaresample_production() -> Any:
        return sw.oaresample(
            raw.x, filter_bandwidth=capture.analysis_bandwidth, **oaresample_kws
        )

    def oaresample_no_lowpass() -> Any:
        return sw.oaresample(raw.x, filter_bandwidth=None, **oaresample_kws)

    def stft_only() -> Any:
        return sw.stft(
            raw.x,
            fs=design['fs_sdr'],
            window=design['window'],
            nperseg=nfft,
            noverlap=nfft // 2,
            axis=1,
            truncate=False,
            return_axis_arrays=False,
        )

    layers += [
        Layer('  sw.oaresample, production args', oaresample_production),
        Layer('  sw.oaresample, no low-pass', oaresample_no_lowpass),
        Layer('  sw.stft at the design block', stft_only),
        Layer('aligned stage', aligned_stage, prepare_aligned),
        Layer(
            'body on the stage',
            lambda: power.evaluate_channel_power_time_series(x, capture, spec),
        ),
        Layer(
            'registry call, stages memoized',
            lambda: info.func(iq=iq, capture=capture, as_xarray=False, **kws),
        ),
        Layer('registry call, cold', registry_cold),
    ]

    def timed(layer: Layer) -> tuple[float, float]:
        if layer.prepare is not None:
            layer.prepare()
            device_synchronize()
        return time_call(layer.func)

    flags = ', '.join(
        f'{label} {"on" if value else "off"}'
        for label, value in (('trigger', trigger), ('overwrite_x', overwrite_x))
    )
    print(
        f'\nchannel_power_time_series on the ADA capture ({duration} s, '
        f'{raw.x.shape[0]} ports), median of {repeat}; {flags}'
    )
    print(f'{"layer":<32} {"host ms":>8} {"dev ms":>8}')
    for layer in layers:
        timed(layer)
        times = [timed(layer) for _ in range(repeat)]
        host = statistics.median(t[0] for t in times)
        dev = statistics.median(t[1] for t in times)
        print(f'{layer.label:<32} {host * 1e3:8.2f} {dev * 1e3:8.2f}', flush=True)
    pending.clear()
    sw.arrays.free_cupy_mempool()


@dataclasses.dataclass
class Row:
    period: float
    kind: str
    variant: str
    host_s: float
    device_s: float
    rel_err: float | None
    """the largest relative difference from the reference; None for a stage without the result"""
    tol: float

    @property
    def passed(self) -> bool:
        return self.rel_err is None or self.rel_err <= self.tol


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--backend', choices=('cupy', 'numpy'), default='cupy')
    parser.add_argument('--duration', type=float, default=0.1)
    parser.add_argument(
        '--periods', type=float, nargs='+', default=list(DEFAULT_PERIODS)
    )
    parser.add_argument('--repeat', type=int, default=10)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--skip-variants', action='store_true')
    parser.add_argument('--skip-measurement', action='store_true')
    parser.add_argument(
        '--trigger',
        action='store_true',
        help='condition the acquisition with the cellular_5g_pss_sync trigger',
    )
    parser.add_argument(
        '--overwrite-x',
        action='store_true',
        help='let the resampling stage consume the raw waveform in place',
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

    size = round(args.duration * SAMPLE_RATE)
    iq_np = np.asarray(
        testing.circular_awgn(
            None, SAMPLE_RATE, ports=PORTS, count=size, seed=args.seed
        ),
        dtype=np.complex64,
    )
    iq = xp.asarray(iq_np)
    in_bytes = iq.nbytes
    print(
        f'waveform {tuple(iq.shape)} complex64 ({in_bytes / 2**20:.0f} MB) at '
        f'{SAMPLE_RATE / 1e6:g} MS/s'
    )

    rows: list[Row] = []
    if not args.skip_measurement:
        profile_measurement(
            args.duration,
            args.backend,
            xp,
            args.repeat,
            args.trigger,
            args.overwrite_x,
        )
    periods = [] if args.skip_variants else args.periods
    print(
        f'\n{"period s":>9} {"bins":>5} {"kind":<5} {"variant":<24} {"host ms":>8} '
        f'{"dev ms":>8} {"GB/s":>6} {"rel err":>9} {"tol":>9}  check'
    )
    for period in periods:
        count = round(period * SAMPLE_RATE)
        if size % count:
            parser.error(f'period {period} is not a whole number of bins of {size}')
        bins = size // count
        for kind in DETECTORS:
            ref = reference(iq_np, kind, count)
            tol = tolerance(kind, count)
            for name, func in make_variants(iq, kind, count, xp).items():
                for _ in range(args.warmup):
                    func()
                device_synchronize()
                times = [time_call(func) for _ in range(args.repeat)]
                host = statistics.median(t[0] for t in times)
                dev = statistics.median(t[1] for t in times)
                result = to_numpy(func())
                # a stage that stops before the reduction has no result to compare
                err = (
                    float(np.max(np.abs(result / ref - 1)))
                    if result.shape == ref.shape
                    else None
                )
                row = Row(period, kind, name, host, dev, err, tol)
                rows.append(row)
                err_text = f'{err:9.2e}' if err is not None else f'{"-":>9}'
                check = '-' if err is None else ('ok' if row.passed else 'FAIL')
                print(
                    f'{period:9.5f} {bins:5d} {kind:<5} {name:<24} {host * 1e3:8.2f} '
                    f'{dev * 1e3:8.2f} {in_bytes / dev / 1e9:6.1f} {err_text} {tol:9.2e}  '
                    f'{check}',
                    flush=True,
                )
            sw.arrays.free_cupy_mempool()

    failed = [f'{r.variant}/{r.kind}@{r.period:g}' for r in rows if not r.passed]
    if failed:
        print(f'\nFAIL: {", ".join(failed)}')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
