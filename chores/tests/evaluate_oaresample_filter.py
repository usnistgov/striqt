#!/usr/bin/env python3
"""Evaluate the STFT-domain analysis filter of `oaresample` against the FIR path.

The sweep covers the oaresample path at each ``MIN_OARESAMPLE_FFT_SIZE`` and COLA
window given on the command line, for the ADA 3750 MHz capture (125 MS/s radio clock,
107.52 MS/s output, 100 MHz analysis bandwidth). The default path is the baseline:
`sw.resample` (brick-wall FFT resampler, hamming COLA design) followed by the 4001-tap
FIR of `sw.design_fir_lpf`. Both low-passes are specified with the same band edges,
``bw/2 - transition/2`` to ``bw/2 + transition/2``.

Per design the tables report:

- geometry: ``nfft``, ``nfft_out``, STFT frames per hop, and the overlap (in source
  samples) that the sensor pads at each end of an acquisition
- tone rejection: the output level (dB) of a unit tone at frequencies across the
  passband, the transition band, and the stopband; the roundoff floor of a float32
  FFT is near -140 dB
- noise leakage: for white noise input, the mean PSD past the transition band over
  the mean PSD inside the passband (dB), which is the overlap-add leakage floor of
  each window
- impulse response: the peak level (dB) relative to ``bw/fs_sdr``, the width (in
  output samples) of the response above -60 dBc, the rms level (dBc) over a fixed
  span each side beyond the FIR half-length ``FILTER_SIZE // 2``, and the rms
  difference from the FIR path's response
- time per call (CUDA event time on cupy, wall time on numpy), and a closing table of the time of
  each correction stage per record (resample, and the FIR on the default path) with
  the ratio of record duration to correction time

Usage:
    pixi run -e test39 python chores/tests/evaluate_oaresample_filter.py --backend numpy
    pixi run -e cupy python chores/tests/evaluate_oaresample_filter.py
    pixi run -e cupy python chores/tests/evaluate_oaresample_filter.py --duration 0.05
        --min-fft-size 8191 16383 32767 --window hamming blackman

Run with its own cache directory beside a test session or another striqt script
(``STRIQT_CACHE_DIR=$(mktemp -d)``): two processes sharing the window shelf corrupt it on
macOS Python 3.9.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import platform
import time
from typing import Any

import numpy as np

import striqt.waveform as sw
from striqt.analysis import testing
from striqt.sensor.lib.compute import corrections

cp: Any = None

# the ADA denver-3750 capture (profile_cuda_streams.CAPTURE) and the air7201b clock
MASTER_CLOCK_RATE = 125e6
SAMPLE_RATE = 107.52e6
ANALYSIS_BANDWIDTH = 100e6
TRANSITION_BW = corrections.FIR_TRANSITION_BW
FILTER_SIZE = corrections.FILTER_SIZE
# design_resampler defaults
BW_LO = 0.25e6
MIN_OVERSAMPLING = 1.1

DEFAULT_MIN_FFT_SIZES = (4096 - 1, 4 * 4096 - 1, 16 * 4096 - 1)
DEFAULT_WINDOWS = ('hamming', 'blackman')

PSD_SEGMENT = 8192
IMPULSE_FLOOR_DBC = -60.0
# output samples each side of the impulse over which the tail is averaged, past the
# FIR half-length: a fixed span keeps the figure independent of the record length
TAIL_SPAN = 100_000


@dataclasses.dataclass(frozen=True)
class Design:
    name: str
    """row label"""
    window: str
    """COLA window of the resampler"""
    min_fft_size: int
    """the `min_fft_size` given to `design_cola_resampler`"""
    resampler: sw.fourier.ResamplerDesign
    """the rates and FFT sizes"""
    fir: bool
    """True runs the default path: `sw.resample` and the FIR; False runs `oaresample`"""

    @property
    def nfft(self) -> int:
        return self.resampler['nfft']

    @property
    def nfft_out(self) -> int:
        return self.resampler['nfft_out']

    @property
    def fs_sdr(self) -> float:
        return self.resampler['fs_sdr']

    @property
    def noverlap(self) -> int:
        """overlap (in source samples) of consecutive STFT blocks"""
        overlap_scale = sw.fourier.design_oafilter(
            self.nfft,
            window=self.window,
            nfft_out=self.nfft_out,
            nfft=self.nfft,
            extend=True,
        )[2]
        return round(self.nfft * overlap_scale)

    @property
    def frames_per_hop(self) -> int:
        return round(self.nfft / (self.nfft - self.noverlap))

    @property
    def overlap(self) -> int:
        """source samples padded at each end of an acquisition for this design"""
        if self.fir:
            return math.ceil((FILTER_SIZE // 2 + 1) * self.nfft / self.nfft_out)
        noverlap = self.noverlap
        lead_step = self.nfft // math.gcd(self.nfft, self.nfft_out)
        return math.ceil(noverlap / lead_step) * lead_step


def build_designs(min_fft_sizes: list[int], windows: list[str]) -> list[Design]:
    default = sw.fourier.design_cola_resampler(
        MASTER_CLOCK_RATE,
        SAMPLE_RATE,
        bw=ANALYSIS_BANDWIDTH,
        bw_lo=BW_LO,
        min_oversampling=MIN_OVERSAMPLING,
        min_fft_size=256,
        window=corrections.RESAMPLE_COLA_WINDOW,
    )
    designs = [
        Design('resample+fir', corrections.RESAMPLE_COLA_WINDOW, 256, default, True)
    ]
    for window in windows:
        for min_fft_size in min_fft_sizes:
            resampler = sw.fourier.design_cola_resampler(
                MASTER_CLOCK_RATE,
                SAMPLE_RATE,
                bw=ANALYSIS_BANDWIDTH,
                bw_lo=BW_LO,
                min_oversampling=MIN_OVERSAMPLING,
                min_fft_size=min_fft_size,
                window=window,
            )
            name = f'oa {window} {min_fft_size}'
            designs.append(Design(name, window, min_fft_size, resampler, False))
    return designs


def signal_count(duration: float, design: Design) -> int:
    """return the sample count at `fs_sdr` of a record `design` resamples whole.

    The count is `duration` rounded up to a multiple of `nfft`: `sw.resample` then
    yields a whole number of output samples and `oaresample` a whole number of STFT
    hops.
    """
    return math.ceil(duration * design.fs_sdr / design.nfft) * design.nfft


def fir_stage(y: Any, xp: Any) -> Any:
    """return `y` low-passed by the analysis FIR of the default path."""
    h = sw.design_fir_lpf(
        bw=ANALYSIS_BANDWIDTH,
        fs=SAMPLE_RATE,
        transition_bw=TRANSITION_BW,
        numtaps=FILTER_SIZE,
        xp=xp,
    )
    return sw.oaconvolve(y, h[xp.newaxis, :], 'same', axes=1)


def resample_stage(design: Design, x: Any, xp: Any) -> Any:
    """return `x` resampled under `design`; low-passed too unless `design.fir`."""
    if design.fir:
        ny = round(x.shape[1] * design.nfft_out / design.nfft)
        return sw.resample(x, ny, axis=1)
    return sw.oaresample(
        x,
        up=design.nfft_out,
        down=design.nfft,
        fs=design.fs_sdr,
        window=design.window,
        axis=1,
        filter_bandwidth=ANALYSIS_BANDWIDTH,
        transition_bandwidth=TRANSITION_BW,
    )


def run(design: Design, x: Any, xp: Any) -> Any:
    """return `x` resampled and low-passed under `design`, shape ``(1, n_out)``."""
    y = resample_stage(design, x, xp)
    return fir_stage(y, xp) if design.fir else y


def to_numpy(x: Any) -> np.ndarray:
    return x.get() if hasattr(x, 'get') else np.asarray(x)


def interior(y: np.ndarray, design: Design) -> np.ndarray:
    """return `y` without the edge where the STFT or the FIR transient falls."""
    margin = max(design.nfft_out, FILTER_SIZE)
    return y[0, margin:-margin]


def db(power: float) -> float:
    return 10 * math.log10(power) if power > 0 else -math.inf


def rms_power(x: np.ndarray) -> float:
    return float(np.mean(np.abs(x) ** 2))


# %% evaluations


def tone_frequencies() -> list[tuple[str, float]]:
    edge = ANALYSIS_BANDWIDTH / 2
    t = TRANSITION_BW
    return [
        ('dc', 0.0),
        ('0.5 bw/2', 0.5 * edge),
        ('0.9 bw/2', 0.9 * edge),
        ('bw/2 - T/2', edge - t / 2),
        ('bw/2', edge),
        ('bw/2 + T/2', edge + t / 2),
        ('bw/2 + T', edge + t),
        ('bw/2 + 2T', edge + 2 * t),
        ('bw/2 + 5T', edge + 5 * t),
        ('0.98 fs/2', 0.98 * SAMPLE_RATE / 2),
    ]


def tone_levels(design: Design, count: int, xp: Any) -> dict[str, float]:
    levels = {}
    for label, frequency in tone_frequencies():
        x = testing.tone(None, design.fs_sdr, frequency=frequency, count=count, xp=xp)
        y = to_numpy(run(design, x, xp))
        levels[label] = db(rms_power(interior(y, design)))
    return levels


def noise_leakage(design: Design, count: int, xp: Any) -> float:
    """return the stopband-to-passband PSD ratio (dB) of white noise input."""
    x = testing.circular_awgn(None, design.fs_sdr, count=count, xp=xp, seed=1)
    y = interior(to_numpy(run(design, x, xp)), design)
    seg = PSD_SEGMENT
    frames = y[: (y.size // seg) * seg].reshape(-1, seg) * np.hanning(seg)
    psd = (np.abs(np.fft.fft(frames, axis=1)) ** 2).mean(0)
    freqs = np.fft.fftfreq(seg, 1 / SAMPLE_RATE)
    edge = ANALYSIS_BANDWIDTH / 2
    passband = np.abs(freqs) < 0.8 * edge
    stopband = np.abs(freqs) > edge + TRANSITION_BW
    return db(psd[stopband].mean() / psd[passband].mean())


@dataclasses.dataclass
class ImpulseResult:
    response: np.ndarray
    """the output about the impulse, ``FILTER_SIZE`` samples each side"""
    peak_dB: float
    """peak level (dB) relative to ``bw/fs_sdr``"""
    peak_offset: int
    """output samples between the peak and the expected impulse position"""
    width: int
    """output samples above `IMPULSE_FLOOR_DBC` below the peak"""
    tail_dBc: float
    """rms level (dBc) over `TAIL_SPAN` samples each side beyond ``FILTER_SIZE // 2``"""


def impulse_response(design: Design, count: int, xp: Any) -> ImpulseResult:
    # an input index that maps onto a whole output sample
    step = design.nfft // math.gcd(design.nfft, design.nfft_out)
    index_in = (count // 2) // step * step
    index_out = index_in * design.nfft_out // design.nfft
    x = testing.dirac_delta(
        None, design.fs_sdr, time=index_in / design.fs_sdr, count=count, xp=xp
    )
    y = to_numpy(run(design, x, xp))[0]
    mag = np.abs(y)
    peak = int(np.argmax(mag))
    expected = ANALYSIS_BANDWIDTH / design.fs_sdr
    half = FILTER_SIZE // 2
    tail = np.concatenate([
        y[peak - half - TAIL_SPAN : peak - half],
        y[peak + half + 1 : peak + half + 1 + TAIL_SPAN],
    ])
    if tail.size != 2 * TAIL_SPAN:
        raise ValueError('the record is too short for the tail span; raise --duration')
    return ImpulseResult(
        response=y[index_out - FILTER_SIZE : index_out + FILTER_SIZE + 1],
        peak_dB=db(mag[peak] ** 2 / expected**2),
        peak_offset=peak - index_out,
        width=int(np.count_nonzero(mag > mag[peak] * 10 ** (IMPULSE_FLOOR_DBC / 20))),
        tail_dBc=db(rms_power(tail) / mag[peak] ** 2),
    )


def _median_time(func: Any, repeat: int, *args: Any) -> float:
    """return the median time (s) of `func(*args)` after one warm-up call.

    On cupy the time is the device time between CUDA events around each call, from
    `cupyx.profiler.benchmark`; on numpy the wall time.
    """
    if cp is not None:
        from cupyx.profiler import benchmark  # ty: ignore[unresolved-import]

        result = benchmark(func, args, n_repeat=repeat, n_warmup=1)
        return float(np.median(result.gpu_times[0]))

    func(*args)
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        func(*args)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def stage_times(design: Design, count: int, xp: Any, repeat: int) -> dict[str, float]:
    """return the median time (s) of each correction stage on a noise record.

    The keys are ``'resample'`` and, for the default path, ``'filter'``.
    """
    x = testing.circular_awgn(None, design.fs_sdr, count=count, xp=xp, seed=2)
    times = {'resample': _median_time(resample_stage, repeat, design, x, xp)}
    if design.fir:
        y = resample_stage(design, x, xp)
        times['filter'] = _median_time(fir_stage, repeat, y, xp)
    return times


def wall_time(design: Design, count: int, xp: Any, repeat: int) -> float:
    """return the median time (s) of `run` on a noise record."""
    x = testing.circular_awgn(None, design.fs_sdr, count=count, xp=xp, seed=2)
    return _median_time(run, repeat, design, x, xp)


# %% report


def print_table(title: str, header: list[str], rows: list[list[str]]) -> None:
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(header)]
    print(f'\n== {title} ==')
    print('  '.join(h.rjust(w) for h, w in zip(header, widths)))
    for row in rows:
        print('  '.join(c.rjust(w) for c, w in zip(row, widths)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument(
        '--min-fft-size',
        type=int,
        nargs='+',
        default=list(DEFAULT_MIN_FFT_SIZES),
        help='MIN_OARESAMPLE_FFT_SIZE candidates',
    )
    parser.add_argument(
        '--window',
        nargs='+',
        default=list(DEFAULT_WINDOWS),
        choices=('hamming', 'blackman', 'blackmanharris'),
        help='OARESAMPLE_COLA_WINDOW candidates',
    )
    parser.add_argument(
        '--duration', type=float, default=0.02, help='record length (s)'
    )
    parser.add_argument('--repeat', type=int, default=5, help='timing repetitions')
    parser.add_argument('--backend', choices=('cupy', 'numpy'), default='cupy')
    parser.add_argument(
        '--save', default=None, help='.npz path for the impulse responses'
    )
    args = parser.parse_args()

    global cp
    if args.backend == 'cupy':
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

    designs = build_designs(args.min_fft_size, args.window)
    counts = {d.name: signal_count(args.duration, d) for d in designs}
    fs_sdr = designs[0].fs_sdr
    print(
        f'{min(counts.values())} to {max(counts.values())} samples per design '
        f'at {fs_sdr / 1e6:g} MS/s -> {SAMPLE_RATE / 1e6:g} MS/s, '
        f'bw {ANALYSIS_BANDWIDTH / 1e6:g} MHz, transition {TRANSITION_BW / 1e3:g} kHz'
    )

    rows = []
    for d in designs:
        rows.append([
            d.name,
            str(d.nfft),
            str(d.nfft_out),
            str(d.frames_per_hop),
            str(d.overlap),
            f'{d.overlap / fs_sdr * 1e6:.1f}',
        ])
    print_table(
        'geometry',
        ['design', 'nfft', 'nfft_out', 'frames/hop', 'overlap', 'overlap (us)'],
        rows,
    )

    labels = [label for label, _ in tone_frequencies()]
    rows = []
    for d in designs:
        levels = tone_levels(d, counts[d.name], xp)
        rows.append([d.name] + [f'{levels[label]:.1f}' for label in labels])
    print_table('tone level (dB) by frequency', ['design'] + labels, rows)

    impulses = {d.name: impulse_response(d, counts[d.name], xp) for d in designs}
    reference = impulses[designs[0].name].response
    rows = []
    for d in designs:
        r = impulses[d.name]
        diff = db(rms_power(r.response - reference) / rms_power(reference))
        rows.append([
            d.name,
            f'{noise_leakage(d, counts[d.name], xp):.1f}',
            f'{r.peak_dB:+.3f}',
            str(r.peak_offset),
            str(r.width),
            f'{r.tail_dBc:.1f}',
            f'{diff:.1f}' if d is not designs[0] else '-',
            f'{wall_time(d, counts[d.name], xp, args.repeat) * 1e3:.1f}',
        ])
    print_table(
        'leakage, impulse response, and time',
        [
            'design',
            'noise leak (dB)',
            'peak (dB)',
            'peak offs',
            f'width >{IMPULSE_FLOOR_DBC:g} dBc',
            'tail (dBc)',
            'vs fir (dB)',
            'ms/call',
        ],
        rows,
    )

    rows = []
    for d in designs:
        n = counts[d.name]
        record = n / d.fs_sdr
        times = stage_times(d, n, xp, args.repeat)
        total = sum(times.values())
        rows.append([
            d.name,
            f'{record * 1e3:.2f}',
            f'{times["resample"] * 1e3:.1f}',
            f'{times["filter"] * 1e3:.1f}' if 'filter' in times else '-',
            f'{total * 1e3:.1f}',
            f'{record / total:.1f}',
        ])
    print_table(
        'correction time per record',
        [
            'design',
            'record (ms)',
            'resample (ms)',
            'filter (ms)',
            'total (ms)',
            'x realtime',
        ],
        rows,
    )

    if args.save:
        responses = {d.name: impulses[d.name].response for d in designs}
        np.savez(args.save, **responses)  # ty: ignore[invalid-argument-type]
        print(f'\nimpulse responses saved to {args.save}')


if __name__ == '__main__':
    main()
