"""roundoff error budgets of a sweep, capture by capture.

The correction stage's own roundoff, from the FFT sizes `correct_iq` actually runs on a
capture, feeds each measurement's registered tolerance function as its `input_error`.
"""

from __future__ import annotations as __

import math
from math import isfinite
from typing import TYPE_CHECKING, Literal

import striqt.analysis as sa

from ... import specs
from .. import util
from . import corrections

if TYPE_CHECKING:
    import striqt.waveform as sw
else:
    sw = util.lazy_import('striqt.waveform')


def _next_fast_len(n: int, array_backend: specs.types.ArrayBackend) -> int:
    """`corrections._get_next_fast_len`, falling back to scipy's rule when the backend's
    own is not importable: a budget for a cupy sweep is still wanted on a host without
    cupy, and the two rules disagree by a few samples at most, below what the
    sqrt(log2 N) model resolves"""
    try:
        return corrections._get_next_fast_len(n, array_backend)
    except ImportError:
        return corrections._get_next_fast_len(n, 'numpy')


def _oaconvolve_nfft(
    signal_size: int, filter_size: int, array_backend: specs.types.ArrayBackend
) -> int:
    """the FFT size that scipy.signal.oaconvolve (and its cupyx port) picks for a long
    signal: the block length minimizing FFT cost per output sample, the Lambert W
    solution of its `_calc_oa_lens`, rounded up to a fast size"""
    from scipy.special import lambertw

    overlap = filter_size - 1
    opt_size = -overlap * lambertw(-1 / (2 * math.e * overlap), k=-1).real
    block = _next_fast_len(math.ceil(opt_size), array_backend)
    whole = _next_fast_len(signal_size + overlap, array_backend)
    return min(block, whole)


def _correction_overlaps(
    capture: specs.SensorCapture,
    source: specs.Source,
    analysis: specs.AnalysisGroup | None,
    array_backend: specs.types.ArrayBackend,
) -> tuple[int, int]:
    """`corrections.get_correction_overlaps` for `array_backend`, with the same fallback
    as `_next_fast_len` when that backend is not importable"""
    source = source.replace(array_backend=array_backend)
    try:
        return corrections.get_correction_overlaps(capture, source, analysis)
    except ImportError:
        source = source.replace(array_backend='numpy')
        return corrections.get_correction_overlaps(capture, source, analysis)


def correction_error(
    capture: specs.SensorCapture,
    source: specs.Source,
    analysis: specs.AnalysisGroup | None = None,
    *,
    array_backend: specs.types.ArrayBackend | None = None,
    stage: Literal['pre_filter', 'pre_align'] = 'pre_align',
) -> float:
    """the relative rms amplitude error that `correct_iq` leaves in a capture's IQ.

    Sized from the FFTs the correction actually runs: `_resample` transforms the whole
    padded acquisition and inverts at the output length, the FIR low-pass is an
    overlap-add convolution of `FILTER_SIZE` taps, and the voltage scale rounds once.
    `stage` names the `AcquiredIQ` stage whose error is wanted: 'pre_filter' stops
    before the FIR. `array_backend` defaults to the source's.
    """
    if array_backend is None:
        array_backend = source.array_backend
    design = corrections.design_resampler(capture, source.master_clock_rate)
    resampled = corrections.needs_resample(design, capture)
    nffts: list[int] = []

    if resampled and corrections.USE_OARESAMPLE:
        nffts += [design['nfft'], design['nfft_out']]
    elif resampled:
        lead, tail = _correction_overlaps(capture, source, analysis, array_backend)
        n_in = round(capture.duration * design['fs_sdr']) + lead + tail
        n_out = round(n_in * capture.sample_rate / design['fs_sdr'])
        nffts += [n_in, n_out]

    filtered = stage == 'pre_align' and isfinite(capture.analysis_bandwidth)
    if filtered and not (resampled and corrections.USE_OARESAMPLE):
        n_out = round(capture.duration * capture.sample_rate)
        fir_nfft = _oaconvolve_nfft(n_out, corrections.FILTER_SIZE, array_backend)
        nffts += [fir_nfft, fir_nfft]

    return sw.fourier.fft_tolerance_rms(
        'complex64', nffts, n_elementwise=1, array_backend=array_backend
    )


def capture_tolerances(
    capture: specs.SensorCapture,
    source: specs.Source,
    analysis: specs.AnalysisGroup,
    *,
    array_backend: specs.types.ArrayBackend | None = None,
) -> dict[str, sa.specs.Tolerance]:
    """the error budget of each measurement in `analysis` for one corrected capture,
    keyed by measurement name (see `AnalysisRegistry.tolerances`)"""
    if array_backend is None:
        array_backend = source.array_backend
    input_error = correction_error(
        capture, source, analysis, array_backend=array_backend
    )
    return sa.registry.tolerances(
        capture, analysis, array_backend=array_backend, input_error=input_error
    )


def sweep_tolerances(
    sweep: specs.Sweep, source_id: specs.types.SourceID | None = None
) -> list[tuple[specs.SensorCapture, dict[str, sa.specs.Tolerance]]]:
    """`capture_tolerances` for every capture the sweep will run, in run order, after
    the per-source `adjust_captures` overrides"""
    result = []
    for capture in specs.helpers.loop_captures(sweep, source_id=source_id):
        if sweep.adjust_captures:
            changes = specs.helpers.adjust_captures(
                capture.to_dict(), sweep.adjust_captures, source_id
            )
            capture = capture.replace(**changes)
        result.append((
            capture,
            capture_tolerances(capture, sweep.source, sweep.analysis),
        ))
    return result


def _loosest(bounds) -> sa.specs.ErrorBound:
    bounds = list(bounds)
    return sa.specs.ErrorBound(
        rms=max(b.rms for b in bounds), peak=max(b.peak for b in bounds)
    )


def worst_case_tolerances(
    entries: list[tuple[specs.SensorCapture, dict[str, sa.specs.Tolerance]]],
) -> dict[str, sa.specs.Tolerance]:
    """the loosest budget of each analysis product over the captures of a sweep.

    `off_peak_dBc` takes the value closest to the peak, since that leaves more of the
    output unchecked; None (everything resolved) is the strictest and loses to any.
    """
    by_name: dict[str, list[sa.specs.Tolerance]] = {}
    for _, tolerances in entries:
        for name, tol in tolerances.items():
            by_name.setdefault(name, []).append(tol)

    result = {}
    for name, tols in by_name.items():
        depths = [t.off_peak_dBc for t in tols if t.off_peak_dBc is not None]
        result[name] = sa.specs.Tolerance(
            units=tols[0].units,
            rtol=max(t.rtol for t in tols),
            on_peak=_loosest(t.on_peak for t in tols),
            off_peak_dBc=_loosest(depths) if depths else None,
        )
    return result
