"""budget the roundoff error of a sweep, capture by capture.

The correction stage's own roundoff, from the FFT sizes `correct_iq` runs on a
capture, feeds each measurement's registered tolerance function as its `input_error`.
"""

from __future__ import annotations as __

import math
from collections.abc import Iterable
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
    """return the fast FFT length (in samples) at or above `n` for `array_backend`.

    Falls back to scipy's rule when the backend's own is not importable: a budget for
    a cupy sweep is still wanted on a host without cupy, and the two rules disagree by
    a few samples at most, below what the sqrt(log2 N) roundoff model resolves.
    """
    try:
        return corrections._get_next_fast_len(n, array_backend)
    except ImportError:
        return corrections._get_next_fast_len(n, 'numpy')


def _oaconvolve_nfft(
    signal_size: int, filter_size: int, array_backend: specs.types.ArrayBackend
) -> int:
    """return the FFT size (in samples) that `scipy.signal.oaconvolve` picks.

    The size follows `scipy.signal._signaltools._calc_oa_lens` (shared by its cupyx
    port): a fast length for `array_backend` that depends on `filter_size` alone,
    except that it never exceeds the fast length of one transform over the whole
    convolution of a `signal_size`-sample input.
    """
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
    """return the (leading, trailing) overlap sample counts of the correction stage.

    The counts are those `corrections.get_correction_overlaps` gives for
    `array_backend` rather than for `source.array_backend`, with the same fallback to
    numpy as `_next_fast_len` when that backend is not importable.
    """
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
    """bound the rms amplitude error that `correct_iq` leaves in a capture's IQ.

    The bound grows with the FFT sizes the correction runs, which follow from
    `capture.duration`, the ratio of `source.master_clock_rate` to
    `capture.sample_rate`, whether `capture.analysis_bandwidth` is finite, and the
    alignment overlap of the trigger in `analysis`. A capture that is neither
    resampled nor filtered is bounded by the roundoff of its voltage scaling alone.

    Args:
        analysis: the measurements the sweep runs, whose `signal_trigger` sets the
            alignment overlap; None budgets no trigger
        stage: the `AcquiredIQ` stage whose error is wanted; ``'pre_filter'`` stops
            before the FIR low-pass
        array_backend: the backend whose FFT roundoff model applies, or None for
            `source.array_backend`

    Returns:
        the rms error relative to the output rms (dimensionless and positive),
        including the `fft_tolerance_rms` safety margin
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
    """budget the roundoff error of each measurement in `analysis` for one capture.

    The `correction_error` of the capture enters as the `input_error` of every
    measurement's tolerance function.

    Args:
        array_backend: the backend whose roundoff model applies, or None for
            `source.array_backend`

    Returns:
        a `Tolerance` per measurement, keyed by measurement name; measurements
        without a tolerance function are omitted (see `AnalysisRegistry.tolerances`)
    """
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
    """budget the roundoff error of every capture the sweep runs, in run order.

    The captures are exactly those `specs.helpers.loop_captures` expands for
    `source_id`: the per-source `adjust_captures` overrides are applied there, a
    looped field keeps the loop's value over any adjustment, and a `repeat` loop is
    listed once.

    Returns:
        ``(capture, tolerances)`` pairs, with `tolerances` as `capture_tolerances`
        returns for the sweep's source backend
    """
    result = []
    for capture in specs.helpers.loop_captures(sweep, source_id=source_id):
        result.append((
            capture,
            capture_tolerances(capture, sweep.source, sweep.analysis),
        ))
    return result


def _loosest(bounds: Iterable[sa.specs.ErrorBound]) -> sa.specs.ErrorBound:
    bounds = list(bounds)
    return sa.specs.ErrorBound(
        rms=max(b.rms for b in bounds), peak=max(b.peak for b in bounds)
    )


def worst_case_tolerances(
    entries: list[tuple[specs.SensorCapture, dict[str, sa.specs.Tolerance]]],
) -> dict[str, sa.specs.Tolerance]:
    """return the loosest budget of each analysis product over the captures of a sweep.

    Every field of the result is at least as loose as that field in each entry:
    `rtol` and both bounds of `on_peak` are no smaller than any entry's, and
    `off_peak_dBc` is no deeper below the peak than any entry's, since a shallower
    depth leaves more of the output unchecked; None (everything resolved) is the
    strictest and loses to any finite depth. `units` are those of the first entry.

    Args:
        entries: ``(capture, tolerances)`` pairs as `sweep_tolerances` returns

    Returns:
        one `Tolerance` per measurement name appearing in any entry
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
