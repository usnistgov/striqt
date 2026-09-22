"""striqt.sensor.lib.compute.tolerance: the correction stage's roundoff budget, its
composition into each measurement's registered tolerance, and the sweep-level
reductions that check-sweep prints"""

from __future__ import annotations

import numpy as np
import pytest
from sweep_strategies import SOURCE
from synthetic_sources import (
    ANALYSIS,
    FILTER_ONLY,
    RESAMPLE_FILTER,
    RESAMPLE_ONLY,
    SCALE_ONLY,
    acquire_corrected,
    make_capture,
    make_sweep,
)

import striqt.analysis as sa
import striqt.sensor as ss
import striqt.waveform as sw
from striqt.sensor.lib.compute import corrections, tolerance

PRESETS = {
    'scale_only': SCALE_ONLY,
    'filter_only': FILTER_ONLY,
    'resample_only': RESAMPLE_ONLY,
    'resample_filter': RESAMPLE_FILTER,
}


def tone_capture(preset):
    return make_capture('single_tone', **preset, frequency_offset=1e6, snr=None)


# %% _oaconvolve_nfft


def test_oaconvolve_nfft_matches_scipy_block_choice():
    """the Lambert W port reproduces the block length scipy picks, after the fast-size
    rounding that fftconvolve applies to a block `_calc_oa_lens` leaves at s1 + s2 - 1"""
    from scipy.signal import _signaltools

    for signal_size in (25_000, 400_000):
        block, _, _, _ = _signaltools._calc_oa_lens(
            signal_size, corrections.FILTER_SIZE
        )
        expected = corrections._get_next_fast_len(block, 'numpy')
        assert (
            tolerance._oaconvolve_nfft(signal_size, corrections.FILTER_SIZE, 'numpy')
            == expected
        )


def test_oaconvolve_nfft_of_a_short_signal_is_the_whole_convolution():
    n = tolerance._oaconvolve_nfft(1000, corrections.FILTER_SIZE, 'numpy')
    assert n == corrections._get_next_fast_len(
        1000 + corrections.FILTER_SIZE - 1, 'numpy'
    )


# %% correction_error


@pytest.mark.parametrize('preset', list(PRESETS), ids=list(PRESETS))
def test_correction_error_is_sized_from_the_acquisition_correct_iq_runs(preset):
    capture = tone_capture(PRESETS[preset])
    stages = acquire_corrected('single_tone', capture, analysis=ANALYSIS)

    n_in = stages.raw.pre_align.shape[1]
    nffts = []
    if corrections.needs_resample(stages.resampler, capture):
        nffts += [n_in, round(n_in * capture.sample_rate / stages.resampler['fs_sdr'])]
    if np.isfinite(capture.analysis_bandwidth):
        fir = tolerance._oaconvolve_nfft(
            round(capture.duration * capture.sample_rate),
            corrections.FILTER_SIZE,
            'numpy',
        )
        nffts += [fir, fir]
    expected = sw.fourier.fft_tolerance_rms('complex64', nffts, n_elementwise=1)

    assert tolerance.correction_error(capture, SOURCE, ANALYSIS) == expected


def test_correction_error_orders_the_presets_by_work_done():
    errors = {
        k: tolerance.correction_error(tone_capture(p), SOURCE)
        for k, p in PRESETS.items()
    }
    assert errors['scale_only'] < errors['resample_only'] < errors['resample_filter']
    assert errors['scale_only'] < errors['filter_only'] < errors['resample_filter']


def test_correction_error_uses_the_backend_fft_constant():
    capture = tone_capture(RESAMPLE_FILTER)
    numpy = tolerance.correction_error(capture, SOURCE)
    cupy = tolerance.correction_error(capture, SOURCE, array_backend='cupy')
    assert cupy > numpy
    assert (
        tolerance.correction_error(capture, SOURCE.replace(array_backend='cupy'))
        == cupy
    )


# %% capture_tolerances / sweep_tolerances


def test_capture_tolerances_feed_the_correction_error_into_each_measurement():
    capture = tone_capture(RESAMPLE_FILTER)
    with_correction = tolerance.capture_tolerances(capture, SOURCE, ANALYSIS)
    exact_input = sa.registry.tolerances(capture, ANALYSIS)

    assert set(with_correction) == set(exact_input)
    assert set(with_correction) <= set(ANALYSIS.to_dict())
    for name, tol in with_correction.items():
        assert isinstance(tol, sa.specs.Tolerance)
        assert tol.rms > exact_input[name].rms
        assert tol.peak > exact_input[name].peak


def test_sweep_tolerances_follow_the_looped_captures():
    captures = (tone_capture(RESAMPLE_FILTER), tone_capture(SCALE_ONLY))
    sweep = make_sweep('single_tone', captures, analysis=ANALYSIS)
    entries = tolerance.sweep_tolerances(sweep)

    assert [c for c, _ in entries] == list(ss.specs.helpers.loop_captures(sweep))
    for capture, tols in entries:
        assert tols == tolerance.capture_tolerances(capture, sweep.source, ANALYSIS)
    # iq_waveform passes the correction error straight through, so it alone orders the
    # captures; the spectral products also differ in nfft between the two presets
    assert entries[1][1]['iq_waveform'].rms < entries[0][1]['iq_waveform'].rms


# %% worst_case_tolerances


def _tol(rms, peak, floor_dBc=None, rtol=1e-6):
    return sa.specs.Tolerance(
        units='dB', rtol=rtol, rms=rms, peak=peak, floor_dBc=floor_dBc
    )


def test_worst_case_takes_the_loosest_field_of_each_product():
    entries = [
        (None, {'a': _tol(1e-3, 2e-3, -100.0), 'b': _tol(1e-4, 1e-4)}),
        (None, {'a': _tol(2e-3, 1e-3, -90.0, rtol=2e-6), 'b': _tol(5e-5, 2e-4)}),
    ]
    worst = tolerance.worst_case_tolerances(entries)
    assert worst['a'] == _tol(2e-3, 2e-3, -90.0, rtol=2e-6)
    assert worst['b'] == _tol(1e-4, 2e-4)


def test_worst_case_floor_ignores_captures_without_one():
    entries = [(None, {'a': _tol(1e-3, 1e-3)}), (None, {'a': _tol(1e-3, 1e-3, -80.0)})]
    assert tolerance.worst_case_tolerances(entries)['a'].floor_dBc == pytest.approx(
        -80.0
    )
    assert tolerance.worst_case_tolerances(entries[:1])['a'].floor_dBc is None
