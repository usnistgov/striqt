"""striqt.sensor.lib.compute.tolerance: the roundoff budget of the correction stage,
the composition of that budget into the registered tolerance of each measurement,
and the sweep-level reductions printed by ``check-sweep``"""

from __future__ import annotations

import pytest
from sweep_strategies import SOURCE
from synthetic_sources import (
    ANALYSIS,
    FILTER_ONLY,
    LO_SHIFT_CAPTURE,
    RESAMPLE_FILTER,
    RESAMPLE_ONLY,
    SCALE_ONLY,
    preset_sweep,
    tone_capture,
)

import striqt.analysis as sa
import striqt.sensor as ss
from striqt.sensor.lib.compute import corrections, tolerance

PRESETS = {
    'scale_only': SCALE_ONLY,
    'filter_only': FILTER_ONLY,
    'resample_only': RESAMPLE_ONLY,
    'resample_filter': RESAMPLE_FILTER,
}


# %% _oaconvolve_nfft


@pytest.mark.parametrize(
    'signal_size',
    [1000, 25_000, 400_000],
    ids=['whole_convolution', '25k_samples', '400k_samples'],
)
def test_oaconvolve_nfft_matches_scipy_block_choice(signal_size):
    """the Lambert W port reproduces the block length chosen by scipy, after the
    fast-size rounding that `fftconvolve` applies to a block that `_calc_oa_lens`
    leaves at s1 + s2 - 1"""
    from scipy.signal import _signaltools

    block, _, _, _ = _signaltools._calc_oa_lens(signal_size, corrections.FILTER_SIZE)
    expected = corrections._get_next_fast_len(block, 'numpy')
    assert (
        tolerance._oaconvolve_nfft(signal_size, corrections.FILTER_SIZE, 'numpy')
        == expected
    )


# %% correction_error


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
        assert tol.on_peak.rms > exact_input[name].on_peak.rms
        assert tol.on_peak.peak > exact_input[name].on_peak.peak


def test_sweep_tolerances_follow_the_looped_captures():
    captures = (tone_capture(RESAMPLE_FILTER), tone_capture(SCALE_ONLY))
    sweep = preset_sweep('single_tone', captures, analysis=ANALYSIS)
    entries = tolerance.sweep_tolerances(sweep)

    assert [c for c, _ in entries] == list(ss.specs.sequencing.loop_captures(sweep))
    for capture, tols in entries:
        assert tols == tolerance.capture_tolerances(capture, sweep.source, ANALYSIS)
    # iq_waveform passes the correction error straight through, so it alone orders the
    # captures; the spectral products also differ in nfft between the 2 presets
    assert (
        entries[1][1]['iq_waveform'].on_peak.rms
        < entries[0][1]['iq_waveform'].on_peak.rms
    )


def test_sweep_tolerances_use_the_looped_value_over_an_adjustment():
    """`loop_captures` applies the loop point last. A looped field keeps the loop value
    over an `adjust_captures` entry, and the budget is that of the looped capture.
    ``lo_shift='none'`` skips the shift stage, and the budget differs from that of a
    shifted capture."""
    loops = (ss.specs.List(field='lo_shift', values=('none', 'left')),)
    sweep = preset_sweep(
        'single_tone',
        (tone_capture(LO_SHIFT_CAPTURE),),
        loops=loops,
        adjust_captures={'defaults': {'lo_shift': 'right'}},
    )
    expected = list(ss.specs.sequencing.loop_captures(sweep))
    entries = tolerance.sweep_tolerances(sweep)

    assert [c for c, _ in entries] == expected
    assert [c.lo_shift for c, _ in entries] == ['none', 'left']
    budgets = [
        tolerance.capture_tolerances(c, sweep.source, ANALYSIS) for c in expected
    ]
    assert [t for _, t in entries] == budgets
    # the adjusted value would have changed the first budget, so the check discriminates
    adjusted = expected[0].replace(lo_shift='right')
    assert tolerance.capture_tolerances(adjusted, sweep.source, ANALYSIS) != budgets[0]


# %% worst_case_tolerances


def _tol(rms, peak, off_peak_dBc=None, rtol=1e-6):
    bound = sa.specs.ErrorBound
    return sa.specs.Tolerance(
        units='dB',
        rtol=rtol,
        on_peak=bound(rms=rms, peak=peak),
        off_peak_dBc=None
        if off_peak_dBc is None
        else bound(rms=off_peak_dBc[0], peak=off_peak_dBc[1]),
    )


def test_worst_case_takes_the_loosest_field_of_each_product():
    entries = [
        (None, {'a': _tol(1e-3, 2e-3, (-110.0, -100.0)), 'b': _tol(1e-4, 1e-4)}),
        (
            None,
            {'a': _tol(2e-3, 1e-3, (-120.0, -90.0), rtol=2e-6), 'b': _tol(5e-5, 2e-4)},
        ),
    ]
    worst = tolerance.worst_case_tolerances(entries)
    assert worst['a'] == _tol(2e-3, 2e-3, (-110.0, -90.0), rtol=2e-6)
    assert worst['b'] == _tol(1e-4, 2e-4)


def test_worst_case_off_peak_depth_ignores_captures_without_one():
    entries = [
        (None, {'a': _tol(1e-3, 1e-3)}),
        (None, {'a': _tol(1e-3, 1e-3, (-90.0, -80.0))}),
    ]
    worst = tolerance.worst_case_tolerances(entries)['a']
    assert worst.off_peak_dBc == sa.specs.ErrorBound(rms=-90.0, peak=-80.0)
    assert tolerance.worst_case_tolerances(entries[:1])['a'].off_peak_dBc is None
