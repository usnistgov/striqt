"""striqt.sensor.lib.compute.corrections: the per-capture signal path of
AcquiredIQ.preprocess, the resampler design, and the acquisition overlap that the
resampler and analysis filter consume
"""

from __future__ import annotations

import dataclasses
import gc
from collections.abc import Callable
from math import inf, isfinite
from typing import TypeVar

import numpy as np
import pytest
from hypothesis import example, given, settings, strategies as st
from numeric_checks import (
    FIR_LEAKAGE,
    assert_close,
    rms,
    to_numpy,
    tone_frequency,
)
from sweep_strategies import SOURCE as FUNCTION_SOURCE
from synthetic_sources import (
    FILTER_ONLY,
    FILTER_SIZE,
    LO_SHIFT_CAPTURE,
    PRESETS,
    RESAMPLE_FILTER,
    RESAMPLE_ONLY,
    SCALE_ONLY,
    FakeTrigger,
    acquire_corrected,
    aligned,
    build_acquired_iq,
    expected_corrected,
    expected_raw,
    filtered,
    fs_sdr,
    generator,
    preset_capture,
    source_count,
    unfiltered,
)

import striqt.analysis as sa
import striqt.waveform as sw
from striqt.sensor.lib.compute import (
    attach_corrections,
    correction_error,
    corrections,
    get_trigger_from_spec,
)
from striqt.sensor.lib.sources import buffers
from striqt.waveform.fourier import on_peak_roundoff

# 0.5 ms is a whole number of samples at the source and capture rate of every preset,
# so the impulse falls on an output sample. The brick-wall resample of an impulse is
# exactly fs/fs_sdr there and zero on every other output sample
IMPULSE_TIME = 0.5e-3
IMPULSE_POWER_DB = -6.0

# 1 MHz lies on the FFT grid of the padded acquisition of both resample presets
# (18750 and 13200 samples at 6.25 MS/s), so its resample is exact up to roundoff;
# the Gibbs ripple of an off-grid tone is tested at the kernel (tests/waveform)
TONE_FREQUENCY = 1e6
# above bw/2 plus the 250 kHz transition band of the analysis filter, below fs/2
OUT_OF_BAND_FREQUENCY = 2.9e6

TRIGGER_LAGS = (0.0, 2e-6)

PRESET_PARAMS = pytest.mark.parametrize(
    'preset', list(PRESETS.values()), ids=list(PRESETS)
)
CHEAP_PRESETS = pytest.mark.parametrize(
    'preset', [SCALE_ONLY, RESAMPLE_ONLY], ids=['scale_only', 'resample_only']
)
FILTERED_PRESETS = pytest.mark.parametrize(
    'preset', [RESAMPLE_FILTER, FILTER_ONLY], ids=['resample_filter', 'filter_only']
)
RESAMPLED_PRESETS = pytest.mark.parametrize(
    'preset', [RESAMPLE_FILTER, RESAMPLE_ONLY], ids=['resample_filter', 'resample_only']
)


def _resample_tol(preset, raw, filtered=False) -> dict:
    """return the `assert_close` keywords for the correction of the acquisition `raw`,
    through the analysis filter when `filtered` is True, or an empty dict (exact
    equality) for a preset served at the firmware rate without resampling"""
    if not preset.get('host_resample', True):
        return {}
    # _build_iq wraps an array of any namespace in the numpy-backed SOURCE spec, so
    # the backend follows the array
    backend = 'cupy' if sw.is_cupy_array(raw.x) else 'numpy'
    sigma = correction_error(
        raw.capture, raw.source_spec, array_backend=backend, filtered=filtered
    )
    return {'sigma': sigma}


def _build_iq(binding, capture, overlaps, xp=np):
    """return the `AcquiredIQ` of `binding` for `capture` and `overlaps` without a
    controller: the generator at `fs_sdr` from index ``-overlaps[0]``"""
    x = expected_raw(binding, capture, overlaps=overlaps, xp=xp)
    return build_acquired_iq(x, capture)


def _build_iq_for_trigger(binding, capture, trigger, xp):
    """return `_build_iq` with the tail overlap sized for `trigger`.

    `Controller.acquire` sizes the overlap from the trigger named in the analysis
    spec. A stand-in trigger has no name there, and the sizing is repeated here.
    """
    lag_pad = corrections._get_max_trigger_lag(FUNCTION_SOURCE, capture, trigger)
    overlaps = corrections._get_resampler_overlaps(
        capture, FUNCTION_SOURCE, min_overlap=lag_pad
    )
    return _build_iq(binding, capture, overlaps, xp=xp)


def _impulse_indexes(x):
    return list(np.argmax(np.abs(to_numpy(x)), axis=1))


def assert_impulse_at(x, capture, time):
    """assert that every port of `x` peaks on the output sample of `time` (s)"""
    index = round(time * capture.sample_rate)
    assert _impulse_indexes(x) == [index] * to_numpy(x).shape[0]


# %% select: time origin


@pytest.mark.parametrize(
    'preset', [SCALE_ONLY, FILTER_ONLY], ids=['scale_only', 'filter_only']
)
def test_output_is_the_generator_from_index_zero(preset, array_backend):
    """a sawtooth exposes a trim that is off by any number of samples, unlike a tone.
    The discontinuities of a ramp ring through a brick-wall resampler, and the
    resample presets are left to the impulse test."""
    capture = preset_capture('sawtooth', **preset, period=1e-4)
    stages = acquire_corrected('sawtooth', capture, array_backend=array_backend)
    expected = expected_corrected('sawtooth', capture)

    assert expected.shape[1] == round(capture.duration * capture.sample_rate)
    assert np.array_equal(to_numpy(unfiltered(stages.corrected)), expected)


@PRESET_PARAMS
def test_tone_phase_matches_the_generator(preset, array_backend, resample_path):
    capture = preset_capture(
        'single_tone', **preset, frequency_offset=TONE_FREQUENCY, snr=None
    )
    stages = acquire_corrected('single_tone', capture, array_backend=array_backend)
    expected = expected_corrected('single_tone', capture)

    assert_close(
        unfiltered(stages.corrected), expected, **_resample_tol(preset, stages.raw)
    )


# %% select: level


@PRESET_PARAMS
def test_impulse_level_through_the_stages(
    preset, array_backend, subtests, resample_path
):
    capture = preset_capture(
        'dirac_delta', **preset, time=IMPULSE_TIME, power=IMPULSE_POWER_DB
    )
    stages = acquire_corrected('dirac_delta', capture, array_backend=array_backend)
    fs = capture.sample_rate
    fs_sdr = stages.resampler['fs_sdr']
    index = round(IMPULSE_TIME * fs)
    amplitude = 10 ** (IMPULSE_POWER_DB / 20)
    impulse = expected_corrected('dirac_delta', capture)

    with subtests.test(stage='unfiltered'):
        x_unfiltered = unfiltered(stages.corrected)
        if resample_path == 'oaresample' and preset is RESAMPLE_FILTER:
            # the low-pass runs on the STFT bins inside the resampler, so the
            # unfiltered stage of a filtered, resampled capture is the filtered one
            assert np.array_equal(
                to_numpy(x_unfiltered), to_numpy(filtered(stages.corrected))
            )
        else:
            # a brick-wall resample keeps the N_out central bins of the flat
            # spectrum and scales by N_out/N_in, so the on-sample impulse becomes
            # fs/fs_sdr; the inverse FFT of a flat spectrum concentrates roundoff on
            # the impulse's on-peak sample the way a tone's FFT does in its on-peak
            # bin
            ratio = fs / fs_sdr
            tol = _resample_tol(preset, stages.raw)
            if tol:
                tol['atol'] = on_peak_roundoff(np.complex64) * amplitude * ratio
            assert_close(x_unfiltered, ratio * impulse, **tol)

    with subtests.test(stage='filtered'):
        x_filtered = to_numpy(filtered(stages.corrected))
        assert_impulse_at(x_filtered, capture, IMPULSE_TIME)
        if isfinite(capture.analysis_bandwidth):
            # the center tap of a unit-DC-gain windowed-sinc lowpass at cutoff bw/2
            # is bw/fs, up to the passband error that the DC normalization moves
            peak = amplitude * capture.analysis_bandwidth / fs_sdr
            assert_close(np.abs(x_filtered[:, index]), peak, rtol=FIR_LEAKAGE)
        else:
            assert np.array_equal(x_filtered, to_numpy(unfiltered(stages.corrected)))


@CHEAP_PRESETS
@pytest.mark.parametrize(
    'voltage_scale',
    [0.5, np.array([0.5, 2.0], dtype='float32')],
    ids=['scalar', 'per_port'],
)
def test_voltage_scale(preset, voltage_scale, array_backend):
    """the scales are powers of 2, the scaling is exact, and only the resample rounds"""
    capture = preset_capture(
        'single_tone', **preset, frequency_offset=TONE_FREQUENCY, snr=None
    )
    stages = acquire_corrected('single_tone', capture, array_backend=array_backend)
    xp = sw.array_namespace(stages.raw.x)
    scale = voltage_scale if np.isscalar(voltage_scale) else xp.asarray(voltage_scale)

    corrected = attach_corrections(dataclasses.replace(stages.raw, voltage_scale=scale))

    expected = np.reshape(voltage_scale, (-1, 1)) * expected_corrected(
        'single_tone', capture
    )
    assert_close(unfiltered(corrected), expected, **_resample_tol(preset, stages.raw))


# %% select: conjugation


@CHEAP_PRESETS
def test_conjugate_flips_only_the_flagged_port(preset, array_backend):
    capture = preset_capture(
        'single_tone', **preset, frequency_offset=TONE_FREQUENCY, snr=None
    )
    stages = acquire_corrected('single_tone', capture, array_backend=array_backend)

    corrected = attach_corrections(
        dataclasses.replace(stages.raw, conjugate=(True, False))
    )

    expected = expected_corrected('single_tone', capture)
    expected[0] = np.conj(expected[0])
    assert_close(unfiltered(corrected), expected, **_resample_tol(preset, stages.raw))


def test_ignore_highside_lo_bypasses_the_conjugate(corrections_flags):
    corrections_flags(ignore_highside_lo=True)
    capture = preset_capture(
        'single_tone', **SCALE_ONLY, frequency_offset=TONE_FREQUENCY, snr=None
    )
    stages = acquire_corrected('single_tone', capture)

    corrected = attach_corrections(
        dataclasses.replace(stages.raw, conjugate=(True, True))
    )

    expected = expected_corrected('single_tone', capture)
    assert np.array_equal(unfiltered(corrected), expected)


def test_overwrite_x_false_leaves_the_acquisition_intact(array_backend):
    """the scale-only path returns the acquisition buffer when `voltage_scale` is 1.
    An in-place conjugate there would corrupt a reused acquisition."""
    capture = preset_capture(
        'single_tone', **SCALE_ONLY, frequency_offset=TONE_FREQUENCY, snr=None
    )
    stages = acquire_corrected('single_tone', capture, array_backend=array_backend)
    iq = dataclasses.replace(stages.raw, conjugate=(True, False))
    before = to_numpy(iq.x).copy()

    unfiltered(attach_corrections(iq, overwrite_x=False))

    assert np.array_equal(to_numpy(iq.x), before)


# %% select: analysis filter


@FILTERED_PRESETS
@pytest.mark.parametrize(
    'frequency, passes',
    [(TONE_FREQUENCY, True), (OUT_OF_BAND_FREQUENCY, False)],
    ids=['in_band', 'out_of_band'],
)
def test_analysis_filter_passband(
    preset, frequency, passes, array_backend, resample_path
):
    """the filtered waveform is the resampled tone, with no delay, inside
    `analysis_bandwidth`, and rejects the tone past the transition band on both paths"""
    capture = preset_capture(
        'single_tone', **preset, frequency_offset=frequency, snr=None
    )
    stages = acquire_corrected('single_tone', capture, array_backend=array_backend)
    expected = to_numpy(expected_corrected('single_tone', capture))
    x_filtered = to_numpy(filtered(stages.corrected))

    if passes:
        assert_close(x_filtered, expected, atol=FIR_LEAKAGE)
    else:
        assert rms(x_filtered) < FIR_LEAKAGE * rms(expected)


# %% select: laziness and memoization


def _count_calls(monkeypatch, module, name):
    """replace `module.name` with a counting wrapper.

    Returns:
        the list that receives one entry per call of the wrapper
    """
    calls = []
    original = getattr(module, name)

    def counting(*args, **kwargs):
        calls.append(None)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, name, counting)
    return calls


def test_the_analysis_filter_runs_only_for_a_filtered_selection(monkeypatch):
    capture = preset_capture(
        'single_tone', **FILTER_ONLY, frequency_offset=TONE_FREQUENCY, snr=None
    )
    iq = acquire_corrected('single_tone', capture).corrected
    calls = _count_calls(monkeypatch, sw, 'oaconvolve')

    unfiltered(iq)
    unfiltered(iq)
    assert calls == []

    filtered(iq)
    filtered(iq)
    assert len(calls) == 1


def test_select_returns_one_object_per_effective_conditioning(subtests):
    capture = preset_capture(
        'single_tone', **FILTER_ONLY, frequency_offset=TONE_FREQUENCY, snr=None
    )
    iq = acquire_corrected('single_tone', capture).corrected

    with subtests.test('a repeated selection is the memo entry'):
        assert filtered(iq) is filtered(iq)
        assert unfiltered(iq) is unfiltered(iq)
    with subtests.test('filtered and unfiltered are distinct'):
        assert filtered(iq) is not unfiltered(iq)
    with subtests.test('without a trigger, aligned collapses onto unaligned'):
        assert aligned(iq) is filtered(iq)


def test_an_infinite_bandwidth_collapses_filtered_onto_unfiltered():
    capture = preset_capture(
        'single_tone', **SCALE_ONLY, frequency_offset=TONE_FREQUENCY, snr=None
    )
    iq = acquire_corrected('single_tone', capture).corrected
    assert filtered(iq) is unfiltered(iq)


def test_the_oaresample_path_collapses_filtered_onto_unfiltered(corrections_flags):
    """oaresample applies the low-pass inside the resampler, and no FIR stage runs"""
    corrections_flags(use_oaresample=True)
    capture = preset_capture('dirac_delta', **RESAMPLE_FILTER, time=IMPULSE_TIME)
    iq = acquire_corrected('dirac_delta', capture).corrected
    assert filtered(iq) is unfiltered(iq)


@pytest.mark.parametrize('overwrite_x', [False, True], ids=['copy', 'in_place'])
def test_overwrite_x_decides_whether_the_scaling_consumes_x(overwrite_x):
    capture = preset_capture(
        'single_tone', **SCALE_ONLY, frequency_offset=TONE_FREQUENCY, snr=None
    )
    raw = dataclasses.replace(
        acquire_corrected('single_tone', capture).raw, voltage_scale=0.5
    )
    before = raw.x.copy()
    iq = attach_corrections(raw, overwrite_x=overwrite_x)

    out = unfiltered(iq)

    assert np.shares_memory(out, iq.x) == overwrite_x
    assert np.array_equal(iq.x, before) != overwrite_x


def test_replace_starts_with_empty_stages(subtests):
    capture = preset_capture(
        'single_tone', **SCALE_ONLY, frequency_offset=TONE_FREQUENCY, snr=None
    )
    iq = acquire_corrected('single_tone', capture).corrected
    unfiltered(iq)
    assert iq.preprocess.outputs

    with subtests.test('repeated access is one preprocessor'):
        assert iq.preprocess is iq.preprocess
        assert isinstance(iq.preprocess, corrections.Preprocessor)
    with subtests.test('a replace copy starts a preprocessor of its own'):
        copy = dataclasses.replace(iq)
        assert copy.preprocess is not iq.preprocess
        assert not copy.preprocess.outputs
        assert copy.preprocess.iq is copy
    with subtests.test('attach_corrections starts a preprocessor of its own'):
        attached = attach_corrections(iq)
        assert attached.preprocess is not iq.preprocess
        assert not attached.preprocess.outputs
    with pytest.raises(TypeError, match='preprocess'):
        dataclasses.replace(iq, preprocess=iq.preprocess)


def test_a_preprocessor_outlives_its_acquisition_only_as_an_error():
    capture = preset_capture(
        'single_tone', **SCALE_ONLY, frequency_offset=TONE_FREQUENCY, snr=None
    )
    iq = acquire_corrected('single_tone', capture).corrected
    pp = iq.preprocess
    unfiltered(iq)
    del iq
    gc.collect()

    with pytest.raises(RuntimeError, match='released'):
        pp(prefer_filtered=False)


class _AlignedInputTrigger(FakeTrigger):
    """a trigger that selects the aligned waveform while computing the alignment"""

    def __call__(
        self, iq: sw.typing.Array | sa.AcquiredIQ, capture: sa.specs.Capture
    ) -> sw.typing.Array:
        assert isinstance(iq, sa.AcquiredIQ)
        iq.preprocess(prefer_aligned=True)
        return super().__call__(iq, capture)


def test_a_trigger_that_selects_aligned_input_is_rejected(xp):
    capture = preset_capture('dirac_delta', **SCALE_ONLY, time=IMPULSE_TIME)
    trigger = _AlignedInputTrigger(TRIGGER_LAGS)
    raw = _build_iq_for_trigger('dirac_delta', capture, trigger, xp)
    iq = attach_corrections(raw, signal_trigger=trigger)

    with pytest.raises(RuntimeError, match='prefer_aligned'):
        aligned(iq)


def test_attach_corrections_falls_back_to_the_trigger_in_info(xp):
    capture = preset_capture('dirac_delta', **SCALE_ONLY, time=IMPULSE_TIME)
    trigger = FakeTrigger(TRIGGER_LAGS)
    raw = _build_iq_for_trigger('dirac_delta', capture, trigger, xp)
    raw.info = raw.info.replace(signal_trigger=trigger)

    implicit = aligned(attach_corrections(raw))
    explicit = aligned(attach_corrections(raw, signal_trigger=trigger))

    assert implicit is not filtered(attach_corrections(raw))
    assert np.array_equal(to_numpy(implicit), to_numpy(explicit))


def test_a_pss_trigger_with_frequency_domain_measurements_skips_the_fir():
    """the sync detectors and the spectral measurements select unfiltered input. A
    triggered capture with only those enabled never runs the analysis FIR."""
    capture = preset_capture(
        'single_tone',
        **{**FILTER_ONLY, 'duration': 10e-3, 'port': 0},
        frequency_offset=TONE_FREQUENCY,
        snr=None,
    )
    analysis = sa.specs.BuiltinAnalyses.from_dict({
        'cellular_5g_pss_sync': {'subcarrier_spacing': 30e3, 'shared_spectrum': True},
        'power_spectral_density': {'window': 'hann', 'frequency_resolution': 8e3},
    })
    source = FUNCTION_SOURCE.replace(signal_trigger='cellular_5g_pss_sync')
    raw = acquire_corrected(
        'single_tone', capture, source=source, analysis=analysis
    ).raw
    trigger = get_trigger_from_spec(source, analysis)
    iq = attach_corrections(raw, signal_trigger=trigger)
    options = sa.EvaluationOptions(as_xarray='delayed', registry=sa.registry)

    with sa.registry.cache_context(capture):
        results = sa.analyze_by_spec(iq, analysis, capture, options)

    assert set(results) == {'cellular_5g_pss_sync', 'power_spectral_density'}
    assert iq.preprocess.shifts is not None
    assert iq.preprocess.filtered is None


# %% select: trigger alignment


@CHEAP_PRESETS
def test_trigger_shifts_aligned_and_leaves_unaligned(preset, xp, subtests):
    capture = preset_capture('dirac_delta', **preset, time=IMPULSE_TIME)
    trigger = FakeTrigger(TRIGGER_LAGS)
    raw = _build_iq_for_trigger('dirac_delta', capture, trigger, xp)
    fs = capture.sample_rate
    size_out = round(capture.duration * fs)
    tol = _resample_tol(preset, raw, filtered=True)
    ratio = fs / raw.resampler['fs_sdr']
    if tol:
        # the resample concentrates its roundoff on the on-peak sample of the impulse,
        # which the rms model of _resample_tol instead spreads over the whole window
        # (as in test_impulse_level_through_the_stages)
        tol['atol'] = on_peak_roundoff(np.complex64) * ratio

    corrected = attach_corrections(raw, signal_trigger=trigger)

    with subtests.test(stage='filtered'):
        assert_impulse_at(filtered(corrected), capture, IMPULSE_TIME)

    for port, lag in enumerate(TRIGGER_LAGS):
        shift = round(lag * fs)
        with subtests.test(stage='aligned', port=port):
            expected = generator(
                'dirac_delta',
                capture,
                sample_rate=fs,
                start_index=shift,
                count=size_out,
                ports=1,
            )
            assert aligned(corrected).shape[1] == size_out
            assert_close(aligned(corrected)[port : port + 1], ratio * expected, **tol)


# %% select: experimental oaresample path (STRIQT_USE_OARESAMPLE=1)


@RESAMPLED_PRESETS
def test_oaresample_acquisition(preset, corrections_flags):
    corrections_flags(use_oaresample=True)
    capture = preset_capture('dirac_delta', **preset, time=IMPULSE_TIME)

    stages = acquire_corrected('dirac_delta', capture)

    assert_impulse_at(filtered(stages.corrected), capture, IMPULSE_TIME)


def _stopband_to_passband_power(x, capture):
    """return the ratio of the mean PSD past the transition band to the mean PSD inside
    the passband of the analysis filter, from a windowed periodogram of `x`"""
    seg = 4096
    frames = x[0, : (x.shape[1] // seg) * seg].reshape(-1, seg) * np.hanning(seg)
    psd = (np.abs(np.fft.fft(frames, axis=1)) ** 2).mean(0)
    freqs = np.fft.fftfreq(seg, 1 / capture.sample_rate)
    edge = capture.analysis_bandwidth / 2
    passband = np.abs(freqs) < 0.8 * edge
    stopband = np.abs(freqs) > edge + corrections.FIR_TRANSITION_BW
    return psd[stopband].mean() / psd[passband].mean()


def test_oaresample_leaks_in_band_noise_no_more_than_the_fir(corrections_flags):
    """white noise through the resampled, filtered preset: the power that the
    oaresample path leaves past the transition band is within the suite's FIR leakage
    bound of what the time-domain FIR leaves there"""
    capture = preset_capture('noise', **RESAMPLE_FILTER, noise_psd=1e-12)
    ratios = {}
    for path in ('resample', 'oaresample'):
        corrections_flags(use_oaresample=path == 'oaresample')
        stages = acquire_corrected('noise', capture)
        x = to_numpy(filtered(stages.corrected))
        ratios[path] = _stopband_to_passband_power(x, capture)
    assert ratios['resample'] < FIR_LEAKAGE**2
    assert ratios['oaresample'] <= ratios['resample'] + FIR_LEAKAGE**2


# %% select: LO shift


@pytest.mark.parametrize('lo_shift', ['left', 'right'])
def test_lo_shift_design_moves_the_lo_out_of_band(lo_shift):
    """the LO leakage is at 0 Hz in the baseband of the radio. The shifted passband
    excludes 0 Hz and fits inside the Nyquist band of the radio."""
    capture = preset_capture('single_tone', **LO_SHIFT_CAPTURE, lo_shift=lo_shift)
    design = corrections.design_resampler(capture, FUNCTION_SOURCE.master_clock_rate)
    half_bw = capture.analysis_bandwidth / 2

    assert np.sign(design['lo_offset']) == (1 if lo_shift == 'right' else -1)
    assert abs(design['lo_offset']) > half_bw
    assert abs(design['lo_offset']) + half_bw <= design['fs_sdr'] / 2


@pytest.mark.parametrize(
    'lo_shift, frequency', [('left', 1e6), ('right', -1e6)], ids=['left', 'right']
)
def test_lo_shift_is_removed_from_the_output(lo_shift, frequency):
    capture = preset_capture(
        'single_tone',
        **LO_SHIFT_CAPTURE,
        lo_shift=lo_shift,
        frequency_offset=frequency,
        snr=None,
    )
    stages = acquire_corrected('single_tone', capture)
    x = filtered(stages.corrected)
    fs = capture.sample_rate

    assert abs(tone_frequency(x[0], fs) - frequency) < fs / x.shape[1]


# %% needs_resample, design_resampler


@pytest.mark.parametrize(
    'design, host_resample, expected',
    [
        ({'nfft': 300, 'nfft_out': 288}, True, True),
        ({'nfft': 8192, 'nfft_out': 8192}, True, False),
        ({'nfft': 300, 'nfft_out': 288}, False, False),
    ],
    ids=['resamples', 'unity_ratio', 'firmware_rate'],
)
def test_needs_resample(design, host_resample, expected):
    capture = preset_capture('single_tone', host_resample=host_resample)
    assert corrections.needs_resample(design, capture) is expected


def test_design_without_host_resample_is_the_identity():
    capture = preset_capture('single_tone', **SCALE_ONLY)
    design = corrections.design_resampler(capture, FUNCTION_SOURCE.master_clock_rate)

    assert design['fs_sdr'] == capture.sample_rate
    assert design['nfft'] == design['nfft_out']
    assert design['lo_offset'] == 0


@pytest.mark.parametrize(
    'kws, match',
    [
        ({'analysis_bandwidth': 7e6}, 'analysis bandwidth must be smaller'),
        ({**SCALE_ONLY, 'lo_shift': 'left'}, 'lo_shift requires host_resample'),
        ({**SCALE_ONLY, 'sample_rate': 250e6}, 'upsampling requires host_resample'),
    ],
    ids=[
        'bandwidth_above_sample_rate',
        'lo_shift_without_host_resample',
        'upsample_without_host_resample',
    ],
)
def test_design_resampler_rejects(kws, match):
    capture = preset_capture('single_tone', **kws)
    with pytest.raises(ValueError, match=match):
        corrections.design_resampler(capture, FUNCTION_SOURCE.master_clock_rate)


def test_design_resampler_needs_a_clock_rate():
    capture = preset_capture('single_tone', **SCALE_ONLY)
    with pytest.raises(TypeError, match='master_clock_rate'):
        corrections.design_resampler(capture, None)


# %% validate_fir_band

FIR_TRANSITION_BW = corrections.FIR_TRANSITION_BW


@pytest.mark.parametrize(
    'analysis_bandwidth, match',
    [(0.9e6, 'leaves no room'), (200e3, 'leaves no passband')],
    ids=['within_the_transition_of_nyquist', 'below_the_transition_band'],
)
def test_fir_band_edges_are_rejected(analysis_bandwidth, match):
    capture = preset_capture(
        'single_tone', sample_rate=1e6, analysis_bandwidth=analysis_bandwidth
    )
    with pytest.raises(ValueError, match=match):
        corrections.validate_fir_band(capture)


@pytest.mark.parametrize('offset', [-1e3, 0.0], ids=['inside', 'on_the_edge'])
@pytest.mark.parametrize('edge', ['nyquist', 'zero'])
def test_fir_band_edges_agree_with_the_filter_design(edge, offset):
    """the rejection is exactly where `scipy.signal.firls` stops designing the filter:
    a band of zero width, at either end, is already too narrow"""
    fs = 1e6
    if edge == 'nyquist':
        bw = fs - FIR_TRANSITION_BW + offset
    else:
        bw = FIR_TRANSITION_BW - offset
    capture = preset_capture('single_tone', sample_rate=fs, analysis_bandwidth=bw)

    try:
        sw.design_fir_lpf(bw=bw, fs=fs, transition_bw=FIR_TRANSITION_BW)
    except ValueError:
        designs = False
    else:
        designs = True

    try:
        corrections.validate_fir_band(capture)
    except ValueError:
        validates = False
    else:
        validates = True

    assert validates is designs is (offset != 0)


# %% validate_oaresample_shift

# a 10 kHz resampler bin: 1000 bins in, the middle 500 out, so the output band spans
# bins [250, 750) and can shift by at most 250 bins (2.5 MHz) either way
SHIFT_DESIGN = {'fs_sdr': 10e6, 'nfft': 1000, 'nfft_out': 500}


@pytest.mark.parametrize(
    'lo_offset', [0.0, 2.5e6, -2.5e6], ids=['none', 'high_edge', 'low_edge']
)
def test_oaresample_shift_within_the_source_band_is_accepted(lo_offset):
    design = {**SHIFT_DESIGN, 'lo_offset': lo_offset}
    assert corrections.validate_oaresample_shift(design) is None


@pytest.mark.parametrize(
    'design, match',
    [
        ({**SHIFT_DESIGN, 'nfft_out': 2000, 'lo_offset': 1e6}, 'only supported'),
        ({**SHIFT_DESIGN, 'lo_offset': 1.5e4}, 'not a multiple'),
        ({**SHIFT_DESIGN, 'lo_offset': 2.51e6}, 'outside the source bandwidth'),
        ({**SHIFT_DESIGN, 'lo_offset': -2.51e6}, 'outside the source bandwidth'),
    ],
    ids=['upsampling', 'off_the_bin_grid', 'above_the_band', 'below_the_band'],
)
def test_oaresample_shift_rejects(design, match):
    with pytest.raises(ValueError, match=match):
        corrections.validate_oaresample_shift(design)


# %% get_correction_overlaps: properties over the capture domain

sample_rates = st.integers(min_value=1000, max_value=20000).map(lambda k: k * 1e3)
sample_counts = st.integers(min_value=100, max_value=40000)
bandwidth_fractions = st.one_of(st.just(inf), st.floats(min_value=0.3, max_value=0.95))
max_lags = st.floats(min_value=0, max_value=1e-4)

# 10 ms SoapySDR records at the 125 MHz clock rate, at its divisors 62.5, 10, and
# 1 MS/s, and through the 15.625 -> 15.36 MS/s resampler, with and without the
# analysis filter
SOAPY_RECORDS = (
    {'sample_rate': 125e6, 'count': 1_250_000, 'bandwidth_fraction': inf},
    {'sample_rate': 125e6, 'count': 1_250_000, 'bandwidth_fraction': 0.32},
    {'sample_rate': 62.5e6, 'count': 625_000, 'bandwidth_fraction': inf},
    {'sample_rate': 15.36e6, 'count': 153_600, 'bandwidth_fraction': 10e6 / 15.36e6},
    {'sample_rate': 1e6, 'count': 10_000, 'bandwidth_fraction': 0.5},
    {'sample_rate': 10e6, 'count': 100_000, 'bandwidth_fraction': 0.8},
)

F = TypeVar('F', bound=Callable[..., None])


def soapy_record_examples(**extra: float) -> Callable[[F], F]:
    """return a decorator that stacks an `example` of each `SOAPY_RECORDS` point, plus
    `extra`, on a property test"""

    def decorate(test: F) -> F:
        for point in SOAPY_RECORDS:
            test = example(**point, **extra)(test)
        return test

    return decorate


def _drawn_capture(sample_rate, count, bandwidth_fraction):
    if bandwidth_fraction == inf:
        bw = inf
    else:
        bw = round(bandwidth_fraction * sample_rate / 1e3) * 1e3
    return preset_capture(
        'single_tone',
        sample_rate=sample_rate,
        duration=count / sample_rate,
        analysis_bandwidth=bw,
    )


@given(
    sample_rate=sample_rates,
    count=sample_counts,
    bandwidth_fraction=bandwidth_fractions,
    max_lag=max_lags,
)
# the FIR pad is 2000 output samples, which the 12.5 MS/s -> 12.065 MS/s resample
# scales to 2000 * 12.5 / 12.065 source samples
@example(sample_rate=12065e3, count=24977, bandwidth_fraction=0.5, max_lag=0.0)
@soapy_record_examples(max_lag=0.0)
def test_overlaps_cover_the_filter_and_trigger_pads(
    sample_rate, count, bandwidth_fraction, max_lag, resample_path
):
    capture = _drawn_capture(sample_rate, count, bandwidth_fraction)
    trigger = FakeTrigger(0.0, max_lag=max_lag)
    lag_pad = corrections._get_max_trigger_lag(FUNCTION_SOURCE, capture, trigger)
    lead, tail = corrections._get_resampler_overlaps(
        capture, FUNCTION_SOURCE, min_overlap=lag_pad
    )
    fs = fs_sdr(capture)
    design = corrections.design_resampler(capture, FUNCTION_SOURCE.master_clock_rate)

    assert lead >= 0 and tail >= 0
    if resample_path == 'oaresample' and corrections.needs_resample(design, capture):
        # the STFT low-pass has no transient beyond a block; each pad is one COLA
        # overlap of the resampler, and the tail adds the trigger lag
        overlap_scale = sw.fourier.design_oafilter(
            design['nfft'],
            window=design['window'],
            nfft_out=design['nfft_out'],
            nfft=design['nfft'],
            extend=True,
        )[2]
        noverlap = round(design['nfft'] * overlap_scale)
        assert lead >= noverlap
        assert tail >= noverlap + lag_pad
    else:
        # the FIR transient spans FILTER_SIZE//2 samples at the output rate, where
        # the analysis filter stage applies the filter, at each end of the record
        filter_out = FILTER_SIZE // 2 if isfinite(capture.analysis_bandwidth) else 0
        assert lead * capture.sample_rate / fs >= filter_out
        # a trigger shift of up to max_lag then reads that far past the transient
        assert (
            tail * capture.sample_rate / fs
            >= filter_out + max_lag * capture.sample_rate
        )
    # the sum of the two is an overlap that read_iq accepts, and it asks the source
    # for exactly the padded resampler input
    read_count = buffers.get_read_count(capture, FUNCTION_SOURCE, overlap=lead + tail)
    assert read_count == source_count(capture, FUNCTION_SOURCE) + lead + tail


@settings(report_multiple_bugs=False, max_examples=50)
@given(
    sample_rate=sample_rates,
    count=sample_counts,
    bandwidth_fraction=bandwidth_fractions,
)
@example(sample_rate=1e7, count=23076, bandwidth_fraction=inf)
@soapy_record_examples()
def test_overlaps_pad_to_a_whole_resampler_block(
    sample_rate, count, bandwidth_fraction
):
    """the lead and tail overlaps need not be even or equal. Together the overlaps pad
    the source-rate record to a length that resamples to a whole number of output
    samples, as `_resample` requires. `read_iq` reads exactly that padded record. The
    example is a 25:24 ratio with a record length of an exact half sample."""
    capture = _drawn_capture(sample_rate, count, bandwidth_fraction)
    lead, tail = corrections._get_resampler_overlaps(capture, FUNCTION_SOURCE)
    fs = fs_sdr(capture)
    padded = source_count(capture, FUNCTION_SOURCE) + lead + tail

    assert lead >= 0 and tail >= 0
    assert sw.isroundmod(padded * capture.sample_rate, fs)
    assert (
        buffers.get_read_count(capture, FUNCTION_SOURCE, overlap=lead + tail) == padded
    )


DOCUMENTED_OVERLAPS = {
    'resample': {
        'resample_filter': (3125, 3125),
        'resample_only': (50, 50),
        'scale_only': (512, 512),
        'filter_only': (4608, 4608),
    },
    # the oaresample path sizes the blocks of the resampled presets from
    # MIN_OARESAMPLE_FFT_SIZE with the blackman window; a preset served at the
    # firmware rate pads for the FIR as on the default path
    'oaresample': {
        'resample_filter': (12500, 12500),
        'resample_only': (12000, 17500),
        'scale_only': (512, 512),
        'filter_only': (4608, 4608),
    },
}


@pytest.mark.parametrize('name', list(PRESETS))
def test_preset_overlaps_are_acquirable(name, resample_path):
    """the preset overlaps equal the documented values of each path (the default
    path's are in `synthetic_sources`). A change in the overlap sizing then fails here
    rather than as unrelated acquisition failures."""
    capture = preset_capture('single_tone', **PRESETS[name])
    overlaps = corrections.get_correction_overlaps(capture, FUNCTION_SOURCE)
    assert overlaps == DOCUMENTED_OVERLAPS[resample_path][name]


# %% acquisitions with odd or non-integral source-rate pads

# 6 MS/s for 1 ms without an analysis filter pads to (175, 175): the 25:24 resampler
# makes the lead a multiple of 25, and 7 of them is odd
ODD_OVERLAP_CAPTURE = {**RESAMPLE_ONLY, 'duration': 1e-3}
# 4 MS/s comes from 125 MHz / 31, so 1 ms is 4032.26 source samples
NONINTEGRAL_CAPTURE = {**RESAMPLE_ONLY, 'sample_rate': 4e6, 'duration': 1e-3}


@pytest.mark.parametrize(
    'preset',
    [
        pytest.param(ODD_OVERLAP_CAPTURE, id='odd_overlap'),
        pytest.param(NONINTEGRAL_CAPTURE, id='nonintegral_source_duration'),
    ],
)
def test_impulse_acquisition(preset):
    capture = preset_capture('dirac_delta', **preset, time=IMPULSE_TIME)
    stages = acquire_corrected('dirac_delta', capture)
    size_out = round(capture.duration * capture.sample_rate)
    assert filtered(stages.corrected).shape[1] == size_out
    assert_impulse_at(filtered(stages.corrected), capture, IMPULSE_TIME)


# %% source_record_size


def test_source_record_size_rounds_an_exact_half_sample_up():
    """23076 samples at 10 MS/s from the 125 MS/s clock resample 25:24, and the source
    record is exactly 24037.5 samples. The float product ``duration * fs_sdr`` falls
    below the half and would round down."""
    capture = _drawn_capture(1e7, 23076, inf)
    design = corrections.design_resampler(capture, FUNCTION_SOURCE.master_clock_rate)
    assert (design['nfft'], design['nfft_out']) == (300, 288)
    assert corrections.source_record_size(capture, design) == 24038
    assert round(capture.duration * design['fs_sdr']) == 24037


def test_oaresample_pads_use_the_exact_record_size(corrections_flags):
    """7680 samples at 7.68 MS/s from the 125 MS/s clock are a 7812.5-sample source
    record. `source_record_size` rounds the record up to 7813 where the float product
    rounds down to 7812. The oaresample pads bring the 7813-sample record to a whole
    number of STFT hops. Pads sized for 7812 samples slip the output by one source
    sample."""
    corrections_flags(use_oaresample=True)
    capture = preset_capture(
        'dirac_delta',
        port=(0, 1),
        sample_rate=7.68e6,
        duration=7680 / 7.68e6,
        analysis_bandwidth=inf,
        time=IMPULSE_TIME,
    )
    design = corrections.design_resampler(capture, FUNCTION_SOURCE.master_clock_rate)
    record = corrections.source_record_size(capture, design)
    assert record == 7813
    assert round(7680 * design['nfft'] / design['nfft_out']) == 7812

    stages = acquire_corrected('dirac_delta', capture)

    assert_impulse_at(filtered(stages.corrected), capture, IMPULSE_TIME)


# %% _resample


def test_resample_rejects_a_length_that_does_not_divide():
    """6.25 MS/s to 6 MS/s is 25:24. One sample more than the (50, 50) overlaps
    leaves a padded length that is not a multiple of 25."""
    capture = preset_capture('dirac_delta', **RESAMPLE_ONLY, time=IMPULSE_TIME)
    lead, tail = corrections.get_correction_overlaps(capture, FUNCTION_SOURCE)
    fs = fs_sdr(capture)
    count = lead + 1 + source_count(capture, FUNCTION_SOURCE) + tail
    assert not sw.isroundmod(count * capture.sample_rate, fs)

    iq = _build_iq('dirac_delta', capture, (lead + 1, tail))

    with pytest.raises(ValueError, match=f'{count} samples at {fs} S/s'):
        unfiltered(attach_corrections(iq))
