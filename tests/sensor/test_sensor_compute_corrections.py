"""striqt.sensor.lib.compute.corrections: the per-capture signal path of correct_iq,
the resampler design, and the acquisition overlap that the resampler and analysis
filter consume
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from math import inf, isfinite
from typing import TypeVar

import numpy as np
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st
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
    build_acquired_iq,
    expected_corrected,
    expected_raw,
    fs_sdr,
    generator,
    preset_capture,
    source_count,
)

import striqt.sensor as ss
import striqt.waveform as sw
from striqt.sensor.lib.compute import correction_error, corrections
from striqt.sensor.lib.sources import buffers
from striqt.waveform.fourier import on_peak_roundoff

# 0.5 ms is a whole number of samples at every preset's source and capture rate, so
# the impulse sits on an output sample. The brick-wall resample of an impulse is
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


def _resample_tol(preset, raw, stage='pre_filter') -> dict:
    """assert_close keywords for the correction of the acquisition `raw` up to `stage`
    of the corrected AcquiredIQ, or exact equality for a preset that the firmware rate
    serves without resampling"""
    if not preset.get('host_resample', True):
        return {}
    # _build_iq wraps an array of any namespace in the numpy-backed SOURCE spec, so
    # the backend follows the array
    backend = 'cupy' if sw.is_cupy_array(raw.pre_align) else 'numpy'
    sigma = correction_error(
        raw.capture, raw.source_spec, array_backend=backend, stage=stage
    )
    return {'sigma': sigma}


def _build_iq(binding, capture, overlaps, xp=np):
    """the AcquiredIQ that `binding` produces for `capture` and `overlaps`, without a
    controller: the generator at fs_sdr from index -overlaps[0]"""
    x = expected_raw(binding, capture, overlaps=overlaps, xp=xp)
    return build_acquired_iq(x, capture)


def _build_iq_for_trigger(binding, capture, trigger, xp):
    """_build_iq with the tail overlap that `trigger` needs.

    Controller.acquire() sizes the overlap from the trigger named in the analysis
    spec, which a stand-in trigger cannot be, so the sizing is repeated here.
    """
    lag_pad = corrections._get_max_trigger_lag(FUNCTION_SOURCE, capture, trigger)
    overlaps = corrections._get_resampler_overlaps(
        capture, FUNCTION_SOURCE, min_overlap=lag_pad
    )
    return _build_iq(binding, capture, overlaps, xp=xp)


def _impulse_indexes(x):
    return list(np.argmax(np.abs(to_numpy(x)), axis=1))


def assert_impulse_at(x, capture, time):
    """every port of `x` peaks on the output sample of `time`"""
    index = round(time * capture.sample_rate)
    assert _impulse_indexes(x) == [index] * to_numpy(x).shape[0]


# %% correct_iq: time origin


@pytest.mark.parametrize(
    'preset', [SCALE_ONLY, FILTER_ONLY], ids=['scale_only', 'filter_only']
)
def test_output_is_the_generator_from_index_zero(preset, array_backend):
    """A sawtooth rather than a tone, so that a trim that is off by any number of
    samples shows. The resample presets are left to the impulse test, because a
    ramp's discontinuities ring through a brick-wall resampler."""
    capture = preset_capture('sawtooth', **preset, period=1e-4)
    stages = acquire_corrected('sawtooth', capture, array_backend=array_backend)
    expected = expected_corrected('sawtooth', capture)

    assert expected.shape[1] == round(capture.duration * capture.sample_rate)
    assert np.array_equal(to_numpy(stages.corrected.pre_filter), expected)


@PRESET_PARAMS
def test_tone_phase_matches_the_generator(preset, array_backend):
    capture = preset_capture(
        'single_tone', **preset, frequency_offset=TONE_FREQUENCY, snr=None
    )
    stages = acquire_corrected('single_tone', capture, array_backend=array_backend)
    expected = expected_corrected('single_tone', capture)

    assert_close(
        stages.corrected.pre_filter, expected, **_resample_tol(preset, stages.raw)
    )


# %% correct_iq: level


@PRESET_PARAMS
def test_impulse_level_through_the_stages(preset, array_backend, subtests):
    capture = preset_capture(
        'dirac_delta', **preset, time=IMPULSE_TIME, power=IMPULSE_POWER_DB
    )
    stages = acquire_corrected('dirac_delta', capture, array_backend=array_backend)
    fs = capture.sample_rate
    fs_sdr = stages.resampler['fs_sdr']
    index = round(IMPULSE_TIME * fs)
    amplitude = 10 ** (IMPULSE_POWER_DB / 20)
    impulse = expected_corrected('dirac_delta', capture)

    with subtests.test(stage='pre_filter'):
        # a brick-wall resample keeps the N_out central bins of the flat spectrum
        # and scales by N_out/N_in, so the on-sample impulse becomes fs/fs_sdr;
        # the inverse FFT of a flat spectrum concentrates roundoff on the impulse's
        # on-peak sample the way a tone's FFT does in its on-peak bin
        ratio = fs / fs_sdr
        tol = _resample_tol(preset, stages.raw)
        if tol:
            tol['atol'] = on_peak_roundoff(np.complex64) * amplitude * ratio
        assert_close(stages.corrected.pre_filter, ratio * impulse, **tol)

    with subtests.test(stage='pre_align'):
        pre_align = to_numpy(stages.corrected.pre_align)
        assert_impulse_at(pre_align, capture, IMPULSE_TIME)
        if isfinite(capture.analysis_bandwidth):
            # the center tap of a unit-DC-gain windowed-sinc lowpass at cutoff bw/2
            # is bw/fs, up to the passband error that the DC normalization moves
            peak = amplitude * capture.analysis_bandwidth / fs_sdr
            assert_close(np.abs(pre_align[:, index]), peak, rtol=FIR_LEAKAGE)
        else:
            assert np.array_equal(pre_align, to_numpy(stages.corrected.pre_filter))


@CHEAP_PRESETS
@pytest.mark.parametrize(
    'voltage_scale',
    [0.5, np.array([0.5, 2.0], dtype='float32')],
    ids=['scalar', 'per_port'],
)
def test_voltage_scale(preset, voltage_scale, array_backend):
    """powers of two, so the scaling itself is exact and only the resample rounds"""
    capture = preset_capture(
        'single_tone', **preset, frequency_offset=TONE_FREQUENCY, snr=None
    )
    stages = acquire_corrected('single_tone', capture, array_backend=array_backend)
    xp = sw.array_namespace(stages.raw.pre_align)
    scale = voltage_scale if np.isscalar(voltage_scale) else xp.asarray(voltage_scale)

    corrected = ss.correct_iq(dataclasses.replace(stages.raw, voltage_scale=scale))

    expected = np.reshape(voltage_scale, (-1, 1)) * expected_corrected(
        'single_tone', capture
    )
    assert_close(corrected.pre_filter, expected, **_resample_tol(preset, stages.raw))


# %% correct_iq: conjugation


@CHEAP_PRESETS
def test_conjugate_flips_only_the_flagged_port(preset, array_backend):
    capture = preset_capture(
        'single_tone', **preset, frequency_offset=TONE_FREQUENCY, snr=None
    )
    stages = acquire_corrected('single_tone', capture, array_backend=array_backend)

    corrected = ss.correct_iq(dataclasses.replace(stages.raw, conjugate=(True, False)))

    expected = expected_corrected('single_tone', capture)
    expected[0] = np.conj(expected[0])
    assert_close(corrected.pre_filter, expected, **_resample_tol(preset, stages.raw))


def test_ignore_highside_lo_bypasses_the_conjugate(corrections_flags):
    corrections_flags(ignore_highside_lo=True)
    capture = preset_capture(
        'single_tone', **SCALE_ONLY, frequency_offset=TONE_FREQUENCY, snr=None
    )
    stages = acquire_corrected('single_tone', capture)

    corrected = ss.correct_iq(dataclasses.replace(stages.raw, conjugate=(True, True)))

    expected = expected_corrected('single_tone', capture)
    assert np.array_equal(corrected.pre_filter, expected)


def test_overwrite_x_false_leaves_the_acquisition_intact(array_backend):
    """the scale-only path returns the acquisition buffer itself when voltage_scale
    is 1, so an in-place conjugate there would corrupt a reused acquisition"""
    capture = preset_capture(
        'single_tone', **SCALE_ONLY, frequency_offset=TONE_FREQUENCY, snr=None
    )
    stages = acquire_corrected('single_tone', capture, array_backend=array_backend)
    iq = dataclasses.replace(stages.raw, conjugate=(True, False))
    before = to_numpy(iq.pre_align).copy()

    ss.correct_iq(iq, overwrite_x=False)

    assert np.array_equal(to_numpy(iq.pre_align), before)


# %% correct_iq: analysis filter


@FILTERED_PRESETS
@pytest.mark.parametrize(
    'frequency, passes',
    [(TONE_FREQUENCY, True), (OUT_OF_BAND_FREQUENCY, False)],
    ids=['in_band', 'out_of_band'],
)
def test_analysis_filter_passband(preset, frequency, passes, array_backend):
    """pre_filter is the resampled tone either way; pre_align keeps it, with no
    delay, only inside analysis_bandwidth"""
    capture = preset_capture(
        'single_tone', **preset, frequency_offset=frequency, snr=None
    )
    stages = acquire_corrected('single_tone', capture, array_backend=array_backend)
    pre_filter = to_numpy(stages.corrected.pre_filter)
    pre_align = to_numpy(stages.corrected.pre_align)

    if passes:
        assert_close(pre_align, pre_filter, atol=FIR_LEAKAGE)
    else:
        assert rms(pre_align) < FIR_LEAKAGE * rms(pre_filter)


# %% correct_iq: trigger alignment


@CHEAP_PRESETS
def test_trigger_shifts_aligned_and_leaves_pre_align(preset, xp, subtests):
    capture = preset_capture('dirac_delta', **preset, time=IMPULSE_TIME)
    trigger = FakeTrigger(TRIGGER_LAGS)
    raw = _build_iq_for_trigger('dirac_delta', capture, trigger, xp)
    fs = capture.sample_rate
    size_out = round(capture.duration * fs)
    tol = _resample_tol(preset, raw, stage='pre_align')
    ratio = fs / raw.resampler['fs_sdr']
    if tol:
        # the resample concentrates its roundoff on the impulse's own on-peak sample,
        # which the rms model of _resample_tol instead spreads over the whole window
        # (as in test_impulse_level_through_the_stages)
        tol['atol'] = on_peak_roundoff(np.complex64) * ratio

    corrected = ss.correct_iq(raw, signal_trigger=trigger)

    with subtests.test(stage='pre_align'):
        assert_impulse_at(corrected.pre_align, capture, IMPULSE_TIME)

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
            assert corrected.aligned.shape[1] == size_out
            assert_close(corrected.aligned[port : port + 1], ratio * expected, **tol)


# %% correct_iq: experimental oaresample path (STRIQT_USE_OARESAMPLE=1)


@RESAMPLED_PRESETS
def test_oaresample_acquisition(preset, corrections_flags):
    corrections_flags(use_oaresample=True)
    capture = preset_capture('dirac_delta', **preset, time=IMPULSE_TIME)

    stages = acquire_corrected('dirac_delta', capture)

    assert_impulse_at(stages.corrected.pre_align, capture, IMPULSE_TIME)


# %% correct_iq: LO shift


@pytest.mark.parametrize('lo_shift', ['left', 'right'])
def test_lo_shift_design_moves_the_lo_out_of_band(lo_shift):
    """the LO leakage sits at 0 Hz in the radio's baseband, so the shifted passband
    must exclude it while still fitting the radio's Nyquist band"""
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
    x = stages.corrected.pre_align
    fs = capture.sample_rate

    assert abs(tone_frequency(x[0], fs) - frequency) < fs / x.shape[1]


# %% needs_resample, design_resampler


@pytest.mark.parametrize(
    'design, host_resample, expected',
    [
        ({'nfft': 550, 'nfft_out': 528}, True, True),
        ({'nfft': 8192, 'nfft_out': 8192}, True, False),
        ({'nfft': 550, 'nfft_out': 528}, False, False),
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
    """the rejection is exactly where scipy.signal.firls stops designing the filter:
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
    """stack an `example` of each SOAPY_RECORDS point, plus `extra`, on a property"""

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
    sample_rate, count, bandwidth_fraction, max_lag
):
    capture = _drawn_capture(sample_rate, count, bandwidth_fraction)
    trigger = FakeTrigger(0.0, max_lag=max_lag)
    lag_pad = corrections._get_max_trigger_lag(FUNCTION_SOURCE, capture, trigger)
    lead, tail = corrections._get_resampler_overlaps(
        capture, FUNCTION_SOURCE, min_overlap=lag_pad
    )
    fs = fs_sdr(capture)

    assert lead >= 0 and tail >= 0
    # the FIR transient spans FILTER_SIZE//2 samples at the output rate, where
    # correct_iq applies the filter, at each end of the record
    filter_out = FILTER_SIZE // 2 if isfinite(capture.analysis_bandwidth) else 0
    assert lead * capture.sample_rate / fs >= filter_out
    # a trigger shift of up to max_lag then reads that far past the transient
    assert tail * capture.sample_rate / fs >= filter_out + max_lag * capture.sample_rate
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
    """the overlaps need not be even or equal: together they pad the source-rate
    record to a length that resamples to a whole number of output samples, as
    _resample requires. read_iq reads exactly that padded record. The example is a
    25:24 ratio whose record length is an exact half sample"""
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
    'resample_filter': (3125, 3125),
    'resample_only': (350, 350),
    'scale_only': (512, 512),
    'filter_only': (4608, 4608),
}


@pytest.mark.parametrize('name', list(PRESETS))
def test_preset_overlaps_are_acquirable(name):
    """Pin the preset overlaps that synthetic_sources documents, so that a change in
    the overlap sizing shows up as a harness change rather than as unrelated
    acquisition failures."""
    capture = preset_capture('single_tone', **PRESETS[name])
    overlaps = corrections.get_correction_overlaps(capture, FUNCTION_SOURCE)
    assert overlaps == DOCUMENTED_OVERLAPS[name]


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
    assert stages.corrected.pre_align.shape[1] == size_out
    assert_impulse_at(stages.corrected.pre_align, capture, IMPULSE_TIME)


# %% source_record_size


def test_source_record_size_rounds_an_exact_half_sample_up():
    """23076 samples at 10 MS/s from the 125 MS/s clock resample 25:24, so the source
    record is 24037.5 samples exactly; the float product duration * fs_sdr falls
    below the half and would round down"""
    capture = _drawn_capture(1e7, 23076, inf)
    design = corrections.design_resampler(capture, FUNCTION_SOURCE.master_clock_rate)
    assert (design['nfft'], design['nfft_out']) == (550, 528)
    assert corrections.source_record_size(capture, design) == 24038
    assert round(capture.duration * design['fs_sdr']) == 24037


def test_oaresample_pads_use_the_exact_record_size(corrections_flags):
    """7680 samples at 7.68 MS/s from the 125 MS/s clock are a 7812.5-sample source
    record, which `source_record_size` rounds up to 7813 where the float product
    rounds down to 7812. The oaresample pads must bring the 7813-sample record to a
    whole number of STFT hops, or the output slips by one source sample"""
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

    assert_impulse_at(stages.corrected.pre_align, capture, IMPULSE_TIME)


# %% _resample


def test_resample_rejects_a_length_that_does_not_divide():
    """6.25 MS/s to 6 MS/s is 25:24, so one sample more than the (350, 350) overlaps
    leaves a padded length that is not a multiple of 25"""
    capture = preset_capture('dirac_delta', **RESAMPLE_ONLY, time=IMPULSE_TIME)
    lead, tail = corrections.get_correction_overlaps(capture, FUNCTION_SOURCE)
    fs = fs_sdr(capture)
    count = lead + 1 + source_count(capture, FUNCTION_SOURCE) + tail
    assert not sw.isroundmod(count * capture.sample_rate, fs)

    iq = _build_iq('dirac_delta', capture, (lead + 1, tail))

    with pytest.raises(ValueError, match=f'{count} samples at {fs} S/s'):
        ss.correct_iq(iq)
