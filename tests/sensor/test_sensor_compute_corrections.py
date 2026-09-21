"""striqt.sensor.lib.compute.corrections: the per-capture signal path of correct_iq
(time origin, level, band placement and the pre_filter/pre_align/aligned stages), the
resampler design, and the sizing of the extra acquisition overlap that the resampler
and the analysis filter consume"""

from __future__ import annotations

import dataclasses
from math import inf, isfinite

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from numeric_checks import (
    FIR_LEAKAGE,
    assert_close,
    assert_tone_level,
    rms,
    single_backend_rms,
    to_numpy,
    tone_frequency,
    tone_peak_roundoff,
)
from soapy_factories import MCR, soapy_capture, source_spec
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
    make_capture,
)

import striqt.sensor as ss
import striqt.waveform as sw
from striqt.sensor.lib.compute import corrections
from striqt.sensor.lib.sources import buffers

SOAPY_SOURCE = source_spec()

# 0.5 ms is a whole number of samples at every preset's source and capture rate, so
# the impulse sits on an output sample: the brick-wall resample of an impulse is
# exactly fs/fs_sdr there and zero on every other output sample
IMPULSE_TIME = 0.5e-3
IMPULSE_POWER_DB = -6.0

# 1 MHz lies on the FFT grid of the padded acquisition of both resample presets
# (25000 and 13200 samples at 6.25 MS/s), so its resample is exact up to roundoff;
# the Gibbs ripple of an off-grid tone is the kernel's business (tests/waveform)
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

D1_REASON = (
    '_get_resample_overlap splits its pad into two equal halves: it asserts '
    'pad_end % 2 == 0, which fails whenever round(duration*fs_sdr) is odd, and when '
    'pad_end is 2 mod 4 each half is odd and Controller.read_iq rejects the overlaps '
    'as not even, so most host-resampled (sample_rate, duration) pairs cannot acquire'
)


D90_REASON = (
    '_get_resampler_overlaps compares its pad against _get_filter_overlap, which is '
    'FILTER_SIZE//2 + 1 counted at fs_sdr, but correct_iq applies the FIR after '
    'resampling, so the transient spans FILTER_SIZE//2 samples of capture.sample_rate; '
    'whenever fs_sdr > sample_rate the lead overlap is short by that ratio'
)


def _capture(**kws):
    """a 10 ms capture at the SoapyCapture defaults of no analysis filter and host
    resampling, unless `kws` says otherwise"""
    return soapy_capture(**{
        'duration': 10e-3,
        'analysis_bandwidth': inf,
        'host_resample': True,
        **kws,
    })


def _resample_tol(preset, raw) -> dict:
    """assert_close keywords for one float32 FFT resample of the acquisition `raw`,
    or exact equality for a preset that the firmware rate serves without resampling"""
    if not preset.get('host_resample', True):
        return {}
    n_in = raw.pre_align.shape[1]
    n_out = round(n_in * raw.capture.sample_rate / raw.resampler['fs_sdr'])
    return {'sigma': single_backend_rms(np.complex64, (n_in, n_out))}


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


@PRESET_PARAMS
def test_impulse_lands_on_its_output_sample(preset, array_backend, subtests):
    capture = make_capture('dirac_delta', **preset, time=IMPULSE_TIME)
    stages = acquire_corrected('dirac_delta', capture, array_backend=array_backend)
    fs = capture.sample_rate

    for name in ('pre_filter', 'pre_align'):
        x = getattr(stages.corrected, name)
        with subtests.test(stage=name):
            assert x.shape[1] == round(capture.duration * fs)
            assert_impulse_at(x, capture, IMPULSE_TIME)


@pytest.mark.parametrize(
    'preset', [SCALE_ONLY, FILTER_ONLY], ids=['scale_only', 'filter_only']
)
def test_output_is_the_generator_from_index_zero(preset, array_backend):
    """A sawtooth rather than a tone, so that a trim that is off by any number of
    samples shows. The resample presets are left to the impulse test, because a
    ramp's discontinuities ring through a brick-wall resampler."""
    capture = make_capture('sawtooth', **preset, period=1e-4)
    stages = acquire_corrected('sawtooth', capture, array_backend=array_backend)
    expected = expected_corrected('sawtooth', capture)

    assert expected.shape[1] == round(capture.duration * capture.sample_rate)
    assert np.array_equal(to_numpy(stages.corrected.pre_filter), expected)


@PRESET_PARAMS
def test_tone_phase_matches_the_generator(preset, array_backend):
    capture = make_capture(
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
    capture = make_capture(
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
        # the inverse FFT of a flat spectrum concentrates roundoff at the peak the
        # way a tone's FFT does in its bin
        ratio = fs / fs_sdr
        tol = _resample_tol(preset, stages.raw)
        if tol:
            tol['atol'] = tone_peak_roundoff(np.complex64) * amplitude * ratio
        assert_close(stages.corrected.pre_filter, ratio * impulse, **tol)

    with subtests.test(stage='pre_align'):
        pre_align = to_numpy(stages.corrected.pre_align)
        if isfinite(capture.analysis_bandwidth):
            # the centre tap of a unit-DC-gain windowed-sinc lowpass at cutoff bw/2
            # is bw/fs, up to the passband error that the DC normalization moves
            peak = amplitude * capture.analysis_bandwidth / fs_sdr
            assert_close(np.abs(pre_align[:, index]), peak, rtol=FIR_LEAKAGE)
        else:
            assert np.array_equal(pre_align, to_numpy(stages.corrected.pre_filter))


@PRESET_PARAMS
def test_tone_level_and_frequency(preset, array_backend, subtests):
    capture = make_capture(
        'single_tone', **preset, frequency_offset=TONE_FREQUENCY, snr=None
    )
    stages = acquire_corrected('single_tone', capture, array_backend=array_backend)
    x = to_numpy(stages.corrected.pre_align)
    fs = capture.sample_rate

    with subtests.test('level'):
        assert_tone_level(x)
    for port in range(x.shape[0]):
        with subtests.test('frequency', port=port):
            assert abs(tone_frequency(x[port], fs) - TONE_FREQUENCY) < fs / x.shape[1]


@CHEAP_PRESETS
@pytest.mark.parametrize(
    'voltage_scale',
    [0.5, np.array([0.5, 2.0], dtype='float32')],
    ids=['scalar', 'per_port'],
)
def test_voltage_scale(preset, voltage_scale, array_backend):
    """powers of two, so the scaling itself is exact and only the resample rounds"""
    capture = make_capture(
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
    capture = make_capture(
        'single_tone', **preset, frequency_offset=TONE_FREQUENCY, snr=None
    )
    stages = acquire_corrected('single_tone', capture, array_backend=array_backend)

    corrected = ss.correct_iq(dataclasses.replace(stages.raw, conjugate=(True, False)))

    expected = expected_corrected('single_tone', capture)
    expected[0] = np.conj(expected[0])
    assert_close(corrected.pre_filter, expected, **_resample_tol(preset, stages.raw))


def test_ignore_highside_lo_bypasses_the_conjugate(corrections_flags):
    corrections_flags(ignore_highside_lo=True)
    capture = make_capture(
        'single_tone', **SCALE_ONLY, frequency_offset=TONE_FREQUENCY, snr=None
    )
    stages = acquire_corrected('single_tone', capture)

    corrected = ss.correct_iq(dataclasses.replace(stages.raw, conjugate=(True, True)))

    expected = expected_corrected('single_tone', capture)
    assert np.array_equal(corrected.pre_filter, expected)


def test_overwrite_x_false_leaves_the_acquisition_intact(array_backend):
    """the scale-only path hands the acquisition buffer itself on when voltage_scale
    is 1, so an in-place conjugate there would corrupt a reused acquisition"""
    capture = make_capture(
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
    capture = make_capture(
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
    capture = make_capture('dirac_delta', **preset, time=IMPULSE_TIME)
    trigger = FakeTrigger(TRIGGER_LAGS)
    raw = _build_iq_for_trigger('dirac_delta', capture, trigger, xp)
    fs = capture.sample_rate
    size_out = round(capture.duration * fs)
    tol = _resample_tol(preset, raw)
    ratio = fs / raw.resampler['fs_sdr']

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


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason='_oaresample trims nfft_out samples from the front of the sw.oaresample '
    'output instead of the resampled lead overlap, then asserts that the remainder '
    'is exactly the capture plus its tail pad; with the lead that '
    '_get_oaresample_overlaps requests it never is, so correct_iq cannot complete '
    'under STRIQT_USE_OARESAMPLE=1',
)
@RESAMPLED_PRESETS
def test_oaresample_impulse_lands_on_its_output_sample(preset, corrections_flags):
    corrections_flags(use_oaresample=True)
    capture = make_capture('dirac_delta', **preset, time=IMPULSE_TIME)
    overlaps = corrections.get_correction_overlaps(capture, FUNCTION_SOURCE)

    corrected = ss.correct_iq(_build_iq('dirac_delta', capture, overlaps))

    size_out = round(capture.duration * capture.sample_rate)
    assert corrected.pre_align.shape[1] == size_out
    assert_impulse_at(corrected.pre_align, capture, IMPULSE_TIME)


@pytest.mark.xfail(
    strict=True,
    raises=ValueError,
    reason='_get_oaresample_overlaps returns odd overlaps (43750, 9375 for the '
    'resample_filter preset), which Controller.read_iq rejects, so no capture '
    'acquires under STRIQT_USE_OARESAMPLE=1',
)
@RESAMPLED_PRESETS
def test_oaresample_acquisition(preset, corrections_flags):
    corrections_flags(use_oaresample=True)
    capture = make_capture('dirac_delta', **preset, time=IMPULSE_TIME)

    stages = acquire_corrected('dirac_delta', capture)

    assert_impulse_at(stages.corrected.pre_align, capture, IMPULSE_TIME)


# %% correct_iq: LO shift


@pytest.mark.parametrize('lo_shift', ['left', 'right'])
def test_lo_shift_design_moves_the_lo_out_of_band(lo_shift):
    """the LO leakage sits at 0 Hz in the radio's baseband, so the shifted passband
    must exclude it while still fitting the radio's Nyquist band"""
    capture = make_capture('single_tone', **LO_SHIFT_CAPTURE, lo_shift=lo_shift)
    design = corrections.design_resampler(capture, FUNCTION_SOURCE.master_clock_rate)
    half_bw = capture.analysis_bandwidth / 2

    assert np.sign(design['lo_offset']) == (1 if lo_shift == 'right' else -1)
    assert abs(design['lo_offset']) > half_bw
    assert abs(design['lo_offset']) + half_bw <= design['fs_sdr'] / 2


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason='_resample calls sw.resample without shift=, so the lo_offset that '
    'design_resampler moved the radio LO by (and that the synthetic source '
    'reproduces) is never removed and the tone stays at frequency_offset + lo_offset',
)
@pytest.mark.parametrize(
    'lo_shift, frequency', [('left', 1e6), ('right', -1e6)], ids=['left', 'right']
)
def test_lo_shift_is_removed_from_the_output(lo_shift, frequency):
    capture = make_capture(
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
    capture = make_capture('single_tone', host_resample=host_resample)
    assert corrections.needs_resample(design, capture) is expected


def test_design_without_host_resample_is_the_identity():
    capture = make_capture('single_tone', **SCALE_ONLY)
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
    capture = make_capture('single_tone', **kws)
    with pytest.raises(ValueError, match=match):
        corrections.design_resampler(capture, FUNCTION_SOURCE.master_clock_rate)


def test_design_resampler_needs_a_clock_rate():
    capture = make_capture('single_tone', **SCALE_ONLY)
    with pytest.raises(TypeError, match='master_clock_rate'):
        corrections.design_resampler(capture, None)


# %% get_correction_overlaps: properties over the capture domain

sample_rates = st.integers(min_value=1000, max_value=20000).map(lambda k: k * 1e3)
sample_counts = st.integers(min_value=100, max_value=40000)
bandwidth_fractions = st.one_of(st.just(inf), st.floats(min_value=0.3, max_value=0.95))
max_lags = st.floats(min_value=0, max_value=1e-4)


def _drawn_capture(sample_rate, count, bandwidth_fraction):
    if bandwidth_fraction == inf:
        bw = inf
    else:
        bw = round(bandwidth_fraction * sample_rate / 1e3) * 1e3
    return make_capture(
        'single_tone',
        sample_rate=sample_rate,
        duration=count / sample_rate,
        analysis_bandwidth=bw,
    )


def _overlaps_or_skip(capture, min_overlap=0):
    """the resampler overlaps of `capture`, skipping the example when the design
    falls into one of the registered defects that abort inside
    _get_resample_overlap: the swapped ceildiv (xfail-audit #43) or the odd pad
    (test_overlaps_are_even)"""
    try:
        return corrections._get_resampler_overlaps(
            capture, FUNCTION_SOURCE, min_overlap=min_overlap
        )
    except AssertionError:
        assume(False)


@given(
    sample_rate=sample_rates,
    count=sample_counts,
    bandwidth_fraction=bandwidth_fractions,
    max_lag=max_lags,
)
def test_overlaps_cover_the_filter_and_trigger_pads(
    sample_rate, count, bandwidth_fraction, max_lag
):
    capture = _drawn_capture(sample_rate, count, bandwidth_fraction)
    trigger = FakeTrigger(0.0, max_lag=max_lag)
    lag_pad = corrections._get_max_trigger_lag(FUNCTION_SOURCE, capture, trigger)
    lead, tail = _overlaps_or_skip(capture, lag_pad)
    fs = fs_sdr(capture)

    assert lead >= 0 and tail >= 0
    if isfinite(capture.analysis_bandwidth):
        # the FIR transient spans FILTER_SIZE//2 output samples. Upsampling designs are
        # short of that by fs/sample_rate (xfail-audit #90), pinned by
        # test_upsampled_overlap_covers_the_output_rate_filter_pad
        assume(fs <= capture.sample_rate)
        assert lead * capture.sample_rate / fs >= FILTER_SIZE // 2
    # a trigger shift of up to max_lag reads that far past the capture
    assert tail >= max_lag * fs
    # the sum of the two is an overlap that read_iq accepts, and it asks the source
    # for exactly the padded resampler input
    read_count = buffers.get_read_count(capture, FUNCTION_SOURCE, overlap=lead + tail)
    assert read_count == round(capture.duration * fs) + lead + tail


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=D90_REASON)
def test_upsampled_overlap_covers_the_output_rate_filter_pad():
    """a capture whose resampler upsamples: 12.065 MS/s from a 12.5 MS/s fs_sdr, where
    the (2061, 2061) overlaps buy only 1989 of the 2000 output samples the FIR
    transient spans, so the trimmed output keeps ~11 samples of edge transient at
    each end"""
    capture = _drawn_capture(12065e3, 24977, 0.5)
    lead, _ = corrections._get_resampler_overlaps(capture, FUNCTION_SOURCE)

    assert lead * capture.sample_rate / fs_sdr(capture) >= FILTER_SIZE // 2


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=D1_REASON)
@settings(report_multiple_bugs=False, max_examples=50)
@given(
    sample_rate=sample_rates,
    count=sample_counts,
    bandwidth_fraction=bandwidth_fractions,
)
def test_overlaps_are_even(sample_rate, count, bandwidth_fraction):
    capture = _drawn_capture(sample_rate, count, bandwidth_fraction)
    design = corrections.design_resampler(capture, FUNCTION_SOURCE.master_clock_rate)
    # the small-FFT case is xfail-audit #43, not this defect
    assume(
        not isfinite(capture.analysis_bandwidth)
        or design['nfft'] >= corrections._get_filter_overlap(capture)
    )

    lead, tail = corrections.get_correction_overlaps(capture, FUNCTION_SOURCE)

    assert lead % 2 == 0 and tail % 2 == 0


DOCUMENTED_OVERLAPS = {
    'resample_filter': (6250, 6250),
    'resample_only': (350, 350),
    'scale_only': (512, 512),
    'filter_only': (12800, 12800),
}


@pytest.mark.parametrize('name', list(PRESETS))
def test_preset_overlaps_are_acquirable(name):
    """The harness presets were chosen to dodge the two overlap defects above. Pin
    the sizes that synthetic_sources documents, so that a change here shows up as
    a harness change rather than as unrelated acquisition failures."""
    capture = make_capture('single_tone', **PRESETS[name])
    overlaps = corrections.get_correction_overlaps(capture, FUNCTION_SOURCE)
    assert overlaps == DOCUMENTED_OVERLAPS[name]


# %% acquisitions that the overlap defects block

# 6.144 MS/s for 2 ms without an analysis filter pads to (3125, 3125)
ODD_OVERLAP_CAPTURE = {**RESAMPLE_ONLY, 'sample_rate': 6.144e6}
# 4 MS/s comes from 125 MHz / 31, so 1 ms is 4032.26 source samples
NONINTEGRAL_CAPTURE = {**RESAMPLE_ONLY, 'sample_rate': 4e6, 'duration': 1e-3}


@pytest.mark.parametrize(
    'preset',
    [
        pytest.param(
            ODD_OVERLAP_CAPTURE,
            id='odd_overlap',
            marks=pytest.mark.xfail(strict=True, raises=ValueError, reason=D1_REASON),
        ),
        pytest.param(NONINTEGRAL_CAPTURE, id='nonintegral_source_duration'),
    ],
)
def test_impulse_acquisition(preset):
    capture = make_capture('dirac_delta', **preset, time=IMPULSE_TIME)
    stages = acquire_corrected('dirac_delta', capture)
    assert_impulse_at(stages.corrected.pre_align, capture, IMPULSE_TIME)


def test_correct_iq_trims_an_odd_lead_pad():
    """the even-overlap requirement is Controller.read_iq's, not the correction's"""
    capture = make_capture('dirac_delta', **ODD_OVERLAP_CAPTURE, time=IMPULSE_TIME)
    overlaps = corrections.get_correction_overlaps(capture, FUNCTION_SOURCE)

    corrected = ss.correct_iq(_build_iq('dirac_delta', capture, overlaps))

    assert corrected.pre_align.shape[1] == round(capture.duration * capture.sample_rate)
    assert_impulse_at(corrected.pre_align, capture, IMPULSE_TIME)


# %% get_correction_overlaps: the soapy captures


def _assert_valid_overlaps(capture):
    low, high = corrections.get_correction_overlaps(capture, SOAPY_SOURCE)
    filter_pad = corrections._get_filter_overlap(capture)
    assert low >= filter_pad and high >= filter_pad
    assert low > 0 and high > 0


VALID_OVERLAP_CAPTURES = {
    'infinite_bandwidth_at_the_clock_rate': {'sample_rate': MCR},
    'infinite_bandwidth_at_62.5_MHz': {'sample_rate': 62.5e6},
    'infinite_bandwidth_at_15.36_MHz': {'sample_rate': 15.36e6},
    'finite_bandwidth_without_host_resampling': {
        'analysis_bandwidth': 40e6,
        'host_resample': False,
    },
    # fs_sdr 15.625 MS/s -> 15.36 MS/s designs a 6250-point FFT, larger than the
    # filter overlap, so the block sizing works out regardless of the ceildiv order
    'finite_bandwidth_with_a_large_resampler_fft': {
        'sample_rate': 15.36e6,
        'analysis_bandwidth': 10e6,
    },
}


@pytest.mark.parametrize(
    'kws',
    list(VALID_OVERLAP_CAPTURES.values()),
    ids=list(VALID_OVERLAP_CAPTURES),
)
def test_soapy_capture_overlaps_cover_the_filter(kws):
    _assert_valid_overlaps(_capture(**kws))


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason='_get_resample_overlap swaps the ceildiv arguments, so a small resampler '
    'FFT gives a block size of one FFT rather than enough to cover the filter '
    'overlap; the pad assertions in _get_resample_overlap/_get_resampler_overlaps '
    'then fail for finite analysis_bandwidth with host_resample',
)
@pytest.mark.parametrize(
    'sample_rate, analysis_bandwidth',
    [(MCR, 40e6), (1e6, 0.5e6), (10e6, 8e6)],
)
def test_finite_bandwidth_with_a_small_resampler_fft(sample_rate, analysis_bandwidth):
    _assert_valid_overlaps(
        _capture(sample_rate=sample_rate, analysis_bandwidth=analysis_bandwidth)
    )
