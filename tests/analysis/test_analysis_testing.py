"""striqt.analysis.testing: closed forms, windowing, and seeding of the generators.

The measurement tests assert against the closed forms documented in
`striqt.analysis.testing`, so these tests pin the closed forms: a unit tone of mean
power 1, noise of total power `noise_psd * sample_rate`, and amplitudes of
`10**(power/20)`. Each generator keeps the `xp` fixture on one closed-form test and
on the windowing test, which run the cupy branches on the Jetson; the remaining
checks run on numpy.
"""

from __future__ import annotations

import numpy as np
import pytest
from numeric_checks import (
    assert_close,
    elementwise_rtol,
    rms,
    to_numpy,
    tone_frequency,
)

from striqt.analysis import testing
from striqt.waveform.arrays import mean_atol

FS = 1e6
DURATION = 1e-4
SIZE = round(DURATION * FS)

# long enough that the statistical tests below are decided far from the bounds
STAT_SIZE = 1 << 14
STAT_DURATION = STAT_SIZE / FS

# 6 sigma on the mean power of `STAT_SIZE` complex gaussian samples
STAT_RTOL = 6 / np.sqrt(STAT_SIZE)

GENERATORS = {
    'tone': (testing.tone, {'frequency': 1e5}),
    'single_tone': (
        testing.single_tone,
        {'frequency_offset': 1e5, 'lo_offset': -2e5, 'snr': 6},
    ),
    'circular_awgn': (testing.circular_awgn, {'power': 1e-3}),
    'sawtooth': (testing.sawtooth, {'period': 1e-5, 'power': 3}),
    'dirac_delta': (testing.dirac_delta, {'time': 3.7e-5, 'power': -4}),
}
SEEDED = ['single_tone', 'circular_awgn']

# samples before index 0 in the longer call of the windowing test
PREROLL = 11

# (start_index, count) windows inside the longer call, None for the default count
WINDOWS = {
    'head': (0, 7),
    'interior': (13, 29),
    'last_sample': (SIZE - 1, 1),
    'preroll': (-PREROLL, 5),
    'across_zero': (-7, 20),
    'preroll_to_end': (-10, None),
}

INVALID_KWS = {
    'ports0': {'ports': 0},
    'negative_count': {'count': -1},
    'start_past_duration': {'start_index': SIZE + 1},
}


def mean_power(x, axis=None):
    x = to_numpy(x)
    return np.mean(np.abs(x) ** 2, axis=axis)


def generate(func, xp=np, duration=DURATION, fs=FS, **kws):
    """call `func` on the `xp` backend and return the result as a numpy array."""
    return to_numpy(func(duration, fs, xp=xp, **kws))


# %% shape and dtype


@pytest.mark.namespaces('numpy')
@pytest.mark.parametrize('name', list(GENERATORS), ids=list(GENERATORS))
@pytest.mark.parametrize('dtype', [None, 'complex128'], ids=['default', 'complex128'])
def test_generator_dtype(xp, name, dtype):
    """the dtype is complex64 unless requested. The windowing test below covers the
    shape on every backend."""
    func, kws = GENERATORS[name]
    requested = {} if dtype is None else {'dtype': dtype}
    x = func(DURATION, FS, xp=xp, **kws, **requested)
    assert x.dtype == np.dtype(dtype or 'complex64')


# %% tone


def test_tone_has_unit_power_and_zero_start_phase(xp):
    x = testing.tone(DURATION, FS, frequency=1e5, xp=xp)
    assert mean_power(x) == pytest.approx(1.0, abs=mean_atol(to_numpy(x), SIZE))
    assert to_numpy(x)[0, 0] == pytest.approx(1.0)


@pytest.mark.parametrize('frequency', [0.0, 1e5, -2e5], ids='f{:g}'.format)
def test_tone_peak_bin_is_at_frequency(frequency):
    x = testing.tone(DURATION, FS, frequency=frequency)
    assert tone_frequency(x[0], FS) == pytest.approx(frequency)


def test_tone_ports_are_identical():
    x = testing.tone(DURATION, FS, frequency=1e5, ports=3)
    assert np.array_equal(x[0], x[1])
    assert np.array_equal(x[0], x[2])


def test_tone_preroll_follows_the_closed_form():
    frequency = 1e5
    x = generate(testing.tone, frequency=frequency, start_index=-PREROLL, count=PREROLL)
    i = np.arange(-PREROLL, 0)
    expected = np.exp(2j * np.pi * frequency / FS * i).astype('complex64')
    assert_close(x[0], expected, rtol=elementwise_rtol(np.complex64))


# %% single_tone


@pytest.mark.parametrize(
    ('frequency_offset', 'lo_offset'), [(1e5, 0.0), (1e5, -2e5), (0.0, 3e5)]
)
def test_single_tone_frequency_includes_lo_offset(frequency_offset, lo_offset):
    x = testing.single_tone(
        DURATION, FS, frequency_offset=frequency_offset, lo_offset=lo_offset
    )
    assert tone_frequency(x[0], FS) == pytest.approx(frequency_offset + lo_offset)


@pytest.mark.parametrize('snr', [0, 6, 20], ids='snr{}'.format)
def test_single_tone_snr_sets_the_noise_power(xp, snr):
    kws = {'frequency_offset': 1e5, 'xp': xp}
    x = testing.single_tone(STAT_DURATION, FS, snr=snr, **kws)
    noiseless = testing.single_tone(STAT_DURATION, FS, **kws)
    noise_power = mean_power(to_numpy(x) - to_numpy(noiseless))
    assert noise_power == pytest.approx(10 ** (-snr / 10), rel=STAT_RTOL)


def test_single_tone_ports_share_the_tone_and_not_the_noise():
    quiet = testing.single_tone(DURATION, FS, frequency_offset=1e5, ports=2)
    assert np.array_equal(quiet[0], quiet[1])

    noisy = testing.single_tone(DURATION, FS, snr=6, ports=2)
    assert not np.array_equal(noisy[0], noisy[1])


# %% circular_awgn


@pytest.mark.parametrize(
    ('power', 'start_index'),
    [(1.0, 0), (1e-3, 0), (4.0, 0), (1e-3, -STAT_SIZE)],
    ids=['power1', 'power0.001', 'power4', 'power0.001_preroll'],
)
def test_circular_awgn_power_matches_the_request(xp, power, start_index):
    x = testing.circular_awgn(
        None, FS, power=power, start_index=start_index, count=STAT_SIZE, xp=xp
    )
    assert mean_power(x) == pytest.approx(power, rel=STAT_RTOL)


def test_circular_awgn_components_split_the_power():
    x = testing.circular_awgn(STAT_DURATION, FS, power=2.0)
    assert np.var(x.real) == pytest.approx(1.0, rel=STAT_RTOL)
    assert np.var(x.imag) == pytest.approx(1.0, rel=STAT_RTOL)


def test_circular_awgn_ports_are_independent():
    x = testing.circular_awgn(STAT_DURATION, FS, ports=2)
    assert not np.array_equal(x[0], x[1])
    correlation = np.abs(np.mean(x[0] * x[1].conj())) / (rms(x[0]) * rms(x[1]))
    assert correlation < STAT_RTOL


def test_circular_awgn_preroll_is_independent_of_the_capture():
    kws = {'count': STAT_SIZE, 'ports': 2}
    preroll = testing.circular_awgn(None, FS, start_index=-STAT_SIZE, **kws)
    capture = testing.circular_awgn(None, FS, start_index=0, **kws)

    assert not np.array_equal(preroll, capture)
    assert not np.array_equal(preroll, capture[:, ::-1])
    for a, b in [(preroll[0], capture[0]), (preroll[0], preroll[1])]:
        correlation = np.abs(np.mean(a * b.conj())) / (rms(a) * rms(b))
        assert correlation < STAT_RTOL


# %% seeding


@pytest.mark.parametrize('name', SEEDED, ids=SEEDED)
def test_seed_determines_the_noise(name):
    """`seed` defaults to 0, equal seeds reproduce the samples, and a different seed
    changes the samples"""
    func, kws = GENERATORS[name]
    x0 = generate(func, seed=0, **kws)
    assert np.array_equal(x0, generate(func, **kws))
    assert not np.array_equal(x0, generate(func, seed=1, **kws))


# %% sawtooth


@pytest.mark.parametrize(
    ('power', 'start_index'),
    [(0, 0), (3, 0), (-6, 0), (3, -64)],
    ids=['power0', 'power3', 'power-6', 'power3_preroll'],
)
def test_sawtooth_ramps_and_resets(xp, power, start_index):
    # a binary-exact sample rate and period make i/fs % period exact, so the ramp
    # reaches the last step rather than rounding up to the reset. A start index that
    # is a multiple of the period starts the window at a reset.
    fs = 2**20
    samples_per_period = 16
    period = samples_per_period / fs
    size = 4 * samples_per_period

    x = to_numpy(
        testing.sawtooth(
            size / fs,
            fs,
            period=period,
            power=power,
            start_index=start_index,
            count=size,
            xp=xp,
        )
    ).real[0]
    amplitude = 10 ** (power / 20)

    assert x[0] == 0
    assert x.max() == pytest.approx(
        amplitude * (samples_per_period - 1) / samples_per_period, rel=1e-6
    )

    ramp = x.reshape(-1, samples_per_period)
    expected = np.arange(samples_per_period) * (amplitude / samples_per_period)
    assert ramp == pytest.approx(np.broadcast_to(expected, ramp.shape), rel=1e-6)


def test_sawtooth_ports_are_identical():
    x = testing.sawtooth(DURATION, FS, period=1e-5, ports=2)
    assert np.array_equal(x[0], x[1])


# %% dirac_delta


@pytest.mark.parametrize(
    ('time', 'power'),
    [(0.0, 0), (3.7e-5, -4), (9.9e-5, 6)],
    ids=['t0_p0', 't37us_pm4', 't99us_p6'],
)
def test_dirac_delta_has_one_nonzero_sample(xp, time, power):
    x = to_numpy(testing.dirac_delta(DURATION, FS, time=time, power=power, xp=xp))
    _, indices = np.nonzero(x)

    assert indices.tolist() == [round(time * FS)]
    assert x[0, indices[0]] == pytest.approx(10 ** (power / 20))


def test_dirac_delta_outside_the_window_is_zero():
    x = testing.dirac_delta(DURATION, FS, time=2 * DURATION, ports=2)
    assert not x.any()


@pytest.mark.parametrize(
    'time', [3.7e-5, -5 / FS, -PREROLL / FS], ids=['in_capture', 'in_preroll', 'first']
)
def test_dirac_delta_preroll_has_the_impulse_only_at_its_time(time):
    kws = {'start_index': -PREROLL, 'count': PREROLL + SIZE}
    x = generate(testing.dirac_delta, time=time, power=-4, **kws)
    _, indices = np.nonzero(x)
    assert indices.tolist() == [round(time * FS) + PREROLL]
    assert x[0, indices[0]] == pytest.approx(10 ** (-4 / 20))


# %% windowing invariant


@pytest.mark.parametrize('name', list(GENERATORS), ids=list(GENERATORS))
@pytest.mark.parametrize('window', list(WINDOWS), ids=list(WINDOWS))
def test_window_is_a_slice_of_the_longer_call(xp, name, window):
    """a call with `start_index` and `count` equals the slice of one longer call that
    starts `PREROLL` samples before index 0, on both sides of index 0 and with the
    default count"""
    func, kws = GENERATORS[name]
    start_index, count = WINDOWS[window]
    kws = {'ports': 2, **kws}
    whole = generate(func, xp, start_index=-PREROLL, count=PREROLL + SIZE, **kws)
    sliced = generate(func, xp, start_index=start_index, count=count, **kws)

    expected_count = SIZE - start_index if count is None else count
    assert sliced.shape == (2, expected_count)
    offset = start_index + PREROLL
    assert np.array_equal(sliced, whole[:, offset : offset + expected_count])


# %% argument validation


@pytest.mark.parametrize('kws', list(INVALID_KWS.values()), ids=list(INVALID_KWS))
def test_invalid_arguments_raise(kws):
    """every generator checks `ports`, `count`, and `start_index` through
    `_resolve_count`, so one generator covers the rejections"""
    func, defaults = GENERATORS['tone']
    with pytest.raises(ValueError):
        func(DURATION, FS, **defaults, **kws)


def test_nonpositive_sawtooth_period_raises():
    with pytest.raises(ValueError):
        testing.sawtooth(DURATION, FS, period=0)
