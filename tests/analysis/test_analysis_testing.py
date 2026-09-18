"""closed forms, windowing and seeding of the striqt.analysis.testing generators.

The measurement tests assert against the closed forms documented in that module, so
these tests pin the closed forms themselves: a unit tone of mean power 1, noise of
total power `noise_psd * sample_rate`, and amplitudes of `10**(power/20)`.
"""

from __future__ import annotations

import numpy as np
import pytest
from numeric_checks import mean_atol, rms, to_numpy, tone_frequency

from striqt.analysis import testing

FS = 1e6
DURATION = 1e-4
SIZE = round(DURATION * FS)

# long enough that the statistical tests below are decided far from their bounds
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
    'noise': (testing.noise, {'noise_psd': 1e-9}),
    'sawtooth': (testing.sawtooth, {'period': 1e-5, 'power': 3}),
    'dirac_delta': (testing.dirac_delta, {'time': 3.7e-5, 'power': -4}),
}

WINDOWS = [(0, SIZE), (0, 7), (13, 29), (SIZE - 1, 1)]
WINDOW_IDS = [f'start{start}_count{count}' for start, count in WINDOWS]

INVALID_KWS = {
    'ports0': {'ports': 0},
    'negative_start': {'start_index': -1},
    'negative_count': {'count': -1},
    'start_past_duration': {'start_index': SIZE + 1},
}


def mean_power(x, axis=None):
    x = to_numpy(x)
    return np.mean(np.abs(x) ** 2, axis=axis)


# %% tone


@pytest.mark.parametrize('ports', [1, 2], ids='ports{}'.format)
def test_tone_shape_and_dtype(xp, ports):
    x = testing.tone(DURATION, FS, ports=ports, xp=xp)
    assert x.shape == (ports, SIZE)
    assert x.dtype == np.dtype('complex64')
    assert testing.tone(DURATION, FS, xp=xp, dtype='complex128').dtype == np.dtype(
        'complex128'
    )


def test_tone_has_unit_power_and_zero_start_phase(xp):
    x = testing.tone(DURATION, FS, frequency=1e5, xp=xp)
    assert mean_power(x) == pytest.approx(1.0, abs=mean_atol(to_numpy(x), SIZE))
    assert to_numpy(x)[0, 0] == pytest.approx(1.0)


@pytest.mark.parametrize('frequency', [0.0, 1e5, -2e5], ids='f{:g}'.format)
def test_tone_peak_bin_is_at_frequency(xp, frequency):
    x = testing.tone(DURATION, FS, frequency=frequency, xp=xp)
    assert tone_frequency(to_numpy(x)[0], FS) == pytest.approx(frequency)


def test_tone_ports_are_identical(xp):
    x = to_numpy(testing.tone(DURATION, FS, frequency=1e5, ports=3, xp=xp))
    assert np.array_equal(x[0], x[1])
    assert np.array_equal(x[0], x[2])


# %% single_tone


def test_single_tone_is_a_product_of_tones(xp):
    kws = {'xp': xp}
    x = testing.single_tone(DURATION, FS, frequency_offset=1e5, lo_offset=-2e5, **kws)
    expected = testing.tone(DURATION, FS, frequency=1e5, **kws) * testing.tone(
        DURATION, FS, frequency=-2e5, **kws
    )
    assert np.array_equal(to_numpy(x), to_numpy(expected))


@pytest.mark.parametrize(
    ('frequency_offset', 'lo_offset'), [(1e5, 0.0), (1e5, -2e5), (0.0, 3e5)]
)
def test_single_tone_frequency_includes_lo_offset(xp, frequency_offset, lo_offset):
    x = testing.single_tone(
        DURATION, FS, frequency_offset=frequency_offset, lo_offset=lo_offset, xp=xp
    )
    peak = tone_frequency(to_numpy(x)[0], FS)
    assert peak == pytest.approx(frequency_offset + lo_offset)


@pytest.mark.parametrize('snr', [0, 6, 20], ids='snr{}'.format)
def test_single_tone_snr_sets_the_noise_power(xp, snr):
    kws = {'frequency_offset': 1e5, 'xp': xp}
    x = testing.single_tone(STAT_DURATION, FS, snr=snr, **kws)
    noiseless = testing.single_tone(STAT_DURATION, FS, **kws)
    noise_power = mean_power(to_numpy(x) - to_numpy(noiseless))
    assert noise_power == pytest.approx(10 ** (-snr / 10), rel=STAT_RTOL)


def test_single_tone_seed_changes_only_the_noise(xp):
    kws = {'snr': 6, 'xp': xp}
    x0 = to_numpy(testing.single_tone(DURATION, FS, seed=0, **kws))
    assert np.array_equal(
        x0, to_numpy(testing.single_tone(DURATION, FS, seed=0, **kws))
    )
    assert not np.array_equal(
        x0, to_numpy(testing.single_tone(DURATION, FS, seed=1, **kws))
    )


def test_single_tone_ports_share_the_tone_and_not_the_noise(xp):
    quiet = to_numpy(
        testing.single_tone(DURATION, FS, frequency_offset=1e5, ports=2, xp=xp)
    )
    assert np.array_equal(quiet[0], quiet[1])

    noisy = to_numpy(testing.single_tone(DURATION, FS, snr=6, ports=2, xp=xp))
    assert not np.array_equal(noisy[0], noisy[1])


# %% circular_awgn


@pytest.mark.parametrize('power', [1.0, 1e-3, 4.0], ids='power{:g}'.format)
def test_circular_awgn_power_matches_the_request(xp, power):
    x = testing.circular_awgn(STAT_DURATION, FS, power=power, xp=xp)
    assert mean_power(x) == pytest.approx(power, rel=STAT_RTOL)


def test_circular_awgn_components_split_the_power(xp):
    x = to_numpy(testing.circular_awgn(STAT_DURATION, FS, power=2.0, xp=xp))
    assert np.var(x.real) == pytest.approx(1.0, rel=STAT_RTOL)
    assert np.var(x.imag) == pytest.approx(1.0, rel=STAT_RTOL)


def test_circular_awgn_ports_are_independent(xp):
    x = to_numpy(testing.circular_awgn(STAT_DURATION, FS, ports=2, xp=xp))
    assert not np.array_equal(x[0], x[1])
    correlation = np.abs(np.mean(x[0] * x[1].conj())) / (rms(x[0]) * rms(x[1]))
    assert correlation < STAT_RTOL


def test_circular_awgn_seed_determines_the_samples(xp):
    x0 = to_numpy(testing.circular_awgn(DURATION, FS, seed=0, xp=xp))
    assert np.array_equal(x0, to_numpy(testing.circular_awgn(DURATION, FS, xp=xp)))
    assert not np.array_equal(
        x0, to_numpy(testing.circular_awgn(DURATION, FS, seed=1, xp=xp))
    )


@pytest.mark.parametrize('ports', [1, 2], ids='ports{}'.format)
def test_circular_awgn_shape_and_dtype(xp, ports):
    x = testing.circular_awgn(DURATION, FS, ports=ports, xp=xp)
    assert x.shape == (ports, SIZE)
    assert x.dtype == np.dtype('complex64')
    assert testing.circular_awgn(
        DURATION, FS, xp=xp, dtype='complex128'
    ).dtype == np.dtype('complex128')


# %% noise


@pytest.mark.parametrize('noise_psd', [1e-9, 1e-17], ids='psd{:g}'.format)
def test_noise_is_circular_awgn_at_the_integrated_power(xp, noise_psd):
    """the contract that keeps `noise` and `circular_awgn` in step"""
    x = testing.noise(DURATION, FS, noise_psd=noise_psd, ports=2, seed=3, xp=xp)
    expected = testing.circular_awgn(
        DURATION, FS, power=noise_psd * FS, ports=2, seed=3, xp=xp
    )
    assert np.array_equal(to_numpy(x), to_numpy(expected))


def test_noise_power_matches_the_psd(xp):
    psd = 1e-9
    x = testing.noise(STAT_DURATION, FS, noise_psd=psd, xp=xp)
    assert mean_power(x) == pytest.approx(psd * FS, rel=STAT_RTOL)


def test_noise_ports_are_independent(xp):
    x = to_numpy(testing.noise(STAT_DURATION, FS, noise_psd=1e-9, ports=2, xp=xp))
    assert not np.array_equal(x[0], x[1])
    correlation = np.abs(np.mean(x[0] * x[1].conj())) / (rms(x[0]) * rms(x[1]))
    assert correlation < STAT_RTOL


def test_noise_seed_determines_the_samples(xp):
    kws = {'noise_psd': 1e-9, 'xp': xp}
    x0 = to_numpy(testing.noise(DURATION, FS, seed=0, **kws))
    assert np.array_equal(x0, to_numpy(testing.noise(DURATION, FS, seed=0, **kws)))
    assert not np.array_equal(x0, to_numpy(testing.noise(DURATION, FS, seed=1, **kws)))


def test_noise_shape_and_dtype(xp):
    x = testing.noise(DURATION, FS, ports=2, xp=xp)
    assert x.shape == (2, SIZE)
    assert x.dtype == np.dtype('complex64')


# %% sawtooth


@pytest.mark.parametrize('power', [0, 3, -6], ids='power{}'.format)
def test_sawtooth_ramps_and_resets(xp, power):
    # a binary-exact sample rate and period, so that i/fs % period is exact and the
    # ramp reaches its last step rather than rounding up to the reset
    fs = 2**20
    samples_per_period = 16
    period = samples_per_period / fs
    size = 4 * samples_per_period

    x = to_numpy(
        testing.sawtooth(size / fs, fs, period=period, power=power, xp=xp)
    ).real[0]
    amplitude = 10 ** (power / 20)

    assert x[0] == 0
    assert x.max() == pytest.approx(
        amplitude * (samples_per_period - 1) / samples_per_period, rel=1e-6
    )

    ramp = x.reshape(-1, samples_per_period)
    expected = np.arange(samples_per_period) * (amplitude / samples_per_period)
    assert ramp == pytest.approx(np.broadcast_to(expected, ramp.shape), rel=1e-6)


@pytest.mark.parametrize('power', [0, -6], ids='power{}'.format)
def test_sawtooth_follows_the_closed_form(xp, power):
    """At an exact reset (`t == k * period`) the ramp has a jump discontinuity
    from `amplitude` back to `0`. `period` (1e-5) and `FS` (1e6) are not exact
    binary fractions, so `i / FS` at a reset sample only approximates that
    instant rather than landing on it exactly: the sample is a floating-point
    tie between the discontinuity's two endpoints, and which side a backend's
    modulo resolves it to is not guaranteed to agree between `numpy` and
    `cupy`. Away from the resets the closed form is unambiguous and is checked
    pointwise; at the resets only the weaker, backend-agnostic invariant --
    close to `0` or close to `amplitude` -- is checked.
    """
    period = 1e-5
    x = to_numpy(testing.sawtooth(DURATION, FS, period=period, power=power, xp=xp))
    amplitude = 10 ** (power / 20)

    t = np.arange(SIZE, dtype='int64') / FS
    expected = (t % period) * (amplitude / period)

    samples_per_period = round(period * FS)
    is_boundary = np.arange(SIZE) % samples_per_period == 0

    assert np.array_equal(x.imag, np.zeros_like(x.imag))

    interior = x.real[0][~is_boundary]
    assert interior == pytest.approx(expected[~is_boundary], rel=1e-6)

    boundary = x.real[0][is_boundary]
    tol = max(1e-6 * abs(amplitude), 1e-9)
    assert np.all((np.abs(boundary) <= tol) | (np.abs(boundary - amplitude) <= tol))


def test_sawtooth_ports_are_identical(xp):
    x = to_numpy(testing.sawtooth(DURATION, FS, period=1e-5, ports=2, xp=xp))
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


def test_dirac_delta_outside_the_window_is_zero(xp):
    x = testing.dirac_delta(DURATION, FS, time=2 * DURATION, ports=2, xp=xp)
    assert not to_numpy(x).any()


# %% windowing invariant


@pytest.mark.parametrize('name', list(GENERATORS), ids=list(GENERATORS))
@pytest.mark.parametrize(('start_index', 'count'), WINDOWS, ids=WINDOW_IDS)
@pytest.mark.parametrize('ports', [1, 2], ids='ports{}'.format)
def test_window_matches_the_whole_capture(xp, name, start_index, count, ports):
    func, kws = GENERATORS[name]
    whole = to_numpy(func(DURATION, FS, ports=ports, xp=xp, **kws))
    window = to_numpy(
        func(
            DURATION,
            FS,
            ports=ports,
            start_index=start_index,
            count=count,
            xp=xp,
            **kws,
        )
    )

    assert window.shape == (ports, count)
    assert np.array_equal(window, whole[:, start_index : start_index + count])


def test_count_defaults_to_the_remainder_of_the_capture(xp):
    x = testing.noise(DURATION, FS, start_index=SIZE - 10, xp=xp)
    assert x.shape == (1, 10)


# %% argument validation


@pytest.mark.parametrize('kws', list(INVALID_KWS.values()), ids=list(INVALID_KWS))
@pytest.mark.parametrize('name', list(GENERATORS), ids=list(GENERATORS))
def test_invalid_arguments_raise(name, kws):
    func, defaults = GENERATORS[name]
    with pytest.raises(ValueError):
        func(DURATION, FS, **defaults, **kws)


def test_nonpositive_sawtooth_period_raises():
    with pytest.raises(ValueError):
        testing.sawtooth(DURATION, FS, period=0)
