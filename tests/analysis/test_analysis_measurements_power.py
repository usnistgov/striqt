"""the time-selective power measurements: channel_power_time_series,
cyclic_channel_power and iq_waveform.

`tests/waveform/test_power_analysis.py` property-tests the underlying
`iq_to_bin_power` and `iq_to_cyclic_power` kernels, so these tests cover only what
the measurement wrapping adds: coordinate values, axis order, dtype, the dB
conversion, detector and statistic selection, and slicing.

Inputs are the closed-form generators in `striqt.analysis.testing`, whose levels
`tests/analysis/test_analysis_testing.py` pins. A unit-amplitude tone has mean
``|x|**2 == 1``, so the measurements' dBm scale is ``10*log10(mean |x|**2)`` with no
offset, and an amplitude of ``10**(power/20)`` reads `power` dBm.

Tolerances come from `dB_tol`: the float32 dB conversion budget of `numeric_checks`
plus the roundoff of the linear averaging that precedes it. They land near 1e-5 dB,
which is far below any level a measurement is read at.
"""

from __future__ import annotations

import fractions
import re

import msgspec
import numpy as np
import pytest
from numeric_checks import (
    DB_PER_NEPER,
    ROUNDOFF_SAFETY,
    assert_close,
    log_conversion_tol,
    unit_roundoff,
)

import striqt.analysis as sa
from striqt.analysis import testing

FS = 1e6

DURATION = 1e-4
SIZE = round(DURATION * FS)
DETECTOR_PERIOD = fractions.Fraction(1, 100_000)
BIN_SIZE = round(float(DETECTOR_PERIOD) * FS)
BIN_COUNT = SIZE // BIN_SIZE

CYCLIC_PERIOD = 1e-4
CYCLIC_LAGS = round(CYCLIC_PERIOD / float(DETECTOR_PERIOD))


def capture(duration=DURATION, sample_rate=FS) -> sa.specs.Capture:
    return sa.specs.Capture(duration=duration, sample_rate=sample_rate)


def cpts(iq, duration=DURATION, **kws):
    kws.setdefault('detector_period', DETECTOR_PERIOD)
    return sa.measurements.channel_power_time_series(
        iq, capture(duration=duration), **kws
    )


def cyclic(iq, duration=DURATION, **kws):
    kws.setdefault('cyclic_period', CYCLIC_PERIOD)
    kws.setdefault('detector_period', DETECTOR_PERIOD)
    return sa.measurements.cyclic_channel_power(iq, capture(duration=duration), **kws)


def validate_cyclic(capture: sa.specs.Capture, spec: sa.specs.CyclicChannelPower):
    """call the registered validator the way the registry wrapper does"""
    validate = sa.registry[sa.specs.CyclicChannelPower].validate
    assert validate is not None
    return validate(sa.specs.helpers.to_analysis_capture(capture), spec)


def dB_tol(n_terms=1):
    """tolerances in dB for a level averaged over `n_terms` float32 power samples"""
    tol = log_conversion_tol(np.float32, 10, complex_input=True)
    accum = DB_PER_NEPER * ROUNDOFF_SAFETY * n_terms * unit_roundoff(np.float32)
    return {'rtol': tol['rtol'], 'atol': tol['atol'] + accum}


# %% channel_power_time_series


class TestChannelPowerTimeSeries:
    @pytest.mark.parametrize('time', [0.0, 3.7e-5, 9.9e-5], ids='t{:g}'.format)
    def test_impulse_lands_in_its_detector_bin(self, time):
        iq = testing.dirac_delta(DURATION, FS, time=time, power=6.0)
        da = cpts(iq, power_detectors=('peak',))

        peak = da.values[0, 0]
        expected_bin = int(time // float(DETECTOR_PERIOD))
        assert np.argmax(peak) == expected_bin
        assert_close(peak[expected_bin], 6.0, **dB_tol())

        empty = np.delete(peak, expected_bin)
        assert np.isneginf(empty).all()

    def test_time_elapsed_is_exact_multiples_of_the_detector_period(self):
        iq = testing.tone(DURATION, FS, frequency=1e5)
        da = cpts(iq)

        expected = (np.arange(BIN_COUNT) * float(DETECTOR_PERIOD)).astype('float32')
        assert da.sizes['time_elapsed'] == BIN_COUNT
        np.testing.assert_array_equal(da.coords['time_elapsed'].values, expected)

    def test_sawtooth_peak_exceeds_rms_by_the_ramp_ratio(self):
        # a ramp of N samples reaching amplitude A takes the values A*i/N, whose peak
        # power is (A*(N-1)/N)**2 and mean power A**2*(N-1)*(2*N-1)/(6*N**2); the ratio
        # 6*(N-1)/(2*N-1) approaches the continuous-ramp value of 3 as N grows. A whole
        # number of ramps per detector bin makes it exact.
        period = 1e-3
        n = round(period * FS)
        duration = 2 * period
        detector_period = fractions.Fraction(1, 1000)

        iq = testing.sawtooth(duration, FS, period=period, power=0.0)
        da = cpts(iq, duration=duration, detector_period=detector_period)

        ratio_dB = da.sel(power_detector='peak') - da.sel(power_detector='rms')
        expected = 10 * np.log10(6 * (n - 1) / (2 * n - 1))
        assert_close(ratio_dB.values, expected, **dB_tol(n))
        assert abs(expected - 10 * np.log10(3)) < 5e-3

    @pytest.mark.parametrize(
        'detectors',
        [('rms', 'peak'), ('peak', 'rms'), ('max', 'min', 'median', 'mean')],
        ids=['default_order', 'reversed', 'four_named'],
    )
    def test_power_detector_coordinate_follows_the_request(self, detectors):
        iq = testing.tone(DURATION, FS, frequency=1e5)
        da = cpts(iq, power_detectors=detectors)

        assert tuple(da.coords['power_detector'].values) == detectors

    def test_float_quantile_detector_is_rejected(self):
        """`stat_ufunc_from_shorthand` accepts a quantile, but `power_detectors` is
        typed as a tuple of str (unlike `cyclic_statistics`), so the spec refuses one"""
        iq = testing.tone(DURATION, FS, frequency=1e5)
        with pytest.raises(msgspec.ValidationError, match='Expected `str`'):
            cpts(iq, power_detectors=('peak', 0.5))

    @pytest.mark.parametrize(
        ('kwargs', 'duration', 'message'),
        [
            (
                {'detector_period': fractions.Fraction(1, 300_000)},
                DURATION,
                (
                    'detector_period must be a counting-number multiple of the '
                    'sample period'
                ),
            ),
            (
                {},
                10.5 * float(DETECTOR_PERIOD),
                'duration must be a counting-number multiple of detector_period',
            ),
        ],
        ids=['detector_period_3.33_samples', 'duration_10.5_detector_periods'],
    )
    def test_unevenly_tiled_capture_is_rejected(self, kwargs, duration, message):
        """`detector_period` is 10/3 samples at `sample_rate`, then the capture is 10.5
        detector periods long. `iq_to_bin_power` and `axis_to_blocks` reject both once
        they have IQ; the validator rejects them before any is acquired."""
        iq = testing.tone(duration, FS, frequency=1e5)
        with pytest.raises(
            msgspec.ValidationError, match=re.escape(message)
        ) as excinfo:
            cpts(iq, duration=duration, as_xarray=False, **kwargs)

        assert str(excinfo.value).endswith('$.channel_power_time_series')

        validate = sa.registry[sa.specs.ChannelPowerTimeSeries].validate
        assert validate is not None
        with pytest.raises(ValueError, match=re.escape(message)):
            validate(
                sa.specs.helpers.to_analysis_capture(capture(duration=duration)),
                sa.specs.ChannelPowerTimeSeries(**{
                    'detector_period': DETECTOR_PERIOD,
                    **kwargs,
                }),
            )

    @pytest.mark.parametrize('power', [0.0, -13.0, 7.5], ids='{:g}dBm'.format)
    def test_absolute_level_is_the_amplitude_in_dBm(self, power):
        amplitude = np.float32(10 ** (power / 20))
        iq = testing.tone(DURATION, FS, frequency=1e5) * amplitude
        da = cpts(iq)

        assert da.attrs['units'] == 'dBm'
        assert_close(da.values, power, **dB_tol(BIN_SIZE))


# %% cyclic_channel_power


class TestCyclicChannelPower:
    @pytest.mark.parametrize('cycles', [1, 4], ids='cycles{}'.format)
    def test_lag_count_is_independent_of_duration(self, cycles):
        duration = cycles * CYCLIC_PERIOD
        iq = testing.tone(duration, FS, frequency=1e5)
        da = cyclic(iq, duration=duration)

        assert da.sizes['cyclic_lag'] == CYCLIC_LAGS
        expected = (np.arange(CYCLIC_LAGS) * float(DETECTOR_PERIOD)).astype('float32')
        np.testing.assert_array_equal(da.coords['cyclic_lag'].values, expected)

    @pytest.mark.parametrize(
        'statistics',
        [('min', 'mean', 'max'), ('max', 'min'), ('min', 0.5, 'max')],
        ids=['default_order', 'reversed_pair', 'with_quantile'],
    )
    def test_axis_order_and_statistic_coordinate(self, statistics):
        detectors = ('peak', 'rms')
        iq = testing.tone(DURATION, FS, frequency=1e5)
        da = cyclic(iq, power_detectors=detectors, cyclic_statistics=statistics)

        assert da.shape == (1, len(detectors), len(statistics), CYCLIC_LAGS)
        assert tuple(da.coords['power_detector'].values) == detectors
        assert tuple(da.coords['cyclic_statistic'].values) == statistics

    @pytest.mark.parametrize(
        ('kwargs', 'duration', 'message'),
        [
            (
                {},
                1.5 * CYCLIC_PERIOD,
                'duration must be a counting-number multiple of cyclic_period',
            ),
            (
                {'cyclic_period': 2.5 * float(DETECTOR_PERIOD)},
                DURATION,
                'cyclic_period must be a counting-number multiple of detector_period',
            ),
            (
                {'detector_period': fractions.Fraction(1, 300_000)},
                DURATION,
                'detector_period must be a counting-number multiple of the sample period',
            ),
        ],
        ids=[
            'duration_1.5_cycles',
            'cyclic_period_2.5_detector_periods',
            'detector_period_3.33_samples',
        ],
    )
    def test_unevenly_nested_periods_are_rejected(self, kwargs, duration, message):
        """the three couplings the docstring states: `duration` is 1.5 cycles,
        `cyclic_period` is 2.5 detector periods, and `detector_period` is 10/3 samples
        at `sample_rate`. Each is caught before any IQ is touched, so the error carries
        the measurement's field path."""
        iq = testing.tone(duration, FS, frequency=1e5)
        with pytest.raises(
            msgspec.ValidationError, match=re.escape(message)
        ) as excinfo:
            cyclic(iq, duration=duration, as_xarray=False, **kwargs)

        assert str(excinfo.value).endswith('$.cyclic_channel_power')

        spec_kwargs = {
            'cyclic_period': CYCLIC_PERIOD,
            'detector_period': DETECTOR_PERIOD,
            **kwargs,
        }
        with pytest.raises(ValueError, match=re.escape(message)):
            validate_cyclic(
                capture(duration=duration),
                sa.specs.CyclicChannelPower(**spec_kwargs),
            )

    def test_validator_returns_the_lag_count(self):
        lag_count = validate_cyclic(
            capture(),
            sa.specs.CyclicChannelPower(
                cyclic_period=CYCLIC_PERIOD, detector_period=DETECTOR_PERIOD
            ),
        )

        assert lag_count == CYCLIC_LAGS

    def test_statistics_are_taken_in_linear_power(self):
        """cycles that alternate between 0 dBm and 20 dBm average to 17.03 dBm in
        linear power but to 10 dBm in dB, so the mean statistic distinguishes them"""
        duration = 2 * CYCLIC_PERIOD
        cycle_size = round(CYCLIC_PERIOD * FS)
        amplitudes = np.repeat(np.float32([1.0, 10.0]), cycle_size)
        iq = testing.tone(duration, FS, frequency=1e5) * amplitudes

        da = cyclic(iq, duration=duration)

        tol = dB_tol(BIN_SIZE + 2)
        assert_close(da.sel(cyclic_statistic='min').values, 0.0, **tol)
        assert_close(da.sel(cyclic_statistic='max').values, 20.0, **tol)

        linear_mean_dB = 10 * np.log10((1.0 + 100.0) / 2)
        assert_close(da.sel(cyclic_statistic='mean').values, linear_mean_dB, **tol)
        assert abs(linear_mean_dB - (0.0 + 20.0) / 2) > 7


# %% iq_waveform


class TestIqWaveform:
    @staticmethod
    def waveform(duration=DURATION):
        return testing.single_tone(duration, FS, frequency_offset=1e5, snr=10)

    @pytest.mark.parametrize(
        ('start_time_sec', 'stop_time_sec', 'start', 'stop'),
        [
            (None, None, 0, SIZE),
            (2e-5, 6e-5, 20, 60),
            (None, 2.5e-5, 0, 25),
            (9.6e-5, None, 96, SIZE),
            (None, 2 * DURATION, 0, SIZE),
            (None, DURATION, 0, SIZE),
        ],
        ids=[
            'unbounded',
            'both_bounds',
            'stop_only',
            'start_only',
            'stop_past_end',
            'stop_at_end',
        ],
    )
    def test_time_bounds_slice_by_sample_index(
        self, start_time_sec, stop_time_sec, start, stop
    ):
        iq = self.waveform()
        da = sa.measurements.iq_waveform(
            iq, capture(), start_time_sec=start_time_sec, stop_time_sec=stop_time_sec
        )

        assert da.sizes['iq_index'] == stop - start
        assert np.array_equal(da.values, iq[:, start:stop])
        indices = da.coords['iq_index']
        assert indices.dtype == np.dtype('uint64')
        np.testing.assert_array_equal(indices.values, np.arange(start, stop))

    @pytest.mark.parametrize(
        'stop_time_sec', [None, 3 * DURATION], ids=['start_only', 'both_bounds']
    )
    def test_bounds_past_the_capture_end_are_empty(self, stop_time_sec):
        iq = self.waveform()
        da = sa.measurements.iq_waveform(
            iq, capture(), start_time_sec=2 * DURATION, stop_time_sec=stop_time_sec
        )

        assert da.sizes['iq_index'] == 0
        assert da.coords['iq_index'].size == 0

    def test_reversed_window_is_empty(self):
        """current behaviour, not a validated contract: a window that ends before it
        starts yields an empty result rather than an error"""
        iq = self.waveform()
        da = sa.measurements.iq_waveform(
            iq, capture(), start_time_sec=6e-5, stop_time_sec=2e-5
        )

        assert da.sizes['iq_index'] == 0
        assert da.coords['iq_index'].size == 0
