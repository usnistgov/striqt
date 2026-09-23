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

Tolerances are the `sa.specs.Tolerance` each measurement registers with
`tolerance=`, evaluated the way `sa.registry.tolerances` does: the float32 dB
conversion budget plus the roundoff of squaring, averaging and storing the level. They
land near 2e-5 dB, which is far below any level a measurement is read at.
"""

from __future__ import annotations

import fractions
import re

import msgspec
import numpy as np
import pytest
from numeric_checks import assert_close

import striqt.analysis as sa
import striqt.waveform as sw
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


def tolerance(
    spec: sa.specs.Analysis, duration=DURATION, input_error=0.0
) -> sa.specs.Tolerance:
    """call the registered tolerance function the way `registry.tolerances` does"""
    func = sa.registry[type(spec)].tolerance
    assert func is not None
    return func(
        sa.specs.helpers.to_analysis_capture(capture(duration=duration)),
        spec,
        input_error=input_error,
    )


CPTS_SPEC = sa.specs.ChannelPowerTimeSeries(detector_period=DETECTOR_PERIOD)
CYCLIC_SPEC = sa.specs.CyclicChannelPower(
    cyclic_period=CYCLIC_PERIOD, detector_period=DETECTOR_PERIOD
)


# %% channel_power_time_series


class TestChannelPowerTimeSeries:
    @pytest.mark.parametrize('time', [0.0, 3.7e-5, 9.9e-5], ids='t{:g}'.format)
    def test_impulse_lands_in_its_detector_bin(self, time):
        iq = testing.dirac_delta(DURATION, FS, time=time, power=6.0)
        da = cpts(iq, power_detectors=('peak',))

        peak = da.values[0, 0]
        expected_bin = int(time // float(DETECTOR_PERIOD))
        assert np.argmax(peak) == expected_bin
        tol = tolerance(CPTS_SPEC.replace(power_detectors=('peak',)))
        assert_close(peak[expected_bin], 6.0, rtol=tol.rtol, atol=tol.on_peak.peak)

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
        # the ratio subtracts two levels that each carry the registered budget
        tol = tolerance(
            CPTS_SPEC.replace(detector_period=detector_period), duration=duration
        )
        assert_close(
            ratio_dB.values, expected, rtol=tol.rtol, atol=2 * tol.on_peak.peak
        )
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

        assert str(excinfo.value).startswith('$.channel_power_time_series: ')

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
        tol = tolerance(CPTS_SPEC)
        assert_close(da.values, power, rtol=tol.rtol, atol=tol.on_peak.peak)

    def test_tolerance_accumulates_over_the_detector_bin_not_the_capture(self):
        """each level averages `detector_period * sample_rate` power samples, so the
        exact-input budget grows with the detector period and, at a fixed detector
        period, is the same for a capture of any duration"""
        base = tolerance(CPTS_SPEC)
        longer_bin = tolerance(CPTS_SPEC.replace(detector_period=5 * DETECTOR_PERIOD))
        longer_capture = tolerance(CPTS_SPEC, duration=10 * DURATION)

        assert longer_bin.on_peak.peak > base.on_peak.peak
        assert longer_capture == base


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

        assert str(excinfo.value).startswith('$.cyclic_channel_power: ')

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

        tol = tolerance(CYCLIC_SPEC, duration=duration)
        kws = {'rtol': tol.rtol, 'atol': tol.on_peak.peak}
        assert_close(da.sel(cyclic_statistic='min').values, 0.0, **kws)
        assert_close(da.sel(cyclic_statistic='max').values, 20.0, **kws)

        linear_mean_dB = 10 * np.log10((1.0 + 100.0) / 2)
        assert_close(da.sel(cyclic_statistic='mean').values, linear_mean_dB, **kws)
        assert abs(linear_mean_dB - (0.0 + 20.0) / 2) > 7

    def test_tolerance_accumulates_over_the_cycles_in_the_capture(self):
        """a cyclic statistic reduces `duration / cyclic_period` detector samples on
        top of the detector's own averaging, so unlike `channel_power_time_series` the
        exact-input budget grows with the capture duration"""
        one_cycle = tolerance(CYCLIC_SPEC)
        many_cycles = tolerance(CYCLIC_SPEC, duration=10 * CYCLIC_PERIOD)

        assert many_cycles.on_peak.peak > one_cycle.on_peak.peak
        assert one_cycle.on_peak.peak > tolerance(CPTS_SPEC).on_peak.peak


# %% registered tolerances

POWER_SPECS = [CPTS_SPEC, CYCLIC_SPEC]
POWER_IDS = [type(spec).__name__ for spec in POWER_SPECS]


@pytest.mark.parametrize('spec', POWER_SPECS, ids=POWER_IDS)
def test_registered_tolerance_is_a_dB_budget_that_grows_with_input_error(spec):
    """`registry.tolerances` finds the budget by measurement name; with exact input
    it is a pure roundoff bound with no floor, and an amplitude error in the IQ adds
    a peak term over the output and a floor below which elements go unchecked"""
    name = sa.registry[type(spec)].name
    group = sa.registry.tospec()(**{name: spec})

    exact = sa.registry.tolerances(capture(), group)[name]
    assert exact == tolerance(spec)
    assert isinstance(exact, sa.specs.Tolerance)
    assert exact.units == 'dB'
    assert exact.on_peak.peak >= exact.on_peak.rms > 0
    assert exact.off_peak_dBc is None

    noisy = sa.registry.tolerances(capture(), group, input_error=1e-4)[name]
    assert noisy.on_peak.peak > noisy.on_peak.rms > exact.on_peak.rms
    assert noisy.off_peak_dBc is not None and noisy.off_peak_dBc.peak < 0

    noisier = sa.registry.tolerances(capture(), group, input_error=1e-3)[name]
    assert noisier.on_peak.rms > noisy.on_peak.rms
    assert noisier.on_peak.peak > noisy.on_peak.peak


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

    def test_tolerance_accumulates_only_for_the_averaging_detectors(self):
        """peak and min select one sample exactly, so their budget does not grow with
        the detector period the way the rms mean's does"""
        rms = tolerance(CPTS_SPEC.replace(power_detectors=('rms',)))
        peak = tolerance(CPTS_SPEC.replace(power_detectors=('peak',)))
        longer = CPTS_SPEC.replace(detector_period=5 * DETECTOR_PERIOD)
        assert peak.on_peak.peak < rms.on_peak.peak
        assert tolerance(longer.replace(power_detectors=('peak',))) == peak

    def test_tolerance_is_the_input_error_on_the_envelope_level(self):
        """the slice adds no error of its own, so exact IQ passes through with a zero
        budget, and an rms amplitude error `r` in the IQ reads on the envelope level
        ``20*log10|iq|`` as an rms of ``20*log10(1 + r)`` dB with a larger peak over
        the slice"""
        exact = tolerance(sa.specs.IQWaveform())
        assert exact == sa.specs.Tolerance(
            units='dB', rtol=0.0, on_peak=sa.specs.ErrorBound(rms=0.0, peak=0.0)
        )

        r = 1e-4
        noisy = tolerance(sa.specs.IQWaveform(), input_error=r)
        assert noisy.units == 'dB'
        assert noisy.rtol == exact.rtol
        assert noisy.on_peak.rms == pytest.approx(sw.level_tolerance_dB(r))
        assert noisy.on_peak.peak > noisy.on_peak.rms
        assert noisy.off_peak_dBc is not None and noisy.off_peak_dBc.peak < 0

        shorter = tolerance(sa.specs.IQWaveform(stop_time_sec=1e-5), input_error=r)
        assert shorter.on_peak.rms == noisy.on_peak.rms
        assert shorter.on_peak.peak < noisy.on_peak.peak
