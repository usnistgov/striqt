"""striqt.waveform.lib.power.detect: property tests of the binned power detectors,
cyclic statistics, the sample CCDF, and the roundoff models, including numpy-vs-cupy
agreement within ulp budgets.
"""

from __future__ import annotations

import functools

import numpy as np
import pytest
from conftest import (
    as_xp,
    iq_waveforms,
)
from hypothesis import given, settings, strategies as st
from hypothesis.extra.numpy import arrays
from numeric_checks import (
    assert_close,
    blocks,
    func_id,
    numpy_and_cupy,
    reference_power,
)
from numpy.testing import assert_array_equal

from striqt.waveform.fourier import peak_factor
from striqt.waveform.lib.arrays import accum_rtol, float_dtype_like
from striqt.waveform.lib.power import (
    bin_power_rms,
    bin_power_rtol,
    iq_to_bin_power,
    iq_to_cyclic_power,
    sample_ccdf,
    stat_rtol,
    stat_ufunc_from_shorthand,
)

NAMED_STATS = {
    'min': np.min,
    'max': np.max,
    'peak': np.max,
    'mean': np.mean,
    'rms': np.mean,
    'median': np.median,
}
BIN_STATS = {**NAMED_STATS, 0.9: functools.partial(np.quantile, q=0.9)}


class TestStatUfuncFromShorthand:
    DATA = np.random.default_rng(0).normal(size=(6, 5)).astype(np.float32)
    # a reduction over the 5 samples of an axis rounds at most 5 times per backend
    RTOL = accum_rtol(np.float32, 5, n_impl=2)

    def _check(self, xp, kind, axis, expected):
        ufunc = stat_ufunc_from_shorthand(kind, xp=xp, axis=axis)
        result = ufunc(as_xp(xp, self.DATA))
        assert_close(result, expected, rtol=self.RTOL)

    @pytest.mark.parametrize('kind', sorted(NAMED_STATS))
    @pytest.mark.parametrize('axis', [0, 1])
    def test_named_statistics(self, xp, kind, axis):
        self._check(xp, kind, axis, NAMED_STATS[kind](self.DATA, axis=axis))

    @pytest.mark.parametrize('q', [0.0, 0.25, 0.9, 1.0])
    def test_quantile(self, xp, q):
        self._check(xp, q, 1, np.quantile(self.DATA, q=q, axis=1))

    def test_callable(self):
        ufunc = stat_ufunc_from_shorthand(np.std, axis=0)
        assert_array_equal(ufunc(self.DATA), np.std(self.DATA, axis=0))

    @pytest.mark.parametrize(
        'kind, match',
        [
            ('average', 'kind argument'),
            ('rms2', 'kind argument'),
            ('', 'kind argument'),
            (('mean',), 'invalid statistic'),
            (1.5, r'quantile 1\.5 is outside'),
            (-0.1, r'quantile -0\.1 is outside'),
        ],
        ids=[
            'average',
            'rms2',
            'empty',
            'tuple',
            'quantile_above_1',
            'quantile_below_0',
        ],
    )
    def test_invalid_kind_raises(self, kind, match):
        with pytest.raises(ValueError, match=match):
            stat_ufunc_from_shorthand(kind)


class TestRoundoffModels:
    """the tolerance models of the binned power detectors"""

    @pytest.mark.parametrize('func', [stat_rtol, bin_power_rtol], ids=func_id)
    def test_unknown_statistic_raises(self, func):
        with pytest.raises(ValueError, match='bogus'):
            func(np.float32, kind='bogus')


def bin_power_reference(iq, size, kind, axis):
    axis %= iq.ndim
    power = reference_power(iq)
    return BIN_STATS[kind](blocks(power, size, axis), axis=axis + 1)


BIN_SIZES = [1, 5, 16]
for_each_bin_size = pytest.mark.parametrize('size', BIN_SIZES)
for_each_bin_kind = pytest.mark.parametrize('kind', sorted(BIN_STATS, key=str))


@st.composite
def cyclic_power_cases(draw, channels=(1, 2)):
    """draw (iq, bin size, bins per cycle, cycle count) arguments for
    `iq_to_cyclic_power`."""
    size = draw(st.sampled_from([2, 4, 8]))
    bins_per_cycle = draw(st.sampled_from([2, 3, 5]))
    n_cycles = draw(st.integers(min_value=1, max_value=6))
    n = size * bins_per_cycle * n_cycles
    iq = draw(
        iq_waveforms(min_size=n, max_size=n, channels=draw(st.sampled_from(channels)))
    )
    return iq, size, bins_per_cycle, n_cycles


def bin_power_bound(dtype, size, outputs, n_impl=1, kind='mean'):
    """return the worst-case rtol over `outputs` binned statistics of `size` samples:
    the deterministic part plus the peak of the rms accumulation model."""
    peak = max(peak_factor(outputs), 1.0)
    return bin_power_rtol(dtype, n_impl, kind) + peak * bin_power_rms(
        dtype, size, n_impl, kind
    )


class TestIqToBinPower:
    @pytest.mark.namespaces('numpy', 'cupy')
    @pytest.mark.parametrize('kind', ['mean', 'max'])
    @pytest.mark.parametrize('size', [2**14, 2**17])
    def test_long_bin_statistics_stay_within_the_reduction_model(self, xp, size, kind):
        """a mean over 2**14 to 2**17 float32 samples stays within `bin_power_bound`,
        below the ``n*u`` bound of recursive summation, and a max over the bin is
        exact"""
        rng = np.random.default_rng(size)
        iq = rng.standard_normal((4, size)) + 1j * rng.standard_normal((4, size))
        iq = (iq / np.sqrt(2)).astype(np.complex64)
        power = reference_power(iq)
        expected = power.mean(axis=1) if kind == 'mean' else power.max(axis=1)

        result = iq_to_bin_power(xp.asarray(iq), Ts=1, Tbin=size, kind=kind, axis=1)

        assert result.dtype == np.float32
        rtol = bin_power_bound(np.float32, size, outputs=result.size, kind=kind)
        assert_close(result[:, 0], expected, rtol=rtol)

    def test_quantile_of_a_long_heavy_tailed_bin_is_exact_on_every_numpy(self):
        """numpy < 2.3 casts a scalar q to float32 (numpy.lib._function_base_impl,
        quantile). The cast shifts the virtual index (n-1)*q by about n*2**-25 samples
        before the interpolation between the 2 neighboring order statistics. On a
        heavy-tailed bin of 2**20 samples the error is 124 u against a budget of
        28 u."""
        n, q = 2**20, 0.999
        rng = np.random.default_rng(0)
        power = np.exp(3 * rng.standard_normal((2, n)))
        phase = rng.uniform(0, 2 * np.pi, size=(2, n))
        iq = (np.sqrt(power) * np.exp(1j * phase)).astype(np.complex64)

        result = iq_to_bin_power(iq, Ts=1, Tbin=n, kind=q, axis=1)

        assert result.dtype == np.float32
        expected = np.quantile(reference_power(iq), q, axis=1, keepdims=True)
        rtol = bin_power_bound(np.float32, n, outputs=result.size, kind=q)
        assert_close(result, expected, rtol=rtol)

    @for_each_bin_size
    # 30 examples per bin size: the 18-item grid over kind and channels at 100
    # examples each took 2 s
    @settings(max_examples=30)
    @given(
        kind=st.sampled_from(['mean', 'max', 0.9]),
        channels=st.sampled_from([None, 3]),
        negative_axis=st.booleans(),
        data=st.data(),
    )
    def test_matches_reference(self, size, kind, channels, negative_axis, data):
        """each bin is the statistic of ``|iq|**2`` over `size` samples, for a 1-D
        waveform (axis 0) and a (channel, sample) array (axis 1 or -1)"""
        waveforms = iq_waveforms(
            min_size=size, max_size=32 * size, multiple_of=size, channels=channels
        )
        iq = data.draw(waveforms)
        axis = -1 if negative_axis else iq.ndim - 1
        Ts = 1e-6

        result = iq_to_bin_power(iq, Ts, size * Ts, kind=kind, axis=axis)

        assert result.dtype == float_dtype_like(iq)
        expected = bin_power_reference(iq, size, kind, axis)
        assert result.shape == expected.shape
        rtol = bin_power_bound(float_dtype_like(iq), size, result.size, kind=kind)
        assert_close(result, expected, rtol=rtol)

    @pytest.mark.parametrize('size', BIN_SIZES[1:])
    @given(data=st.data(), remainder=st.integers(min_value=1, max_value=4))
    def test_truncate(self, size, data, remainder):
        iq = data.draw(
            iq_waveforms(min_size=size, max_size=16 * size, multiple_of=size)
        )
        iq = np.concatenate([iq, iq[:remainder]]) if remainder < size else iq[:-1]
        Ts = 1e-6

        with pytest.raises(ValueError):
            iq_to_bin_power(iq, Ts, size * Ts, truncate=False)

        result = iq_to_bin_power(iq, Ts, size * Ts, truncate=True)
        expected = bin_power_reference(iq, size, 'mean', 0)
        assert result.shape == (iq.shape[0] // size,)
        rtol = bin_power_bound(float_dtype_like(iq), size, result.size)
        assert_close(result, expected, rtol=rtol)

    @given(ratio=st.floats(min_value=1.1, max_value=9.9).filter(lambda r: r % 1 > 0.01))
    def test_bin_period_must_be_multiple(self, ratio):
        iq = np.ones(64, dtype=np.complex64)
        with pytest.raises(ValueError, match='multiple'):
            iq_to_bin_power(iq, 1.0, ratio)

    def test_truncate_allows_fractional_bin_period(self):
        """``truncate=True`` rounds a fractional Tbin/Ts to whole samples"""
        iq = np.ones(64, dtype=np.complex64)
        result = iq_to_bin_power(iq, 1.0, 4.4, truncate=True)
        assert result.shape == (64 // 4,)
        assert_array_equal(result, 1.0)

    @pytest.mark.parametrize('size', BIN_SIZES[1:])
    @given(data=st.data())
    def test_randomize(self, size, data):
        """every random bin is the mean power of some `size` contiguous samples, i.e.,
        a value of the sliding-window mean of |iq|**2"""
        iq = data.draw(
            iq_waveforms(min_size=4 * size, max_size=32 * size, multiple_of=size)
        )
        Ts = 1e-6

        result = iq_to_bin_power(iq, Ts, size * Ts, randomize=True)

        assert result.shape == (iq.shape[0] // size,)
        assert result.dtype == float_dtype_like(iq)
        power = reference_power(iq)
        windows = np.lib.stride_tricks.sliding_window_view(power, size).mean(axis=1)
        matches = np.isclose(
            result[:, np.newaxis],
            windows[np.newaxis, :],
            rtol=bin_power_bound(float_dtype_like(iq), size, result.size),
        )
        assert np.all(matches.any(axis=1))

    def test_randomize_requires_axis_0(self):
        iq = np.ones((2, 64), dtype=np.complex64)
        with pytest.raises(ValueError, match='axis=0'):
            iq_to_bin_power(iq, 1.0, 4.0, randomize=True, axis=1)


class TestIqToCyclicPower:
    """`iq_to_cyclic_power` on (channel, sample) input with ``axis=1``.

    The expectations follow doc/papers/cyclic_power.tex: the detector output is
    folded into a (cycle, lag) matrix with cyclic_period / detector_period lags per
    cycle, and each cyclic statistic reduces the cycle axis.
    """

    DETECTORS = ('rms', 'peak')
    CYCLE_STATS = ('min', 'mean', 'max', 0.9)

    @staticmethod
    def evaluate(iq, size, lags, **kws):
        """run `iq_to_cyclic_power` with unit sample period, `size` samples per
        detector bin, and `lags` bins per cycle, on axis 1 of a 2-D `iq` unless
        `axis` is given."""
        kws = {'detectors': TestIqToCyclicPower.DETECTORS, **kws}
        kws = {'cycle_stats': TestIqToCyclicPower.CYCLE_STATS, **kws}
        if iq.ndim > 1:
            kws.setdefault('axis', 1)
        return iq_to_cyclic_power(iq, 1.0, size, size * lags, **kws)

    @given(case=cyclic_power_cases(channels=(None, 1, 2)), negative_axis=st.booleans())
    def test_matches_reference(self, case, negative_axis):
        """each statistic is the column-wise reduction of the binned power of
        `bin_power_reference` reshaped to (cycle, lag), including a quantile
        statistic, for a 1-D power time series on the default axis 0 and for a
        (channel, sample) array on axis 1 or -1"""
        iq, size, bins_per_cycle, n_cycles = case
        if iq.ndim == 1 and not negative_axis:
            axis_kws = {}
        else:
            axis_kws = {'axis': -1 if negative_axis else 1}

        result = self.evaluate(iq, size, bins_per_cycle, **axis_kws)

        assert set(result) == set(self.DETECTORS)
        dtype = float_dtype_like(iq)
        outputs = iq.shape[-1] // size
        for detector in self.DETECTORS:
            assert set(result[detector]) == set(self.CYCLE_STATS)
            binned = bin_power_reference(iq, size, detector, iq.ndim - 1)
            by_cycle = binned.reshape(iq.shape[:-1] + (n_cycles, bins_per_cycle))
            rtol = bin_power_bound(dtype, size, outputs, kind=detector)
            rtol += accum_rtol(dtype, n_cycles)
            for stat in self.CYCLE_STATS:
                value = result[detector][stat]
                assert value.shape == iq.shape[:-1] + (bins_per_cycle,)
                assert value.dtype == dtype
                expected = BIN_STATS[stat](by_cycle, axis=-2)
                assert_close(value, expected, rtol=rtol)

    @given(case=cyclic_power_cases())
    def test_lag_average_of_mean_rms_is_the_waveform_power(self, case):
        """the fold keeps every detector sample (K = L * M), so averaging the mean rms
        trace over lags recovers the mean power of the waveform"""
        iq, size, bins_per_cycle, _ = case

        kws = {'detectors': ('rms',), 'cycle_stats': ('mean',)}
        result = self.evaluate(iq, size, bins_per_cycle, **kws)

        expected = reference_power(iq, axis=1)
        rtol = accum_rtol(float_dtype_like(iq), iq.shape[1])
        assert_close(result['rms']['mean'].mean(axis=1), expected, rtol=rtol)

    @pytest.mark.parametrize('n_cycles', [1, 3, 8], ids=lambda n: f'{n}_cycles')
    @pytest.mark.parametrize(
        'periods_per_cycle', [1, 2, 5], ids=lambda q: f'{q}_periods_per_cycle'
    )
    def test_commensurate_signal_has_no_spread(
        self, periods_per_cycle, n_cycles, subtests
    ):
        """a signal with a period that divides `cyclic_period` repeats identically in
        every cycle, so every statistic returns the one-cycle power envelope. The lag
        count is fixed by cyclic_period / detector_period rather than by n_cycles."""
        size, lags = 4, 10
        rng = np.random.default_rng(0)
        envelope = rng.uniform(0.5, 2.0, size=(2, lags // periods_per_cycle))
        amplitude = np.tile(envelope, (1, periods_per_cycle * n_cycles))
        iq = np.repeat(amplitude, size, axis=1).astype(np.complex64)

        result = self.evaluate(iq, size, lags)

        # constant amplitude within each bin makes the rms and peak detectors agree
        expected = np.tile(envelope**2, (1, periods_per_cycle))
        rtol = accum_rtol(np.float32, size)
        for detector in self.DETECTORS:
            for stat in self.CYCLE_STATS:
                with subtests.test(detector=detector, stat=stat):
                    assert result[detector][stat].shape == (2, lags)
                    assert_close(result[detector][stat], expected, rtol=rtol)

    def test_pulse_period_selectivity(self):
        """pulses with a period that divides the cycle stay at fixed lags with zero
        spread. Pulses with a period coprime to the lag count visit every lag: the max
        trace is the pulse power everywhere, the min trace is zero, and the mean is
        the pulse power divided by the period."""
        size, lags, n_cycles, pulse = 2, 8, 12, 4.0

        def pulse_train(period):
            power = np.zeros(lags * n_cycles)
            power[::period] = pulse
            iq = np.repeat(np.sqrt(power), size).astype(np.complex64)
            kws = {'detectors': ('rms',), 'cycle_stats': ('min', 'mean', 'max')}
            return self.evaluate(iq[np.newaxis], size, lags, **kws)['rms']

        aligned = pulse_train(4)
        expected = np.zeros((1, lags))
        expected[0, ::4] = pulse
        for stat in ('min', 'mean', 'max'):
            assert_array_equal(aligned[stat], expected)

        smeared = pulse_train(3)
        assert_array_equal(smeared['max'], np.full((1, lags), pulse))
        assert_array_equal(smeared['min'], np.zeros((1, lags)))
        rtol = accum_rtol(np.float32, n_cycles)
        assert_close(smeared['mean'], np.full((1, lags), pulse / 3), rtol=rtol)

    @pytest.mark.parametrize(
        'drift_per_cycle, bleeds', [(1, False), (2, True)], ids=['within_bin', 'bleeds']
    )
    def test_cycle_period_mismatch_bleeds_into_the_next_lag(
        self, drift_per_cycle, bleeds
    ):
        """a period mismatch drifts a feature by (n_cycles - 1) * drift_per_cycle
        samples over the capture. The peak trace keeps the pulse in one lag with no
        spread while the drift is below the bin size. Once the drift reaches the bin
        size, the pulse leaks into the next lag."""
        size, lags, n_cycles = 8, 4, 5
        iq = np.zeros((1, size * lags * n_cycles), dtype=np.complex64)
        for cycle in range(n_cycles):
            iq[0, cycle * size * lags + cycle * drift_per_cycle] = 1.0
        assert ((n_cycles - 1) * drift_per_cycle >= size) == bleeds

        kws = {'detectors': ('peak',), 'cycle_stats': ('min', 'max')}
        peak = self.evaluate(iq, size, lags, **kws)['peak']

        assert peak['max'][0].tolist() == [1.0, 1.0 if bleeds else 0.0, 0.0, 0.0]
        assert peak['min'][0].tolist() == [0.0 if bleeds else 1.0, 0.0, 0.0, 0.0]

    @pytest.mark.parametrize(
        'n, kws, match',
        [
            (64, {'detectors': None}, 'detectors'),
            (64, {'cyclic_period': 10.0}, 'cyclic period'),
            (4 * 7, {}, 'truncate'),
        ],
        ids=['detectors_none', 'cyclic_period_not_multiple', 'misaligned_no_truncate'],
    )
    def test_argument_errors(self, n, kws, match):
        iq = np.ones((1, n), dtype=np.complex64)
        with pytest.raises(ValueError, match=match):
            iq_to_cyclic_power(
                iq, 1.0, 4.0, **{'cyclic_period': 16.0, 'axis': 1, **kws}
            )

    def test_truncate_drops_trailing_bins_of_the_time_axis(self, subtests):
        """a 2-port capture at 107.52 MS/s with 1/28 ms detector bins (3840 samples)
        contains 25 bins. ``truncate=True`` folds 20 bins into 2 cycles of 10 lags and
        drops the last 5. A unit-envelope input has power exactly 1 in every bin, so
        each statistic is exactly 1 at every lag."""
        Ts = 1 / 107.52e6
        detector_period = 1 / 28000
        kws = {'cyclic_period': 10 * detector_period, 'axis': 1}
        iq = np.ones((2, 96000), dtype=np.complex64)

        with pytest.raises(ValueError, match='truncate'):
            iq_to_cyclic_power(iq, Ts, detector_period, truncate=False, **kws)

        result = iq_to_cyclic_power(iq, Ts, detector_period, truncate=True, **kws)
        for detector, stats in result.items():
            for stat, value in stats.items():
                with subtests.test(detector=detector, stat=stat):
                    assert value.shape == (2, 10)
                    assert_array_equal(value, 1.0)


def ccdf_reference(a, edges, density):
    counts = np.array([(a > e).sum() for e in edges])
    return counts / a.size if density else counts


@st.composite
def ccdf_cases(draw):
    """draw (samples, sorted edges, density) arguments for `sample_ccdf`."""
    sizes = st.integers(min_value=1, max_value=200)
    sample_values = st.floats(min_value=-100, max_value=100, allow_nan=False)
    a = draw(arrays(dtype=np.float64, shape=sizes, elements=sample_values))
    edge_values = st.floats(min_value=-120, max_value=120, allow_nan=False)
    edges = draw(st.lists(edge_values, min_size=1, max_size=20, unique=True))
    return a, np.asarray(sorted(edges)), draw(st.booleans())


class TestSampleCcdf:
    @given(case=ccdf_cases())
    def test_matches_brute_force(self, case):
        a, edges, density = case
        result = sample_ccdf(a, edges, density=density)
        expected = ccdf_reference(a, edges, density)
        assert result.shape == edges.shape
        if density:
            assert result.dtype == np.float64
            rtol = accum_rtol(np.float64, 1)
            assert_close(result, expected, rtol=rtol)
        else:
            assert_array_equal(result, expected)

    def test_samples_equal_to_edge_are_not_counted(self):
        a = np.array([1.0, 1.0, 2.0])
        edges = np.array([0.0, 1.0, 2.0])
        assert_array_equal(sample_ccdf(a, edges, density=False), [3, 1, 0])


class TestNumpyCupyCrossComparison:
    """the numpy and cupy results of each detector compared against each other.

    Tolerances come from the ulp budgets in `striqt.waveform.lib.power`.
    """

    @for_each_bin_size
    @for_each_bin_kind
    @given(data=st.data())
    def test_iq_to_bin_power(self, cupy_available, size, kind, data):
        waveforms = iq_waveforms(
            min_size=size, max_size=32 * size, multiple_of=size, channels=2
        )
        iq = data.draw(waveforms)
        Ts = 1e-6
        result_np, result_cp = numpy_and_cupy(
            cupy_available, iq_to_bin_power, iq, Ts, size * Ts, kind=kind, axis=1
        )

        assert result_cp.dtype == result_np.dtype
        rtol = bin_power_bound(float_dtype_like(iq), size, result_np.size, 2, kind)
        assert_close(result_cp, result_np, rtol=rtol)

    @given(case=cyclic_power_cases(channels=(2,)))
    def test_iq_to_cyclic_power(self, cupy_available, case):
        iq, size, bins_per_cycle, n_cycles = case
        Ts = 1e-6
        detector_period = size * Ts
        cyclic_period = size * bins_per_cycle * Ts
        kws = {'detector_period': detector_period, 'cyclic_period': cyclic_period}

        result_np = iq_to_cyclic_power(iq, Ts, axis=1, **kws)
        result_cp = iq_to_cyclic_power(cupy_available.asarray(iq), Ts, axis=1, **kws)

        dtype = float_dtype_like(iq)
        outputs = iq.shape[1] // size
        rtol = bin_power_bound(dtype, size, outputs, 2) + accum_rtol(dtype, n_cycles, 2)
        for detector, stats in result_np.items():
            for stat, value in stats.items():
                assert_close(result_cp[detector][stat], value, rtol=rtol)

    @given(case=ccdf_cases())
    def test_sample_ccdf(self, cupy_available, case):
        """counting is exact, so the backends agree exactly"""
        a, edges, density = case
        result_np, result_cp = numpy_and_cupy(
            cupy_available, sample_ccdf, a, edges, density=density
        )
        assert_array_equal(result_cp, result_np)
