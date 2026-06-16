"""Tests for dB conversion functions in striqt.waveform.lib.power_analysis"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from striqt.waveform.lib.power_analysis import (
    dBlinmean,
    dBlinsum,
    dBtopow,
    envtodB,
    envtopow,
    powtodB,
    unit_dB_to_linear,
    unit_linear_to_dB,
    unit_dB_to_wave,
    unit_wave_to_dB,
    unit_wave_to_linear,
)

# Import to_numpy helper from conftest
from conftest import to_numpy


def make_array(xp, data, dtype=None):
    """Create an array using the appropriate method for each namespace.

    For dask, uses from_array() to properly wrap numpy arrays.
    For numpy/cupy, uses array() directly.
    """
    np_arr = np.array(data, dtype=dtype)
    if hasattr(xp, 'from_array'):
        # dask.array
        return xp.from_array(np_arr, chunks=-1)
    else:
        # numpy, cupy
        return xp.asarray(np_arr)


class TestUnitConversions:
    """Tests for unit string conversion functions."""

    @pytest.mark.parametrize(
        'dB_unit,linear_unit',
        [
            ('dBm', 'mW'),
            ('dBW', 'W'),
            ('dB', 'unitless'),
            ('dBm/Hz', 'mW/Hz'),
            ('dBW/Hz', 'W/Hz'),
        ],
    )
    def test_unit_dB_to_linear(self, dB_unit: str, linear_unit: str):
        assert unit_dB_to_linear(dB_unit) == linear_unit

    @pytest.mark.parametrize(
        'linear_unit,dB_unit',
        [
            ('mW', 'dBm'),
            ('W', 'dBW'),
            ('unitless', 'dB'),
            ('mW/Hz', 'dBm/Hz'),
            ('W/Hz', 'dBW/Hz'),
        ],
    )
    def test_unit_linear_to_dB(self, linear_unit: str, dB_unit: str):
        assert unit_linear_to_dB(linear_unit) == dB_unit

    @pytest.mark.parametrize(
        'dB_unit,wave_unit',
        [
            ('dBm', '√mW'),
            ('dBW', '√W'),
            ('dB', '√unitless'),
        ],
    )
    def test_unit_dB_to_wave(self, dB_unit: str, wave_unit: str):
        assert unit_dB_to_wave(dB_unit) == wave_unit

    @pytest.mark.parametrize(
        'wave_unit,dB_unit',
        [
            ('√mW', 'dBm'),
            ('√W', 'dBW'),
            ('√unitless', 'dB'),
        ],
    )
    def test_unit_wave_to_dB(self, wave_unit: str, dB_unit: str):
        assert unit_wave_to_dB(wave_unit) == dB_unit

    @pytest.mark.parametrize(
        'wave_unit,linear_unit',
        [
            ('√mW', 'mW'),
            ('√W', 'W'),
            ('√unitless', 'unitless'),
        ],
    )
    def test_unit_wave_to_linear(self, wave_unit: str, linear_unit: str):
        assert unit_wave_to_linear(wave_unit) == linear_unit


class TestPowtodB:
    """Tests for powtodB: 10*log10(abs(x))"""

    def test_basic_conversion(self, xp):
        """Test basic power to dB conversion."""
        power = make_array(xp, [1.0, 10.0, 100.0, 1000.0])
        expected_dB = np.array([0.0, 10.0, 20.0, 30.0])
        result = to_numpy(powtodB(power))
        assert_allclose(result, expected_dB, rtol=1e-10)

    def test_fractional_power(self, xp):
        """Test conversion of fractional power values."""
        power = make_array(xp, [0.1, 0.01, 0.001])
        expected_dB = np.array([-10.0, -20.0, -30.0])
        result = to_numpy(powtodB(power))
        assert_allclose(result, expected_dB, rtol=1e-10)

    def test_negative_values_with_abs(self, xp):
        """Test that abs=True handles negative values."""
        power = make_array(xp, [-1.0, -10.0, -100.0])
        expected_dB = np.array([0.0, 10.0, 20.0])
        result = to_numpy(powtodB(power, abs=True))
        assert_allclose(result, expected_dB, rtol=1e-10)

    def test_complex_values(self, xp):
        """Test conversion of complex values (uses magnitude)."""
        power = make_array(xp, [1 + 0j, 0 + 10j, 3 + 4j])  # magnitudes: 1, 10, 5
        expected_dB = np.array([0.0, 10.0, 10 * np.log10(5)])
        result = to_numpy(powtodB(power))
        assert_allclose(result, expected_dB, rtol=1e-10)

    def test_eps_parameter(self, xp):
        """Test epsilon parameter for avoiding log(0)."""
        power = make_array(xp, [0.0, 1.0])
        eps = 1e-10
        result = to_numpy(powtodB(power, eps=eps))
        expected = np.array([10 * np.log10(eps), 10 * np.log10(1 + eps)])
        assert_allclose(result, expected, rtol=1e-6)

    def test_2d_array(self, xp):
        """Test conversion of 2D arrays."""
        power = make_array(xp, [[1.0, 10.0], [100.0, 1000.0]])
        expected_dB = np.array([[0.0, 10.0], [20.0, 30.0]])
        result = to_numpy(powtodB(power))
        assert_allclose(result, expected_dB, rtol=1e-10)

    def test_overwrite_x(self, xp_name):
        """Test in-place computation with overwrite_x=True."""
        name, xp = xp_name
        if name == 'dask':
            pytest.skip('dask does not support in-place operations')
        power = make_array(xp, [1.0, 10.0, 100.0], dtype=np.float64)
        result = powtodB(power, overwrite_x=True)
        expected_dB = np.array([0.0, 10.0, 20.0])
        assert_allclose(to_numpy(result), expected_dB, rtol=1e-10)


class TestDBtopow:
    """Tests for dBtopow: 10**(x/10)"""

    def test_basic_conversion(self, xp):
        """Test basic dB to power conversion."""
        dB = make_array(xp, [0.0, 10.0, 20.0, 30.0])
        expected_power = np.array([1.0, 10.0, 100.0, 1000.0])
        result = to_numpy(dBtopow(dB))
        assert_allclose(result, expected_power, rtol=1e-10)

    def test_negative_dB(self, xp):
        """Test conversion of negative dB values."""
        dB = make_array(xp, [-10.0, -20.0, -30.0])
        expected_power = np.array([0.1, 0.01, 0.001])
        result = to_numpy(dBtopow(dB))
        assert_allclose(result, expected_power, rtol=1e-10)

    def test_roundtrip_powtodB_dBtopow(self, xp):
        """Test that powtodB and dBtopow are inverses."""
        power = make_array(xp, [0.001, 0.1, 1.0, 10.0, 100.0, 1000.0])
        roundtrip = to_numpy(dBtopow(powtodB(power)))
        assert_allclose(roundtrip, to_numpy(power), rtol=1e-10)

    def test_roundtrip_dBtopow_powtodB(self, xp):
        """Test that dBtopow and powtodB are inverses."""
        dB = make_array(xp, [-30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0])
        roundtrip = to_numpy(powtodB(dBtopow(dB)))
        assert_allclose(roundtrip, to_numpy(dB), rtol=1e-10)

    def test_2d_array(self, xp):
        """Test conversion of 2D arrays."""
        dB = make_array(xp, [[0.0, 10.0], [20.0, 30.0]])
        expected_power = np.array([[1.0, 10.0], [100.0, 1000.0]])
        result = to_numpy(dBtopow(dB))
        assert_allclose(result, expected_power, rtol=1e-10)


class TestEnvtopow:
    """Tests for envtopow: abs(x)**2"""

    def test_real_values(self, xp):
        """Test envelope to power for real values."""
        env = make_array(xp, [1.0, 2.0, 3.0, 10.0])
        expected_power = np.array([1.0, 4.0, 9.0, 100.0])
        result = to_numpy(envtopow(env))
        assert_allclose(result, expected_power, rtol=1e-10)

    def test_complex_values(self, xp):
        """Test envelope to power for complex values."""
        env = make_array(xp, [1 + 0j, 0 + 2j, 3 + 4j])  # magnitudes: 1, 2, 5
        expected_power = np.array([1.0, 4.0, 25.0])
        result = to_numpy(envtopow(env))
        assert_allclose(result, expected_power, rtol=1e-10)

    def test_negative_real_values(self, xp):
        """Test that negative real values are handled correctly."""
        env = make_array(xp, [-1.0, -2.0, -3.0])
        expected_power = np.array([1.0, 4.0, 9.0])
        result = to_numpy(envtopow(env))
        assert_allclose(result, expected_power, rtol=1e-10)


class TestEnvtodB:
    """Tests for envtodB: 20*log10(abs(x))"""

    def test_basic_conversion(self, xp):
        """Test basic envelope to dB conversion."""
        env = make_array(xp, [1.0, 10.0, 100.0])
        expected_dB = np.array([0.0, 20.0, 40.0])
        result = to_numpy(envtodB(env))
        assert_allclose(result, expected_dB, rtol=1e-10)

    def test_fractional_envelope(self, xp):
        """Test conversion of fractional envelope values."""
        env = make_array(xp, [0.1, 0.01, 0.001])
        expected_dB = np.array([-20.0, -40.0, -60.0])
        result = to_numpy(envtodB(env))
        assert_allclose(result, expected_dB, rtol=1e-10)

    def test_complex_values(self, xp):
        """Test conversion of complex envelope values."""
        env = make_array(xp, [1 + 0j, 0 + 10j, 3 + 4j])  # magnitudes: 1, 10, 5
        expected_dB = np.array([0.0, 20.0, 20 * np.log10(5)])
        result = to_numpy(envtodB(env))
        assert_allclose(result, expected_dB, rtol=1e-10)

    def test_relationship_to_powtodB(self, xp):
        """Test that envtodB(x) == powtodB(envtopow(x))."""
        env = make_array(xp, [0.5, 1.0, 2.0, 5.0, 10.0])
        result_direct = to_numpy(envtodB(env))
        result_via_power = to_numpy(powtodB(envtopow(env)))
        assert_allclose(result_direct, result_via_power, rtol=1e-10)


class TestDBlinmean:
    """Tests for dBlinmean: mean in linear space, result in dB."""

    def test_equal_values(self, xp):
        """Test mean of equal dB values returns the same value."""
        dB = make_array(xp, [10.0, 10.0, 10.0, 10.0])
        result = to_numpy(dBlinmean(dB))
        assert_allclose(result, 10.0, rtol=1e-10)

    def test_known_values(self, xp):
        """Test with known values where linear mean is calculable."""
        # Two values: 0 dB (1 linear) and 10 dB (10 linear)
        # Linear mean = (1 + 10) / 2 = 5.5
        # dB result = 10*log10(5.5) ≈ 7.404
        dB = make_array(xp, [0.0, 10.0])
        expected = 10 * np.log10(5.5)
        result = to_numpy(dBlinmean(dB))
        assert_allclose(result, expected, rtol=1e-10)

    def test_axis_parameter(self, xp):
        """Test mean along specific axis."""
        # 2x3 array
        dB = make_array(xp, [[0.0, 10.0, 20.0], [0.0, 10.0, 20.0]])
        # Mean along axis 0 should give same values (rows are identical)
        result = to_numpy(dBlinmean(dB, axis=0))
        expected = np.array([0.0, 10.0, 20.0])
        assert_allclose(result, expected, rtol=1e-10)

    def test_axis_1(self, xp):
        """Test mean along axis 1."""
        # Each row: [0 dB, 10 dB] -> linear [1, 10] -> mean 5.5 -> ~7.404 dB
        dB = make_array(xp, [[0.0, 10.0], [0.0, 10.0]])
        expected_single = 10 * np.log10(5.5)
        result = to_numpy(dBlinmean(dB, axis=1))
        expected = np.array([expected_single, expected_single])
        assert_allclose(result, expected, rtol=1e-10)

    def test_full_array_mean(self, xp):
        """Test mean over entire array (axis=None)."""
        dB = make_array(xp, [[0.0, 10.0], [20.0, 30.0]])
        # Linear: [1, 10, 100, 1000], mean = 277.75
        expected = 10 * np.log10(277.75)
        result = to_numpy(dBlinmean(dB, axis=None))
        assert_allclose(result, expected, rtol=1e-10)

    def test_single_value(self, xp):
        """Test that single value returns itself."""
        dB = make_array(xp, [15.0])
        result = to_numpy(dBlinmean(dB))
        assert_allclose(result, 15.0, rtol=1e-10)

    def test_large_dynamic_range(self, xp):
        """Test with large dynamic range values."""
        # -30 dB (0.001) and 30 dB (1000)
        # Linear mean = (0.001 + 1000) / 2 = 500.0005
        dB = make_array(xp, [-30.0, 30.0])
        expected = 10 * np.log10(500.0005)
        result = to_numpy(dBlinmean(dB))
        assert_allclose(result, expected, rtol=1e-6)


class TestDBlinsum:
    """Tests for dBlinsum: sum in linear space, result in dB."""

    def test_equal_values(self, xp):
        """Test sum of equal dB values."""
        # Four values of 10 dB (10 linear each)
        # Linear sum = 40, dB = 10*log10(40) ≈ 16.02
        dB = make_array(xp, [10.0, 10.0, 10.0, 10.0])
        expected = 10 * np.log10(40)
        result = to_numpy(dBlinsum(dB))
        assert_allclose(result, expected, rtol=1e-10)

    def test_known_values(self, xp):
        """Test with known values where linear sum is calculable."""
        # 0 dB (1 linear) and 10 dB (10 linear)
        # Linear sum = 11
        dB = make_array(xp, [0.0, 10.0])
        expected = 10 * np.log10(11)
        result = to_numpy(dBlinsum(dB))
        assert_allclose(result, expected, rtol=1e-10)

    def test_axis_parameter(self, xp):
        """Test sum along specific axis."""
        # 2x3 array, sum along axis 0
        dB = make_array(xp, [[0.0, 10.0, 20.0], [0.0, 10.0, 20.0]])
        # Sum along axis 0: [2, 20, 200] in linear
        expected = 10 * np.log10(np.array([2, 20, 200]))
        result = to_numpy(dBlinsum(dB, axis=0))
        assert_allclose(result, expected, rtol=1e-10)

    def test_axis_1(self, xp):
        """Test sum along axis 1."""
        # Each row: [0 dB, 10 dB] -> linear [1, 10] -> sum 11
        dB = make_array(xp, [[0.0, 10.0], [0.0, 10.0]])
        expected_single = 10 * np.log10(11)
        result = to_numpy(dBlinsum(dB, axis=1))
        expected = np.array([expected_single, expected_single])
        assert_allclose(result, expected, rtol=1e-10)

    def test_full_array_sum(self, xp):
        """Test sum over entire array (axis=None)."""
        dB = make_array(xp, [[0.0, 10.0], [20.0, 30.0]])
        # Linear: [1, 10, 100, 1000], sum = 1111
        expected = 10 * np.log10(1111)
        result = to_numpy(dBlinsum(dB, axis=None))
        assert_allclose(result, expected, rtol=1e-10)

    def test_single_value(self, xp):
        """Test that single value returns itself."""
        dB = make_array(xp, [15.0])
        result = to_numpy(dBlinsum(dB))
        assert_allclose(result, 15.0, rtol=1e-10)

    def test_3dB_rule(self, xp):
        """Test the 3 dB rule: doubling power adds ~3 dB."""
        # Two equal power sources: sum should be ~3 dB higher
        base_dB = 20.0
        dB = make_array(xp, [base_dB, base_dB])
        result = to_numpy(dBlinsum(dB))
        # Doubling power = +3.0103 dB
        expected = base_dB + 10 * np.log10(2)
        assert_allclose(result, expected, rtol=1e-10)

    def test_10dB_rule(self, xp):
        """Test the 10 dB rule: 10x power adds 10 dB."""
        # Ten equal power sources: sum should be 10 dB higher
        base_dB = 0.0
        dB = make_array(xp, [base_dB] * 10)
        result = to_numpy(dBlinsum(dB))
        expected = base_dB + 10.0  # 10*log10(10) = 10
        assert_allclose(result, expected, rtol=1e-10)


class TestDBlinmeanVsDBlinsum:
    """Tests comparing dBlinmean and dBlinsum behavior."""

    def test_mean_vs_sum_relationship(self, xp):
        """Test that dBlinmean = dBlinsum - 10*log10(N)."""
        dB = make_array(xp, [0.0, 10.0, 20.0, 30.0])
        N = 4
        mean_result = to_numpy(dBlinmean(dB))
        sum_result = to_numpy(dBlinsum(dB))
        # mean = sum / N, so in dB: mean_dB = sum_dB - 10*log10(N)
        expected_mean = sum_result - 10 * np.log10(N)
        assert_allclose(mean_result, expected_mean, rtol=1e-10)

    def test_consistency_across_axes(self, xp):
        """Test consistency of mean/sum relationship across different axes."""
        # Use fixed data for reproducibility
        dB_data = [
            [-15.2, 8.3, -5.1, 12.7, 3.9, -18.4, 7.2, -2.8, 16.5, -9.6],
            [4.1, -11.3, 19.8, -7.5, 1.2, 14.6, -3.9, 10.4, -16.7, 5.8],
            [-8.9, 17.1, -1.4, 9.3, -12.6, 6.7, -4.2, 15.9, -0.3, 11.5],
            [2.6, -14.8, 8.9, -6.1, 18.3, -10.7, 13.2, -5.4, 7.6, -17.9],
            [16.4, -3.7, 11.8, -9.2, 4.5, -15.1, 0.8, 19.3, -8.6, 12.1],
        ]
        dB = make_array(xp, dB_data)

        for axis in [0, 1, None]:
            mean_result = to_numpy(dBlinmean(dB, axis=axis))
            sum_result = to_numpy(dBlinsum(dB, axis=axis))
            N = (
                np.array(dB_data).shape[axis]
                if axis is not None
                else np.array(dB_data).size
            )
            expected_mean = sum_result - 10 * np.log10(N)
            assert_allclose(mean_result, expected_mean, rtol=1e-10)


class TestEdgeCases:
    """Tests for edge cases and special values."""

    def test_very_small_values(self, xp):
        """Test handling of very small power values."""
        power = make_array(xp, [1e-15, 1e-12, 1e-9])
        expected_dB = np.array([-150.0, -120.0, -90.0])
        result = to_numpy(powtodB(power))
        assert_allclose(result, expected_dB, rtol=1e-10)

    def test_very_large_values(self, xp):
        """Test handling of very large power values."""
        power = make_array(xp, [1e9, 1e12, 1e15])
        expected_dB = np.array([90.0, 120.0, 150.0])
        result = to_numpy(powtodB(power))
        assert_allclose(result, expected_dB, rtol=1e-10)

    def test_mixed_dtypes_float32(self, xp):
        """Test that float32 input works correctly."""
        power = make_array(xp, [1.0, 10.0, 100.0], dtype=np.float32)
        expected_dB = np.array([0.0, 10.0, 20.0])
        result = to_numpy(powtodB(power))
        assert_allclose(result, expected_dB, rtol=1e-5)

    def test_mixed_dtypes_float64(self, xp):
        """Test that float64 input works correctly."""
        power = make_array(xp, [1.0, 10.0, 100.0], dtype=np.float64)
        expected_dB = np.array([0.0, 10.0, 20.0])
        result = to_numpy(powtodB(power))
        assert_allclose(result, expected_dB, rtol=1e-10)

    def test_empty_array(self, xp):
        """Test handling of empty arrays."""
        power = make_array(xp, [])
        result = to_numpy(powtodB(power))
        assert_array_equal(result, np.array([]))

    def test_scalar_like_array(self, xp):
        """Test handling of 0-d arrays."""
        power = make_array(xp, 100.0)
        result = to_numpy(powtodB(power))
        assert_allclose(result, 20.0, rtol=1e-10)


class TestZeroValuesAndInfinity:
    """Tests for zero values producing -inf when eps==0."""

    def test_powtodB_zero_produces_neg_inf(self, xp):
        """Test that powtodB(0) produces -inf when eps=0."""
        power = make_array(xp, [0.0, 1.0, 0.0])
        result = to_numpy(powtodB(power, eps=0))
        assert result[0] == -np.inf
        assert_allclose(result[1], 0.0, rtol=1e-10)
        assert result[2] == -np.inf

    def test_envtodB_zero_produces_neg_inf(self, xp):
        """Test that envtodB(0) produces -inf when eps=0."""
        env = make_array(xp, [0.0, 1.0, 0.0])
        result = to_numpy(envtodB(env, eps=0))
        assert result[0] == -np.inf
        assert_allclose(result[1], 0.0, rtol=1e-10)
        assert result[2] == -np.inf

    def test_powtodB_all_zeros(self, xp):
        """Test array of all zeros produces all -inf."""
        power = make_array(xp, [0.0, 0.0, 0.0])
        result = to_numpy(powtodB(power, eps=0))
        assert np.all(result == -np.inf)

    def test_envtodB_all_zeros(self, xp):
        """Test array of all zeros produces all -inf."""
        env = make_array(xp, [0.0, 0.0, 0.0])
        result = to_numpy(envtodB(env, eps=0))
        assert np.all(result == -np.inf)

    def test_powtodB_eps_avoids_neg_inf(self, xp):
        """Test that eps parameter prevents -inf for zero values."""
        power = make_array(xp, [0.0, 1.0])
        eps = 1e-20
        result = to_numpy(powtodB(power, eps=eps))
        # With eps, zero becomes 10*log10(eps) which is finite
        assert np.isfinite(result[0])
        assert_allclose(result[0], 10 * np.log10(eps), rtol=1e-6)

    def test_envtodB_eps_avoids_neg_inf(self, xp):
        """Test that eps parameter prevents -inf for zero values."""
        env = make_array(xp, [0.0, 1.0])
        eps = 1e-20
        result = to_numpy(envtodB(env, eps=eps))
        # With eps, zero becomes 20*log10(eps) which is finite
        assert np.isfinite(result[0])
        assert_allclose(result[0], 20 * np.log10(eps), rtol=1e-6)

    def test_powtodB_2d_with_zeros(self, xp):
        """Test 2D array with scattered zeros."""
        power = make_array(xp, [[0.0, 10.0], [100.0, 0.0]])
        result = to_numpy(powtodB(power, eps=0))
        assert result[0, 0] == -np.inf
        assert_allclose(result[0, 1], 10.0, rtol=1e-10)
        assert_allclose(result[1, 0], 20.0, rtol=1e-10)
        assert result[1, 1] == -np.inf


class TestInPlaceOutputs:
    """Tests for in-place output operations (overwrite_x parameter)."""

    def test_powtodB_overwrite_x_true(self, xp_name):
        """Test that powtodB with overwrite_x=True computes in-place."""
        name, xp = xp_name
        if name == 'dask':
            pytest.skip('dask does not support in-place operations')

        power = make_array(xp, [1.0, 10.0, 100.0], dtype=np.float64)

        result = powtodB(power, overwrite_x=True)
        expected = np.array([0.0, 10.0, 20.0])

        # Verify result is correct
        assert_allclose(to_numpy(result), expected, rtol=1e-10)

    def test_dBtopow_overwrite_x_true(self, xp_name):
        """Test that dBtopow with overwrite_x=True computes in-place."""
        name, xp = xp_name
        if name == 'dask':
            pytest.skip('dask does not support in-place operations')

        dB = make_array(xp, [0.0, 10.0, 20.0], dtype=np.float64)

        result = dBtopow(dB, overwrite_x=True)
        expected = np.array([1.0, 10.0, 100.0])

        assert_allclose(to_numpy(result), expected, rtol=1e-10)

    def test_envtodB_overwrite_x_true(self, xp_name):
        """Test that envtodB with overwrite_x=True computes in-place."""
        name, xp = xp_name
        if name == 'dask':
            pytest.skip('dask does not support in-place operations')

        env = make_array(xp, [1.0, 10.0, 100.0], dtype=np.float64)

        result = envtodB(env, overwrite_x=True)
        expected = np.array([0.0, 20.0, 40.0])

        assert_allclose(to_numpy(result), expected, rtol=1e-10)

    def test_envtopow_overwrite_x_true(self, xp_name):
        """Test that envtopow with overwrite_x=True computes in-place."""
        name, xp = xp_name
        if name == 'dask':
            pytest.skip('dask does not support in-place operations')

        env = make_array(xp, [1.0, 2.0, 3.0], dtype=np.float64)

        result = envtopow(env, overwrite_x=True)
        expected = np.array([1.0, 4.0, 9.0])

        assert_allclose(to_numpy(result), expected, rtol=1e-10)

    def test_powtodB_overwrite_x_false_preserves_input(self, xp_name):
        """Test that powtodB with overwrite_x=False does not modify input array."""
        name, xp = xp_name
        if name == 'dask':
            pytest.skip('dask arrays are immutable')

        power = make_array(xp, [1.0, 10.0, 100.0], dtype=np.float64)
        original_power = to_numpy(power).copy()

        powtodB(power, overwrite_x=False)

        # Input should be unchanged
        assert_allclose(to_numpy(power), original_power, rtol=1e-10)

    def test_dBtopow_overwrite_x_false_preserves_input(self, xp_name):
        """Test that dBtopow with overwrite_x=False does not modify input array."""
        name, xp = xp_name
        if name == 'dask':
            pytest.skip('dask arrays are immutable')

        dB = make_array(xp, [0.0, 10.0, 20.0], dtype=np.float64)
        original_dB = to_numpy(dB).copy()

        dBtopow(dB, overwrite_x=False)

        # Input should be unchanged
        assert_allclose(to_numpy(dB), original_dB, rtol=1e-10)

    def test_envtodB_overwrite_x_false_preserves_input(self, xp_name):
        """Test that envtodB with overwrite_x=False does not modify input array."""
        name, xp = xp_name
        if name == 'dask':
            pytest.skip('dask arrays are immutable')

        env = make_array(xp, [1.0, 10.0, 100.0], dtype=np.float64)
        original_env = to_numpy(env).copy()

        envtodB(env, overwrite_x=False)

        # Input should be unchanged
        assert_allclose(to_numpy(env), original_env, rtol=1e-10)

    def test_envtopow_overwrite_x_false_preserves_input(self, xp_name):
        """Test that envtopow with overwrite_x=False does not modify input array."""
        name, xp = xp_name
        if name == 'dask':
            pytest.skip('dask arrays are immutable')

        env = make_array(xp, [1.0, 2.0, 3.0], dtype=np.float64)
        original_env = to_numpy(env).copy()

        envtopow(env, overwrite_x=False)

        # Input should be unchanged
        assert_allclose(to_numpy(env), original_env, rtol=1e-10)

    def test_dBlinmean_overwrite_x_true_modifies_input(self, xp_name):
        """Test that dBlinmean with overwrite_x=True modifies the input array."""
        name, xp = xp_name
        if name == 'dask':
            pytest.skip('dask does not support in-place operations')

        dB = make_array(xp, [0.0, 10.0, 20.0, 30.0], dtype=np.float64)
        original_values = to_numpy(dB).copy()

        result = dBlinmean(dB, overwrite_x=True)

        # Input array should be modified (contains intermediate linear values)
        modified_values = to_numpy(dB)
        # The array should have been overwritten with linear power values
        # (10^(dB/10)) before the mean was computed
        assert not np.allclose(modified_values, original_values)

    def test_dBlinmean_overwrite_x_false_preserves_input(self, xp_name):
        """Test that dBlinmean with overwrite_x=False preserves the input array."""
        name, xp = xp_name
        if name == 'dask':
            pytest.skip('dask arrays are immutable')

        dB = make_array(xp, [0.0, 10.0, 20.0, 30.0], dtype=np.float64)
        original_values = to_numpy(dB).copy()

        result = dBlinmean(dB, overwrite_x=False)

        # Input array should be unchanged
        assert_allclose(to_numpy(dB), original_values, rtol=1e-10)

    def test_dBlinsum_overwrite_x_true_modifies_input(self, xp_name):
        """Test that dBlinsum with overwrite_x=True modifies the input array."""
        name, xp = xp_name
        if name == 'dask':
            pytest.skip('dask does not support in-place operations')

        dB = make_array(xp, [0.0, 10.0, 20.0, 30.0], dtype=np.float64)
        original_values = to_numpy(dB).copy()

        result = dBlinsum(dB, overwrite_x=True)

        # Input array should be modified
        modified_values = to_numpy(dB)
        assert not np.allclose(modified_values, original_values)

    def test_dBlinsum_overwrite_x_false_preserves_input(self, xp_name):
        """Test that dBlinsum with overwrite_x=False preserves the input array."""
        name, xp = xp_name
        if name == 'dask':
            pytest.skip('dask arrays are immutable')

        dB = make_array(xp, [0.0, 10.0, 20.0, 30.0], dtype=np.float64)
        original_values = to_numpy(dB).copy()

        result = dBlinsum(dB, overwrite_x=False)

        # Input array should be unchanged
        assert_allclose(to_numpy(dB), original_values, rtol=1e-10)

    def test_powtodB_overwrite_x_2d_array(self, xp_name):
        """Test overwrite_x with 2D arrays."""
        name, xp = xp_name
        if name == 'dask':
            pytest.skip('dask does not support in-place operations')

        power = make_array(xp, [[1.0, 10.0], [100.0, 1000.0]], dtype=np.float64)

        result = powtodB(power, overwrite_x=True)
        expected = np.array([[0.0, 10.0], [20.0, 30.0]])

        assert_allclose(to_numpy(result), expected, rtol=1e-10)

    def test_dBlinmean_overwrite_x_with_axis(self, xp_name):
        """Test overwrite_x with axis parameter."""
        name, xp = xp_name
        if name == 'dask':
            pytest.skip('dask does not support in-place operations')

        dB = make_array(xp, [[0.0, 10.0], [0.0, 10.0]], dtype=np.float64)
        original_values = to_numpy(dB).copy()

        # Mean along axis 0
        result = dBlinmean(dB, axis=0, overwrite_x=True)
        expected = np.array([0.0, 10.0])

        assert_allclose(to_numpy(result), expected, rtol=1e-10)
        # Input should be modified
        assert not np.allclose(to_numpy(dB), original_values)

    def test_dBlinsum_overwrite_x_with_axis(self, xp_name):
        """Test overwrite_x with axis parameter."""
        name, xp = xp_name
        if name == 'dask':
            pytest.skip('dask does not support in-place operations')

        dB = make_array(xp, [[0.0, 10.0], [0.0, 10.0]], dtype=np.float64)
        original_values = to_numpy(dB).copy()

        # Sum along axis 0: [2, 20] in linear -> [3.01, 13.01] in dB
        result = dBlinsum(dB, axis=0, overwrite_x=True)
        expected = 10 * np.log10(np.array([2, 20]))

        assert_allclose(to_numpy(result), expected, rtol=1e-10)
        # Input should be modified
        assert not np.allclose(to_numpy(dB), original_values)


class TestMinDtype:
    """Tests for min_dtype parameter that promotes low-precision inputs."""

    def test_powtodB_float16_without_min_dtype_has_rounding_errors(self, xp_name):
        """Demonstrate that float16 without min_dtype promotion has rounding errors."""
        name, xp = xp_name
        if name == 'dask':
            pytest.skip('dask does not support float16')

        # Values that expose float16 precision limits
        power = make_array(xp, [0.001, 0.0001, 0.00001], dtype=np.float16)
        expected_dB = np.array([-30.0, -40.0, -50.0])

        # With min_dtype=None, computation stays in float16 and has errors
        result = to_numpy(powtodB(power, min_dtype=None))

        # float16 has ~3 decimal digits of precision, so large errors expected
        # This test documents the problem that min_dtype solves
        assert not np.allclose(result, expected_dB, rtol=1e-4)

    def test_powtodB_float16_with_min_dtype_float32_mitigates_errors(self, xp_name):
        """Test that min_dtype='float32' mitigates rounding errors for float16 input."""
        name, xp = xp_name
        if name == 'dask':
            pytest.skip('dask does not support float16')

        # Values that expose float16 precision limits
        power = make_array(xp, [0.001, 0.0001, 0.00001], dtype=np.float16)
        expected_dB = np.array([-30.0, -40.0, -50.0])

        # With min_dtype='float32', computation is promoted and accurate
        result = to_numpy(powtodB(power, min_dtype='float32'))

        # Should now be accurate within 0.01 dB (dB values are already relative)
        assert_allclose(result, expected_dB, atol=0.006)

    def test_dBtopow_float16_with_min_dtype_float32(self, xp_name):
        """Test that dBtopow with min_dtype='float32' handles float16 input."""
        name, xp = xp_name
        if name == 'dask':
            pytest.skip('dask does not support float16')

        dB = make_array(xp, [-30.0, -40.0, -50.0], dtype=np.float16)
        expected_power = np.array([0.001, 0.0001, 0.00001])

        result = to_numpy(dBtopow(dB, min_dtype='float32'))

        assert_allclose(result, expected_power, rtol=1e-5)

    def test_envtopow_float16_with_min_dtype_float32(self, xp_name):
        """Test that envtopow with min_dtype='float32' handles float16 input."""
        name, xp = xp_name
        if name == 'dask':
            pytest.skip('dask does not support float16')

        # Small envelope values that would lose precision in float16
        env = make_array(xp, [0.01, 0.001, 0.0001], dtype=np.float16)
        expected_power = np.array([0.0001, 0.000001, 0.00000001])

        result = to_numpy(envtopow(env, min_dtype='float32'))

        assert_allclose(result, expected_power, rtol=1e-3)

    def test_envtodB_float16_with_min_dtype_float32(self, xp_name):
        """Test that envtodB with min_dtype='float32' handles float16 input."""
        name, xp = xp_name
        if name == 'dask':
            pytest.skip('dask does not support float16')

        env = make_array(xp, [0.01, 0.001, 0.0001], dtype=np.float16)
        expected_dB = np.array([-40.0, -60.0, -80.0])

        result = to_numpy(envtodB(env, min_dtype='float32'))

        # dB values are already relative, use atol
        assert_allclose(result, expected_dB, atol=0.006)

    def test_powtodB_float64_unaffected_by_min_dtype_float32(self, xp):
        """Test that float64 input is not downgraded by min_dtype='float32'."""
        power = make_array(xp, [1e-15, 1e-14, 1e-13], dtype=np.float64)
        expected_dB = np.array([-150.0, -140.0, -130.0])

        result = to_numpy(powtodB(power, min_dtype='float32'))

        # float64 precision should be preserved (not downgraded to float32)
        # dB values are already relative, use atol
        assert_allclose(result, expected_dB, atol=1e-6)

    def test_min_dtype_default_is_float32(self, xp_name):
        """Test that the default min_dtype='float32' is applied."""
        name, xp = xp_name
        if name == 'dask':
            pytest.skip('dask does not support float16')

        # float16 input with default min_dtype should work correctly
        power = make_array(xp, [0.001, 0.01, 0.1], dtype=np.float16)
        expected_dB = np.array([-30.0, -20.0, -10.0])

        # Default min_dtype='float32' should promote and give accurate results
        result = to_numpy(powtodB(power))

        # dB values are already relative, use atol
        assert_allclose(result, expected_dB, atol=0.006)

    def test_roundtrip_float16_with_min_dtype(self, xp_name):
        """Test roundtrip conversion preserves values with min_dtype promotion."""
        name, xp = xp_name
        if name == 'dask':
            pytest.skip('dask does not support float16')

        original_dB = make_array(xp, [-30.0, -20.0, -10.0, 0.0, 10.0], dtype=np.float16)

        # Roundtrip: dB -> power -> dB with min_dtype promotion
        power = dBtopow(original_dB, min_dtype='float32')
        roundtrip_dB = powtodB(power, min_dtype='float32')

        # Should recover original values within reasonable tolerance
        # dB values are already relative, use atol
        assert_allclose(to_numpy(roundtrip_dB), to_numpy(original_dB), atol=0.5)

    def test_dBlinmean_float16_with_min_dtype_float32(self, xp_name):
        """Test that dBlinmean with min_dtype='float32' handles float16 input."""
        name, xp = xp_name
        if name == 'dask':
            pytest.skip('dask does not support float16')

        # Known dB values where linear mean is calculable
        dB = make_array(xp, [-30.0, -20.0, -10.0, 0.0], dtype=np.float16)
        # Linear: [0.001, 0.01, 0.1, 1.0], mean = 0.27775
        expected = 10 * np.log10(0.27775)

        result = to_numpy(dBlinmean(dB, min_dtype='float32'))

        # dB values are already relative, use atol
        assert_allclose(result, expected, atol=0.01)

    def test_dBlinmean_float16_random_values(self, xp_name):
        """Test dBlinmean with random float16 values and min_dtype promotion."""
        name, xp = xp_name
        if name == 'dask':
            pytest.skip('dask does not support float16')

        # Generate reproducible random dB values in typical range
        rng = np.random.default_rng(42)
        dB_float64 = rng.uniform(-60, 10, size=100)

        # Compute reference result in float64
        linear_f64 = 10 ** (dB_float64 / 10)
        expected = 10 * np.log10(np.mean(linear_f64))

        # Convert to float16 and compute with min_dtype promotion
        dB_float16 = make_array(xp, dB_float64.astype(np.float16), dtype=np.float16)
        result = to_numpy(dBlinmean(dB_float16, min_dtype='float32'))

        # Should be accurate within 0.1 dB despite float16 input
        assert_allclose(result, expected, atol=0.1)

    def test_dBlinmean_float16_2d_with_axis(self, xp_name):
        """Test dBlinmean with float16 2D array and axis parameter."""
        name, xp = xp_name
        if name == 'dask':
            pytest.skip('dask does not support float16')

        # 2D array: mean along axis 1
        dB = make_array(xp, [[-30.0, -20.0], [-10.0, 0.0]], dtype=np.float16)
        # Row 0: linear [0.001, 0.01], mean = 0.0055 -> -22.596 dB
        # Row 1: linear [0.1, 1.0], mean = 0.55 -> -2.596 dB
        expected = np.array([10 * np.log10(0.0055), 10 * np.log10(0.55)])

        result = to_numpy(dBlinmean(dB, axis=1, min_dtype='float32'))

        assert_allclose(result, expected, atol=0.01)

    def test_dBlinsum_float16_with_min_dtype_float32(self, xp_name):
        """Test that dBlinsum with min_dtype='float32' handles float16 input."""
        name, xp = xp_name
        if name == 'dask':
            pytest.skip('dask does not support float16')

        # Known dB values where linear sum is calculable
        dB = make_array(xp, [-30.0, -20.0, -10.0, 0.0], dtype=np.float16)
        # Linear: [0.001, 0.01, 0.1, 1.0], sum = 1.111
        expected = 10 * np.log10(1.111)

        result = to_numpy(dBlinsum(dB, min_dtype='float32'))

        # dB values are already relative, use atol
        assert_allclose(result, expected, atol=0.01)

    def test_dBlinsum_float16_random_values(self, xp_name):
        """Test dBlinsum with random float16 values and min_dtype promotion."""
        name, xp = xp_name
        if name == 'dask':
            pytest.skip('dask does not support float16')

        # Generate reproducible random dB values in typical range
        rng = np.random.default_rng(123)
        dB_float64 = rng.uniform(-50, 0, size=50)

        # Compute reference result in float64
        linear_f64 = 10 ** (dB_float64 / 10)
        expected = 10 * np.log10(np.sum(linear_f64))

        # Convert to float16 and compute with min_dtype promotion
        dB_float16 = make_array(xp, dB_float64.astype(np.float16), dtype=np.float16)
        result = to_numpy(dBlinsum(dB_float16, min_dtype='float32'))

        # Should be accurate within 0.1 dB despite float16 input
        assert_allclose(result, expected, atol=1e-3)

    def test_dBlinsum_float16_2d_with_axis(self, xp_name):
        """Test dBlinsum with float16 2D array and axis parameter."""
        name, xp = xp_name
        if name == 'dask':
            pytest.skip('dask does not support float16')

        # 2D array: sum along axis 1
        dB = make_array(xp, [[-30.0, -20.0], [-10.0, 0.0]], dtype=np.float16)
        # Row 0: linear [0.001, 0.01], sum = 0.011 -> -19.586 dB
        # Row 1: linear [0.1, 1.0], sum = 1.1 -> 0.414 dB
        expected = np.array([10 * np.log10(0.011), 10 * np.log10(1.1)])

        result = to_numpy(dBlinsum(dB, axis=1, min_dtype='float32'))

        assert_allclose(result, expected, atol=1e-3)