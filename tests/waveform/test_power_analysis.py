"""Property-based tests for striqt.waveform.power_analysis using Hypothesis.

Covers the dB/linear conversions and dB-domain statistics: identities and algebraic
rules, dtype handling, input preservation, edge cases, complex inputs, numpy/cupy/dask
compatibility, and numpy-vs-cupy agreement within ulp budgets for the library calls.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import assume, given, settings, HealthCheck
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays, array_shapes
from numpy.testing import assert_allclose

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

from conftest import (
    to_numpy,
    positive_power_arrays,
    dB_arrays,
    envelope_arrays,
    available_namespaces,
    convert_array,
    for_each_namespace,
)

# Roundoff budgets for the dB conversions, in ulps per library call (1 ulp <= 2u
# relative, u = unit roundoff). Sized to cover CUDA's single-precision bounds (log10f 2,
# powf 8, hypotf 3 ulp) as well as libm (~1 ulp). Measured with
# chores/tests/measure_db_accuracy.py: libm log10 1.1-1.9 ulp; CUDA libdevice on a
# Jetson TX2i log10f 2.0, powf 5 (|x| <= 30 dB), complex |z| 1.6 input ulps.
LOG10_ULP = 2
POW_ULP = 8
HYPOT_ULP = 3
ROUNDOFF_SAFETY = 2
DB_PER_NEPER = 10 / np.log(10)


def unit_roundoff(dtype):
    return np.finfo(dtype).eps / 2


def log_conversion_tol(dtype, scale, complex_input=False, n_impl=1):
    """(rtol, atol) for scale*log10(|x|), against exact math (n_impl=1) or a second
    implementation (n_impl=2).

    Roundoff on the input side of the log becomes an absolute dB error, which is what
    bounds the result near 0 dB where ulps of the output are meaningless.
    """
    u = unit_roundoff(dtype)
    rtol = 2 * u * (LOG10_ULP + 0.5)
    input_ulp = (HYPOT_ULP if complex_input else 0) + 0.5
    atol = (scale / np.log(10)) * 2 * u * input_ulp
    return ROUNDOFF_SAFETY * n_impl * rtol, ROUNDOFF_SAFETY * n_impl * atol


def pow_conversion_rtol(dtype, max_abs_dB, n_impl=1):
    """rtol for 10**(x/10) with |x| <= max_abs_dB.

    Rounding x/10 perturbs the exponent, so its effect scales with |x|.
    """
    u = unit_roundoff(dtype)
    rtol = u * (np.log(10) * max_abs_dB / 10 + 2 * POW_ULP)
    return ROUNDOFF_SAFETY * n_impl * rtol


def envelope_power_rtol(dtype, complex_input=False, n_impl=1):
    """rtol for |x|**2"""
    u = unit_roundoff(dtype)
    rtol = 2 * u * (0.5 + (2 * HYPOT_ULP if complex_input else 0))
    return ROUNDOFF_SAFETY * n_impl * rtol


def linear_stat_tol(dtype, max_abs_dB, n, n_impl=1):
    """(rtol, atol) in dB for dBlinmean/dBlinsum over n terms with |x| <= max_abs_dB"""
    u = unit_roundoff(dtype)
    rel_linear = pow_conversion_rtol(dtype, max_abs_dB) + ROUNDOFF_SAFETY * n * u
    rtol, atol = log_conversion_tol(dtype, 10)
    return n_impl * rtol, n_impl * (atol + DB_PER_NEPER * rel_linear)


def roundtrip_power_rtol(dtype, max_abs_dB):
    """rtol for dBtopow(powtodB(x)) with |powtodB(x)| <= max_abs_dB"""
    rtol_dB, atol_dB = log_conversion_tol(dtype, 10)
    dB_err = rtol_dB * max_abs_dB + atol_dB
    return np.log(10) / 10 * dB_err + pow_conversion_rtol(dtype, max_abs_dB)


def roundtrip_dB_tol(dtype, max_abs_dB):
    """(rtol, atol) for powtodB(dBtopow(x)) with |x| <= max_abs_dB"""
    rtol, atol = log_conversion_tol(dtype, 10)
    return rtol, atol + DB_PER_NEPER * pow_conversion_rtol(dtype, max_abs_dB)


def linear_tolerance_dB(rtol):
    """express a relative tolerance on a linear power as a tolerance in dB"""
    return 10 * np.log10(1 + rtol)


def dB_tolerance(rtol, atol, max_abs_dB):
    """express (rtol, atol) on a dB-valued output as its worst-case tolerance in dB"""
    return atol + rtol * max_abs_dB


class TestUnitConversionProperties:
    """Property: Unit conversions form bijections (invertible mappings)."""

    UNIT_PAIRS = [
        ('dBm', 'mW'),
        ('dBW', 'W'),
        ('dB', 'unitless'),
        ('dBm/Hz', 'mW/Hz'),
        ('dBW/Hz', 'W/Hz'),
    ]

    WAVE_PAIRS = [
        ('dBm', '√mW'),
        ('dBW', '√W'),
        ('dB', '√unitless'),
    ]

    @pytest.mark.parametrize('dB_unit,linear_unit', UNIT_PAIRS)
    def test_dB_linear_roundtrip(self, dB_unit: str, linear_unit: str):
        """Property: dB → linear → dB is identity."""
        assert unit_linear_to_dB(unit_dB_to_linear(dB_unit)) == dB_unit

    @pytest.mark.parametrize('dB_unit,linear_unit', UNIT_PAIRS)
    def test_linear_dB_roundtrip(self, dB_unit: str, linear_unit: str):
        """Property: linear → dB → linear is identity."""
        assert unit_dB_to_linear(unit_linear_to_dB(linear_unit)) == linear_unit

    @pytest.mark.parametrize('dB_unit,wave_unit', WAVE_PAIRS)
    def test_dB_wave_roundtrip(self, dB_unit: str, wave_unit: str):
        """Property: dB → wave → dB is identity."""
        assert unit_wave_to_dB(unit_dB_to_wave(dB_unit)) == dB_unit


class TestConversionIdentities:
    """Properties: Mathematical identities that must hold for dB conversions."""

    @given(power=positive_power_arrays(dtype=np.float64))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_powtodB_dBtopow_roundtrip(self, power):
        """Property: dBtopow(powtodB(x)) ≈ x for all positive x."""
        roundtrip = dBtopow(powtodB(power))
        rtol = roundtrip_power_rtol(power.dtype, np.abs(powtodB(power)).max())
        assert_allclose(roundtrip, power, rtol=rtol)

    @given(dB=dB_arrays(min_value=-140, max_value=100, dtype=np.float64))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_dBtopow_powtodB_roundtrip(self, dB):
        """Property: powtodB(dBtopow(x)) ≈ x."""
        roundtrip = powtodB(dBtopow(dB))
        rtol, atol = roundtrip_dB_tol(dB.dtype, np.abs(dB).max())
        assert_allclose(roundtrip, dB, rtol=rtol, atol=atol)

    @given(
        env=envelope_arrays(
            include_complex=False,
            dtype=np.float64,
            min_magnitude=1e-6,
            max_magnitude=1e6,
        )
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_envtodB_equals_powtodB_envtopow(self, env):
        """Property: envtodB(x) = powtodB(envtopow(x))."""
        direct = envtodB(env)
        via_power = powtodB(envtopow(env))
        rtol, atol = log_conversion_tol(env.dtype, 20, n_impl=2)
        assert_allclose(direct, via_power, rtol=rtol, atol=atol)

    @given(env=envelope_arrays(include_complex=False, dtype=np.float64))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_envtopow_is_square(self, env):
        """Property: envtopow(x) = |x|² for real positive x."""
        result = envtopow(env)
        expected = np.abs(env) ** 2
        rtol = envelope_power_rtol(env.dtype, n_impl=2)
        assert_allclose(result, expected, rtol=rtol)

    @given(power=positive_power_arrays(dtype=np.float64))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_powtodB_is_10log10(self, power):
        """Property: powtodB(x) = 10 * log10(x)."""
        result = powtodB(power)
        expected = 10 * np.log10(power)
        rtol, atol = log_conversion_tol(power.dtype, 10, n_impl=2)
        assert_allclose(result, expected, rtol=rtol, atol=atol)

    @given(env=envelope_arrays(include_complex=False, dtype=np.float64))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_envtodB_is_20log10(self, env):
        """Property: envtodB(x) = 20 * log10(|x|)."""
        result = envtodB(env)
        expected = 20 * np.log10(np.abs(env))
        rtol, atol = log_conversion_tol(env.dtype, 20, n_impl=2)
        assert_allclose(result, expected, rtol=rtol, atol=atol)


class TestAlgebraicProperties:
    """Properties: Algebraic rules for dB arithmetic."""

    @given(
        base_dB=st.floats(
            min_value=-140, max_value=100, allow_nan=False, allow_infinity=False
        )
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_3dB_rule(self, base_dB):
        """Property: Doubling power adds ~3.01 dB.

        dBlinsum([x, x]) = x + 10*log10(2) ≈ x + 3.01
        """
        dB = np.array([base_dB, base_dB], dtype=np.float64)
        result = dBlinsum(dB)
        expected = base_dB + 10 * np.log10(2)
        rtol, atol = linear_stat_tol(np.float64, abs(base_dB), 2)
        assert_allclose(result, expected, rtol=rtol, atol=atol)

    @given(
        base_dB=st.floats(
            min_value=-140, max_value=100, allow_nan=False, allow_infinity=False
        )
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_10dB_rule(self, base_dB):
        """Property: 10x power adds exactly 10 dB.

        dBlinsum([x]*10) = x + 10
        """
        dB = np.array([base_dB] * 10, dtype=np.float64)
        result = dBlinsum(dB)
        expected = base_dB + 10.0
        rtol, atol = linear_stat_tol(np.float64, abs(base_dB), 10)
        assert_allclose(result, expected, rtol=rtol, atol=atol)

    @given(
        dB=dB_arrays(
            min_value=-50,
            max_value=50,
            min_size=2,
            max_size=50,
            dtype=np.float64,
        )
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_mean_sum_relationship(self, dB):
        """Property: dBlinmean = dBlinsum - 10*log10(N).

        Linear mean = linear sum / N, so in dB:
        mean_dB = sum_dB - 10*log10(N)
        """
        N = dB.size
        mean_result = dBlinmean(dB, axis=None)
        sum_result = dBlinsum(dB, axis=None)
        expected_mean = sum_result - 10 * np.log10(N)
        rtol, atol = linear_stat_tol(dB.dtype, np.abs(dB).max(), N, n_impl=2)
        assert_allclose(mean_result, expected_mean, rtol=rtol, atol=atol)

    @given(
        dB=st.floats(
            min_value=-140, max_value=100, allow_nan=False, allow_infinity=False
        )
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_single_value_mean_identity(self, dB):
        """Property: dBlinmean([x]) = x."""
        arr = np.array([dB], dtype=np.float64)
        result = dBlinmean(arr)
        rtol, atol = linear_stat_tol(np.float64, abs(dB), 1)
        assert_allclose(result, dB, rtol=rtol, atol=atol)

    @given(
        dB=st.floats(
            min_value=-140, max_value=100, allow_nan=False, allow_infinity=False
        )
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_single_value_sum_identity(self, dB):
        """Property: dBlinsum([x]) = x."""
        arr = np.array([dB], dtype=np.float64)
        result = dBlinsum(arr)
        rtol, atol = linear_stat_tol(np.float64, abs(dB), 1)
        assert_allclose(result, dB, rtol=rtol, atol=atol)

    @given(
        dB=st.floats(
            min_value=-140, max_value=100, allow_nan=False, allow_infinity=False
        ),
        n=st.integers(min_value=1, max_value=20),
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_equal_values_mean(self, dB, n):
        """Property: dBlinmean([x, x, ..., x]) = x."""
        arr = np.array([dB] * n, dtype=np.float64)
        result = dBlinmean(arr)
        rtol, atol = linear_stat_tol(np.float64, abs(dB), n)
        assert_allclose(result, dB, rtol=rtol, atol=atol)


class TestDtypePreservation:
    """Properties: Output dtype should be at least min_dtype."""

    @given(power=positive_power_arrays(dtype=np.float32))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_powtodB_float32_preserved(self, power):
        """Property: float32 input with min_dtype='float32' → float32 output."""
        result = powtodB(power, min_dtype='float32')
        assert result.dtype == np.float32

    @given(power=positive_power_arrays(dtype=np.float64))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_powtodB_float64_preserved(self, power):
        """Property: float64 input is never downgraded."""
        result = powtodB(power, min_dtype='float32')
        assert result.dtype == np.float64

    @given(dB=dB_arrays(dtype=np.float32))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_dBtopow_float32_preserved(self, dB):
        """Property: float32 input with min_dtype='float32' → float32 output."""
        result = dBtopow(dB, min_dtype='float32')
        assert result.dtype == np.float32

    @given(env=envelope_arrays(include_complex=False, dtype=np.float32))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_envtodB_float32_preserved(self, env):
        """Property: float32 input with min_dtype='float32' → float32 output."""
        result = envtodB(env, min_dtype='float32')
        assert result.dtype == np.float32

    @given(env=envelope_arrays(include_complex=False, dtype=np.float32))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_envtopow_float32_preserved(self, env):
        """Property: float32 input with min_dtype='float32' → float32 output."""
        result = envtopow(env, min_dtype='float32')
        assert result.dtype == np.float32


class TestInputPreservation:
    """Properties: overwrite_x=False must not modify input arrays."""

    @given(power=positive_power_arrays(dtype=np.float64))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_powtodB_preserves_input(self, power):
        """Property: powtodB(x, overwrite_x=False) does not modify x."""
        original = power.copy()
        powtodB(power, overwrite_x=False)
        assert_allclose(power, original, rtol=0)

    @given(dB=dB_arrays(dtype=np.float64))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_dBtopow_preserves_input(self, dB):
        """Property: dBtopow(x, overwrite_x=False) does not modify x."""
        original = dB.copy()
        dBtopow(dB, overwrite_x=False)
        assert_allclose(dB, original, rtol=0)

    @given(env=envelope_arrays(include_complex=False, dtype=np.float64))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_envtodB_preserves_input(self, env):
        """Property: envtodB(x, overwrite_x=False) does not modify x."""
        original = env.copy()
        envtodB(env, overwrite_x=False)
        assert_allclose(env, original, rtol=0)

    @given(env=envelope_arrays(include_complex=False, dtype=np.float64))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_envtopow_preserves_input(self, env):
        """Property: envtopow(x, overwrite_x=False) does not modify x."""
        original = env.copy()
        envtopow(env, overwrite_x=False)
        assert_allclose(env, original, rtol=0)

    @given(dB=dB_arrays(dtype=np.float64, min_value=-50, max_value=50))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_dBlinmean_preserves_input(self, dB):
        """Property: dBlinmean(x, overwrite_x=False) does not modify x."""
        original = dB.copy()
        dBlinmean(dB, overwrite_x=False)
        assert_allclose(dB, original, rtol=0)

    @given(dB=dB_arrays(dtype=np.float64, min_value=-50, max_value=50))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_dBlinsum_preserves_input(self, dB):
        """Property: dBlinsum(x, overwrite_x=False) does not modify x."""
        original = dB.copy()
        dBlinsum(dB, overwrite_x=False)
        assert_allclose(dB, original, rtol=0)


class TestEdgeCases:
    """Properties: Behavior at edge cases (zeros, extreme values)."""

    @given(n=st.integers(min_value=1, max_value=10))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_powtodB_zero_produces_neg_inf(self, n):
        """Property: powtodB(0) = -inf when eps=0."""
        power = np.zeros(n, dtype=np.float64)
        result = powtodB(power, eps=0)
        assert np.all(result == -np.inf)

    @given(n=st.integers(min_value=1, max_value=10))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_envtodB_zero_produces_neg_inf(self, n):
        """Property: envtodB(0) = -inf when eps=0."""
        env = np.zeros(n, dtype=np.float64)
        result = envtodB(env, eps=0)
        assert np.all(result == -np.inf)

    @given(eps=st.floats(min_value=1e-30, max_value=1e-10, allow_nan=False))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_powtodB_eps_avoids_neg_inf(self, eps):
        """Property: powtodB(0, eps=ε) is finite for ε > 0."""
        power = np.array([0.0], dtype=np.float64)
        result = powtodB(power, eps=eps)
        assert np.isfinite(result[0])
        rtol, atol = log_conversion_tol(np.float64, 10, n_impl=2)
        assert_allclose(result[0], 10 * np.log10(eps), rtol=rtol, atol=atol)

    @given(eps=st.floats(min_value=1e-30, max_value=1e-10, allow_nan=False))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_envtodB_eps_avoids_neg_inf(self, eps):
        """Property: envtodB(0, eps=ε) is finite for ε > 0."""
        env = np.array([0.0], dtype=np.float64)
        result = envtodB(env, eps=eps)
        assert np.isfinite(result[0])
        rtol, atol = log_conversion_tol(np.float64, 20, n_impl=2)
        assert_allclose(result[0], 20 * np.log10(eps), rtol=rtol, atol=atol)

    def test_empty_array(self):
        """Property: Empty arrays produce empty results."""
        empty = np.array([], dtype=np.float64)
        assert powtodB(empty).shape == (0,)
        assert dBtopow(empty).shape == (0,)
        assert envtodB(empty).shape == (0,)
        assert envtopow(empty).shape == (0,)

    @given(
        value=st.floats(
            min_value=1e-10, max_value=1e10, allow_nan=False, allow_infinity=False
        )
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_scalar_like_array(self, value):
        """Property: 0-d arrays work correctly."""
        scalar = np.array(value, dtype=np.float64)
        result = powtodB(scalar)
        expected = 10 * np.log10(value)
        rtol, atol = log_conversion_tol(np.float64, 10, n_impl=2)
        assert_allclose(result, expected, rtol=rtol, atol=atol)


class TestComplexValues:
    """Properties: Correct handling of complex-valued inputs."""

    @given(
        env=envelope_arrays(
            include_complex=True,
            min_magnitude=1e-6,
            max_magnitude=1e6,
            dtype=np.float64,
        )
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_envtopow_complex_is_magnitude_squared(self, env):
        """Property: envtopow(z) = |z|² for complex z."""
        result = envtopow(env)
        expected = np.abs(env) ** 2
        rtol = envelope_power_rtol(np.float64, complex_input=True, n_impl=2)
        assert_allclose(result.real, expected, rtol=rtol)

    @given(
        env=envelope_arrays(
            include_complex=True,
            min_magnitude=1e-6,
            max_magnitude=1e6,
            dtype=np.float64,
        )
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_envtodB_complex_is_20log10_magnitude(self, env):
        """Property: envtodB(z) = 20*log10(|z|) for complex z."""
        result = envtodB(env)
        expected = 20 * np.log10(np.abs(env))
        rtol, atol = log_conversion_tol(np.float64, 20, complex_input=True, n_impl=2)
        assert_allclose(result.real, expected, rtol=rtol, atol=atol)


class TestAxisParameter:
    """Properties: Correct behavior with axis parameter."""

    @given(data=st.data())
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_dBlinmean_axis_reduces_dimension(self, data):
        """Property: dBlinmean along axis reduces that dimension."""
        shape = data.draw(
            st.tuples(
                st.integers(min_value=2, max_value=10),
                st.integers(min_value=2, max_value=10),
            )
        )
        dB = data.draw(
            arrays(
                dtype=np.float64,
                shape=shape,
                elements=st.floats(
                    min_value=-50, max_value=50, allow_nan=False, allow_infinity=False
                ),
            )
        )
        axis = data.draw(st.integers(min_value=0, max_value=1))

        result = dBlinmean(dB, axis=axis)
        expected_shape = list(shape)
        del expected_shape[axis]
        assert result.shape == tuple(expected_shape)

    @given(data=st.data())
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_dBlinsum_axis_reduces_dimension(self, data):
        """Property: dBlinsum along axis reduces that dimension."""
        shape = data.draw(
            st.tuples(
                st.integers(min_value=2, max_value=10),
                st.integers(min_value=2, max_value=10),
            )
        )
        dB = data.draw(
            arrays(
                dtype=np.float64,
                shape=shape,
                elements=st.floats(
                    min_value=-50, max_value=50, allow_nan=False, allow_infinity=False
                ),
            )
        )
        axis = data.draw(st.integers(min_value=0, max_value=1))

        result = dBlinsum(dB, axis=axis)
        expected_shape = list(shape)
        del expected_shape[axis]
        assert result.shape == tuple(expected_shape)

    @given(data=st.data())
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_mean_sum_relationship_with_axis(self, data):
        """Property: mean/sum relationship holds for any axis."""
        shape = data.draw(
            st.tuples(
                st.integers(min_value=2, max_value=10),
                st.integers(min_value=2, max_value=10),
            )
        )
        dB = data.draw(
            arrays(
                dtype=np.float64,
                shape=shape,
                elements=st.floats(
                    min_value=-50, max_value=50, allow_nan=False, allow_infinity=False
                ),
            )
        )
        axis = data.draw(st.integers(min_value=0, max_value=1))

        N = shape[axis]
        mean_result = dBlinmean(dB, axis=axis)
        sum_result = dBlinsum(dB, axis=axis)
        expected_mean = sum_result - 10 * np.log10(N)
        rtol, atol = linear_stat_tol(np.float64, 50, N, n_impl=2)
        assert_allclose(mean_result, expected_mean, rtol=rtol, atol=atol)


class TestMinDtypePromotion:
    """Properties: min_dtype promotes low-precision inputs."""

    @given(power=positive_power_arrays(min_value=1e-3, max_value=1e3, dtype=np.float32))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_float16_promoted_to_float32(self, power):
        """Property: float16 input with min_dtype='float32' → float32 output.

        The expected value is computed from the float16-quantized input, not the
        original float32 draw.
        """
        power_f16 = power.astype(np.float16)
        result = powtodB(power_f16, min_dtype='float32')
        assert result.dtype == np.float32

        expected = powtodB(power_f16.astype(np.float32), min_dtype='float32')
        rtol, atol = log_conversion_tol(np.float32, 10, n_impl=2)
        assert_allclose(result, expected, rtol=rtol, atol=atol)

    @given(dB=dB_arrays(min_value=-30, max_value=30, dtype=np.float32))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_dBtopow_float16_promoted(self, dB):
        """Property: float16 input with min_dtype='float32' → float32 output."""
        dB_f16 = dB.astype(np.float16)
        result = dBtopow(dB_f16, min_dtype='float32')
        assert result.dtype == np.float32


class TestMultiBackendRoundtrip:
    """Properties: Roundtrip conversions work across all backends."""

    @given(data=for_each_namespace(positive_power_arrays(dtype=np.float64)))
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None
    )
    def test_powtodB_dBtopow_roundtrip_multibackend(self, data):
        """Property: dBtopow(powtodB(x)) ≈ x for all backends."""
        arr, xp_name, xp = data

        dB_result = powtodB(arr)
        roundtrip = dBtopow(dB_result)

        original_np = to_numpy(arr)
        roundtrip_np = to_numpy(roundtrip)

        rtol = roundtrip_power_rtol(np.float64, np.abs(powtodB(original_np)).max())
        assert_allclose(roundtrip_np, original_np, rtol=rtol)

    @given(
        data=for_each_namespace(
            dB_arrays(min_value=-100, max_value=100, dtype=np.float64)
        )
    )
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None
    )
    def test_dBtopow_powtodB_roundtrip_multibackend(self, data):
        """Property: powtodB(dBtopow(x)) ≈ x for all backends."""
        arr, xp_name, xp = data

        pow_result = dBtopow(arr)
        roundtrip = powtodB(pow_result)

        original_np = to_numpy(arr)
        roundtrip_np = to_numpy(roundtrip)

        rtol, atol = roundtrip_dB_tol(np.float64, np.abs(original_np).max())
        assert_allclose(roundtrip_np, original_np, rtol=rtol, atol=atol)


class TestMultiBackendDtypePreservation:
    """Properties: Dtype preservation works across all backends."""

    @given(data=for_each_namespace(positive_power_arrays(dtype=np.float32)))
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None
    )
    def test_powtodB_float32_preserved_multibackend(self, data):
        """Property: float32 input → float32 output for all backends."""
        arr, xp_name, xp = data

        result = powtodB(arr, min_dtype='float32')
        result_np = to_numpy(result)

        assert result_np.dtype == np.float32, (
            f'Expected float32 but got {result_np.dtype} for backend {xp_name}'
        )

    @given(data=for_each_namespace(dB_arrays(dtype=np.float32)))
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None
    )
    def test_dBtopow_float32_preserved_multibackend(self, data):
        """Property: float32 input → float32 output for all backends."""
        arr, xp_name, xp = data

        result = dBtopow(arr, min_dtype='float32')
        result_np = to_numpy(result)

        assert result_np.dtype == np.float32, (
            f'Expected float32 but got {result_np.dtype} for backend {xp_name}'
        )


class TestMultiBackendAlgebraicProperties:
    """Properties: Algebraic rules hold across all backends."""

    @given(
        data=for_each_namespace(
            dB_arrays(
                min_value=-50,
                max_value=50,
                min_size=2,
                max_size=20,
                dtype=np.float64,
            )
        )
    )
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None
    )
    def test_mean_sum_relationship_multibackend(self, data):
        """Property: dBlinmean = dBlinsum - 10*log10(N) for all backends."""
        arr, xp_name, xp = data

        arr_np = to_numpy(arr)
        N = arr_np.size

        mean_result = dBlinmean(arr, axis=None)
        sum_result = dBlinsum(arr, axis=None)

        mean_np = to_numpy(mean_result)
        sum_np = to_numpy(sum_result)

        expected_mean = sum_np - 10 * np.log10(N)
        rtol, atol = linear_stat_tol(np.float64, np.abs(arr_np).max(), N, n_impl=2)
        assert_allclose(mean_np, expected_mean, rtol=rtol, atol=atol)


class TestMultiBackendComplexValues:
    """Properties: Complex value handling works across all backends."""

    @given(
        data=for_each_namespace(
            envelope_arrays(
                include_complex=True,
                min_magnitude=1e-6,
                max_magnitude=1e6,
                dtype=np.float64,
            )
        )
    )
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None
    )
    def test_envtopow_complex_multibackend(self, data):
        """Property: envtopow(z) = |z|² for complex z across all backends."""
        arr, xp_name, xp = data

        result = envtopow(arr)

        arr_np = to_numpy(arr)
        result_np = to_numpy(result)
        expected = np.abs(arr_np) ** 2

        rtol = envelope_power_rtol(np.float64, complex_input=True, n_impl=2)
        assert_allclose(result_np.real, expected, rtol=rtol)

    @given(
        data=for_each_namespace(
            envelope_arrays(
                include_complex=True,
                min_magnitude=1e-6,
                max_magnitude=1e6,
                dtype=np.float64,
            )
        )
    )
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None
    )
    def test_envtodB_complex_multibackend(self, data):
        """Property: envtodB(z) = 20*log10(|z|) for complex z across all backends."""
        arr, xp_name, xp = data

        result = envtodB(arr)

        arr_np = to_numpy(arr)
        result_np = to_numpy(result)
        expected = 20 * np.log10(np.abs(arr_np))

        rtol, atol = log_conversion_tol(np.float64, 20, complex_input=True, n_impl=2)
        assert_allclose(result_np.real, expected, rtol=rtol, atol=atol)


class TestNumpyCupyCrossComparison:
    """Cross-comparison tests validating numpy and cupy produce close results.

    Tolerances come from the ulp budgets at the top of this module, so a difference
    larger than the two libraries' rounding bounds fails the test.
    """

    @pytest.fixture
    def cupy_available(self):
        from conftest import _cupy

        if _cupy is None:
            pytest.skip('cupy is not available')
        return _cupy

    @given(
        power=positive_power_arrays(
            min_value=1e-10, max_value=1e10, dtype=np.float64, min_dims=1, max_dims=1
        )
    )
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None
    )
    def test_powtodB_numpy_vs_cupy_float64(self, cupy_available, power):
        """Cross-comparison: powtodB numpy vs cupy (float64)."""
        cp = cupy_available

        result_np = powtodB(power)

        power_cp = cp.asarray(power)
        result_cp = powtodB(power_cp)
        result_cp_np = result_cp.get()

        rtol, atol = log_conversion_tol(np.float64, 10, n_impl=2)
        tol_dB = dB_tolerance(rtol, atol, np.abs(result_np).max())
        assert_allclose(
            result_cp_np, result_np, rtol=rtol, atol=atol, err_msg=f'{tol_dB:.2e} dB'
        )

    @given(
        power=positive_power_arrays(
            min_value=1e-5, max_value=1e5, dtype=np.float32, min_dims=1, max_dims=1
        )
    )
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None
    )
    def test_powtodB_numpy_vs_cupy_float32(self, cupy_available, power):
        """Cross-comparison: powtodB numpy vs cupy (float32)."""
        cp = cupy_available

        result_np = powtodB(power, min_dtype='float32')

        power_cp = cp.asarray(power)
        result_cp = powtodB(power_cp, min_dtype='float32')
        result_cp_np = result_cp.get()

        rtol, atol = log_conversion_tol(np.float32, 10, n_impl=2)
        tol_dB = dB_tolerance(rtol, atol, np.abs(result_np).max())
        assert_allclose(
            result_cp_np, result_np, rtol=rtol, atol=atol, err_msg=f'{tol_dB:.2e} dB'
        )

    @given(
        dB=dB_arrays(
            min_value=-100,
            max_value=100,
            dtype=np.float64,
            min_dims=1,
            max_dims=1,
        )
    )
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None
    )
    def test_dBtopow_numpy_vs_cupy_float64(self, cupy_available, dB):
        """Cross-comparison: dBtopow numpy vs cupy (float64)."""
        cp = cupy_available

        result_np = dBtopow(dB)

        dB_cp = cp.asarray(dB)
        result_cp = dBtopow(dB_cp)
        result_cp_np = result_cp.get()

        rtol = pow_conversion_rtol(np.float64, np.abs(dB).max(), n_impl=2)
        tol_dB = linear_tolerance_dB(rtol)
        assert_allclose(result_cp_np, result_np, rtol=rtol, err_msg=f'{tol_dB:.2e} dB')

    @given(
        dB=dB_arrays(
            min_value=-30,
            max_value=30,
            dtype=np.float32,
            min_dims=1,
            max_dims=1,
        )
    )
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None
    )
    def test_dBtopow_numpy_vs_cupy_float32(self, cupy_available, dB):
        """Cross-comparison: dBtopow numpy vs cupy (float32)."""
        cp = cupy_available

        result_np = dBtopow(dB, min_dtype='float32')

        dB_cp = cp.asarray(dB)
        result_cp = dBtopow(dB_cp, min_dtype='float32')
        result_cp_np = result_cp.get()

        rtol = pow_conversion_rtol(np.float32, np.abs(dB).max(), n_impl=2)
        tol_dB = linear_tolerance_dB(rtol)
        assert_allclose(result_cp_np, result_np, rtol=rtol, err_msg=f'{tol_dB:.2e} dB')

    @given(
        env=envelope_arrays(
            include_complex=False,
            min_magnitude=1e-6,
            max_magnitude=1e6,
            dtype=np.float64,
            min_dims=1,
            max_dims=1,
        )
    )
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None
    )
    def test_envtodB_numpy_vs_cupy_float64(self, cupy_available, env):
        """Cross-comparison: envtodB numpy vs cupy (float64)."""
        cp = cupy_available

        result_np = envtodB(env)

        env_cp = cp.asarray(env)
        result_cp = envtodB(env_cp)
        result_cp_np = result_cp.get()

        rtol, atol = log_conversion_tol(np.float64, 20, n_impl=2)
        tol_dB = dB_tolerance(rtol, atol, np.abs(result_np).max())
        assert_allclose(
            result_cp_np, result_np, rtol=rtol, atol=atol, err_msg=f'{tol_dB:.2e} dB'
        )

    @given(
        env=envelope_arrays(
            include_complex=False,
            min_magnitude=1e-4,
            max_magnitude=1e4,
            dtype=np.float32,
            min_dims=1,
            max_dims=1,
        )
    )
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None
    )
    def test_envtodB_numpy_vs_cupy_float32(self, cupy_available, env):
        """Cross-comparison: envtodB numpy vs cupy (float32)."""
        cp = cupy_available

        result_np = envtodB(env, min_dtype='float32')

        env_cp = cp.asarray(env)
        result_cp = envtodB(env_cp, min_dtype='float32')
        result_cp_np = result_cp.get()

        rtol, atol = log_conversion_tol(np.float32, 20, n_impl=2)
        tol_dB = dB_tolerance(rtol, atol, np.abs(result_np).max())
        assert_allclose(
            result_cp_np, result_np, rtol=rtol, atol=atol, err_msg=f'{tol_dB:.2e} dB'
        )

    @given(
        env=envelope_arrays(
            include_complex=False,
            min_magnitude=1e-6,
            max_magnitude=1e6,
            dtype=np.float64,
            min_dims=1,
            max_dims=1,
        )
    )
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None
    )
    def test_envtopow_numpy_vs_cupy_float64(self, cupy_available, env):
        """Cross-comparison: envtopow numpy vs cupy (float64)."""
        cp = cupy_available

        result_np = envtopow(env)

        env_cp = cp.asarray(env)
        result_cp = envtopow(env_cp)
        result_cp_np = result_cp.get()

        rtol = envelope_power_rtol(np.float64, n_impl=2)
        tol_dB = linear_tolerance_dB(rtol)
        assert_allclose(result_cp_np, result_np, rtol=rtol, err_msg=f'{tol_dB:.2e} dB')

    @given(
        dB=dB_arrays(
            min_value=-50,
            max_value=50,
            min_size=2,
            max_size=50,
            dtype=np.float64,
            min_dims=1,
            max_dims=1,
        )
    )
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None
    )
    def test_dBlinmean_numpy_vs_cupy_float64(self, cupy_available, dB):
        """Cross-comparison: dBlinmean numpy vs cupy (float64)."""
        cp = cupy_available

        result_np = dBlinmean(dB, axis=None)

        dB_cp = cp.asarray(dB)
        result_cp = dBlinmean(dB_cp, axis=None)
        result_cp_np = float(result_cp.get())

        rtol, atol = linear_stat_tol(np.float64, np.abs(dB).max(), dB.size, n_impl=2)
        tol_dB = dB_tolerance(rtol, atol, abs(result_np))
        assert_allclose(
            result_cp_np, result_np, rtol=rtol, atol=atol, err_msg=f'{tol_dB:.2e} dB'
        )

    @given(
        dB=dB_arrays(
            min_value=-50,
            max_value=50,
            min_size=2,
            max_size=50,
            dtype=np.float64,
            min_dims=1,
            max_dims=1,
        )
    )
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None
    )
    def test_dBlinsum_numpy_vs_cupy_float64(self, cupy_available, dB):
        """Cross-comparison: dBlinsum numpy vs cupy (float64)."""
        cp = cupy_available

        result_np = dBlinsum(dB, axis=None)

        dB_cp = cp.asarray(dB)
        result_cp = dBlinsum(dB_cp, axis=None)
        result_cp_np = float(result_cp.get())

        rtol, atol = linear_stat_tol(np.float64, np.abs(dB).max(), dB.size, n_impl=2)
        tol_dB = dB_tolerance(rtol, atol, abs(result_np))
        assert_allclose(
            result_cp_np, result_np, rtol=rtol, atol=atol, err_msg=f'{tol_dB:.2e} dB'
        )
