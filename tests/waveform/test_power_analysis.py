"""Property-based tests for dB conversion functions using Hypothesis.

This module demonstrates how to use Hypothesis to reduce boilerplate and make
test specifications clearer. Key benefits:

1. **Reduced boilerplate**: Custom strategies encapsulate array generation logic
2. **Property-based testing**: Tests express mathematical invariants directly
3. **Automatic edge case discovery**: Hypothesis finds corner cases automatically
4. **Clear specifications**: Each test documents a mathematical property
5. **Multi-backend support**: Tests run against numpy, cupy, and dask arrays

Test Categories:
- Mathematical identities (roundtrips, inverses)
- Algebraic properties (3dB rule, 10dB rule, mean/sum relationship)
- Dtype preservation
- Input preservation (overwrite_x behavior)
- Edge cases (zeros, infinities, extreme values)
- Multi-backend compatibility (numpy, cupy, dask)
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

# Import shared strategies and utilities from conftest
from conftest import (
    to_numpy,
    positive_power_arrays,
    dB_arrays,
    envelope_arrays,
    available_namespaces,
    convert_array,
    for_each_namespace,
)


# =============================================================================
# Unit Conversion Tests - Bijection properties
# =============================================================================

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


# =============================================================================
# Mathematical Identity Tests - Core dB conversion properties (numpy only)
# =============================================================================

class TestConversionIdentities:
    """Properties: Mathematical identities that must hold for dB conversions."""
    
    @given(power=positive_power_arrays(dtype=np.float64))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_powtodB_dBtopow_roundtrip(self, power):
        """Property: dBtopow(powtodB(x)) ≈ x for all positive x.
        
        This is the fundamental inverse relationship.
        Note: Uses float64 for precision; float32 has ~1e-6 relative error.
        """
        roundtrip = dBtopow(powtodB(power))
        assert_allclose(roundtrip, power, rtol=1e-10)
    
    @given(dB=dB_arrays(min_value=-140, max_value=100, dtype=np.float64, filter_near_zero=True))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_dBtopow_powtodB_roundtrip(self, dB):
        """Property: powtodB(dBtopow(x)) ≈ x for all finite x.
        
        Note: Range limited to ±100 dB to avoid underflow/overflow in linear domain.
        10^(-100/10) = 10^-10 is safely representable.
        """
        roundtrip = powtodB(dBtopow(dB))
        assert_allclose(roundtrip, dB, rtol=1e-9)
    
    @given(env=envelope_arrays(include_complex=False, dtype=np.float64, min_magnitude=1e-6, max_magnitude=1e6))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_envtodB_equals_powtodB_envtopow(self, env):
        """Property: envtodB(x) = powtodB(envtopow(x)).
        
        Envelope-to-dB should equal power-to-dB of squared envelope.
        Note: Magnitude limited to avoid squaring overflow/underflow.
        """
        direct = envtodB(env)
        via_power = powtodB(envtopow(env))
        assert_allclose(direct, via_power, rtol=1e-10)

    @given(env=envelope_arrays(include_complex=False, dtype=np.float64))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_envtopow_is_square(self, env):
        """Property: envtopow(x) = |x|² for real positive x."""
        result = envtopow(env)
        expected = np.abs(env) ** 2
        assert_allclose(result, expected, rtol=1e-10)
    
    @given(power=positive_power_arrays(dtype=np.float64))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_powtodB_is_10log10(self, power):
        """Property: powtodB(x) = 10 * log10(x)."""
        result = powtodB(power)
        expected = 10 * np.log10(power)
        assert_allclose(result, expected, rtol=1e-10)
    
    @given(env=envelope_arrays(include_complex=False, dtype=np.float64))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_envtodB_is_20log10(self, env):
        """Property: envtodB(x) = 20 * log10(|x|)."""
        result = envtodB(env)
        expected = 20 * np.log10(np.abs(env))
        assert_allclose(result, expected, rtol=1e-10)


# =============================================================================
# Algebraic Properties - dB arithmetic rules
# =============================================================================

class TestAlgebraicProperties:
    """Properties: Algebraic rules for dB arithmetic."""
    
    @given(base_dB=st.floats(min_value=-140, max_value=100, allow_nan=False, allow_infinity=False))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_3dB_rule(self, base_dB):
        """Property: Doubling power adds ~3.01 dB.
        
        dBlinsum([x, x]) = x + 10*log10(2) ≈ x + 3.01
        """
        dB = np.array([base_dB, base_dB], dtype=np.float64)
        result = dBlinsum(dB)
        expected = base_dB + 10 * np.log10(2)
        assert_allclose(result, expected, rtol=1e-10)
    
    @given(base_dB=st.floats(min_value=-140, max_value=100, allow_nan=False, allow_infinity=False))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_10dB_rule(self, base_dB):
        """Property: 10x power adds exactly 10 dB.
        
        dBlinsum([x]*10) = x + 10
        """
        dB = np.array([base_dB] * 10, dtype=np.float64)
        result = dBlinsum(dB)
        expected = base_dB + 10.0
        assert_allclose(result, expected, rtol=1e-10)
    
    @given(dB=dB_arrays(min_value=-50, max_value=50, min_size=2, max_size=50, dtype=np.float64, filter_near_zero=True))
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
        # Use looser tolerance due to floating-point accumulation
        assert_allclose(mean_result, expected_mean, rtol=1e-6)
    
    @given(dB=st.one_of(
        st.floats(min_value=-140, max_value=-1e-6, allow_nan=False, allow_infinity=False),
        st.floats(min_value=1e-6, max_value=100, allow_nan=False, allow_infinity=False),
    ))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_single_value_mean_identity(self, dB):
        """Property: dBlinmean([x]) = x (mean of single value is itself).
        
        Note: Range limited to ±100 dB to avoid underflow to zero in linear domain.
        Values very close to zero are excluded since relative tolerance is meaningless there.
        """
        arr = np.array([dB], dtype=np.float64)
        result = dBlinmean(arr)
        # Use looser tolerance due to dB→linear→dB roundtrip precision loss
        assert_allclose(result, dB, rtol=1e-9)
    
    @given(dB=st.one_of(
        st.floats(min_value=-140, max_value=-1e-6, allow_nan=False, allow_infinity=False),
        st.floats(min_value=1e-6, max_value=100, allow_nan=False, allow_infinity=False),
    ))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_single_value_sum_identity(self, dB):
        """Property: dBlinsum([x]) = x (sum of single value is itself).
        
        Note: Range limited to ±100 dB to avoid underflow to zero in linear domain.
        Values very close to zero are excluded since relative tolerance is meaningless there.
        """
        arr = np.array([dB], dtype=np.float64)
        result = dBlinsum(arr)
        # Use looser tolerance due to dB→linear→dB roundtrip precision loss
        assert_allclose(result, dB, rtol=1e-9)
    
    @given(dB=st.one_of(
        st.floats(min_value=-140, max_value=-1e-6, allow_nan=False, allow_infinity=False),
        st.floats(min_value=1e-6, max_value=100, allow_nan=False, allow_infinity=False),
    ), n=st.integers(min_value=1, max_value=20))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_equal_values_mean(self, dB, n):
        """Property: dBlinmean([x, x, ..., x]) = x (mean of equal values).
        
        Note: Range limited to ±100 dB to avoid underflow to zero in linear domain.
        Values very close to zero are excluded since relative tolerance is meaningless there.
        """
        arr = np.array([dB] * n, dtype=np.float64)
        result = dBlinmean(arr)
        assert_allclose(result, dB, rtol=1e-8)


# =============================================================================
# Dtype Preservation Tests
# =============================================================================

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


# =============================================================================
# Input Preservation Tests (overwrite_x behavior)
# =============================================================================

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


# =============================================================================
# Edge Case Tests - Zeros and special values
# =============================================================================

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
        assert_allclose(result[0], 10 * np.log10(eps), rtol=1e-6)
    
    @given(eps=st.floats(min_value=1e-30, max_value=1e-10, allow_nan=False))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_envtodB_eps_avoids_neg_inf(self, eps):
        """Property: envtodB(0, eps=ε) is finite for ε > 0."""
        env = np.array([0.0], dtype=np.float64)
        result = envtodB(env, eps=eps)
        assert np.isfinite(result[0])
        assert_allclose(result[0], 20 * np.log10(eps), rtol=1e-6)
    
    def test_empty_array(self):
        """Property: Empty arrays produce empty results."""
        empty = np.array([], dtype=np.float64)
        assert powtodB(empty).shape == (0,)
        assert dBtopow(empty).shape == (0,)
        assert envtodB(empty).shape == (0,)
        assert envtopow(empty).shape == (0,)
    
    @given(value=st.floats(min_value=1e-10, max_value=1e10, allow_nan=False, allow_infinity=False))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_scalar_like_array(self, value):
        """Property: 0-d arrays work correctly."""
        scalar = np.array(value, dtype=np.float64)
        result = powtodB(scalar)
        expected = 10 * np.log10(value)
        assert_allclose(result, expected, rtol=1e-10)


# =============================================================================
# Complex Value Tests
# =============================================================================

class TestComplexValues:
    """Properties: Correct handling of complex-valued inputs."""
    
    @given(env=envelope_arrays(include_complex=True, min_magnitude=1e-6, max_magnitude=1e6, dtype=np.float64))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_envtopow_complex_is_magnitude_squared(self, env):
        """Property: envtopow(z) = |z|² for complex z."""
        result = envtopow(env)
        expected = np.abs(env) ** 2
        assert_allclose(result.real, expected, rtol=1e-6)
    
    @given(env=envelope_arrays(include_complex=True, min_magnitude=1e-6, max_magnitude=1e6, dtype=np.float64))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_envtodB_complex_is_20log10_magnitude(self, env):
        """Property: envtodB(z) = 20*log10(|z|) for complex z."""
        result = envtodB(env)
        expected = 20 * np.log10(np.abs(env))
        # Use atol for values near zero where relative tolerance is meaningless
        assert_allclose(result.real, expected, rtol=1e-6, atol=1e-12)


# =============================================================================
# Axis Parameter Tests
# =============================================================================

class TestAxisParameter:
    """Properties: Correct behavior with axis parameter."""
    
    @given(data=st.data())
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_dBlinmean_axis_reduces_dimension(self, data):
        """Property: dBlinmean along axis reduces that dimension."""
        shape = data.draw(st.tuples(
            st.integers(min_value=2, max_value=10),
            st.integers(min_value=2, max_value=10),
        ))
        dB = data.draw(arrays(
            dtype=np.float64,
            shape=shape,
            elements=st.floats(min_value=-50, max_value=50, allow_nan=False, allow_infinity=False),
        ))
        axis = data.draw(st.integers(min_value=0, max_value=1))
        
        result = dBlinmean(dB, axis=axis)
        expected_shape = list(shape)
        del expected_shape[axis]
        assert result.shape == tuple(expected_shape)
    
    @given(data=st.data())
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_dBlinsum_axis_reduces_dimension(self, data):
        """Property: dBlinsum along axis reduces that dimension."""
        shape = data.draw(st.tuples(
            st.integers(min_value=2, max_value=10),
            st.integers(min_value=2, max_value=10),
        ))
        dB = data.draw(arrays(
            dtype=np.float64,
            shape=shape,
            elements=st.floats(min_value=-50, max_value=50, allow_nan=False, allow_infinity=False),
        ))
        axis = data.draw(st.integers(min_value=0, max_value=1))
        
        result = dBlinsum(dB, axis=axis)
        expected_shape = list(shape)
        del expected_shape[axis]
        assert result.shape == tuple(expected_shape)
    
    @given(data=st.data())
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_mean_sum_relationship_with_axis(self, data):
        """Property: mean/sum relationship holds for any axis."""
        shape = data.draw(st.tuples(
            st.integers(min_value=2, max_value=10),
            st.integers(min_value=2, max_value=10),
        ))
        # Exclude values near zero to avoid precision issues in dB→linear→dB roundtrip
        dB = data.draw(arrays(
            dtype=np.float64,
            shape=shape,
            elements=st.one_of(
                st.floats(min_value=-50, max_value=-1e-6, allow_nan=False, allow_infinity=False),
                st.floats(min_value=1e-6, max_value=50, allow_nan=False, allow_infinity=False),
            ),
        ))
        axis = data.draw(st.integers(min_value=0, max_value=1))
        
        N = shape[axis]
        mean_result = dBlinmean(dB, axis=axis)
        sum_result = dBlinsum(dB, axis=axis)
        expected_mean = sum_result - 10 * np.log10(N)
        assert_allclose(mean_result, expected_mean, rtol=1e-6)


# =============================================================================
# Min Dtype Promotion Tests
# =============================================================================

class TestMinDtypePromotion:
    """Properties: min_dtype promotes low-precision inputs."""
    
    @given(power=positive_power_arrays(min_value=1e-3, max_value=1e3, dtype=np.float32))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_float16_promoted_to_float32(self, power):
        """Property: float16 input with min_dtype='float32' → float32 output.
        
        Note: We use float32 input and cast to float16 to test promotion.
        The comparison is against the float16-converted input (not original float32)
        since float16 quantization changes the input values.
        """
        power_f16 = power.astype(np.float16)
        result = powtodB(power_f16, min_dtype='float32')
        assert result.dtype == np.float32
        
        # Verify result matches computation on the same float16 values
        # (converted back to float32 for the expected computation)
        expected = powtodB(power_f16.astype(np.float32), min_dtype='float32')
        assert_allclose(result, expected, rtol=1e-6)
    
    @given(dB=dB_arrays(min_value=-30, max_value=30, dtype=np.float32))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_dBtopow_float16_promoted(self, dB):
        """Property: float16 input with min_dtype='float32' → float32 output."""
        dB_f16 = dB.astype(np.float16)
        result = dBtopow(dB_f16, min_dtype='float32')
        assert result.dtype == np.float32


# =============================================================================
# Multi-Backend Tests - numpy, cupy, dask compatibility
# =============================================================================

class TestMultiBackendRoundtrip:
    """Properties: Roundtrip conversions work across all backends."""
    
    @given(data=for_each_namespace(positive_power_arrays(dtype=np.float64)))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
    def test_powtodB_dBtopow_roundtrip_multibackend(self, data):
        """Property: dBtopow(powtodB(x)) ≈ x for all backends."""
        arr, xp_name, xp = data
        
        # Perform roundtrip
        dB_result = powtodB(arr)
        roundtrip = dBtopow(dB_result)
        
        # Convert to numpy for comparison
        original_np = to_numpy(arr)
        roundtrip_np = to_numpy(roundtrip)
        
        assert_allclose(roundtrip_np, original_np, rtol=1e-10)
    
    @given(data=for_each_namespace(dB_arrays(min_value=-100, max_value=100, dtype=np.float64, filter_near_zero=True)))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
    def test_dBtopow_powtodB_roundtrip_multibackend(self, data):
        """Property: powtodB(dBtopow(x)) ≈ x for all backends."""
        arr, xp_name, xp = data
        
        # Perform roundtrip
        pow_result = dBtopow(arr)
        roundtrip = powtodB(pow_result)
        
        # Convert to numpy for comparison
        original_np = to_numpy(arr)
        roundtrip_np = to_numpy(roundtrip)
        
        assert_allclose(roundtrip_np, original_np, rtol=1e-9)


class TestMultiBackendDtypePreservation:
    """Properties: Dtype preservation works across all backends."""
    
    @given(data=for_each_namespace(positive_power_arrays(dtype=np.float32)))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
    def test_powtodB_float32_preserved_multibackend(self, data):
        """Property: float32 input → float32 output for all backends."""
        arr, xp_name, xp = data
        
        result = powtodB(arr, min_dtype='float32')
        result_np = to_numpy(result)
        
        assert result_np.dtype == np.float32, (
            f"Expected float32 but got {result_np.dtype} for backend {xp_name}"
        )
    
    @given(data=for_each_namespace(dB_arrays(dtype=np.float32)))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
    def test_dBtopow_float32_preserved_multibackend(self, data):
        """Property: float32 input → float32 output for all backends."""
        arr, xp_name, xp = data
        
        result = dBtopow(arr, min_dtype='float32')
        result_np = to_numpy(result)
        
        assert result_np.dtype == np.float32, (
            f"Expected float32 but got {result_np.dtype} for backend {xp_name}"
        )


class TestMultiBackendAlgebraicProperties:
    """Properties: Algebraic rules hold across all backends."""
    
    @given(data=for_each_namespace(dB_arrays(min_value=-50, max_value=50, min_size=2, max_size=20, dtype=np.float64, filter_near_zero=True)))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
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
        assert_allclose(mean_np, expected_mean, rtol=1e-6)


class TestMultiBackendComplexValues:
    """Properties: Complex value handling works across all backends."""
    
    @given(data=for_each_namespace(envelope_arrays(include_complex=True, min_magnitude=1e-6, max_magnitude=1e6, dtype=np.float64)))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
    def test_envtopow_complex_multibackend(self, data):
        """Property: envtopow(z) = |z|² for complex z across all backends."""
        arr, xp_name, xp = data
        
        result = envtopow(arr)
        
        arr_np = to_numpy(arr)
        result_np = to_numpy(result)
        expected = np.abs(arr_np) ** 2
        
        assert_allclose(result_np.real, expected, rtol=1e-6)
    
    @given(data=for_each_namespace(envelope_arrays(include_complex=True, min_magnitude=1e-6, max_magnitude=1e6, dtype=np.float64)))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
    def test_envtodB_complex_multibackend(self, data):
        """Property: envtodB(z) = 20*log10(|z|) for complex z across all backends."""
        arr, xp_name, xp = data
        
        result = envtodB(arr)
        
        arr_np = to_numpy(arr)
        result_np = to_numpy(result)
        expected = 20 * np.log10(np.abs(arr_np))
        
        assert_allclose(result_np.real, expected, rtol=1e-6, atol=1e-12)


# =============================================================================
# NumPy vs CuPy Cross-Comparison Tests
# =============================================================================

class TestNumpyCupyCrossComparison:
    """Cross-comparison tests validating numpy and cupy produce close results.
    
    These tests run the same computation on both numpy and cupy arrays and
    compare the results. Tolerances are set for IEEE fast-math precision
    (cupy may use fast-math optimizations).
    """
    
    # Fast-math tolerances: ~1e-5 relative for float32, ~1e-12 for float64
    RTOL_FLOAT32 = 1e-5
    RTOL_FLOAT64 = 1e-12
    ATOL = 1e-10
    
    @pytest.fixture
    def cupy_available(self):
        """Skip test if cupy is not available."""
        from conftest import _cupy
        if _cupy is None:
            pytest.skip('cupy is not available')
        return _cupy
    
    @given(power=positive_power_arrays(min_value=1e-10, max_value=1e10, dtype=np.float64, min_dims=1, max_dims=1))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
    def test_powtodB_numpy_vs_cupy_float64(self, cupy_available, power):
        """Cross-comparison: powtodB numpy vs cupy (float64)."""
        cp = cupy_available
        
        # Compute with numpy
        result_np = powtodB(power)
        
        # Compute with cupy
        power_cp = cp.asarray(power)
        result_cp = powtodB(power_cp)
        result_cp_np = result_cp.get()
        
        assert_allclose(result_cp_np, result_np, rtol=self.RTOL_FLOAT64, atol=self.ATOL)
    
    @given(power=positive_power_arrays(min_value=1e-5, max_value=1e5, dtype=np.float32, min_dims=1, max_dims=1))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
    def test_powtodB_numpy_vs_cupy_float32(self, cupy_available, power):
        """Cross-comparison: powtodB numpy vs cupy (float32)."""
        cp = cupy_available
        
        # Compute with numpy
        result_np = powtodB(power, min_dtype='float32')
        
        # Compute with cupy
        power_cp = cp.asarray(power)
        result_cp = powtodB(power_cp, min_dtype='float32')
        result_cp_np = result_cp.get()
        
        assert_allclose(result_cp_np, result_np, rtol=self.RTOL_FLOAT32, atol=self.ATOL)
    
    @given(dB=dB_arrays(min_value=-100, max_value=100, dtype=np.float64, min_dims=1, max_dims=1, filter_near_zero=True))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
    def test_dBtopow_numpy_vs_cupy_float64(self, cupy_available, dB):
        """Cross-comparison: dBtopow numpy vs cupy (float64)."""
        cp = cupy_available
        
        # Compute with numpy
        result_np = dBtopow(dB)
        
        # Compute with cupy
        dB_cp = cp.asarray(dB)
        result_cp = dBtopow(dB_cp)
        result_cp_np = result_cp.get()
        
        assert_allclose(result_cp_np, result_np, rtol=self.RTOL_FLOAT64, atol=self.ATOL)
    
    @given(dB=dB_arrays(min_value=-30, max_value=30, dtype=np.float32, min_dims=1, max_dims=1, filter_near_zero=True))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
    def test_dBtopow_numpy_vs_cupy_float32(self, cupy_available, dB):
        """Cross-comparison: dBtopow numpy vs cupy (float32)."""
        cp = cupy_available
        
        # Compute with numpy
        result_np = dBtopow(dB, min_dtype='float32')
        
        # Compute with cupy
        dB_cp = cp.asarray(dB)
        result_cp = dBtopow(dB_cp, min_dtype='float32')
        result_cp_np = result_cp.get()
        
        assert_allclose(result_cp_np, result_np, rtol=self.RTOL_FLOAT32, atol=self.ATOL)
    
    @given(env=envelope_arrays(include_complex=False, min_magnitude=1e-6, max_magnitude=1e6, dtype=np.float64, min_dims=1, max_dims=1))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
    def test_envtodB_numpy_vs_cupy_float64(self, cupy_available, env):
        """Cross-comparison: envtodB numpy vs cupy (float64)."""
        cp = cupy_available
        
        # Compute with numpy
        result_np = envtodB(env)
        
        # Compute with cupy
        env_cp = cp.asarray(env)
        result_cp = envtodB(env_cp)
        result_cp_np = result_cp.get()
        
        assert_allclose(result_cp_np, result_np, rtol=self.RTOL_FLOAT64, atol=self.ATOL)
    
    @given(env=envelope_arrays(include_complex=False, min_magnitude=1e-4, max_magnitude=1e4, dtype=np.float32, min_dims=1, max_dims=1))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
    def test_envtodB_numpy_vs_cupy_float32(self, cupy_available, env):
        """Cross-comparison: envtodB numpy vs cupy (float32)."""
        cp = cupy_available
        
        # Compute with numpy
        result_np = envtodB(env, min_dtype='float32')
        
        # Compute with cupy
        env_cp = cp.asarray(env)
        result_cp = envtodB(env_cp, min_dtype='float32')
        result_cp_np = result_cp.get()
        
        assert_allclose(result_cp_np, result_np, rtol=self.RTOL_FLOAT32, atol=self.ATOL)
    
    @given(env=envelope_arrays(include_complex=False, min_magnitude=1e-6, max_magnitude=1e6, dtype=np.float64, min_dims=1, max_dims=1))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
    def test_envtopow_numpy_vs_cupy_float64(self, cupy_available, env):
        """Cross-comparison: envtopow numpy vs cupy (float64)."""
        cp = cupy_available
        
        # Compute with numpy
        result_np = envtopow(env)
        
        # Compute with cupy
        env_cp = cp.asarray(env)
        result_cp = envtopow(env_cp)
        result_cp_np = result_cp.get()
        
        assert_allclose(result_cp_np, result_np, rtol=self.RTOL_FLOAT64, atol=self.ATOL)
    
    @given(dB=dB_arrays(min_value=-50, max_value=50, min_size=2, max_size=50, dtype=np.float64, min_dims=1, max_dims=1, filter_near_zero=True))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
    def test_dBlinmean_numpy_vs_cupy_float64(self, cupy_available, dB):
        """Cross-comparison: dBlinmean numpy vs cupy (float64)."""
        cp = cupy_available
        
        # Compute with numpy
        result_np = dBlinmean(dB, axis=None)
        
        # Compute with cupy
        dB_cp = cp.asarray(dB)
        result_cp = dBlinmean(dB_cp, axis=None)
        result_cp_np = float(result_cp.get())
        
        assert_allclose(result_cp_np, result_np, rtol=self.RTOL_FLOAT64, atol=self.ATOL)
    
    @given(dB=dB_arrays(min_value=-50, max_value=50, min_size=2, max_size=50, dtype=np.float64, min_dims=1, max_dims=1, filter_near_zero=True))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
    def test_dBlinsum_numpy_vs_cupy_float64(self, cupy_available, dB):
        """Cross-comparison: dBlinsum numpy vs cupy (float64)."""
        cp = cupy_available
        
        # Compute with numpy
        result_np = dBlinsum(dB, axis=None)
        
        # Compute with cupy
        dB_cp = cp.asarray(dB)
        result_cp = dBlinsum(dB_cp, axis=None)
        result_cp_np = float(result_cp.get())
        
        assert_allclose(result_cp_np, result_np, rtol=self.RTOL_FLOAT64, atol=self.ATOL)
