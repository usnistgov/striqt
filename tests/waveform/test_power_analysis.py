"""Property-based tests for striqt.waveform.power_analysis using Hypothesis.

Covers the dB/linear conversions and dB-domain statistics: identities and algebraic
rules, dtype handling, input preservation, edge cases, complex inputs, numpy/cupy/dask
compatibility, and numpy-vs-cupy agreement within ulp budgets for the library calls.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import (
    dB_arrays,
    envelope_arrays,
    for_each_namespace,
    iq_waveforms,
    positive_power_arrays,
    to_numpy,
)
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from numpy.testing import assert_allclose, assert_array_equal

from striqt.waveform.lib.arrays import float_dtype_like
from striqt.waveform.lib.power_analysis import (
    _arraylike_with_buffer,
    dBlinmean,
    dBlinsum,
    dBtopow,
    envtodB,
    envtopow,
    iq_to_bin_power,
    iq_to_cyclic_power,
    powtodB,
    sample_ccdf,
    stat_ufunc_from_shorthand,
    unit_dB_to_linear,
    unit_dB_to_wave,
    unit_linear_to_dB,
    unit_wave_to_dB,
    unit_wave_to_linear,
)

PROPERTY = settings(
    suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None
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

    @pytest.mark.parametrize('dB_unit,wave_unit', WAVE_PAIRS)
    def test_wave_linear_matches_dB_linear(self, dB_unit: str, wave_unit: str):
        """Property: wave → linear agrees with dB → linear for the same quantity."""
        assert unit_wave_to_linear(wave_unit) == unit_dB_to_linear(dB_unit)

    @pytest.mark.parametrize('unit', ['V', 'counts', ''])
    def test_unknown_units_pass_through(self, unit):
        for func in (
            unit_dB_to_linear,
            unit_linear_to_dB,
            unit_dB_to_wave,
            unit_wave_to_dB,
            unit_wave_to_linear,
        ):
            assert func(unit) == unit


NAMED_STATS = {
    'min': np.min,
    'max': np.max,
    'peak': np.max,
    'mean': np.mean,
    'rms': np.mean,
    'median': np.median,
}


class TestStatUfuncFromShorthand:
    @pytest.fixture
    def data(self):
        return np.random.default_rng(0).normal(size=(6, 5)).astype(np.float32)

    @pytest.mark.parametrize('kind', sorted(NAMED_STATS))
    @pytest.mark.parametrize('axis', [0, 1])
    def test_named_statistics(self, data, kind, axis):
        ufunc = stat_ufunc_from_shorthand(kind, axis=axis)
        assert_array_equal(ufunc(data), NAMED_STATS[kind](data, axis=axis))

    @pytest.mark.parametrize('q', [0.0, 0.25, 0.9, 1.0])
    def test_quantile(self, data, q):
        ufunc = stat_ufunc_from_shorthand(q, axis=1)
        assert_array_equal(ufunc(data), np.quantile(data, q=q, axis=1))

    def test_callable(self, data):
        ufunc = stat_ufunc_from_shorthand(np.std, axis=0)
        assert_array_equal(ufunc(data), np.std(data, axis=0))

    def test_default_namespace_is_numpy(self, data):
        assert_array_equal(
            stat_ufunc_from_shorthand('mean')(data),
            stat_ufunc_from_shorthand('mean', xp=np)(data),
        )

    @pytest.mark.parametrize('kind', ['average', 'rms2', ''])
    def test_unknown_name_raises(self, kind):
        with pytest.raises(ValueError, match='kind argument'):
            stat_ufunc_from_shorthand(kind)

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match='invalid statistic'):
            stat_ufunc_from_shorthand(('mean',))


class TestConversionIdentities:
    """Properties: Mathematical identities that must hold for dB conversions."""

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

    @given(
        dB=st.floats(
            min_value=-140, max_value=100, allow_nan=False, allow_infinity=False
        ),
        n=st.integers(min_value=1, max_value=20),
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_equal_values_sum(self, dB, n):
        """Property: dBlinsum([x, x, ..., x]) = x + 10*log10(n)."""
        arr = np.array([dB] * n, dtype=np.float64)
        result = dBlinsum(arr)
        expected = dB + 10 * np.log10(n)
        rtol, atol = linear_stat_tol(np.float64, abs(dB), n)
        assert_allclose(result, expected, rtol=rtol, atol=atol)


class TestDtypePreservation:
    """Properties: Output dtype should be at least min_dtype."""

    @given(power=positive_power_arrays(dtype=np.float64))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_powtodB_float64_preserved(self, power):
        """Property: float64 input is never downgraded."""
        result = powtodB(power, min_dtype='float32')
        assert result.dtype == np.float64

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


class TestAxisParameter:
    """Properties: Correct behavior with axis parameter."""

    @given(data=st.data())
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_axis_reduces_dimension(self, data):
        """Property: dBlinmean and dBlinsum along axis reduce that dimension."""
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

        expected_shape = list(shape)
        del expected_shape[axis]
        assert dBlinmean(dB, axis=axis).shape == tuple(expected_shape)
        assert dBlinsum(dB, axis=axis).shape == tuple(expected_shape)

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
            dB_arrays(min_value=-140, max_value=100, dtype=np.float64)
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
                max_size=50,
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
                min_magnitude=1e-8,
                max_magnitude=1e8,
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

        assert np.isrealobj(result_np)
        rtol = envelope_power_rtol(np.float64, complex_input=True, n_impl=2)
        assert_allclose(result_np, expected, rtol=rtol)

    @given(
        data=for_each_namespace(
            envelope_arrays(
                include_complex=True,
                min_magnitude=1e-8,
                max_magnitude=1e8,
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


class TestAbsAndEpsBranches:
    """Properties: the `abs` and `eps` arguments of powtodB and envtodB."""

    @given(power=positive_power_arrays(min_value=1e-6, max_value=1e6, dtype=np.float64))
    @PROPERTY
    def test_powtodB_noabs_matches_abs_for_positive(self, power):
        assert_array_equal(powtodB(power, abs=False), powtodB(power, abs=True))

    @given(env=envelope_arrays(include_complex=False, dtype=np.float64))
    @PROPERTY
    def test_envtodB_noabs_matches_abs_for_positive(self, env):
        assert_array_equal(envtodB(env, abs=False), envtodB(env, abs=True))

    @given(power=positive_power_arrays(min_value=1e-6, max_value=1e6, dtype=np.float64))
    @PROPERTY
    def test_noabs_negative_input_is_nan(self, power):
        """Property: without abs, log10 of a negative value is nan, not folded."""
        assert np.all(np.isnan(powtodB(-power, abs=False)))
        assert np.all(np.isnan(envtodB(-power, abs=False)))
        assert np.all(np.isfinite(powtodB(-power, abs=True)))
        assert np.all(np.isfinite(envtodB(-power, abs=True)))

    @given(
        power=positive_power_arrays(min_value=1e-6, max_value=1e6, dtype=np.float64),
        eps=st.floats(min_value=1e-9, max_value=1e-3),
        abs=st.booleans(),
    )
    @PROPERTY
    def test_powtodB_eps_is_added_before_log(self, power, eps, abs):
        expected = 10 * np.log10(power + eps)
        rtol, atol = log_conversion_tol(np.float64, 10, n_impl=2)
        assert_allclose(
            powtodB(power, eps=eps, abs=abs), expected, rtol=rtol, atol=atol
        )

    @given(
        env=envelope_arrays(
            include_complex=False,
            min_magnitude=1e-6,
            max_magnitude=1e6,
            dtype=np.float64,
        ),
        eps=st.floats(min_value=1e-9, max_value=1e-3),
        abs=st.booleans(),
    )
    @PROPERTY
    def test_envtodB_eps_is_added_before_log(self, env, eps, abs):
        expected = 20 * np.log10(env + eps)
        rtol, atol = log_conversion_tol(np.float64, 20, n_impl=2)
        assert_allclose(envtodB(env, eps=eps, abs=abs), expected, rtol=rtol, atol=atol)


class TestArrayLikeHandling:
    """The array-like unpacking and repackaging behind every conversion."""

    def test_min_dtype_none_raises(self):
        with pytest.raises(TypeError, match='min_dtype'):
            _arraylike_with_buffer(np.ones(3), min_dtype=None)

    def test_min_dtype_float16_raises(self):
        with pytest.raises(TypeError, match='float32 or larger'):
            _arraylike_with_buffer(np.ones(3), min_dtype=np.dtype('float16'))

    @pytest.mark.parametrize('obj', ['text', [1.0, 2.0], (1.0, 2.0), object()])
    def test_unsupported_input_raises(self, obj):
        with pytest.raises(TypeError, match='unable to associate'):
            powtodB(obj)

    def test_promotion_allocates_buffer(self):
        x = np.ones(4, dtype=np.float16)
        values, out, xp = _arraylike_with_buffer(
            x, overwrite_x=True, min_dtype='float32'
        )
        assert values is x
        assert out.dtype == np.float32
        assert xp is np

    def test_overwrite_returns_input_as_buffer(self):
        x = np.ones(4, dtype=np.float32)
        values, out, _ = _arraylike_with_buffer(x, overwrite_x=True)
        assert values is x
        assert out is x

    def test_no_overwrite_returns_fresh_buffer(self):
        x = np.ones(4, dtype=np.float32)
        values, out, _ = _arraylike_with_buffer(x, overwrite_x=False)
        assert values is x
        assert out is not x
        assert out.dtype == x.dtype
        assert out.shape == x.shape

    def test_scalar_input_never_overwrites(self):
        x = np.array(2.0, dtype=np.float32)
        _, out, _ = _arraylike_with_buffer(x, overwrite_x=True)
        assert out is not x

    @given(
        value=st.floats(
            min_value=1e-10, max_value=1e10, allow_nan=False, allow_infinity=False
        )
    )
    @PROPERTY
    def test_python_float_returns_python_float(self, value):
        result = powtodB(value)
        assert isinstance(result, float)
        rtol, atol = log_conversion_tol(np.float64, 10, n_impl=2)
        assert_allclose(result, 10 * np.log10(value), rtol=rtol, atol=atol)

    def test_python_int_input(self):
        assert isinstance(dBtopow(10), float)
        assert_allclose(dBtopow(10), 10.0, rtol=pow_conversion_rtol(np.float64, 10))

    @given(power=positive_power_arrays(dtype=np.float64, min_dims=1, max_dims=1))
    @PROPERTY
    def test_pandas_series_roundtrip(self, power):
        pd = pytest.importorskip('pandas')

        index = pd.RangeIndex(10, 10 + power.size)
        series = pd.Series(power, index=index)

        result = powtodB(series)
        assert isinstance(result, pd.Series)
        assert result.index.equals(index)
        assert_array_equal(result.values, powtodB(power))

    @given(
        power=positive_power_arrays(
            dtype=np.float64, min_dims=2, max_dims=2, min_size=1, max_size=6
        )
    )
    @PROPERTY
    def test_pandas_dataframe_roundtrip(self, power):
        pd = pytest.importorskip('pandas')

        columns = [f'c{i}' for i in range(power.shape[1])]
        frame = pd.DataFrame(power, columns=columns)

        result = powtodB(frame)
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == columns
        assert result.index.equals(frame.index)
        assert_array_equal(result.values, powtodB(power))

    @pytest.mark.parametrize(
        'func,units_in,units_out',
        [
            (powtodB, 'mW', 'dBm'),
            (powtodB, 'W/Hz', 'dBW/Hz'),
            (dBtopow, 'dBm', 'mW'),
            (dBtopow, 'dB', 'unitless'),
            (envtodB, '√mW', 'dBm'),
            (envtopow, '√W', 'W'),
        ],
    )
    def test_xarray_units_transform(self, func, units_in, units_out):
        xr = pytest.importorskip('xarray')

        data = np.linspace(0.5, 2.0, 8, dtype=np.float64)
        da = xr.DataArray(
            data, dims=['t'], coords={'t': np.arange(8)}, attrs={'units': units_in}
        )

        result = func(da)
        assert isinstance(result, xr.DataArray)
        assert result.dims == ('t',)
        assert result.attrs['units'] == units_out
        assert da.attrs['units'] == units_in
        assert_array_equal(result.values, func(data))

    def test_xarray_without_units(self):
        xr = pytest.importorskip('xarray')

        da = xr.DataArray(np.ones(4), dims=['t'])
        assert 'units' not in powtodB(da).attrs

    def test_xarray_dataset_values_raise(self):
        xr = pytest.importorskip('xarray')

        ds = xr.Dataset({'a': ('t', np.ones(4))})
        with pytest.raises(TypeError):
            powtodB(ds)

    def test_unknown_container_with_values_raises(self):
        class Container:
            @property
            def values(self):
                return np.ones(4)

        with pytest.raises(TypeError, match='unable to associate an array type'):
            powtodB(Container())

    @given(
        data=for_each_namespace(
            positive_power_arrays(min_value=1e-3, max_value=1e3, dtype=np.float64)
        ),
        eps=st.floats(min_value=1e-9, max_value=1e-3),
        abs=st.booleans(),
    )
    @PROPERTY
    def test_eps_multibackend(self, data, eps, abs):
        """Property: the eps offset is applied by every backend's code path."""
        arr, xp_name, _ = data
        arr_np = to_numpy(arr)

        rtol, atol = log_conversion_tol(np.float64, 10, n_impl=2)
        assert_allclose(
            to_numpy(powtodB(arr, eps=eps, abs=abs)),
            10 * np.log10(arr_np + eps),
            rtol=rtol,
            atol=atol,
            err_msg=xp_name,
        )
        rtol, atol = log_conversion_tol(np.float64, 20, n_impl=2)
        assert_allclose(
            to_numpy(envtodB(arr, eps=eps, abs=abs)),
            20 * np.log10(arr_np + eps),
            rtol=rtol,
            atol=atol,
            err_msg=xp_name,
        )

    @given(
        data=for_each_namespace(
            positive_power_arrays(min_value=1e-3, max_value=1e3, dtype=np.float32)
        )
    )
    @PROPERTY
    def test_float16_promotion_multibackend(self, data):
        arr, xp_name, _ = data
        result = powtodB(arr.astype(np.float16), min_dtype='float32')
        assert to_numpy(result).dtype == np.float32, xp_name


def _blocks(x, size, axis):
    """reshape the axis of x into (n_blocks, size) after truncating the remainder"""
    x = np.moveaxis(x, axis, -1)
    n_blocks = x.shape[-1] // size
    x = x[..., : n_blocks * size].reshape(x.shape[:-1] + (n_blocks, size))
    return np.moveaxis(x, (-2, -1), (axis, axis + 1))


BIN_STATS = {
    'mean': np.mean,
    'rms': np.mean,
    'max': np.max,
    'peak': np.max,
    'min': np.min,
    'median': np.median,
    0.9: lambda x, axis: np.quantile(x, 0.9, axis=axis),
}


def bin_power_reference(iq, size, kind, axis):
    power = np.abs(iq.astype(np.complex128)) ** 2
    return BIN_STATS[kind](_blocks(power, size, axis), axis=axis + 1)


def bin_power_rtol(iq, size, n_impl=1):
    """rtol for a statistic of |x|**2 over `size` samples against exact arithmetic"""
    u = unit_roundoff(float_dtype_like(iq))
    return (
        envelope_power_rtol(float_dtype_like(iq), complex_input=True, n_impl=n_impl)
        + ROUNDOFF_SAFETY * n_impl * size * u
    )


def bin_sizes():
    return st.sampled_from([1, 2, 5, 8, 16])


class TestIqToBinPower:
    @given(
        data=st.data(),
        size=bin_sizes(),
        kind=st.sampled_from(sorted(BIN_STATS, key=str)),
        channels=st.sampled_from([None, 1, 3]),
    )
    @PROPERTY
    def test_matches_reference(self, data, size, kind, channels):
        """Property: each bin is the statistic of |iq|**2 over `size` samples."""
        iq = data.draw(
            iq_waveforms(
                min_size=size, max_size=32 * size, multiple_of=size, channels=channels
            )
        )
        axis = 0 if channels is None else 1
        Ts = 1e-6

        result = iq_to_bin_power(iq, Ts, size * Ts, kind=kind, axis=axis)

        assert result.dtype == float_dtype_like(iq)
        expected = bin_power_reference(iq, size, kind, axis)
        assert result.shape == expected.shape
        assert_allclose(result, expected, rtol=bin_power_rtol(iq, size))

    @given(
        data=st.data(),
        size=bin_sizes().filter(lambda n: n > 1),
        remainder=st.integers(min_value=1, max_value=4),
    )
    @PROPERTY
    def test_truncate(self, data, size, remainder):
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
        assert_allclose(result, expected, rtol=bin_power_rtol(iq, size))

    @given(ratio=st.floats(min_value=1.1, max_value=9.9).filter(lambda r: r % 1 > 0.01))
    @PROPERTY
    def test_bin_period_must_be_multiple(self, ratio):
        iq = np.ones(64, dtype=np.complex64)
        with pytest.raises(ValueError, match='multiple'):
            iq_to_bin_power(iq, 1.0, ratio)

    def test_truncate_allows_fractional_bin_period(self):
        iq = np.ones(64, dtype=np.complex64)
        result = iq_to_bin_power(iq, 1.0, 4.4, truncate=True)
        assert result.shape == (64 // 4,)

    @given(data=st.data(), size=bin_sizes().filter(lambda n: n > 1))
    @PROPERTY
    def test_randomize(self, data, size):
        """Property: random bins keep the shape, dtype and range of contiguous bins."""
        iq = data.draw(
            iq_waveforms(min_size=4 * size, max_size=32 * size, multiple_of=size)
        )
        Ts = 1e-6

        result = iq_to_bin_power(iq, Ts, size * Ts, randomize=True)

        assert result.shape == (iq.shape[0] // size,)
        assert result.dtype == float_dtype_like(iq)
        power = np.abs(iq) ** 2
        assert np.all(result >= power.min() * (1 - bin_power_rtol(iq, size)))
        assert np.all(result <= power.max() * (1 + bin_power_rtol(iq, size)))

    def test_randomize_requires_axis_0(self):
        iq = np.ones((2, 64), dtype=np.complex64)
        with pytest.raises(ValueError, match='axis=0'):
            iq_to_bin_power(iq, 1.0, 4.0, randomize=True, axis=1)

    @pytest.mark.xfail(
        strict=True,
        reason='iq_to_bin_power applies the detector on axis+1 without normalizing '
        'a negative axis first',
    )
    def test_negative_axis(self):
        iq = np.random.default_rng(0).normal(size=(2, 32)).astype(np.complex64)
        assert_array_equal(
            iq_to_bin_power(iq, 1.0, 4.0, axis=-1),
            iq_to_bin_power(iq, 1.0, 4.0, axis=1),
        )


class TestIqToCyclicPower:
    """iq_to_cyclic_power on (channel, sample) input with axis=1, as the
    cyclic_channel_power measurement calls it."""

    DETECTORS = ('rms', 'peak')
    CYCLE_STATS = ('min', 'mean', 'max')

    @given(
        data=st.data(),
        size=st.sampled_from([2, 4, 8]),
        bins_per_cycle=st.sampled_from([2, 3, 5]),
        n_cycles=st.integers(min_value=1, max_value=6),
        channels=st.sampled_from([1, 2]),
    )
    @PROPERTY
    def test_matches_reference(self, data, size, bins_per_cycle, n_cycles, channels):
        n = size * bins_per_cycle * n_cycles
        iq = data.draw(iq_waveforms(min_size=n, max_size=n, channels=channels))
        Ts = 1e-6

        result = iq_to_cyclic_power(
            iq,
            Ts,
            detector_period=size * Ts,
            cyclic_period=size * bins_per_cycle * Ts,
            detectors=self.DETECTORS,
            cycle_stats=self.CYCLE_STATS,
            axis=1,
        )

        assert set(result) == set(self.DETECTORS)
        u = unit_roundoff(float_dtype_like(iq))
        for detector in self.DETECTORS:
            assert set(result[detector]) == set(self.CYCLE_STATS)
            binned = iq_to_bin_power(iq, Ts, size * Ts, kind=detector, axis=1)
            by_cycle = binned.reshape(channels, n_cycles, bins_per_cycle)
            for stat in self.CYCLE_STATS:
                value = result[detector][stat]
                assert value.shape == (channels, bins_per_cycle)
                expected = BIN_STATS[stat](by_cycle, axis=1)
                assert_allclose(value, expected, rtol=ROUNDOFF_SAFETY * n_cycles * u)

            assert np.all(result[detector]['min'] <= result[detector]['mean'])
            assert np.all(result[detector]['mean'] <= result[detector]['max'])

    def test_detectors_none_raises(self):
        iq = np.ones((1, 64), dtype=np.complex64)
        with pytest.raises(ValueError, match='detectors'):
            iq_to_cyclic_power(iq, 1.0, 4.0, 16.0, detectors=None, axis=1)

    def test_cyclic_period_must_be_multiple(self):
        iq = np.ones((1, 64), dtype=np.complex64)
        with pytest.raises(ValueError, match='cyclic period'):
            iq_to_cyclic_power(iq, 1.0, 4.0, 10.0, axis=1)

    def test_misaligned_length_without_truncate_raises(self):
        iq = np.ones((1, 4 * 7), dtype=np.complex64)
        with pytest.raises(ValueError, match='truncate'):
            iq_to_cyclic_power(iq, 1.0, 4.0, 16.0, axis=1)

    @pytest.mark.xfail(
        strict=True,
        reason='iq_to_cyclic_power truncates axis 0 (channels) instead of the bin axis',
    )
    def test_misaligned_length_with_truncate(self):
        iq = np.random.default_rng(0).normal(size=(2, 4 * 7)).astype(np.complex64)
        result = iq_to_cyclic_power(iq, 1.0, 4.0, 16.0, truncate=True, axis=1)
        assert result['rms']['mean'].shape == (2, 4)

    @pytest.mark.xfail(
        strict=True,
        reason='iq_to_cyclic_power indexes power_shape[1], which a 1-D waveform lacks',
    )
    def test_1d_input(self):
        iq = np.random.default_rng(0).normal(size=64).astype(np.complex64)
        result = iq_to_cyclic_power(iq, 1.0, 4.0, 16.0)
        assert result['rms']['mean'].shape == (4,)

    @pytest.mark.xfail(
        strict=True,
        reason='iq_to_cyclic_power normalizes a negative axis only after binning, so '
        'the cycle statistics reduce the wrong axis',
    )
    def test_negative_axis(self):
        iq = np.random.default_rng(0).normal(size=(2, 64)).astype(np.complex64)
        expected = iq_to_cyclic_power(iq, 1.0, 4.0, 16.0, axis=1)
        result = iq_to_cyclic_power(iq, 1.0, 4.0, 16.0, axis=-1)
        for detector, stats in expected.items():
            for stat, value in stats.items():
                assert_array_equal(result[detector][stat], value)


def ccdf_reference(a, edges, density):
    counts = np.array([(a > e).sum() for e in edges])
    return counts / a.size if density else counts


class TestSampleCcdf:
    @given(
        a=arrays(
            dtype=np.float64,
            shape=st.integers(min_value=1, max_value=200),
            elements=st.floats(min_value=-100, max_value=100, allow_nan=False),
        ),
        edges=st.lists(
            st.floats(min_value=-120, max_value=120, allow_nan=False),
            min_size=1,
            max_size=20,
            unique=True,
        ).map(sorted),
        density=st.booleans(),
    )
    @PROPERTY
    def test_matches_brute_force(self, a, edges, density):
        edges = np.asarray(edges)
        result = sample_ccdf(a, edges, density=density)
        expected = ccdf_reference(a, edges, density)
        assert result.shape == edges.shape
        if density:
            assert result.dtype == np.float64
            assert_allclose(
                result, expected, rtol=ROUNDOFF_SAFETY * unit_roundoff(np.float64)
            )
        else:
            assert_array_equal(result, expected)

    def test_samples_equal_to_edge_are_not_counted(self):
        a = np.array([1.0, 1.0, 2.0])
        edges = np.array([0.0, 1.0, 2.0])
        assert_array_equal(sample_ccdf(a, edges, density=False), [3, 1, 0])


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

    @given(data=st.data(), dtype=st.sampled_from([np.float64, np.float32]))
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None
    )
    @pytest.mark.parametrize('abs', [True, False])
    @pytest.mark.parametrize('eps', [0, 1e-6])
    def test_powtodB_numpy_vs_cupy(self, cupy_available, data, dtype, abs, eps):
        """Cross-comparison: powtodB numpy vs cupy, on each fused kernel variant."""
        cp = cupy_available
        lim = {np.float64: 1e10, np.float32: 1e5}[dtype]
        power = data.draw(
            positive_power_arrays(
                min_value=1 / lim, max_value=lim, dtype=dtype, min_dims=1, max_dims=1
            )
        )

        result_np = powtodB(power, min_dtype='float32', abs=abs, eps=eps)
        result_cp_np = powtodB(
            cp.asarray(power), min_dtype='float32', abs=abs, eps=eps
        ).get()

        rtol, atol = log_conversion_tol(dtype, 10, n_impl=2)
        tol_dB = dB_tolerance(rtol, atol, np.abs(result_np).max())
        assert_allclose(
            result_cp_np, result_np, rtol=rtol, atol=atol, err_msg=f'{tol_dB:.2e} dB'
        )

    @given(data=st.data(), dtype=st.sampled_from([np.float64, np.float32]))
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None
    )
    def test_dBtopow_numpy_vs_cupy(self, cupy_available, data, dtype):
        """Cross-comparison: dBtopow numpy vs cupy."""
        cp = cupy_available
        lim = {np.float64: 100, np.float32: 30}[dtype]
        dB = data.draw(
            dB_arrays(
                min_value=-lim, max_value=lim, dtype=dtype, min_dims=1, max_dims=1
            )
        )

        result_np = dBtopow(dB, min_dtype='float32')
        result_cp_np = dBtopow(cp.asarray(dB), min_dtype='float32').get()

        rtol = pow_conversion_rtol(dtype, np.abs(dB).max(), n_impl=2)
        tol_dB = linear_tolerance_dB(rtol)
        assert_allclose(result_cp_np, result_np, rtol=rtol, err_msg=f'{tol_dB:.2e} dB')

    @given(data=st.data(), dtype=st.sampled_from([np.float64, np.float32]))
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None
    )
    @pytest.mark.parametrize('abs', [True, False])
    @pytest.mark.parametrize('eps', [0, 1e-6])
    def test_envtodB_numpy_vs_cupy(self, cupy_available, data, dtype, abs, eps):
        """Cross-comparison: envtodB numpy vs cupy, on each fused kernel variant."""
        cp = cupy_available
        lim = {np.float64: 1e6, np.float32: 1e4}[dtype]
        env = data.draw(
            envelope_arrays(
                include_complex=False,
                min_magnitude=1 / lim,
                max_magnitude=lim,
                dtype=dtype,
                min_dims=1,
                max_dims=1,
            )
        )

        result_np = envtodB(env, min_dtype='float32', abs=abs, eps=eps)
        result_cp_np = envtodB(
            cp.asarray(env), min_dtype='float32', abs=abs, eps=eps
        ).get()

        rtol, atol = log_conversion_tol(dtype, 20, n_impl=2)
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

    @given(
        env=envelope_arrays(
            include_complex=True,
            min_magnitude=1e-4,
            max_magnitude=1e4,
            dtype=np.float32,
            min_dims=1,
            max_dims=1,
        )
    )
    @PROPERTY
    def test_envtopow_complex_numpy_vs_cupy(self, cupy_available, env):
        """Cross-comparison: envtopow on complex input returns a real array on both."""
        cp = cupy_available

        result_np = envtopow(env)
        result_cp_np = envtopow(cp.asarray(env)).get()

        assert np.isrealobj(result_np)
        assert np.isrealobj(result_cp_np)
        rtol = envelope_power_rtol(np.float32, complex_input=True, n_impl=2)
        assert_allclose(result_cp_np, result_np, rtol=rtol)

    @given(
        power=positive_power_arrays(
            min_value=1e-3, max_value=1e3, dtype=np.float32, min_dims=1, max_dims=1
        )
    )
    @PROPERTY
    def test_overwrite_x_numpy_vs_cupy(self, cupy_available, power):
        """Cross-comparison: in-place evaluation writes the result into the input."""
        cp = cupy_available

        expected = powtodB(power)
        power_cp = cp.asarray(power)
        result_cp = powtodB(power_cp, overwrite_x=True)

        assert result_cp is power_cp
        rtol, atol = log_conversion_tol(np.float32, 10, n_impl=2)
        assert_allclose(result_cp.get(), expected, rtol=rtol, atol=atol)

    def test_xarray_cupy_data(self, cupy_available):
        """Cross-comparison: a DataArray wrapping cupy data keeps cupy data."""
        cp = cupy_available
        xr = pytest.importorskip('xarray')

        data = np.linspace(0.5, 2.0, 8, dtype=np.float32)
        da = xr.DataArray(cp.asarray(data), dims=['t'], attrs={'units': 'mW'})

        result = powtodB(da)
        assert isinstance(result.data, cp.ndarray)
        assert result.attrs['units'] == 'dBm'
        rtol, atol = log_conversion_tol(np.float32, 10, n_impl=2)
        assert_allclose(result.data.get(), powtodB(data), rtol=rtol, atol=atol)

    @given(
        data=st.data(),
        size=bin_sizes(),
        kind=st.sampled_from(sorted(BIN_STATS, key=str)),
    )
    @PROPERTY
    def test_iq_to_bin_power_numpy_vs_cupy(self, cupy_available, data, size, kind):
        cp = cupy_available
        iq = data.draw(
            iq_waveforms(
                min_size=size, max_size=32 * size, multiple_of=size, channels=2
            )
        )
        Ts = 1e-6

        result_np = iq_to_bin_power(iq, Ts, size * Ts, kind=kind, axis=1)
        result_cp = iq_to_bin_power(cp.asarray(iq), Ts, size * Ts, kind=kind, axis=1)

        assert result_cp.dtype == result_np.dtype
        assert_allclose(result_cp.get(), result_np, rtol=bin_power_rtol(iq, size, 2))

    @given(
        data=st.data(),
        size=st.sampled_from([2, 4, 8]),
        bins_per_cycle=st.sampled_from([2, 3, 5]),
        n_cycles=st.integers(min_value=1, max_value=6),
    )
    @PROPERTY
    def test_iq_to_cyclic_power_numpy_vs_cupy(
        self, cupy_available, data, size, bins_per_cycle, n_cycles
    ):
        cp = cupy_available
        n = size * bins_per_cycle * n_cycles
        iq = data.draw(iq_waveforms(min_size=n, max_size=n, channels=2))
        Ts = 1e-6
        kws = {
            'detector_period': size * Ts,
            'cyclic_period': size * bins_per_cycle * Ts,
            'axis': 1,
        }

        result_np = iq_to_cyclic_power(iq, Ts, **kws)
        result_cp = iq_to_cyclic_power(cp.asarray(iq), Ts, **kws)

        u = unit_roundoff(float_dtype_like(iq))
        rtol = bin_power_rtol(iq, size, 2) + ROUNDOFF_SAFETY * 2 * n_cycles * u
        for detector, stats in result_np.items():
            for stat, value in stats.items():
                assert_allclose(result_cp[detector][stat].get(), value, rtol=rtol)

    @given(
        a=arrays(
            dtype=np.float64,
            shape=st.integers(min_value=1, max_value=200),
            elements=st.floats(min_value=-100, max_value=100, allow_nan=False),
        ),
        edges=st.lists(
            st.floats(min_value=-120, max_value=120, allow_nan=False),
            min_size=1,
            max_size=20,
            unique=True,
        ).map(sorted),
        density=st.booleans(),
    )
    @PROPERTY
    def test_sample_ccdf_numpy_vs_cupy(self, cupy_available, a, edges, density):
        """Cross-comparison: counting is exact, so the backends agree exactly."""
        cp = cupy_available
        edges = np.asarray(edges)

        result_np = sample_ccdf(a, edges, density=density)
        result_cp = sample_ccdf(cp.asarray(a), cp.asarray(edges), density=density)

        assert_array_equal(result_cp.get(), result_np)

    @pytest.mark.parametrize('kind', ['min', 'max', 'mean', 'median', 0.25])
    def test_stat_ufunc_numpy_vs_cupy(self, cupy_available, kind):
        cp = cupy_available
        data = np.random.default_rng(0).normal(size=(6, 5)).astype(np.float32)

        result_np = stat_ufunc_from_shorthand(kind, xp=np, axis=1)(data)
        result_cp = stat_ufunc_from_shorthand(kind, xp=cp, axis=1)(cp.asarray(data))

        u = unit_roundoff(np.float32)
        assert_allclose(result_cp.get(), result_np, rtol=ROUNDOFF_SAFETY * 2 * 5 * u)
