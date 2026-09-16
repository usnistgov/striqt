"""Property-based tests for striqt.waveform.power_analysis using Hypothesis.

Covers the dB/linear conversions and dB-domain statistics: identities and algebraic
rules, dtype handling, input preservation, edge cases, complex inputs, numpy/cupy/dask
compatibility, and numpy-vs-cupy agreement within ulp budgets for the library calls.
"""

from __future__ import annotations

import functools
from typing import ClassVar

import numpy as np
import pytest
from conftest import (
    dB_arrays,
    envelope_arrays,
    for_each_namespace,
    iq_waveforms,
    numpy_and_cupy,
    positive_power_arrays,
    to_numpy,
)
from hypothesis import given
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


def func_id(value):
    """parametrize id for a function argument; other argument types get the default"""
    return getattr(value, '__name__', None)


# strategy factories for the valid inputs of each conversion, keyed by function
INPUTS = {
    powtodB: positive_power_arrays,
    dBtopow: dB_arrays,
    envtodB: functools.partial(envelope_arrays, include_complex=False),
    envtopow: functools.partial(envelope_arrays, include_complex=False),
    dBlinmean: functools.partial(dB_arrays, min_value=-50, max_value=50),
    dBlinsum: functools.partial(dB_arrays, min_value=-50, max_value=50),
}

# (function, dB per decade) for the two logarithmic conversions
LOG_CONVERSIONS = [(powtodB, 10), (envtodB, 20)]
by_log_conversion = pytest.mark.parametrize(
    'func, scale', LOG_CONVERSIONS, ids=[func_id(f) for f, _ in LOG_CONVERSIONS]
)


class TestUnitConversionProperties:
    """Property: Unit conversions form bijections (invertible mappings)."""

    UNIT_PAIRS: ClassVar = [
        ('dBm', 'mW'),
        ('dBW', 'W'),
        ('dB', 'unitless'),
        ('dBm/Hz', 'mW/Hz'),
        ('dBW/Hz', 'W/Hz'),
    ]

    WAVE_PAIRS: ClassVar = [
        ('dBm', '√mW'),
        ('dBW', '√W'),
        ('dB', '√unitless'),
    ]

    @pytest.mark.parametrize('dB_unit,linear_unit', UNIT_PAIRS)
    def test_dB_linear_roundtrip(self, dB_unit: str, linear_unit: str):
        assert unit_dB_to_linear(dB_unit) == linear_unit
        assert unit_linear_to_dB(unit_dB_to_linear(dB_unit)) == dB_unit
        assert unit_dB_to_linear(unit_linear_to_dB(linear_unit)) == linear_unit

    @pytest.mark.parametrize('dB_unit,wave_unit', WAVE_PAIRS)
    def test_dB_wave_roundtrip(self, dB_unit: str, wave_unit: str):
        assert unit_wave_to_dB(unit_dB_to_wave(dB_unit)) == dB_unit
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
    DATA = np.random.default_rng(0).normal(size=(6, 5)).astype(np.float32)
    # a reduction over the 5 samples of an axis rounds at most 5 times per backend
    RTOL = ROUNDOFF_SAFETY * 2 * 5 * unit_roundoff(np.float32)

    def _check(self, xp, kind, axis, expected):
        ufunc = stat_ufunc_from_shorthand(kind, xp=xp, axis=axis)
        result = ufunc(xp.asarray(self.DATA))
        assert_allclose(to_numpy(result), expected, rtol=self.RTOL)

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
    def test_envtopow_is_square(self, env):
        rtol = envelope_power_rtol(env.dtype, n_impl=2)
        assert_allclose(envtopow(env), np.abs(env) ** 2, rtol=rtol)

    @by_log_conversion
    @given(x=positive_power_arrays(dtype=np.float64))
    def test_log_conversion_is_scaled_log10(self, func, scale, x):
        rtol, atol = log_conversion_tol(x.dtype, scale, n_impl=2)
        assert_allclose(func(x), scale * np.log10(x), rtol=rtol, atol=atol)


class TestAlgebraicProperties:
    """Properties: Algebraic rules for dB arithmetic."""

    @given(
        dB=st.floats(
            min_value=-140, max_value=100, allow_nan=False, allow_infinity=False
        ),
        n=st.integers(min_value=1, max_value=20),
    )
    def test_equal_values(self, dB, n):
        """Property: dBlinmean([x, ..., x]) = x and dBlinsum([x, ..., x]) = x + 10*log10(n)."""
        arr = np.array([dB] * n, dtype=np.float64)
        rtol, atol = linear_stat_tol(np.float64, abs(dB), n)
        assert_allclose(dBlinmean(arr), dB, rtol=rtol, atol=atol)
        assert_allclose(dBlinsum(arr), dB + 10 * np.log10(n), rtol=rtol, atol=atol)

    @given(
        data=for_each_namespace(
            dB_arrays(
                min_value=-50,
                max_value=50,
                dtype=np.float64,
                min_dims=2,
                max_dims=2,
                min_size=2,
                max_size=10,
            )
        ),
        axis=st.sampled_from([None, 0, 1]),
    )
    def test_axis_reduces_dimension_and_relates_mean_to_sum(self, data, axis):
        """Property: dBlinmean = dBlinsum - 10*log10(N) over the reduced axis."""
        dB, xp_name = data
        shape = to_numpy(dB).shape
        if axis is None:
            expected_shape, N = (), to_numpy(dB).size
        else:
            expected_shape, N = shape[:axis] + shape[axis + 1 :], shape[axis]

        mean_result = to_numpy(dBlinmean(dB, axis=axis))
        sum_result = to_numpy(dBlinsum(dB, axis=axis))
        assert mean_result.shape == sum_result.shape == expected_shape

        rtol, atol = linear_stat_tol(np.float64, 50, N, n_impl=2)
        expected_mean = sum_result - 10 * np.log10(N)
        assert_allclose(
            mean_result, expected_mean, rtol=rtol, atol=atol, err_msg=xp_name
        )


class TestInputPreservation:
    @pytest.mark.parametrize('func', INPUTS, ids=func_id)
    @given(data=st.data())
    def test_overwrite_x_false_leaves_input_unchanged(self, func, data):
        x = data.draw(INPUTS[func](dtype=np.float64))
        original = x.copy()
        func(x, overwrite_x=False)
        assert_array_equal(x, original)


class TestEdgeCases:
    """Properties: Behavior at edge cases (zeros, extreme values)."""

    @by_log_conversion
    def test_zero_produces_neg_inf(self, func, scale):
        assert np.all(func(np.zeros(4, dtype=np.float64), eps=0) == -np.inf)

    @by_log_conversion
    @given(eps=st.floats(min_value=1e-30, max_value=1e-10, allow_nan=False))
    def test_eps_avoids_neg_inf(self, func, scale, eps):
        result = func(np.array([0.0], dtype=np.float64), eps=eps)
        assert np.isfinite(result[0])
        rtol, atol = log_conversion_tol(np.float64, scale, n_impl=2)
        assert_allclose(result[0], scale * np.log10(eps), rtol=rtol, atol=atol)

    def test_empty_array(self):
        empty = np.array([], dtype=np.float64)
        for func in (powtodB, dBtopow, envtodB, envtopow):
            assert func(empty).shape == (0,)

    @given(
        value=st.floats(
            min_value=1e-10, max_value=1e10, allow_nan=False, allow_infinity=False
        )
    )
    def test_scalar_like_array(self, value):
        result = powtodB(np.array(value, dtype=np.float64))
        rtol, atol = log_conversion_tol(np.float64, 10, n_impl=2)
        assert_allclose(result, 10 * np.log10(value), rtol=rtol, atol=atol)


class TestMultiBackend:
    """Properties that must hold on every available array namespace."""

    @pytest.mark.parametrize('func', [powtodB, dBtopow, envtodB, envtopow], ids=func_id)
    @given(data=st.data(), dtype=st.sampled_from([np.float32, np.float64]))
    def test_dtype_is_preserved(self, func, data, dtype):
        """Property: with min_dtype='float32', the input precision is kept."""
        arr, xp_name = data.draw(for_each_namespace(INPUTS[func](dtype=dtype)))
        assert to_numpy(func(arr, min_dtype='float32')).dtype == dtype, xp_name

    @pytest.mark.parametrize(
        'func, limits', [(powtodB, (1e-3, 1e3)), (dBtopow, (-30, 30))], ids=func_id
    )
    @given(data=st.data())
    def test_float16_is_promoted_to_float32(self, func, limits, data):
        """The expected value follows the float16-quantized input, not the float32
        draw it was made from."""
        strategy = INPUTS[func](
            dtype=np.float32, min_value=limits[0], max_value=limits[1]
        )
        arr, xp_name = data.draw(for_each_namespace(strategy))
        x16 = arr.astype(np.float16)

        result = func(x16, min_dtype='float32')
        assert to_numpy(result).dtype == np.float32, xp_name

        expected = func(to_numpy(x16).astype(np.float32), min_dtype='float32')
        if func is dBtopow:
            rtol, atol = pow_conversion_rtol(np.float32, 30, n_impl=2), 0
        else:
            rtol, atol = log_conversion_tol(np.float32, 10, n_impl=2)
        assert_allclose(to_numpy(result), expected, rtol=rtol, atol=atol)

    @given(data=for_each_namespace(positive_power_arrays(dtype=np.float64)))
    def test_powtodB_dBtopow_roundtrip(self, data):
        arr, xp_name = data
        original = to_numpy(arr)
        roundtrip = to_numpy(dBtopow(powtodB(arr)))
        rtol = roundtrip_power_rtol(np.float64, np.abs(powtodB(original)).max())
        assert_allclose(roundtrip, original, rtol=rtol, err_msg=xp_name)

    @given(
        data=for_each_namespace(
            dB_arrays(min_value=-140, max_value=100, dtype=np.float64)
        )
    )
    def test_dBtopow_powtodB_roundtrip(self, data):
        arr, xp_name = data
        original = to_numpy(arr)
        roundtrip = to_numpy(powtodB(dBtopow(arr)))
        rtol, atol = roundtrip_dB_tol(np.float64, np.abs(original).max())
        assert_allclose(roundtrip, original, rtol=rtol, atol=atol, err_msg=xp_name)

    @given(data=for_each_namespace(envelope_arrays(dtype=np.float64)))
    def test_complex_envelopes(self, data):
        """Property: envtopow(z) = |z|² and envtodB(z) = 20*log10(|z|) for complex z."""
        arr, xp_name = data
        arr_np = to_numpy(arr)

        power = to_numpy(envtopow(arr))
        assert np.isrealobj(power)
        rtol = envelope_power_rtol(np.float64, complex_input=True, n_impl=2)
        assert_allclose(power, np.abs(arr_np) ** 2, rtol=rtol, err_msg=xp_name)

        dB = to_numpy(envtodB(arr))
        assert np.isrealobj(dB)
        rtol, atol = log_conversion_tol(np.float64, 20, complex_input=True, n_impl=2)
        expected = 20 * np.log10(np.abs(arr_np))
        assert_allclose(dB, expected, rtol=rtol, atol=atol, err_msg=xp_name)


class TestAbsAndEpsBranches:
    """Properties: the `abs` and `eps` arguments of powtodB and envtodB."""

    @by_log_conversion
    @given(x=positive_power_arrays(min_value=1e-6, max_value=1e6, dtype=np.float64))
    def test_noabs_matches_abs_for_positive(self, func, scale, x):
        assert_array_equal(func(x, abs=False), func(x, abs=True))

    @given(power=positive_power_arrays(min_value=1e-6, max_value=1e6, dtype=np.float64))
    def test_noabs_negative_input_is_nan(self, power):
        """Property: without abs, log10 of a negative value is nan, not folded."""
        assert np.all(np.isnan(powtodB(-power, abs=False)))
        assert np.all(np.isnan(envtodB(-power, abs=False)))
        assert np.all(np.isfinite(powtodB(-power, abs=True)))
        assert np.all(np.isfinite(envtodB(-power, abs=True)))

    @by_log_conversion
    @given(
        data=for_each_namespace(
            positive_power_arrays(min_value=1e-3, max_value=1e3, dtype=np.float64)
        ),
        eps=st.floats(min_value=1e-9, max_value=1e-3),
        abs=st.booleans(),
    )
    def test_eps_is_added_before_log(self, func, scale, data, eps, abs):
        """Property: the eps offset is applied by every backend's code path."""
        arr, xp_name = data
        expected = scale * np.log10(to_numpy(arr) + eps)
        rtol, atol = log_conversion_tol(np.float64, scale, n_impl=2)
        result = to_numpy(func(arr, eps=eps, abs=abs))
        assert_allclose(result, expected, rtol=rtol, atol=atol, err_msg=xp_name)


class TestArrayLikeHandling:
    """The array-like unpacking and repackaging behind every conversion."""

    def test_min_dtype_none_raises(self):
        with pytest.raises(TypeError, match='min_dtype'):
            _arraylike_with_buffer(np.ones(3), min_dtype=None)

    @pytest.mark.parametrize(
        'min_dtype', ['float16', np.float16, np.dtype('float16')], ids=repr
    )
    def test_min_dtype_float16_raises(self, min_dtype):
        with pytest.raises(TypeError, match='float32 or larger'):
            _arraylike_with_buffer(np.ones(3), min_dtype=min_dtype)

    @pytest.mark.parametrize('obj', ['text', [1.0, 2.0], (1.0, 2.0), object()])
    def test_unsupported_input_raises(self, obj):
        with pytest.raises(TypeError, match='unable to associate'):
            powtodB(obj)

    def test_promotion_widens_the_input_before_evaluation(self):
        x = np.ones(4, dtype=np.float16)
        values, out, xp = _arraylike_with_buffer(x, min_dtype='float32')
        assert values.dtype == np.float32
        assert out is values
        assert x.dtype == np.float16
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
    def test_python_float_returns_python_float(self, value):
        result = powtodB(value)
        assert isinstance(result, float)
        rtol, atol = log_conversion_tol(np.float64, 10, n_impl=2)
        assert_allclose(result, 10 * np.log10(value), rtol=rtol, atol=atol)

    def test_python_int_input(self):
        assert isinstance(dBtopow(10), float)
        assert_allclose(dBtopow(10), 10.0, rtol=pow_conversion_rtol(np.float64, 10))

    @given(power=positive_power_arrays(dtype=np.float64, min_dims=1, max_dims=1))
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


def bin_kinds():
    return st.sampled_from(sorted(BIN_STATS, key=str))


@st.composite
def cyclic_power_cases(draw, channels=(1, 2)):
    """(iq, bin size, bins per cycle, cycle count) for iq_to_cyclic_power"""
    size = draw(st.sampled_from([2, 4, 8]))
    bins_per_cycle = draw(st.sampled_from([2, 3, 5]))
    n_cycles = draw(st.integers(min_value=1, max_value=6))
    n = size * bins_per_cycle * n_cycles
    iq = draw(
        iq_waveforms(min_size=n, max_size=n, channels=draw(st.sampled_from(channels)))
    )
    return iq, size, bins_per_cycle, n_cycles


class TestIqToBinPower:
    @given(
        data=st.data(),
        size=bin_sizes(),
        kind=bin_kinds(),
        channels=st.sampled_from([None, 1, 3]),
    )
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
    def test_bin_period_must_be_multiple(self, ratio):
        iq = np.ones(64, dtype=np.complex64)
        with pytest.raises(ValueError, match='multiple'):
            iq_to_bin_power(iq, 1.0, ratio)

    def test_truncate_allows_fractional_bin_period(self):
        """truncate=True intentionally rounds a fractional Tbin/Ts to whole samples."""
        iq = np.ones(64, dtype=np.complex64)
        result = iq_to_bin_power(iq, 1.0, 4.4, truncate=True)
        assert result.shape == (64 // 4,)

    @given(data=st.data(), size=bin_sizes().filter(lambda n: n > 1))
    def test_randomize(self, data, size):
        """Property: every random bin is the mean power of some `size` contiguous
        samples, i.e. a value of the sliding-window mean of |iq|**2."""
        iq = data.draw(
            iq_waveforms(min_size=4 * size, max_size=32 * size, multiple_of=size)
        )
        Ts = 1e-6

        result = iq_to_bin_power(iq, Ts, size * Ts, randomize=True)

        assert result.shape == (iq.shape[0] // size,)
        assert result.dtype == float_dtype_like(iq)
        power = np.abs(iq.astype(np.complex128)) ** 2
        windows = np.lib.stride_tricks.sliding_window_view(power, size).mean(axis=1)
        matches = np.isclose(
            result[:, np.newaxis], windows[np.newaxis, :], rtol=bin_power_rtol(iq, size)
        )
        assert np.all(matches.any(axis=1))

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

    @given(case=cyclic_power_cases())
    def test_matches_reference(self, case):
        iq, size, bins_per_cycle, n_cycles = case
        channels = iq.shape[0]
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
        reason='iq_to_cyclic_power(axis=-1) fails upstream in iq_to_bin_power, which '
        'applies its detector on axis+1 == 0 and so reduces the channel axis (see '
        'TestIqToBinPower.test_negative_axis)',
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


@st.composite
def ccdf_cases(draw):
    """(samples, sorted edges, density) arguments for sample_ccdf"""
    a = draw(
        arrays(
            dtype=np.float64,
            shape=st.integers(min_value=1, max_value=200),
            elements=st.floats(min_value=-100, max_value=100, allow_nan=False),
        )
    )
    edges = draw(
        st.lists(
            st.floats(min_value=-120, max_value=120, allow_nan=False),
            min_size=1,
            max_size=20,
            unique=True,
        )
    )
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

    @pytest.mark.parametrize(
        'func, scale, limits',
        [
            (powtodB, 10, {np.float64: 1e10, np.float32: 1e5}),
            (envtodB, 20, {np.float64: 1e6, np.float32: 1e4}),
        ],
        ids=func_id,
    )
    @pytest.mark.parametrize('abs', [True, False])
    @pytest.mark.parametrize('eps', [0, 1e-6])
    @given(data=st.data(), dtype=st.sampled_from([np.float64, np.float32]))
    def test_log_conversions(
        self, cupy_available, func, scale, limits, abs, eps, data, dtype
    ):
        """Cross-comparison on each fused kernel variant."""
        lim = limits[dtype]
        x = data.draw(
            positive_power_arrays(
                min_value=1 / lim, max_value=lim, dtype=dtype, min_dims=1, max_dims=1
            )
        )
        result_np, result_cp = numpy_and_cupy(
            cupy_available, func, x, min_dtype='float32', abs=abs, eps=eps
        )

        rtol, atol = log_conversion_tol(dtype, scale, n_impl=2)
        tol_dB = dB_tolerance(rtol, atol, np.abs(result_np).max())
        assert_allclose(
            result_cp, result_np, rtol=rtol, atol=atol, err_msg=f'{tol_dB:.2e} dB'
        )

    @given(data=st.data(), dtype=st.sampled_from([np.float64, np.float32]))
    def test_dBtopow(self, cupy_available, data, dtype):
        lim = {np.float64: 100, np.float32: 30}[dtype]
        dB = data.draw(
            dB_arrays(
                min_value=-lim, max_value=lim, dtype=dtype, min_dims=1, max_dims=1
            )
        )
        result_np, result_cp = numpy_and_cupy(
            cupy_available, dBtopow, dB, min_dtype='float32'
        )

        rtol = pow_conversion_rtol(dtype, np.abs(dB).max(), n_impl=2)
        tol_dB = linear_tolerance_dB(rtol)
        assert_allclose(result_cp, result_np, rtol=rtol, err_msg=f'{tol_dB:.2e} dB')

    @given(
        env=envelope_arrays(
            min_magnitude=1e-4, max_magnitude=1e4, min_dims=1, max_dims=1
        )
    )
    def test_envtopow(self, cupy_available, env):
        """Cross-comparison: real and complex input both give a real result."""
        result_np, result_cp = numpy_and_cupy(cupy_available, envtopow, env)

        assert np.isrealobj(result_np)
        assert np.isrealobj(result_cp)
        rtol = envelope_power_rtol(
            float_dtype_like(env), complex_input=np.iscomplexobj(env), n_impl=2
        )
        assert_allclose(result_cp, result_np, rtol=rtol)

    @pytest.mark.parametrize('func', [dBlinmean, dBlinsum], ids=func_id)
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
    def test_linear_statistics(self, cupy_available, func, dB):
        result_np, result_cp = numpy_and_cupy(cupy_available, func, dB, axis=None)

        rtol, atol = linear_stat_tol(np.float64, np.abs(dB).max(), dB.size, n_impl=2)
        tol_dB = dB_tolerance(rtol, atol, abs(result_np))
        assert_allclose(
            result_cp, result_np, rtol=rtol, atol=atol, err_msg=f'{tol_dB:.2e} dB'
        )

    @given(
        power=positive_power_arrays(
            min_value=1e-3, max_value=1e3, dtype=np.float32, min_dims=1, max_dims=1
        )
    )
    def test_overwrite_x(self, cupy_available, power):
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

    @given(data=st.data(), size=bin_sizes(), kind=bin_kinds())
    def test_iq_to_bin_power(self, cupy_available, data, size, kind):
        iq = data.draw(
            iq_waveforms(
                min_size=size, max_size=32 * size, multiple_of=size, channels=2
            )
        )
        Ts = 1e-6
        result_np, result_cp = numpy_and_cupy(
            cupy_available, iq_to_bin_power, iq, Ts, size * Ts, kind=kind, axis=1
        )

        assert result_cp.dtype == result_np.dtype
        assert_allclose(result_cp, result_np, rtol=bin_power_rtol(iq, size, 2))

    @given(case=cyclic_power_cases(channels=(2,)))
    def test_iq_to_cyclic_power(self, cupy_available, case):
        cp = cupy_available
        iq, size, bins_per_cycle, n_cycles = case
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

    @given(case=ccdf_cases())
    def test_sample_ccdf(self, cupy_available, case):
        """Cross-comparison: counting is exact, so the backends agree exactly."""
        a, edges, density = case
        result_np, result_cp = numpy_and_cupy(
            cupy_available, sample_ccdf, a, edges, density=density
        )
        assert_array_equal(result_cp, result_np)
