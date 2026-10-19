"""striqt.waveform.lib.power.convert: property tests of the dB conversions and their
roundoff models, including numpy-vs-cupy agreement within ulp budgets.
"""

from __future__ import annotations

import functools
from typing import ClassVar

import numpy as np
import pytest
from conftest import (
    as_xp,
    dB_arrays,
    envelope_arrays,
    positive_power_arrays,
)
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from numeric_checks import (
    FLOAT_DTYPES,
    assert_close,
    by_dtype,
    dtype_id,
    func_id,
    numpy_and_cupy,
    to_numpy,
)
from numpy.testing import assert_array_equal

from striqt.waveform.lib.arrays import float_dtype_like
from striqt.waveform.lib.power import (
    dB_tolerance,
    dBlinmean,
    dBlinsum,
    dBtopow,
    envelope_power_rtol,
    envtodB,
    envtopow,
    level_tolerance_dB,
    linear_stat_tol,
    linear_tolerance_dB,
    log_conversion_tol,
    off_peak_dB_tolerance,
    pow_conversion_rtol,
    powtodB,
    roundtrip_dB_tol,
    roundtrip_power_rtol,
    unit_dB_to_linear,
    unit_dB_to_wave,
    unit_linear_to_dB,
    unit_wave_to_dB,
    unit_wave_to_linear,
)
from striqt.waveform.lib.power.convert import _arraylike_with_buffer

# strategy factories for the valid inputs of each conversion, keyed by function
INPUTS = {
    powtodB: positive_power_arrays,
    dBtopow: dB_arrays,
    envtodB: functools.partial(envelope_arrays, include_complex=False),
    envtopow: functools.partial(envelope_arrays, include_complex=False),
    dBlinmean: functools.partial(dB_arrays, min_value=-50, max_value=50),
    dBlinsum: functools.partial(dB_arrays, min_value=-50, max_value=50),
}

# 1-D variants of the array strategies used by the elementwise conversion tests
one_d_powers = functools.partial(positive_power_arrays, min_dims=1, max_dims=1)
one_d_levels = functools.partial(dB_arrays, min_dims=1, max_dims=1)
ONE_D_ENVELOPES = envelope_arrays(
    min_magnitude=1e-4, max_magnitude=1e4, min_dims=1, max_dims=1
)
LINEAR_STAT_LEVELS = one_d_levels(
    min_value=-50, max_value=50, min_size=2, max_size=50, dtype=np.float64
)
FLOAT32_POWERS = one_d_powers(min_value=1e-3, max_value=1e3, dtype=np.float32)
POSITIVE_SCALARS = st.floats(
    min_value=1e-10, max_value=1e10, allow_nan=False, allow_infinity=False
)

# (function, dB per decade) for the two logarithmic conversions
LOG_CONVERSIONS = [(powtodB, 10), (envtodB, 20)]
by_log_conversion = pytest.mark.parametrize(
    'func, scale', LOG_CONVERSIONS, ids=[func_id(f) for f, _ in LOG_CONVERSIONS]
)
for_each_float_dtype = pytest.mark.parametrize('dtype', FLOAT_DTYPES, ids=dtype_id)

# (input dtype, computation dtype) under the default min_dtype='float32'
BOUNDED_DTYPES = [
    (np.float16, np.float32),
    (np.float32, np.float32),
    (np.float64, np.float64),
    (np.complex64, np.float32),
    (np.complex128, np.float64),
]


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
    def test_unknown_units_pass_through(self, subtests, unit):
        for func in (
            unit_dB_to_linear,
            unit_linear_to_dB,
            unit_dB_to_wave,
            unit_wave_to_dB,
            unit_wave_to_linear,
        ):
            with subtests.test(msg=func.__name__):
                assert func(unit) == unit


class TestConversionIdentities:
    """Properties: Mathematical identities that must hold for dB conversions."""

    @given(env=envelope_arrays(include_complex=False, dtype=np.float64))
    def test_envtopow_is_square(self, env):
        rtol = envelope_power_rtol(env.dtype, n_impl=2)
        assert_close(envtopow(env), np.abs(env) ** 2, rtol=rtol)

    @by_log_conversion
    @for_each_float_dtype
    @given(x=positive_power_arrays(dtype=np.float64))
    def test_log_conversion_is_scaled_log10(self, func, scale, dtype, x):
        """Property: func(x) = scale*log10(x) within the exact-math budget of `dtype`,
        with the oracle evaluated in float64 from the `dtype`-quantized input."""
        x = x.astype(dtype)
        expected = scale * np.log10(x.astype(np.float64))
        tol = log_conversion_tol(dtype, scale, n_impl=1)
        assert_close(func(x), expected, **tol)


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
        tol = linear_stat_tol(np.float64, abs(dB), n)
        assert_close(dBlinmean(arr), dB, **tol)
        assert_close(dBlinsum(arr), dB + 10 * np.log10(n), **tol)

    @pytest.mark.namespaces('numpy', 'cupy', 'dask')
    @pytest.mark.parametrize('axis', [None, 0, 1])
    @given(
        dB=dB_arrays(
            min_value=-50,
            max_value=50,
            dtype=np.float64,
            min_dims=2,
            max_dims=2,
            min_size=2,
            max_size=10,
        )
    )
    def test_axis_reduces_dimension_and_relates_mean_to_sum(self, xp, axis, dB):
        """Property: dBlinmean = dBlinsum - 10*log10(N) over the reduced axis."""
        if axis is None:
            expected_shape, N = (), dB.size
        else:
            expected_shape, N = dB.shape[:axis] + dB.shape[axis + 1 :], dB.shape[axis]

        x = as_xp(xp, dB)
        mean_result = to_numpy(dBlinmean(x, axis=axis))
        sum_result = to_numpy(dBlinsum(x, axis=axis))
        assert mean_result.shape == sum_result.shape == expected_shape

        expected_mean = sum_result - 10 * np.log10(N)
        tol = linear_stat_tol(np.float64, 50, N, n_impl=2)
        assert_close(mean_result, expected_mean, **tol)


class TestInputPreservation:
    @pytest.mark.parametrize('func', INPUTS, ids=func_id)
    @given(data=st.data())
    def test_overwrite_x_false_leaves_input_unchanged(self, func, data):
        x = data.draw(INPUTS[func](dtype=np.float64))
        original = x.copy()
        func(x, overwrite_x=False)
        assert_array_equal(x, original)


class TestInPlaceComplexInput:
    """Properties: a complex input overwritten in place comes back real-valued."""

    @by_log_conversion
    @pytest.mark.parametrize('dtype', [np.complex64, np.complex128], ids=dtype_id)
    def test_overwrite_x_writes_magnitude_dB_into_real_part(
        self, xp, func, scale, dtype
    ):
        """the result is scale*log10(|z|) in the real dtype of z, held in z's real part"""
        magnitude = np.array([1e-3, 0.5, 1.0, 2.0, 1e3], dtype=np.float64)
        phase = np.linspace(0, np.pi, magnitude.size)
        z = as_xp(xp, (magnitude * np.exp(1j * phase)).astype(dtype))

        result = to_numpy(func(z, overwrite_x=True))

        assert result.dtype == np.empty(0, dtype=dtype).real.dtype
        tol = log_conversion_tol(dtype, scale, complex_input=True)
        assert_close(result, scale * np.log10(magnitude), **tol)
        assert_array_equal(to_numpy(z).real, result)


class TestEdgeCases:
    """Properties: Behavior at edge cases (zeros, extreme values)."""

    @by_log_conversion
    def test_zero_produces_neg_inf(self, func, scale):
        assert np.all(func(np.zeros(4, dtype=np.float64)) == -np.inf)

    @by_log_conversion
    @pytest.mark.parametrize('dtype, compute_dtype', BOUNDED_DTYPES, ids=dtype_id)
    def test_bounded_zero_floor_follows_the_computation_dtype(
        self, xp, func, scale, dtype, compute_dtype
    ):
        result = to_numpy(func(as_xp(xp, np.zeros(4, dtype=dtype)), bounded=True))
        assert result.dtype == compute_dtype
        floor = 10 * np.log10(float(np.finfo(compute_dtype).tiny))
        assert np.all(np.isfinite(result))
        assert_close(result, floor, **log_conversion_tol(compute_dtype, scale))

    def test_empty_array(self, subtests):
        empty = np.array([], dtype=np.float64)
        for func in (powtodB, dBtopow, envtodB, envtopow):
            with subtests.test(msg=func.__name__):
                assert func(empty).shape == (0,)


@pytest.mark.namespaces('numpy', 'cupy', 'dask')
class TestMultiBackend:
    """Properties that must hold on every available array namespace."""

    @pytest.mark.parametrize('func', [powtodB, dBtopow, envtodB, envtopow], ids=func_id)
    @for_each_float_dtype
    @given(data=st.data())
    def test_dtype_is_preserved(self, xp, func, dtype, data):
        """Property: with min_dtype='float32', the input precision is kept."""
        x = as_xp(xp, data.draw(INPUTS[func](dtype=dtype)))
        assert to_numpy(func(x, min_dtype='float32')).dtype == dtype

    @pytest.mark.parametrize(
        'func, limits', [(powtodB, (1e-3, 1e3)), (dBtopow, (-30, 30))], ids=func_id
    )
    @given(data=st.data())
    def test_float16_is_promoted_to_float32(self, xp, func, limits, data):
        """The expected value follows the float16-quantized input, not the float32
        draw it was made from."""
        strategy = INPUTS[func](
            dtype=np.float32, min_value=limits[0], max_value=limits[1]
        )
        x16 = data.draw(strategy).astype(np.float16)

        result = func(as_xp(xp, x16), min_dtype='float32')
        assert to_numpy(result).dtype == np.float32

        expected = func(x16.astype(np.float32), min_dtype='float32')
        if func is dBtopow:
            tol = {'rtol': pow_conversion_rtol(np.float32, 30, n_impl=2)}
        else:
            tol = log_conversion_tol(np.float32, 10, n_impl=2)
        assert_close(result, expected, **tol)

    @given(power=positive_power_arrays(dtype=np.float64))
    def test_powtodB_dBtopow_roundtrip(self, xp, power):
        roundtrip = dBtopow(powtodB(as_xp(xp, power)))
        rtol = roundtrip_power_rtol(np.float64, np.abs(powtodB(power)).max())
        assert_close(roundtrip, power, rtol=rtol)

    @given(dB=dB_arrays(min_value=-140, max_value=100, dtype=np.float64))
    def test_dBtopow_powtodB_roundtrip(self, xp, dB):
        roundtrip = powtodB(dBtopow(as_xp(xp, dB)))
        assert_close(roundtrip, dB, **roundtrip_dB_tol(np.float64, np.abs(dB).max()))

    @given(env=envelope_arrays(dtype=np.float64))
    def test_complex_envelopes(self, xp, env):
        """Property: envtopow(z) = |z|² and envtodB(z) = 20*log10(|z|) for complex z."""
        x = as_xp(xp, env)

        power = to_numpy(envtopow(x))
        assert np.isrealobj(power)
        rtol = envelope_power_rtol(np.float64, complex_input=True, n_impl=2)
        assert_close(power, np.abs(env) ** 2, rtol=rtol)

        dB = to_numpy(envtodB(x))
        assert np.isrealobj(dB)
        tol = log_conversion_tol(np.float64, 20, complex_input=True, n_impl=2)
        assert_close(dB, 20 * np.log10(np.abs(env)), **tol)


class TestAbsAndBoundedBranches:
    """Properties: the `abs` and `bounded` arguments of powtodB and envtodB."""

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

    @pytest.mark.namespaces('numpy', 'cupy', 'dask')
    @by_log_conversion
    @given(
        power=positive_power_arrays(min_value=1e-3, max_value=1e3, dtype=np.float64),
        abs=st.booleans(),
        data=st.data(),
    )
    def test_bound_is_applied_on_every_backend(self, xp, func, scale, power, abs, data):
        """Property: the floor is applied by every backend's code path."""
        zero = data.draw(arrays(np.bool_, power.shape))
        # force one zero rather than assume(zero.any()), which would reject about
        # half of the draws for 1-element arrays
        zero.flat[data.draw(st.integers(0, power.size - 1))] = True
        power = np.where(zero, 0.0, power)

        result = func(as_xp(xp, power), bounded=True, abs=abs)

        expected = np.full(power.shape, 10 * np.log10(float(np.finfo(np.float64).tiny)))
        expected[~zero] = scale * np.log10(power[~zero])
        tol = log_conversion_tol(np.float64, scale, n_impl=2)
        assert_close(result, expected, **tol)


class TestArrayLikeHandling:
    """The array-like unpacking and repackaging behind every conversion."""

    @pytest.mark.parametrize(
        'min_dtype, match',
        [
            (None, 'min_dtype'),
            ('float16', 'float32 or larger'),
            (np.float16, 'float32 or larger'),
            (np.dtype('float16'), 'float32 or larger'),
        ],
        ids=['none', 'float16_str', 'float16_type', 'float16_dtype'],
    )
    def test_invalid_min_dtype_raises(self, min_dtype, match):
        with pytest.raises(TypeError, match=match):
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

    @pytest.mark.parametrize(
        'ctor, result_type',
        [(float, float), (np.float64, float), (np.array, np.ndarray)],
        ids=['float', 'float64', 'array'],
    )
    @given(value=POSITIVE_SCALARS)
    def test_scalar_input_returns_a_scalar(self, ctor, result_type, value):
        result = powtodB(ctor(value))
        assert isinstance(result, result_type)
        tol = log_conversion_tol(np.float64, 10, n_impl=2)
        assert_close(result, 10 * np.log10(value), **tol)

    def test_python_int_input(self):
        assert isinstance(dBtopow(10), float)
        assert_close(dBtopow(10), 10.0, rtol=pow_conversion_rtol(np.float64, 10))

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


class TestRoundoffModels:
    """The tolerance models that the rest of this module measures the library against."""

    @for_each_float_dtype
    def test_log_conversion_tol_scales_with_n_impl_and_scale(self, dtype):
        one = log_conversion_tol(dtype, 10)
        two = log_conversion_tol(dtype, 10, n_impl=2)
        assert two['rtol'] == 2 * one['rtol']
        assert two['atol'] == 2 * one['atol']

        double_scale = log_conversion_tol(dtype, 20)
        assert double_scale['atol'] == 2 * one['atol']
        assert double_scale['rtol'] == one['rtol']

    @for_each_float_dtype
    def test_log_conversion_tol_complex_input_raises_atol(self, dtype):
        real = log_conversion_tol(dtype, 20)
        complex_ = log_conversion_tol(dtype, 20, complex_input=True)
        assert complex_['atol'] > real['atol']
        assert complex_['rtol'] == real['rtol']

    @for_each_float_dtype
    def test_pow_conversion_rtol_grows_with_level(self, dtype):
        assert pow_conversion_rtol(dtype, 100) > pow_conversion_rtol(dtype, 10)

    @pytest.mark.parametrize('r', [1e-5, 0.1, 0.9])
    def test_off_peak_dB_tolerance_at_peak(self, r):
        """the low side of a relative error costs at least as much dB as the high side"""
        at_peak = off_peak_dB_tolerance(0, r)
        assert at_peak == pytest.approx(-20 * np.log10(1 - r))
        assert at_peak >= level_tolerance_dB(r)

    def test_off_peak_dB_tolerance_unbounded_past_floor(self):
        err = 1e-3
        depth_dBc = -20 * np.log10(err)
        assert np.isposinf(off_peak_dB_tolerance(depth_dBc, err))
        assert np.isposinf(off_peak_dB_tolerance(depth_dBc + 10, err))
        assert np.isfinite(off_peak_dB_tolerance(depth_dBc - 1, err))

    def test_level_tolerance_dB_power_is_half_of_amplitude(self):
        sigma = np.array([1e-6, 1e-3, 0.1, 1.0])
        np.testing.assert_allclose(
            level_tolerance_dB(sigma, power=True), level_tolerance_dB(sigma) / 2
        )


class TestNumpyCupyCrossComparison:
    """Cross-comparison tests validating numpy and cupy produce close results.

    Tolerances come from the ulp budgets in striqt.waveform.lib.power, so a
    difference larger than the two libraries' rounding bounds fails the test.
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
    @pytest.mark.parametrize('bounded', [False, True], ids=['unbounded', 'bounded'])
    @for_each_float_dtype
    @given(data=st.data())
    def test_log_conversions(
        self, cupy_available, func, scale, limits, abs, bounded, dtype, data
    ):
        """Cross-comparison on each fused kernel variant."""
        lim = limits[dtype]
        x = data.draw(one_d_powers(min_value=1 / lim, max_value=lim, dtype=dtype))
        result_np, result_cp = numpy_and_cupy(
            cupy_available, func, x, min_dtype='float32', abs=abs, bounded=bounded
        )

        tol = log_conversion_tol(dtype, scale, n_impl=2)
        tol_dB = dB_tolerance(max_abs_dB=np.abs(result_np).max(), **tol)
        assert_close(result_cp, result_np, err_msg=f'{tol_dB:.2e} dB', **tol)

    @for_each_float_dtype
    @given(data=st.data())
    def test_dBtopow(self, cupy_available, dtype, data):
        lim = by_dtype(dtype, float32=30, float64=100)
        dB = data.draw(one_d_levels(min_value=-lim, max_value=lim, dtype=dtype))
        result_np, result_cp = numpy_and_cupy(
            cupy_available, dBtopow, dB, min_dtype='float32'
        )

        rtol = pow_conversion_rtol(dtype, np.abs(dB).max(), n_impl=2)
        tol_dB = linear_tolerance_dB(rtol)
        assert_close(result_cp, result_np, rtol=rtol, err_msg=f'{tol_dB:.2e} dB')

    @given(env=ONE_D_ENVELOPES)
    def test_envtopow(self, cupy_available, env):
        """Cross-comparison: real and complex input both give a real result."""
        result_np, result_cp = numpy_and_cupy(cupy_available, envtopow, env)

        assert np.isrealobj(result_np)
        assert np.isrealobj(result_cp)
        rtol = envelope_power_rtol(
            float_dtype_like(env), complex_input=np.iscomplexobj(env), n_impl=2
        )
        assert_close(result_cp, result_np, rtol=rtol)

    @pytest.mark.parametrize('func', [dBlinmean, dBlinsum], ids=func_id)
    @given(dB=LINEAR_STAT_LEVELS)
    def test_linear_statistics(self, cupy_available, func, dB):
        result_np, result_cp = numpy_and_cupy(cupy_available, func, dB, axis=None)

        tol = linear_stat_tol(np.float64, np.abs(dB).max(), dB.size, n_impl=2)
        tol_dB = dB_tolerance(max_abs_dB=abs(result_np), **tol)
        assert_close(result_cp, result_np, err_msg=f'{tol_dB:.2e} dB', **tol)

    @given(power=FLOAT32_POWERS)
    def test_overwrite_x(self, cupy_available, power):
        """Cross-comparison: in-place evaluation writes the result into the input."""
        power_cp = cupy_available.asarray(power)
        result_cp = powtodB(power_cp, overwrite_x=True)

        assert result_cp is power_cp
        tol = log_conversion_tol(np.float32, 10, n_impl=2)
        assert_close(result_cp, powtodB(power), **tol)

    def test_xarray_cupy_data(self, cupy_available):
        """Cross-comparison: a DataArray wrapping cupy data keeps cupy data."""
        xr = pytest.importorskip('xarray')

        data = np.linspace(0.5, 2.0, 8, dtype=np.float32)
        da = xr.DataArray(
            cupy_available.asarray(data), dims=['t'], attrs={'units': 'mW'}
        )

        result = powtodB(da)
        assert isinstance(result.data, cupy_available.ndarray)
        assert result.attrs['units'] == 'dBm'
        tol = log_conversion_tol(np.float32, 10, n_impl=2)
        assert_close(result.data, powtodB(data), **tol)
