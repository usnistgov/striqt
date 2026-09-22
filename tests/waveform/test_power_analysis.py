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
    as_xp,
    dB_arrays,
    envelope_arrays,
    iq_waveforms,
    positive_power_arrays,
)
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from numeric_checks import (
    FLOAT_DTYPES,
    assert_close,
    blocks,
    by_dtype,
    dtype_id,
    func_id,
    numpy_and_cupy,
    reference_power,
    to_numpy,
)
from numpy.testing import assert_array_equal

from striqt.waveform.lib.arrays import accum_rtol, float_dtype_like
from striqt.waveform.lib.power_analysis import (
    DB_PER_NEPER,
    _arraylike_with_buffer,
    bin_power_rtol,
    dB_tolerance,
    dB_tolerance_below_peak,
    dBlinmean,
    dBlinsum,
    dBtopow,
    envelope_power_rtol,
    envtodB,
    envtopow,
    iq_to_bin_power,
    iq_to_cyclic_power,
    level_tolerance_dB,
    linear_stat_tol,
    linear_tolerance_dB,
    log_conversion_tol,
    pow_conversion_rtol,
    powtodB,
    roundtrip_dB_tol,
    roundtrip_power_rtol,
    sample_ccdf,
    stat_ufunc_from_shorthand,
    unit_dB_to_linear,
    unit_dB_to_wave,
    unit_linear_to_dB,
    unit_wave_to_dB,
    unit_wave_to_linear,
)

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


NAMED_STATS = {
    'min': np.min,
    'max': np.max,
    'peak': np.max,
    'mean': np.mean,
    'rms': np.mean,
    'median': np.median,
}
BIN_STATS = {**NAMED_STATS, 0.9: functools.partial(np.quantile, q=0.9)}


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
        ],
        ids=['average', 'rms2', 'empty', 'tuple'],
    )
    def test_invalid_kind_raises(self, kind, match):
        with pytest.raises(ValueError, match=match):
            stat_ufunc_from_shorthand(kind)


class TestConversionIdentities:
    """Properties: Mathematical identities that must hold for dB conversions."""

    @given(env=envelope_arrays(include_complex=False, dtype=np.float64))
    def test_envtopow_is_square(self, env):
        rtol = envelope_power_rtol(env.dtype, n_impl=2)
        assert_close(envtopow(env), np.abs(env) ** 2, rtol=rtol)

    @by_log_conversion
    @given(x=positive_power_arrays(dtype=np.float64))
    def test_log_conversion_is_scaled_log10(self, func, scale, x):
        tol = log_conversion_tol(x.dtype, scale, n_impl=2)
        assert_close(func(x), scale * np.log10(x), **tol)


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
        tol = log_conversion_tol(np.float64, scale, n_impl=2)
        assert_close(result[0], scale * np.log10(eps), **tol)

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

    @pytest.mark.namespaces('numpy', 'cupy', 'dask')
    @by_log_conversion
    @given(
        power=positive_power_arrays(min_value=1e-3, max_value=1e3, dtype=np.float64),
        eps=st.floats(min_value=1e-9, max_value=1e-3),
        abs=st.booleans(),
    )
    def test_eps_is_added_before_log(self, xp, func, scale, power, eps, abs):
        """Property: the eps offset is applied by every backend's code path."""
        result = func(as_xp(xp, power), eps=eps, abs=abs)
        expected = scale * np.log10(power + eps)
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

    def test_dB_tolerance_below_peak_at_peak(self):
        r = 1e-3
        assert dB_tolerance_below_peak(0, r) == pytest.approx(-20 * np.log10(1 - r))

    def test_dB_tolerance_below_peak_unbounded_past_floor(self):
        err = 1e-3
        floor_dBc = -20 * np.log10(err)
        assert np.isposinf(dB_tolerance_below_peak(floor_dBc, err))
        assert np.isposinf(dB_tolerance_below_peak(floor_dBc + 10, err))
        assert np.isfinite(dB_tolerance_below_peak(floor_dBc - 1, err))

    @pytest.mark.parametrize('r', [1e-5, 0.1, 0.9])
    def test_dB_tolerance_below_peak_bounds_one_sided_expansion(self, r):
        """-20*log10(1-r) is at least the one-sided series DB_PER_NEPER*(2r + r**2)."""
        assert dB_tolerance_below_peak(0, r) >= DB_PER_NEPER * (2 * r + r**2)

    def test_bin_power_rtol_grows_with_bin_size(self):
        assert bin_power_rtol(np.float32, 16) > bin_power_rtol(np.float32, 1)

    def test_level_tolerance_dB_power_is_half_of_amplitude(self):
        sigma = np.array([1e-6, 1e-3, 0.1, 1.0])
        np.testing.assert_allclose(
            level_tolerance_dB(sigma, power=True), level_tolerance_dB(sigma) / 2
        )


def bin_power_reference(iq, size, kind, axis):
    power = reference_power(iq)
    return BIN_STATS[kind](blocks(power, size, axis), axis=axis + 1)


BIN_SIZES = [1, 5, 16]
for_each_bin_size = pytest.mark.parametrize('size', BIN_SIZES)
for_each_bin_kind = pytest.mark.parametrize('kind', sorted(BIN_STATS, key=str))


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
    @for_each_bin_size
    @pytest.mark.parametrize('kind', ['mean', 'max', 0.9])
    @pytest.mark.parametrize('channels', [None, 3])
    @given(data=st.data())
    def test_matches_reference(self, size, kind, channels, data):
        """Property: each bin is the statistic of |iq|**2 over `size` samples,
        for a 1-D waveform (axis 0) and a (channel, sample) array (axis 1).

        The kinds are one name from each group of BIN_STATS: the aliases are
        covered by TestStatUfuncFromShorthand, and the cupy cross-comparison runs
        every kind because each is a separate kernel there."""
        waveforms = iq_waveforms(
            min_size=size, max_size=32 * size, multiple_of=size, channels=channels
        )
        iq = data.draw(waveforms)
        axis = 0 if channels is None else 1
        Ts = 1e-6

        result = iq_to_bin_power(iq, Ts, size * Ts, kind=kind, axis=axis)

        assert result.dtype == float_dtype_like(iq)
        expected = bin_power_reference(iq, size, kind, axis)
        assert result.shape == expected.shape
        assert_close(result, expected, rtol=bin_power_rtol(float_dtype_like(iq), size))

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
        assert_close(result, expected, rtol=bin_power_rtol(float_dtype_like(iq), size))

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

    @pytest.mark.parametrize('size', BIN_SIZES[1:])
    @given(data=st.data())
    def test_randomize(self, size, data):
        """Property: every random bin is the mean power of some `size` contiguous
        samples, i.e. a value of the sliding-window mean of |iq|**2."""
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
            rtol=bin_power_rtol(float_dtype_like(iq), size),
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
    cyclic_channel_power measurement calls it.

    The expectations follow doc/papers/cyclic_power.tex: the detector output is
    folded into a (cycle, lag) matrix with cyclic_period / detector_period lags per
    cycle, and each cyclic statistic reduces the cycle axis.
    """

    DETECTORS = ('rms', 'peak')
    CYCLE_STATS = ('min', 'mean', 'max', 0.9)

    @staticmethod
    def evaluate(iq, size, lags, **kws):
        """iq_to_cyclic_power with unit sample period, `size` samples per detector
        bin and `lags` bins per cycle"""
        kws = {'detectors': TestIqToCyclicPower.DETECTORS, **kws}
        kws = {'cycle_stats': TestIqToCyclicPower.CYCLE_STATS, 'axis': 1, **kws}
        return iq_to_cyclic_power(iq, 1.0, size, size * lags, **kws)

    @given(case=cyclic_power_cases())
    def test_matches_reference(self, case):
        """Property: each statistic is the column-wise reduction of the binned power
        reshaped to (cycle, lag), including a quantile statistic."""
        iq, size, bins_per_cycle, n_cycles = case
        channels = iq.shape[0]

        result = self.evaluate(iq, size, bins_per_cycle)

        assert set(result) == set(self.DETECTORS)
        rtol = accum_rtol(float_dtype_like(iq), n_cycles)
        for detector in self.DETECTORS:
            assert set(result[detector]) == set(self.CYCLE_STATS)
            binned = iq_to_bin_power(iq, 1.0, size, kind=detector, axis=1)
            by_cycle = binned.reshape(channels, n_cycles, bins_per_cycle)
            for stat in self.CYCLE_STATS:
                value = result[detector][stat]
                assert value.shape == (channels, bins_per_cycle)
                expected = BIN_STATS[stat](by_cycle, axis=1)
                assert_close(value, expected, rtol=rtol)

    @given(case=cyclic_power_cases())
    def test_lag_average_of_mean_rms_is_the_waveform_power(self, case):
        """Property: the fold keeps every detector sample (K = L * M), so averaging
        the mean rms trace over lags recovers the mean power of the waveform."""
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
        """A signal whose period divides cyclic_period repeats identically in every
        cycle, so every statistic returns the one-cycle power envelope, and the lag
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
        """Pulses whose period divides the cycle stay at fixed lags with zero spread.
        Pulses at a period coprime with the lag count visit every lag: the max
        trace is the pulse power everywhere, the min trace is zero, and the mean
        is the pulse power diluted by the period."""
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
        """A period mismatch drifts a feature by (n_cycles - 1) * drift_per_cycle
        samples over the capture. The peak trace keeps the pulse in one lag with no
        spread while that stays below the bin size, and leaks it into the next lag
        once it reaches the bin size."""
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
        """The paper defines the input as a single power time series, and axis=0 is
        the default."""
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
    """Cross-comparison tests validating numpy and cupy produce close results.

    Tolerances come from the ulp budgets in striqt.waveform.lib.power_analysis, so a
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
    @pytest.mark.parametrize('eps', [0, 1e-6])
    @for_each_float_dtype
    @given(data=st.data())
    def test_log_conversions(
        self, cupy_available, func, scale, limits, abs, eps, dtype, data
    ):
        """Cross-comparison on each fused kernel variant."""
        lim = limits[dtype]
        x = data.draw(one_d_powers(min_value=1 / lim, max_value=lim, dtype=dtype))
        result_np, result_cp = numpy_and_cupy(
            cupy_available, func, x, min_dtype='float32', abs=abs, eps=eps
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
        rtol = bin_power_rtol(float_dtype_like(iq), size, 2)
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
        rtol = bin_power_rtol(dtype, size, 2) + accum_rtol(dtype, n_cycles, 2)
        for detector, stats in result_np.items():
            for stat, value in stats.items():
                assert_close(result_cp[detector][stat], value, rtol=rtol)

    @given(case=ccdf_cases())
    def test_sample_ccdf(self, cupy_available, case):
        """Cross-comparison: counting is exact, so the backends agree exactly."""
        a, edges, density = case
        result_np, result_cp = numpy_and_cupy(
            cupy_available, sample_ccdf, a, edges, density=density
        )
        assert_array_equal(result_cp, result_np)
