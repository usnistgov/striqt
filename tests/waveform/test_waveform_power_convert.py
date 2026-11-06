"""striqt.waveform.lib.power.convert: property tests of the dB conversions and the
roundoff models, including numpy-vs-cupy agreement within ulp budgets.
"""

from __future__ import annotations

import functools

import numpy as np
import pytest
from conftest import (
    as_xp,
    dB_arrays,
    envelope_arrays,
    positive_power_arrays,
)
from hypothesis import given, strategies as st
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
    linear_stat_tol,
    linear_tolerance_dB,
    log_conversion_tol,
    pow_conversion_rtol,
    powtodB,
    roundtrip_dB_tol,
    roundtrip_power_rtol,
)

# deterministic float64 inputs for the tests of a side effect, a type, or the dask
# dispatch, none of which is a numeric property
LEVELS = np.array([[-30.0, -3.0, 0.0], [3.0, 10.0, 30.0]])
POWERS = 10 ** (LEVELS / 10)
PHASES = np.linspace(0, np.pi, LEVELS.size).reshape(LEVELS.shape)
ENVELOPES = np.sqrt(POWERS) * np.exp(1j * PHASES)
VALID_INPUTS = {
    powtodB: POWERS,
    dBtopow: LEVELS,
    envtodB: POWERS,
    envtopow: POWERS,
    dBlinmean: LEVELS,
    dBlinsum: LEVELS,
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

# (function, dB per decade) for the 2 logarithmic conversions
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


class TestConversionIdentities:
    """Properties: Mathematical identities that must hold for dB conversions."""

    @given(env=envelope_arrays(include_complex=False, dtype=np.float64))
    def test_envtopow_is_square(self, env):
        rtol = envelope_power_rtol(env.dtype, n_impl=2)
        assert_close(envtopow(env), np.abs(env) ** 2, rtol=rtol)

    @by_log_conversion
    @for_each_float_dtype
    @pytest.mark.parametrize('abs', [True, False], ids=['abs', 'noabs'])
    @given(x=positive_power_arrays(dtype=np.float64))
    def test_log_conversion_is_scaled_log10(self, func, scale, dtype, abs, x):
        """``func(x) == scale*log10(x)`` for positive `x` with or without `abs`,
        within the exact-math budget of `dtype`, with the oracle evaluated in float64
        from the `dtype`-quantized input"""
        x = x.astype(dtype)
        expected = scale * np.log10(x.astype(np.float64))
        tol = log_conversion_tol(dtype, scale, n_impl=1)
        assert_close(func(x, abs=abs), expected, **tol)


class TestAlgebraicProperties:
    """Properties: Algebraic rules for dB arithmetic."""

    @given(
        dB=st.floats(
            min_value=-140, max_value=100, allow_nan=False, allow_infinity=False
        ),
        n=st.integers(min_value=1, max_value=20),
    )
    def test_equal_values(self, dB, n):
        """``dBlinmean([x, ..., x]) == x`` and
        ``dBlinsum([x, ..., x]) == x + 10*log10(n)``"""
        arr = np.array([dB] * n, dtype=np.float64)
        tol = linear_stat_tol(np.float64, abs(dB), n)
        assert_close(dBlinmean(arr), dB, **tol)
        assert_close(dBlinsum(arr), dB + 10 * np.log10(n), **tol)

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
        """``dBlinmean == dBlinsum - 10*log10(N)`` over the reduced axis, and the axis
        is removed"""
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
    @pytest.mark.parametrize('func', VALID_INPUTS, ids=func_id)
    def test_overwrite_x_false_leaves_input_unchanged(self, func):
        x = VALID_INPUTS[func].copy()
        func(x, overwrite_x=False)
        assert_array_equal(x, VALID_INPUTS[func])


class TestInPlaceComplexInput:
    """a complex input overwritten in place comes back real-valued"""

    @by_log_conversion
    @pytest.mark.parametrize('dtype', [np.complex64, np.complex128], ids=dtype_id)
    def test_overwrite_x_writes_magnitude_dB_into_real_part(
        self, xp, func, scale, dtype
    ):
        """the result is ``scale*log10(|z|)`` in the real dtype of `z`, written into
        the real part of `z`"""
        magnitude = np.array([1e-3, 0.5, 1.0, 2.0, 1e3], dtype=np.float64)
        phase = np.linspace(0, np.pi, magnitude.size)
        z = as_xp(xp, (magnitude * np.exp(1j * phase)).astype(dtype))

        result = to_numpy(func(z, overwrite_x=True))

        assert result.dtype == np.empty(0, dtype=dtype).real.dtype
        tol = log_conversion_tol(dtype, scale, complex_input=True)
        assert_close(result, scale * np.log10(magnitude), **tol)
        assert_array_equal(to_numpy(z).real, result)


class TestEdgeCases:
    """zero inputs, the bounded zero floor, and empty arrays"""

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


class TestMultiBackend:
    """the properties that hold on numpy and cupy"""

    @pytest.mark.parametrize('func', [powtodB, dBtopow, envtodB, envtopow], ids=func_id)
    @for_each_float_dtype
    def test_dtype_is_preserved(self, xp, func, dtype):
        """with ``min_dtype='float32'``, the input precision is kept"""
        x = as_xp(xp, np.array([0.5, 2.0], dtype=dtype))
        assert to_numpy(func(x, min_dtype='float32')).dtype == dtype

    @pytest.mark.parametrize(
        'func, values, reference, tol',
        [
            (
                powtodB,
                (1e-3, 1e3),
                lambda x: 10 * np.log10(x),
                log_conversion_tol(np.float32, 10),
            ),
            (
                dBtopow,
                (-30, 30),
                lambda x: 10 ** (x / 10),
                {'rtol': pow_conversion_rtol(np.float32, 30)},
            ),
        ],
        ids=func_id,
    )
    def test_float16_is_promoted_to_float32(self, xp, func, values, reference, tol):
        """the float32 result follows the float16-quantized input"""
        x16 = np.array(values, dtype=np.float16)

        result = func(as_xp(xp, x16), min_dtype='float32')
        assert to_numpy(result).dtype == np.float32
        assert_close(result, reference(x16.astype(np.float64)), **tol)

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
        """``envtopow(z) == |z|**2`` and ``envtodB(z) == 20*log10(|z|)`` for complex
        `z`"""
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
    """the `abs` and `bounded` arguments of `powtodB` and `envtodB`"""

    @given(power=positive_power_arrays(min_value=1e-6, max_value=1e6, dtype=np.float64))
    def test_noabs_negative_input_is_nan(self, power):
        """without `abs`, log10 of a negative value is nan rather than the value at
        the magnitude"""
        assert np.all(np.isnan(powtodB(-power, abs=False)))
        assert np.all(np.isnan(envtodB(-power, abs=False)))
        assert np.all(np.isfinite(powtodB(-power, abs=True)))
        assert np.all(np.isfinite(envtodB(-power, abs=True)))

    @by_log_conversion
    @pytest.mark.parametrize('abs', [True, False], ids=['abs', 'noabs'])
    def test_bound_is_applied_on_every_backend(self, xp, func, scale, abs):
        """zeros come out at the floor of the computation dtype and the other samples
        at their exact-math level, with or without `abs`"""
        power = np.array([[0.0, 1e-3, 1.0], [1e3, 0.0, 2.0]])
        zero = power == 0

        result = func(as_xp(xp, power), bounded=True, abs=abs)

        expected = np.full(power.shape, 10 * np.log10(float(np.finfo(np.float64).tiny)))
        expected[~zero] = scale * np.log10(power[~zero])
        tol = log_conversion_tol(np.float64, scale, n_impl=2)
        assert_close(result, expected, **tol)


@pytest.mark.namespaces('dask')
class TestDaskNamespace:
    """one deterministic example per conversion on dask"""

    @pytest.mark.parametrize(
        'func, x, kws, reference, tol',
        [
            pytest.param(
                powtodB,
                POWERS,
                {},
                lambda x: 10 * np.log10(x),
                log_conversion_tol(np.float64, 10),
                id='powtodB',
            ),
            pytest.param(
                dBtopow,
                LEVELS,
                {},
                lambda x: 10 ** (x / 10),
                {'rtol': pow_conversion_rtol(np.float64, 30)},
                id='dBtopow',
            ),
            pytest.param(
                envtodB,
                ENVELOPES,
                {},
                lambda x: 20 * np.log10(np.abs(x)),
                log_conversion_tol(np.float64, 20, complex_input=True),
                id='envtodB',
            ),
            pytest.param(
                envtopow,
                ENVELOPES,
                {},
                lambda x: np.abs(x) ** 2,
                {'rtol': envelope_power_rtol(np.float64, complex_input=True)},
                id='envtopow',
            ),
            pytest.param(
                dBlinmean,
                LEVELS,
                {'axis': 1},
                lambda x: 10 * np.log10(np.mean(10 ** (x / 10), axis=1)),
                linear_stat_tol(np.float64, 30, LEVELS.shape[1]),
                id='dBlinmean',
            ),
            pytest.param(
                dBlinsum,
                LEVELS,
                {'axis': 0},
                lambda x: 10 * np.log10(np.sum(10 ** (x / 10), axis=0)),
                linear_stat_tol(np.float64, 30, LEVELS.shape[0]),
                id='dBlinsum',
            ),
        ],
    )
    def test_returns_a_dask_array_at_the_exact_math_value(
        self, xp, func, x, kws, reference, tol
    ):
        result = func(as_xp(xp, x), **kws)
        assert isinstance(result, xp.Array)
        assert_close(result, reference(x), **tol)


class _ValuesOnly:
    """an object with a `values` attribute that belongs to no array container type"""

    values = np.ones(4)


class TestArrayLikeHandling:
    """the array-like unpacking and repackaging behind every conversion"""

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
            powtodB(np.ones(3), min_dtype=min_dtype)

    @pytest.mark.parametrize('obj', ['text', [1.0, 2.0], (1.0, 2.0), object()])
    def test_unsupported_input_raises(self, obj):
        with pytest.raises(TypeError, match='unable to associate'):
            powtodB(obj)

    def test_scalar_input_is_never_overwritten(self):
        x = np.array(2.0, dtype=np.float32)
        result = powtodB(x, overwrite_x=True)
        assert x == 2.0
        assert_close(result, 10 * np.log10(2.0), **log_conversion_tol(np.float32, 10))

    @pytest.mark.parametrize(
        'ctor, result_type',
        [(float, float), (np.float64, float), (np.array, np.ndarray)],
        ids=['float', 'float64', 'array'],
    )
    def test_scalar_input_returns_a_scalar(self, ctor, result_type):
        value = 2.5
        result = powtodB(ctor(value))
        assert isinstance(result, result_type)
        assert_close(result, 10 * np.log10(value), **log_conversion_tol(np.float64, 10))

    def test_python_int_input(self):
        assert isinstance(dBtopow(10), float)
        assert_close(dBtopow(10), 10.0, rtol=pow_conversion_rtol(np.float64, 10))

    @pytest.mark.parametrize(
        'make',
        [
            lambda pd: pd.Series(POWERS[0], index=pd.RangeIndex(10, 10 + 3)),
            lambda pd: pd.DataFrame(POWERS, columns=['a', 'b', 'c']),
        ],
        ids=['series', 'frame'],
    )
    def test_pandas_container_keeps_its_axes(self, make):
        pd = pytest.importorskip('pandas')
        obj = make(pd)

        result = powtodB(obj)

        assert type(result) is type(obj)
        for axis, expected in zip(result.axes, obj.axes):
            assert axis.equals(expected)
        tol = log_conversion_tol(np.float64, 10)
        assert_close(result.values, 10 * np.log10(obj.values), **tol)

    @pytest.mark.parametrize(
        'func,units_in,units_out',
        [
            (powtodB, 'mW', 'dBm'),
            (powtodB, 'W/Hz', 'dBW/Hz'),
            (powtodB, None, None),
            (dBtopow, 'dBm', 'mW'),
            (dBtopow, 'dB', 'unitless'),
            (envtodB, '√mW', 'dBm'),
            (envtopow, '√W', 'W'),
        ],
        ids=lambda v: 'none' if v is None else v if isinstance(v, str) else func_id(v),
    )
    def test_xarray_units_transform(self, func, units_in, units_out):
        """the units attr follows the conversion, and is absent when the input has
        none"""
        xr = pytest.importorskip('xarray')

        data = np.linspace(0.5, 2.0, 8, dtype=np.float64)
        attrs = {} if units_in is None else {'units': units_in}
        da = xr.DataArray(data, dims=['t'], coords={'t': np.arange(8)}, attrs=attrs)

        result = func(da)
        assert isinstance(result, xr.DataArray)
        assert result.dims == ('t',)
        assert result.attrs.get('units') == units_out
        assert da.attrs.get('units') == units_in
        assert_array_equal(result.values, func(data))

    @pytest.mark.parametrize(
        'make',
        [
            lambda: pytest.importorskip('xarray').Dataset({'a': ('t', np.ones(4))}),
            _ValuesOnly,
        ],
        ids=['dataset', 'object_with_values'],
    )
    def test_container_without_an_array_type_raises(self, make):
        with pytest.raises(TypeError, match='unable to associate an array type'):
            powtodB(make())


class TestNumpyCupyCrossComparison:
    """the numpy and cupy results of each conversion compared against each other.

    Tolerances come from the ulp budgets in `striqt.waveform.lib.power`.
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
        """each fused kernel variant agrees between the backends"""
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
        """real and complex input both give a real result"""
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
        """in-place evaluation writes the result into the input"""
        power_cp = cupy_available.asarray(power)
        result_cp = powtodB(power_cp, overwrite_x=True)

        assert result_cp is power_cp
        tol = log_conversion_tol(np.float32, 10, n_impl=2)
        assert_close(result_cp, powtodB(power), **tol)

    def test_xarray_cupy_data(self, cupy_available):
        """a DataArray wrapping cupy data keeps cupy data"""
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
