"""Property-based tests for striqt.waveform.lib.arrays using Hypothesis.

Covers rounding, dtype helpers, binned and sliding windows, blocking, histograms,
axis slicing and indexing, grouped views, padding, the numpy-to-xp conversion
decorator, and the cupy runtime helpers, each against a plain numpy reference.
Known defects are recorded as strict expected failures.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from conftest import _cupy, shaped_arrays, to_numpy
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import array_shapes
from hypothesis.extra.numpy import arrays as np_arrays
from numpy.testing import assert_allclose, assert_array_equal

from striqt.waveform.lib import arrays

PROPERTY = settings(
    suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None
)


@pytest.fixture
def cupy_available():
    if _cupy is None:
        pytest.skip('cupy is not available')
    return _cupy


def float_arrays(shape, dtype=np.float64, min_value=-10.0, max_value=10.0):
    return np_arrays(
        dtype=dtype,
        shape=shape,
        elements=st.floats(
            min_value=min_value,
            max_value=max_value,
            allow_nan=False,
            allow_infinity=False,
            width=np.dtype(dtype).itemsize * 8,
        ),
    )


def mean_atol(x, count):
    """absolute roundoff bound on the mean of `count` samples drawn from `x`"""
    return count * np.finfo(x.dtype).eps * float(np.abs(x).max())


def reference_binned_mean(x, count, axis):
    """mean over contiguous, left-aligned bins of `count` along `axis`"""
    moved = np.moveaxis(x, axis, -1)
    m = moved.shape[-1] // count
    binned = moved[..., : m * count].reshape(moved.shape[:-1] + (m, count))
    return np.moveaxis(binned.mean(axis=-1), -1, axis)


class TestIsRoundMod:
    """Properties: multiples of the divisor are detected, half-multiples are not."""

    @PROPERTY
    @given(
        k=st.integers(min_value=-1000, max_value=1000),
        div=st.floats(min_value=1e-3, max_value=1e3, allow_nan=False),
    )
    def test_integer_multiple_is_round(self, k, div):
        assert arrays.isroundmod(k * div, div)

    @PROPERTY
    @given(
        k=st.integers(min_value=-1000, max_value=1000),
        div=st.floats(min_value=1e-3, max_value=1e3, allow_nan=False),
    )
    def test_half_multiple_is_not_round(self, k, div):
        assert not arrays.isroundmod((k + 0.5) * div, div)

    @PROPERTY
    @given(
        k=st.integers(min_value=-1000, max_value=1000),
        div=st.floats(min_value=1e-3, max_value=1e3, allow_nan=False),
    )
    def test_array_input_is_elementwise(self, k, div):
        values = np.array([k * div, (k + 0.5) * div])
        assert_array_equal(arrays.isroundmod(values, div), [True, False])

    def test_atol_controls_the_decision(self):
        assert not arrays.isroundmod(1.001, 1.0)
        assert arrays.isroundmod(1.001, 1.0, atol=1e-2)


class TestDtypeChangeFloat:
    @pytest.mark.parametrize(
        'dtype,basis,expected',
        [
            (np.complex128, np.float32, np.complex64),
            (np.complex64, np.float64, np.complex128),
            (np.complex64, np.float32, np.complex64),
            (np.float64, np.float32, np.float32),
            (np.float32, np.float64, np.float64),
            (np.float16, np.float64, np.float64),
        ],
    )
    def test_valid_pairs(self, dtype, basis, expected):
        assert arrays.dtype_change_float(dtype, basis) is expected

    @pytest.mark.parametrize(
        'dtype,basis',
        [(np.complex64, np.float16), (np.int32, np.float32), (np.float32, np.int32)],
    )
    def test_invalid_pairs_raise(self, dtype, basis):
        with pytest.raises(ValueError):
            arrays.dtype_change_float(dtype, basis)


class TestFloatDtypeLike:
    @pytest.mark.parametrize(
        'dtype,expected',
        [
            (np.complex64, np.float32),
            (np.complex128, np.float64),
            (np.float16, np.float16),
            (np.float32, np.float32),
            (np.float64, np.float64),
            (np.int32, np.float32),
        ],
    )
    def test_array_dtypes(self, dtype, expected):
        assert arrays.float_dtype_like(np.zeros(3, dtype=dtype)) == expected

    def test_python_scalars(self):
        assert arrays.float_dtype_like(3) == np.float32
        assert arrays.float_dtype_like(2.0) == np.float64

    def test_sequence_input_falls_back_to_numpy(self):
        assert arrays.float_dtype_like([1.0, 2.0]) == np.float64
        assert arrays.float_dtype_like([1, 2]) == np.float32

    @pytest.mark.parametrize(
        'dtype,min_dtype,expected',
        [
            (np.float16, 'float32', np.float32),
            (np.float64, 'float32', np.float64),
            (np.complex64, 'float64', np.float64),
        ],
    )
    def test_min_dtype_is_a_floor(self, dtype, min_dtype, expected):
        x = np.zeros(3, dtype=dtype)
        assert arrays.float_dtype_like(x, min_dtype=min_dtype) == expected


class TestBinnedMean:
    @PROPERTY
    @given(data=st.data())
    def test_left_aligned_bins_match_reshape_mean(self, data):
        x, axis = data.draw(shaped_arrays(dtype=np.float64))
        n = x.shape[axis]
        count = data.draw(st.integers(min_value=1, max_value=n))

        result = arrays.binned_mean(x, count, axis=axis, fft=False)
        ref = reference_binned_mean(x, count, axis)

        assert result.shape == ref.shape
        assert_allclose(result, ref, rtol=0, atol=mean_atol(x, count))

    @PROPERTY
    @given(data=st.data())
    def test_truncate_false_accepts_only_whole_bins(self, data):
        x, axis = data.draw(shaped_arrays(dtype=np.float64, min_side=2))
        n = x.shape[axis]
        count = data.draw(st.integers(min_value=1, max_value=n))

        if n % count == 0:
            result = arrays.binned_mean(x, count, axis=axis, fft=False, truncate=False)
            ref = reference_binned_mean(x, count, axis)
            assert_allclose(result, ref, rtol=0, atol=mean_atol(x, count))
        else:
            with pytest.raises(ValueError):
                arrays.binned_mean(x, count, axis=axis, fft=False, truncate=False)

    @PROPERTY
    @given(data=st.data())
    def test_fft_bins_center_on_the_middle_sample(self, data):
        """Property: with fft=True the sample at index n//2 is the center of the
        middle bin, the bin count is odd, and no further symmetric pair fits."""
        count = data.draw(st.integers(min_value=2, max_value=8))
        if count % 2:
            # even n with odd count is a known defect, tested separately
            n = data.draw(st.integers(min_value=count // 2, max_value=32)) * 2 + 1
        else:
            n = data.draw(st.integers(min_value=count, max_value=64))
        x = data.draw(float_arrays((n,)))

        result = arrays.binned_mean(x, count, fft=True)

        nblocks = result.shape[0]
        assert nblocks % 2 == 1
        assert (nblocks + 2) * count > n
        start = n // 2 - count // 2 - (nblocks // 2) * count
        ref = x[start : start + nblocks * count].reshape(nblocks, count).mean(axis=1)
        assert_allclose(result, ref, rtol=0, atol=mean_atol(x, count))

    @pytest.mark.xfail(
        strict=True,
        raises=ValueError,
        reason=(
            'binned_mean(fft=True) computes a stop index of n+1 when '
            '(n//2 - count//2) % count == 0 for odd count and even n, skips the '
            'slice, and axis_to_blocks then rejects the unaligned length'
        ),
    )
    @pytest.mark.parametrize('n,count', [(8, 3), (14, 3), (1024, 5)])
    def test_fft_bins_odd_count_even_length(self, n, count):
        result = arrays.binned_mean(np.arange(n, dtype=np.float64), count, fft=True)
        assert result.shape[0] % 2 == 1

    @PROPERTY
    @given(
        count=st.integers(min_value=3, max_value=6),
        nbins=st.integers(min_value=1, max_value=8),
        data=st.data(),
    )
    def test_reject_extrema_averages_the_sorted_interior(self, count, nbins, data):
        x = data.draw(float_arrays((nbins * count,)))

        result = arrays.binned_mean(x, count, fft=False, reject_extrema=True)

        interior = np.sort(x.reshape(nbins, count), axis=1)[:, 1:-1]
        assert_allclose(result, interior.mean(axis=1), rtol=0, atol=mean_atol(x, count))

    def test_nan_samples_are_ignored(self):
        x = np.arange(8, dtype=np.float64)
        x[0] = np.nan
        result = arrays.binned_mean(x, 4, fft=False)
        assert_array_equal(result, [2.0, 5.5])

    @PROPERTY
    @given(data=st.data())
    def test_numpy_vs_cupy(self, cupy_available, data):
        cp = cupy_available
        x, axis = data.draw(shaped_arrays(dtype=np.float32, min_side=2))
        count = data.draw(st.integers(min_value=2, max_value=x.shape[axis]))
        if count % 2:
            count += 1
        assume(count <= x.shape[axis])

        for fft in (False, True):
            result_np = arrays.binned_mean(x, count, axis=axis, fft=fft)
            result_cp = arrays.binned_mean(cp.asarray(x), count, axis=axis, fft=fft)
            assert_allclose(
                to_numpy(result_cp), result_np, rtol=0, atol=mean_atol(x, count)
            )

    def test_reject_extrema_cupy(self, cupy_available):
        cp = cupy_available
        x = np.arange(12, dtype=np.float32)
        result = arrays.binned_mean(cp.asarray(x), 3, fft=False, reject_extrema=True)
        assert_array_equal(to_numpy(result), [1.0, 4.0, 7.0, 10.0])


class TestSlidingWindowView:
    @PROPERTY
    @given(data=st.data())
    def test_matches_numpy_for_integer_axis(self, data):
        x, axis = data.draw(shaped_arrays(max_side=6))
        window = data.draw(st.integers(min_value=1, max_value=x.shape[axis]))

        result = arrays.sliding_window_view(x, window, axis=axis)
        expected = np.lib.stride_tricks.sliding_window_view(x, window, axis=axis)

        assert result.shape == expected.shape
        assert_array_equal(result, expected)

    @PROPERTY
    @given(data=st.data())
    def test_matches_numpy_for_tuple_axis(self, data):
        x, _ = data.draw(shaped_arrays(min_dims=2, max_dims=2, max_side=6))
        w0 = data.draw(st.integers(min_value=1, max_value=x.shape[0]))
        w1 = data.draw(st.integers(min_value=1, max_value=x.shape[1]))
        axis = data.draw(st.sampled_from([(0, 1), (1, 0), (-1, -2)]))
        window = (w0, w1) if axis[0] in (0, -2) else (w1, w0)

        result = arrays.sliding_window_view(x, window, axis=axis)
        expected = np.lib.stride_tricks.sliding_window_view(x, window, axis=axis)

        assert_array_equal(result, expected)

    @pytest.mark.xfail(
        strict=True,
        raises=TypeError,
        reason=(
            'sliding_window_view normalizes axis=None only inside '
            '_sliding_window_output_shape and then passes None to '
            'normalize_axis_tuple'
        ),
    )
    def test_default_axis_windows_every_dimension(self):
        x = np.arange(12).reshape(3, 4)
        result = arrays.sliding_window_view(x, (2, 2))
        expected = np.lib.stride_tricks.sliding_window_view(x, (2, 2))
        assert_array_equal(result, expected)

    def test_output_shape_for_default_axis(self):
        assert arrays._sliding_window_output_shape((3, 4), (2, 2), None) == (
            2,
            3,
            2,
            2,
        )

    def test_output_shape_accepts_integer_shapes(self):
        assert arrays._sliding_window_output_shape(6, 3, 0) == (4, 3)

    @pytest.mark.parametrize(
        'array_shape,window_shape,axis,match',
        [
            ((3, 4), (-1,), 0, 'negative'),
            ((3, 4), (5,), 0, 'larger than input'),
            ((3, 4), (2,), None, 'all dimensions'),
            ((3, 4), (2, 2), 0, 'matching length'),
        ],
    )
    def test_invalid_shapes_raise(self, array_shape, window_shape, axis, match):
        with pytest.raises(ValueError, match=match):
            arrays._sliding_window_output_shape(array_shape, window_shape, axis)

    def test_writeable_is_not_supported(self):
        with pytest.raises(NotImplementedError):
            arrays.sliding_window_view(np.arange(4), 2, axis=0, writeable=True)


class TestAxisToBlocks:
    @PROPERTY
    @given(data=st.data())
    def test_blocks_reshape_back_to_the_truncated_input(self, data):
        x, axis = data.draw(shaped_arrays())
        n = x.shape[axis]
        size = data.draw(st.integers(min_value=1, max_value=n))
        pos = axis % x.ndim

        if n % size:
            with pytest.raises(ValueError):
                arrays.axis_to_blocks(x, size, axis=axis)
        result = arrays.axis_to_blocks(x, size, axis=axis, truncate=True)

        m = n // size
        assert result.shape == x.shape[:pos] + (m, size) + x.shape[pos + 1 :]
        truncated = np.take(x, np.arange(m * size), axis=axis)
        assert_array_equal(result.reshape(truncated.shape), truncated)

    @pytest.mark.parametrize('size', [2.0, np.int64(2), '2'])
    def test_non_int_size_raises(self, size):
        with pytest.raises(TypeError):
            arrays.axis_to_blocks(np.zeros(4), size)

    def test_empty_input_raises(self):
        with pytest.raises(IndexError):
            arrays.axis_to_blocks(np.zeros((0,)), 2)


class TestHistogramLastAxis:
    @PROPERTY
    @given(
        shape=array_shapes(min_dims=1, max_dims=3, min_side=1, max_side=4),
        nbins=st.integers(min_value=1, max_value=8),
        lo=st.floats(min_value=-12, max_value=-1),
        hi=st.floats(min_value=1, max_value=12),
        data=st.data(),
    )
    def test_integer_bins_with_range_match_numpy(self, shape, nbins, lo, hi, data):
        x = data.draw(float_arrays(shape + (16,)))
        assume(not np.any(x == hi))

        counts, edges = arrays.histogram_last_axis(x, nbins, range=(lo, hi))

        assert counts.shape == shape + (nbins,)
        assert_array_equal(edges, np.linspace(lo, hi, nbins + 1))
        for row, row_counts in zip(x.reshape(-1, 16), counts.reshape(-1, nbins)):
            expected, _ = np.histogram(row, bins=nbins, range=(lo, hi))
            assert_array_equal(row_counts, expected)

    @PROPERTY
    @given(
        nrows=st.integers(min_value=1, max_value=4),
        edges=st.lists(
            st.integers(min_value=-10, max_value=10),
            min_size=2,
            max_size=8,
            unique=True,
        ).map(sorted),
        data=st.data(),
    )
    def test_explicit_edges_match_numpy(self, nrows, edges, data):
        x = data.draw(float_arrays((2, nrows, 16)))
        edges = np.asarray(edges, dtype=np.float64)
        assume(not np.any(x == edges[-1]))

        counts, returned_edges = arrays.histogram_last_axis(x, edges)

        assert counts.shape == (2, nrows, edges.size - 1)
        assert_array_equal(returned_edges, edges)
        for row, row_counts in zip(
            x.reshape(-1, 16), counts.reshape(-1, edges.size - 1)
        ):
            expected, _ = np.histogram(row, bins=edges)
            assert_array_equal(row_counts, expected)

    def test_values_outside_the_range_are_dropped(self):
        x = np.array([[-5.0, 0.5, 1.5, 7.0]])
        counts, _ = arrays.histogram_last_axis(x, 2, range=(0.0, 2.0))
        assert_array_equal(counts, [[1, 1]])

    def test_default_range_spans_the_data(self):
        x = np.array([[0.0, 1.0, 2.0, 3.0]])
        _, edges = arrays.histogram_last_axis(x, 3)
        assert_array_equal(edges, [0.0, 1.0, 2.0, 3.0])

    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason=(
            'the docstring declares the upper bound inclusive, but the maximum '
            'sample lands in the extra bin that is sliced off'
        ),
    )
    def test_default_range_counts_the_maximum(self):
        x = np.array([[0.0, 1.0, 2.0, 3.0]])
        counts, _ = arrays.histogram_last_axis(x, 3)
        assert_array_equal(counts, [[1, 1, 2]])


class TestAxisSlicing:
    @PROPERTY
    @given(data=st.data())
    def test_axis_index_matches_take(self, data):
        a, axis = data.draw(shaped_arrays(max_dims=4, max_side=5))
        n = a.shape[axis]
        index = np.asarray(
            data.draw(st.lists(st.integers(min_value=0, max_value=n - 1), max_size=6)),
            dtype=np.intp,
        )
        assert_array_equal(
            arrays.axis_index(a, index, axis=axis), np.take(a, index, axis)
        )

    @PROPERTY
    @given(data=st.data())
    def test_axis_index_with_mask_matches_compress(self, data):
        a, axis = data.draw(shaped_arrays(max_dims=4, max_side=5))
        mask = np.asarray(
            data.draw(
                st.lists(st.booleans(), min_size=a.shape[axis], max_size=a.shape[axis])
            )
        )
        assert_array_equal(
            arrays.axis_index(a, mask, axis=axis), np.compress(mask, a, axis=axis)
        )

    @PROPERTY
    @given(data=st.data())
    def test_axis_slice_matches_direct_slicing(self, data):
        a, axis = data.draw(shaped_arrays(max_dims=4, max_side=5))
        n = a.shape[axis]
        bound = st.one_of(st.none(), st.integers(min_value=-n, max_value=n))
        start, stop = data.draw(bound), data.draw(bound)
        step = data.draw(st.one_of(st.none(), st.integers(min_value=1, max_value=3)))

        index = [slice(None)] * a.ndim
        index[axis] = slice(start, stop, step)
        assert_array_equal(
            arrays.axis_slice(a, start, stop, step, axis=axis), a[tuple(index)]
        )

    @pytest.mark.parametrize('ndim', [1, 2, 3, 4])
    def test_pad_slices_select_the_requested_axis(self, ndim):
        a = np.zeros((2,) * ndim)
        for axis in range(-ndim, ndim):
            before, after = arrays._pad_slices_to_dim(ndim, axis)
            expected_shape = list(a.shape)
            del expected_shape[axis]
            assert a[before + (0,) + after].shape == tuple(expected_shape)

    def test_pad_slices_reject_bad_axes(self):
        with pytest.raises(TypeError):
            arrays._pad_slices_to_dim(2, 0.5)
        with pytest.raises(ValueError):
            arrays._pad_slices_to_dim(2, -3)


class TestGroupedViews:
    @PROPERTY
    @given(
        shape=array_shapes(min_dims=1, max_dims=3, min_side=1, max_side=8),
        max_size=st.integers(min_value=1, max_value=64),
        data=st.data(),
    )
    def test_views_partition_the_array(self, shape, max_size, data):
        axis = data.draw(st.integers(min_value=-len(shape), max_value=len(shape) - 1))
        x = np.arange(int(np.prod(shape))).reshape(shape)

        views = list(arrays.grouped_views_along_axis(x, max_size, axis=axis))

        if x.size < max_size:
            assert len(views) == 1 and views[0] is x
            return

        coverage = np.zeros(shape, dtype=int)
        slices = itertools.product(
            *arrays.grouped_slices_along_axis(shape, max_size, axis)
        )
        for view, slice_ in zip(views, slices):
            assert_array_equal(view, x[slice_])
            assert view.size <= max(max_size, shape[axis])
            coverage[slice_] += 1
        assert_array_equal(coverage, 1)

    def test_axis_dimension_is_never_split(self):
        for slices in arrays.grouped_slices_along_axis((4, 100), 10, axis=1)[1:]:
            assert slices == (slice(None, None),)
        assert arrays.grouped_slices_along_axis((4, 100), 10, axis=-1) == (
            arrays.grouped_slices_along_axis((4, 100), 10, axis=1)
        )


class TestPadAlongAxis:
    @PROPERTY
    @given(
        shape=array_shapes(min_dims=1, max_dims=3, min_side=1, max_side=4),
        before=st.integers(min_value=0, max_value=3),
        after=st.integers(min_value=0, max_value=3),
    )
    def test_pads_the_last_axis(self, shape, before, after):
        a = np.arange(int(np.prod(shape)), dtype=np.float64).reshape(shape)
        pad_width = [(0, 0)] * (len(shape) - 1) + [(before, after)]
        expected = np.pad(a, pad_width)

        result = arrays.pad_along_axis(a, [[before, after]], axis=len(shape) - 1)
        assert_array_equal(result, expected)
        if len(shape) == 1:
            assert_array_equal(
                arrays.pad_along_axis(a, [[before, after]], axis=-1), expected
            )

    @pytest.mark.parametrize(
        'shape,axis', [((3, 5), 0), ((3, 5), -1), ((2, 3, 5), 1), ((2, 3, 5), -2)]
    )
    def test_pads_only_the_requested_axis(self, shape, axis):
        a = np.zeros(shape)
        expected_shape = list(shape)
        expected_shape[axis] += 3
        result = arrays.pad_along_axis(a, [[1, 2]], axis=axis)
        assert result.shape == tuple(expected_shape)


@arrays.convert_np_to_xp
def _arange_np(n, xp=None):
    return np.arange(n)


class _AsarrayNamespace:
    def asarray(self, x):
        return ('asarray', x)


class _ArrayOnlyNamespace:
    def array(self, x):
        return ('array', x)


class TestConvertNpToXp:
    def test_numpy_namespaces_pass_through(self):
        import array_api_compat.numpy as anp

        for xp in (None, np, anp):
            result = _arange_np(3, xp=xp)
            assert type(result) is np.ndarray
            assert_array_equal(result, [0, 1, 2])
        assert_array_equal(_arange_np(3), [0, 1, 2])

    def test_other_namespace_converts_with_asarray(self):
        tag, converted = _arange_np(3, xp=_AsarrayNamespace())
        assert tag == 'asarray'
        assert_array_equal(converted[1], [0, 1, 2])

    @pytest.mark.xfail(
        strict=True,
        raises=AttributeError,
        reason=(
            'convert_np_to_xp falls back to xp.array but then unconditionally '
            'calls xp.asarray on the result'
        ),
    )
    def test_other_namespace_converts_with_array(self):
        tag, converted = _arange_np(3, xp=_ArrayOnlyNamespace())
        assert tag == 'array'
        assert_array_equal(converted, [0, 1, 2])

    def test_namespace_without_constructors_raises(self):
        with pytest.raises(AttributeError, match='invalid array module'):
            _arange_np(3, xp=object())

    def test_cupy_namespace_returns_cupy(self, cupy_available):
        cp = cupy_available
        result = _arange_np(3, xp=cp)
        assert isinstance(result, cp.ndarray)
        assert_array_equal(result.get(), [0, 1, 2])


class TestCupyHelpers:
    def test_non_stream_context_is_a_no_op(self):
        with arrays.NonStreamContext(1, a=2) as ctx:
            assert ctx.synchronize() is None
            assert ctx.use() is None

    def test_numpy_arrays_get_a_non_stream_context(self):
        x = np.zeros(4)
        assert isinstance(arrays.array_stream(x), arrays.NonStreamContext)
        assert arrays.sync_if_cupy(x) is None
        assert not arrays.is_cupy_array(x)

    def test_array_namespace(self):
        import array_api_compat.numpy as anp

        x = np.zeros(4)
        assert arrays.array_namespace(x) is np
        assert arrays.array_namespace(x, use_compat=True) is anp

    def test_memory_helpers_run_without_a_device(self):
        assert arrays.free_cupy_mempool() is None
        assert arrays.configure_cupy() is None
        assert arrays.set_cuda_mem_limit(fraction=1.0) is None

    @pytest.mark.skipif(_cupy is not None, reason='cupy is installed')
    def test_pinned_copy_requires_cupy(self):
        with pytest.raises(AssertionError):
            arrays.pinned_array_as_cupy(np.zeros(4, dtype=np.float32))

    def test_cupy_arrays_get_a_stream(self, cupy_available):
        cp = cupy_available
        x = cp.zeros(4)
        stream = arrays.array_stream(x)
        assert isinstance(stream, cp.cuda.Stream)
        with stream:
            stream.synchronize()
        assert arrays.sync_if_cupy(x) is None
        assert arrays.is_cupy_array(x)
        assert arrays.array_namespace(x) is cp

    def test_pinned_copy_round_trips(self, cupy_available):
        import cupyx

        cp = cupy_available
        host = cupyx.zeros_pinned(16, dtype=np.float32)
        host[:] = np.arange(16, dtype=np.float32)

        device = arrays.pinned_array_as_cupy(host)
        cp.cuda.get_current_stream().synchronize()

        assert isinstance(device, cp.ndarray)
        assert_array_equal(device.get(), host)
