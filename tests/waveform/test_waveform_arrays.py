"""Property-based tests for striqt.waveform.lib.arrays using Hypothesis.

Covers rounding, dtype helpers, binned and sliding windows, blocking, histograms,
axis slicing and indexing, grouped views, padding, the numpy-to-xp conversion
decorator, and the cupy runtime helpers, each against a plain numpy reference.
Known defects are recorded as strict expected failures.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import _cupy, as_xp, float_arrays, shaped_arrays
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import array_shapes
from numeric_checks import assert_close, mean_atol, reference_binned_mean, to_numpy
from numpy.testing import assert_array_equal

from striqt.waveform.lib import arrays


class TestIsRoundMod:
    @given(
        k=st.integers(min_value=-1000, max_value=1000),
        div=st.floats(min_value=1e-3, max_value=1e3, allow_nan=False),
    )
    def test_multiples_are_round_and_half_multiples_are_not(self, k, div):
        assert arrays.isroundmod(k * div, div)
        assert not arrays.isroundmod((k + 0.5) * div, div)
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
    """each case runs on every array namespace against the numpy reference"""

    @given(case=shaped_arrays(dtype=np.float64), data=st.data())
    def test_left_aligned_bins_match_reshape_mean(self, xp, case, data):
        x, axis = case
        count = data.draw(st.integers(min_value=1, max_value=x.shape[axis]))

        result = arrays.binned_mean(as_xp(xp, x), count, axis=axis, fft=False)
        ref = reference_binned_mean(x, count, axis)

        assert result.shape == ref.shape
        assert_close(result, ref, atol=mean_atol(x, count))

    @given(case=shaped_arrays(dtype=np.float64, min_side=2), data=st.data())
    def test_truncate_false_rejects_partial_bins(self, case, data):
        x, axis = case
        n = x.shape[axis]
        # n + 1 never divides n, so the filter is satisfiable for every n
        counts = st.integers(min_value=2, max_value=n + 1).filter(lambda c: n % c)
        count = data.draw(counts)

        with pytest.raises(ValueError):
            arrays.binned_mean(x, count, axis=axis, fft=False, truncate=False)

    @given(data=st.data())
    def test_fft_bins_center_on_the_middle_sample(self, xp, data):
        """Property: with fft=True the sample at index n//2 is the center of the
        middle bin, the bin count is odd, and no further symmetric pair fits."""
        count = data.draw(st.integers(min_value=2, max_value=8))
        if count % 2:
            # even n with odd count is a known defect, tested separately
            n = data.draw(st.integers(min_value=count // 2, max_value=32)) * 2 + 1
        else:
            n = data.draw(st.integers(min_value=count, max_value=64))
        x = data.draw(float_arrays((n,)))

        result = to_numpy(arrays.binned_mean(as_xp(xp, x), count, fft=True))

        nblocks = result.shape[0]
        assert nblocks % 2 == 1
        assert (nblocks + 2) * count > n
        start = n // 2 - count // 2 - (nblocks // 2) * count
        ref = x[start : start + nblocks * count].reshape(nblocks, count).mean(axis=1)
        assert_close(result, ref, atol=mean_atol(x, count))

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

    @given(
        count=st.integers(min_value=3, max_value=6),
        nbins=st.integers(min_value=1, max_value=8),
        data=st.data(),
    )
    def test_reject_extrema_averages_the_sorted_interior(self, xp, count, nbins, data):
        x = data.draw(float_arrays((nbins * count,)))

        result = arrays.binned_mean(as_xp(xp, x), count, fft=False, reject_extrema=True)

        interior = np.sort(x.reshape(nbins, count), axis=1)[:, 1:-1]
        assert_close(result, interior.mean(axis=1), atol=mean_atol(x, count))

    def test_nan_samples_are_ignored(self):
        x = np.arange(8, dtype=np.float64)
        x[0] = np.nan
        result = arrays.binned_mean(x, 4, fft=False)
        assert_array_equal(result, [2.0, 5.5])


class TestSlidingWindowView:
    @given(case=shaped_arrays(max_side=6), data=st.data())
    def test_matches_numpy_for_integer_axis(self, case, data):
        x, axis = case
        window = data.draw(st.integers(min_value=1, max_value=x.shape[axis]))

        result = arrays.sliding_window_view(x, window, axis=axis)
        expected = np.lib.stride_tricks.sliding_window_view(x, window, axis=axis)

        assert result.shape == expected.shape
        assert_array_equal(result, expected)

    @pytest.mark.parametrize('axis', [(0, 1), (1, 0), (-1, -2)])
    @given(case=shaped_arrays(min_dims=2, max_dims=2, max_side=6), data=st.data())
    def test_matches_numpy_for_tuple_axis(self, axis, case, data):
        x, _ = case
        w0 = data.draw(st.integers(min_value=1, max_value=x.shape[0]))
        w1 = data.draw(st.integers(min_value=1, max_value=x.shape[1]))
        window = (w0, w1) if axis[0] in (0, -2) else (w1, w0)

        result = arrays.sliding_window_view(x, window, axis=axis)
        expected = np.lib.stride_tricks.sliding_window_view(x, window, axis=axis)

        assert_array_equal(result, expected)

    def test_default_axis_windows_every_dimension(self):
        x = np.arange(12).reshape(3, 4)
        result = arrays.sliding_window_view(x, (2, 2))
        expected = np.lib.stride_tricks.sliding_window_view(x, (2, 2))
        assert_array_equal(result, expected)

    def test_output_shape_for_default_axis(self):
        shape = arrays._sliding_window_output_shape((3, 4), (2, 2), None)
        assert shape == (2, 3, 2, 2)

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
    @given(case=shaped_arrays(), data=st.data())
    def test_blocks_reshape_back_to_the_truncated_input(self, case, data):
        x, axis = case
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

    @pytest.mark.parametrize(
        'shape, size, exc',
        [((4,), 2.0, TypeError), ((4,), '2', TypeError), ((0,), 2, IndexError)],
        ids=['float_size', 'str_size', 'empty_input'],
    )
    def test_argument_errors(self, shape, size, exc):
        with pytest.raises(exc):
            arrays.axis_to_blocks(np.zeros(shape), size)


class TestHistogramLastAxis:
    @given(
        shape=array_shapes(min_dims=1, max_dims=3, min_side=1, max_side=4),
        nbins=st.integers(min_value=1, max_value=8),
        lo=st.floats(min_value=-12, max_value=-1),
        hi=st.floats(min_value=1, max_value=12),
        data=st.data(),
    )
    def test_integer_bins_with_range_match_numpy(self, shape, nbins, lo, hi, data):
        x = data.draw(float_arrays(shape + (16,)))

        counts, edges = arrays.histogram_last_axis(x, nbins, range=(lo, hi))

        assert counts.shape == shape + (nbins,)
        assert_array_equal(edges, np.linspace(lo, hi, nbins + 1))
        for row, row_counts in zip(x.reshape(-1, 16), counts.reshape(-1, nbins)):
            expected, _ = np.histogram(row, bins=nbins, range=(lo, hi))
            assert_array_equal(row_counts, expected)

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

    @pytest.mark.parametrize('bins', [3, [0.0, 1.0, 2.0, 3.0]])
    def test_upper_edge_is_inclusive(self, bins):
        x = np.array([[0.0, 1.0, 2.0, 3.0]])
        counts, _ = arrays.histogram_last_axis(x, bins)
        assert_array_equal(counts, [[1, 1, 2]])


class TestAxisSlicing:
    @given(case=shaped_arrays(max_dims=4, max_side=5), data=st.data())
    def test_axis_index_matches_take(self, case, data):
        a, axis = case
        n = a.shape[axis]
        index = np.asarray(
            data.draw(st.lists(st.integers(min_value=0, max_value=n - 1), max_size=6)),
            dtype=np.intp,
        )
        assert_array_equal(
            arrays.axis_index(a, index, axis=axis), np.take(a, index, axis)
        )

    @given(case=shaped_arrays(max_dims=4, max_side=5), data=st.data())
    def test_axis_index_with_mask_matches_compress(self, case, data):
        a, axis = case
        n = a.shape[axis]
        mask = np.asarray(data.draw(st.lists(st.booleans(), min_size=n, max_size=n)))
        assert_array_equal(
            arrays.axis_index(a, mask, axis=axis), np.compress(mask, a, axis=axis)
        )

    @given(case=shaped_arrays(max_dims=4, max_side=5), data=st.data())
    def test_axis_slice_matches_direct_slicing(self, case, data):
        a, axis = case
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
    def test_pad_slices_select_the_requested_axis(self, subtests, ndim):
        a = np.zeros((2,) * ndim)
        for axis in range(-ndim, ndim):
            with subtests.test(msg=f'axis={axis}'):
                before, after = arrays._pad_slices_to_dim(ndim, axis)
                expected_shape = list(a.shape)
                del expected_shape[axis]
                assert a[before + (0,) + after].shape == tuple(expected_shape)

    @pytest.mark.parametrize(
        'axis, exc', [(0.5, TypeError), (-3, ValueError)], ids=['float', 'out_of_range']
    )
    def test_pad_slices_reject_bad_axes(self, axis, exc):
        with pytest.raises(exc):
            arrays._pad_slices_to_dim(2, axis)


class TestGroupedViews:
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

        coverage = np.zeros(x.size, dtype=int)
        for view in views:
            assert view.size <= max(max_size, shape[axis])
            coverage[view.ravel()] += 1
        assert_array_equal(coverage, 1)

    def test_axis_dimension_is_never_split(self):
        for slices in arrays.grouped_slices_along_axis((4, 100), 10, axis=1)[1:]:
            assert slices == (slice(None, None),)
        assert arrays.grouped_slices_along_axis((4, 100), 10, axis=-1) == (
            arrays.grouped_slices_along_axis((4, 100), 10, axis=1)
        )


class TestPadAlongAxis:
    @pytest.mark.parametrize('axis', [0, -1, 1, -2], ids=['0', '-1', '1', '-2'])
    @given(
        data=st.data(),
        before=st.integers(min_value=0, max_value=3),
        after=st.integers(min_value=0, max_value=3),
    )
    def test_pads_only_the_requested_axis(self, data, axis, before, after):
        min_dims = axis + 1 if axis >= 0 else -axis
        shapes = array_shapes(min_dims=min_dims, max_dims=3, min_side=1, max_side=4)
        shape = data.draw(shapes)
        a = np.arange(int(np.prod(shape)), dtype=np.float64).reshape(shape)
        pad_width = [(0, 0)] * len(shape)
        pad_width[axis] = (before, after)

        result = arrays.pad_along_axis(a, [[before, after]], axis=axis)
        assert_array_equal(result, np.pad(a, pad_width))


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
    def test_numpy_namespaces_pass_through(self, subtests):
        import array_api_compat.numpy as anp

        for xp in (None, np, anp):
            with subtests.test(msg=getattr(xp, '__name__', repr(xp))):
                result = _arange_np(3, xp=xp)
                assert type(result) is np.ndarray
                assert_array_equal(result, [0, 1, 2])
        assert_array_equal(_arange_np(3), [0, 1, 2])

    @pytest.mark.parametrize(
        'namespace,expected_tag',
        [(_AsarrayNamespace(), 'asarray'), (_ArrayOnlyNamespace(), 'array')],
        ids=['asarray', 'array'],
    )
    def test_other_namespace_converts_with_its_constructor(
        self, namespace, expected_tag
    ):
        tag, converted = _arange_np(3, xp=namespace)
        assert tag == expected_tag
        assert type(converted) is np.ndarray
        assert_array_equal(converted, [0, 1, 2])

    def test_namespace_without_constructors_raises(self):
        with pytest.raises(AttributeError, match='invalid array module'):
            _arange_np(3, xp=object())

    def test_cupy_namespace_returns_cupy(self, cupy_available):
        result = _arange_np(3, xp=cupy_available)
        assert isinstance(result, cupy_available.ndarray)
        assert_array_equal(to_numpy(result), [0, 1, 2])


class TestCupyHelpers:
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

    @pytest.mark.skipif(
        _cupy is not None,
        reason='with a device, set_cuda_mem_limit(1.0) caps the pool for the session',
    )
    def test_memory_helpers_run_without_a_device(self):
        assert arrays.free_cupy_mempool() is None
        assert arrays.configure_cupy() is None
        assert arrays.set_cuda_mem_limit(fraction=1.0) is None

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
