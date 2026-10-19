"""striqt.waveform.lib.fourier.windows: window length, normalization, half-band shift,
caching and the ENBW parameter search, on numpy and cupy.
"""

from __future__ import annotations

import functools

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numeric_checks import (
    ATOL,
    FLOAT_DTYPES,
    assert_close,
    dtype_id,
    elementwise_rtol,
    numpy_and_cupy,
)
from numpy.testing import assert_allclose, assert_array_equal

from striqt.waveform.lib.fourier.windows import (
    equivalent_noise_bandwidth,
    find_window_param_from_enbw,
    get_window,
)

WINDOW_NAMES = ['hann', 'hamming', 'blackman', 'blackmanharris', 'bartlett', 'nuttall']


# %% get_window
class TestGetWindow:
    @pytest.mark.parametrize('dtype', FLOAT_DTYPES, ids=dtype_id)
    @given(
        name=st.sampled_from(WINDOW_NAMES),
        nwindow=st.integers(min_value=2, max_value=256),
        nzero=st.integers(min_value=0, max_value=64),
    )
    def test_length_and_dtype(self, name, nwindow, nzero, dtype):
        w = get_window(name, nwindow, nzero=nzero, dtype=dtype)
        assert w.shape == (nwindow + nzero,)
        assert w.dtype == dtype

    def test_dtype_none_keeps_float64(self):
        assert get_window('hann', 16, dtype=None).dtype == np.float64

    @given(
        name=st.sampled_from(WINDOW_NAMES),
        nwindow=st.integers(min_value=8, max_value=256),
    )
    def test_normalization(self, name, nwindow):
        """norm=True gives unit mean square; norm=False peaks at 1 and is non-negative"""
        w = get_window(name, nwindow, norm=True)
        assert_allclose(np.mean(np.abs(w) ** 2), 1.0, rtol=1e-6)

        w = get_window(name, nwindow, fftshift=False, norm=False)
        # blackman's endpoints are zero up to roundoff
        assert np.all(w >= -1e-10)
        assert np.max(w) <= 1.0 + 1e-10

    @given(
        name=st.sampled_from(WINDOW_NAMES),
        nwindow=st.integers(min_value=8, max_value=128),
    )
    def test_fftshift_preserves_energy(self, name, nwindow):
        """an odd size shifts by half a sample, which makes the window complex"""
        w0 = get_window(name, nwindow, fftshift=False, norm=False)
        w = get_window(name, nwindow, fftshift=True, norm=False)
        assert np.iscomplexobj(w) == bool(nwindow % 2)
        assert_allclose(np.sum(np.abs(w) ** 2), np.sum(w0**2), rtol=1e-6)

    @pytest.mark.parametrize('name', ['hann', 'hamming'])
    @pytest.mark.parametrize('nwindow', [8, 16, 64])
    @pytest.mark.parametrize('nzero', [2, 4, 6])
    def test_center_zeros_pads_both_sides(self, name, nwindow, nzero):
        w = get_window(name, nwindow, nzero=nzero, center_zeros=True, norm=False)
        ws = get_window(name, nwindow, norm=False)
        assert w.size == nwindow + nzero
        assert_array_equal(w[: nzero // 2], 0)
        assert_array_equal(w[nzero // 2 + nwindow :], 0)
        assert_array_equal(w[nzero // 2 : nzero // 2 + nwindow], ws)

    def test_equivalent_argument_spellings_share_one_cache_entry(self):
        """the analysis warms the cache with a dtype string and no xp, while stft
        asks with a numpy dtype object and its namespace; both must hit one entry"""
        window, nwindow, nzero = ('kaiser', 11.884), 3840, 3328
        get_window.cache_clear()

        w1 = get_window(window, nwindow, nzero=nzero, dtype='complex64', fftshift=True)
        before = get_window.cache_info()
        w2 = get_window(
            window,
            nwindow,
            nzero,
            dtype=np.dtype('complex64'),
            xp=np,
            norm=True,
            fftshift=True,
        )
        after = get_window.cache_info()

        assert (after.hits, after.misses) == (before.hits + 1, before.misses)
        assert w2 is w1
        assert w1.shape == (nwindow + nzero,)


# %% equivalent_noise_bandwidth and find_window_param_from_enbw
class TestEquivalentNoiseBandwidth:
    def test_enbw_of_rect_and_hann(self):
        assert equivalent_noise_bandwidth('rect', 64) == pytest.approx(1.0, abs=0)
        # the periodic hann window has ENBW exactly 1.5 bins
        assert_allclose(equivalent_noise_bandwidth('hann', 64), 1.5, rtol=1e-6)
        uncached = equivalent_noise_bandwidth('hann', 64, cached=False)
        cached = equivalent_noise_bandwidth('hann', 64)
        assert_allclose(uncached, cached, rtol=0)

    @pytest.mark.parametrize('window', ['kaiser', 'dpss', 'chebwin'])
    def test_find_window_param_reproduces_enbw(self, window):
        target = 1.5
        param = find_window_param_from_enbw(window, target, nfft=1024)
        enbw = equivalent_noise_bandwidth((window, param), 1024)
        # bisect stops at |param error| <= atol=1e-6; the ENBW slope in the
        # parameter is of order 1, so rtol=1e-4 is loose by a factor of 100
        assert_allclose(enbw, target, rtol=1e-4)

    @pytest.mark.parametrize(
        'window, enbw, match',
        [('kaiser', 1.0, 'greater than 1'), ('hann', 1.5, 'window_name')],
        ids=['enbw_at_rect_limit', 'window_without_parameter'],
    )
    def test_find_window_param_errors(self, window, enbw, match):
        with pytest.raises(ValueError, match=match):
            find_window_param_from_enbw(window, enbw)


# %% numpy vs cupy
class TestNumpyCupyCrossComparison:
    @pytest.mark.parametrize(
        'name', ['hamming', 'hann', 'blackman', 'bartlett', 'flattop']
    )
    @pytest.mark.parametrize('nwindow', [8, 64, 129, 512])
    @pytest.mark.parametrize('dtype', FLOAT_DTYPES, ids=dtype_id)
    def test_get_window(self, cupy_available, name, nwindow, dtype):
        design = functools.partial(get_window, name, nwindow, dtype=dtype)
        w_np, w_cp = numpy_and_cupy(cupy_available, design, xp_kwarg='xp')
        assert_close(w_cp, w_np, rtol=elementwise_rtol(dtype), atol=ATOL)
