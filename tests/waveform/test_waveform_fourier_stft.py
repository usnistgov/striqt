"""striqt.waveform.lib.fourier.stft: the STFT and its inverse against linearity, shape
and round-trip identities, the window stacking helpers, and the frequency-domain edits
of an STFT, on numpy and cupy.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import gaussian_iq, iq_waveforms
from hypothesis import given, settings
from hypothesis import strategies as st
from numeric_checks import RTOL_FLOAT64, assert_close, interior, numpy_and_cupy
from numpy.testing import assert_allclose, assert_array_equal
from waveform_strategies import cross_backend_sigma, noise_waveforms, stft_parameters

from striqt.waveform.lib.arrays import accum_rtol
from striqt.waveform.lib.fourier.fft import fft_tolerance_rms, fftfreq
from striqt.waveform.lib.fourier.stft import (
    _find_downsample_copy_range,
    _get_stft_axes,
    _same_base_memory,
    _stack_stft_windows,
    _unstack_stft_windows,
    downsample_stft,
    istft,
    stft,
    zero_stft_by_freq,
)


# %% stft
class TestStft:
    @settings(max_examples=30)
    @given(scale=st.floats(min_value=0.5, max_value=2.0, allow_nan=False))
    def test_linearity(self, scale):
        x = gaussian_iq(512, dtype=np.complex128, seed=42)
        kws = {
            'fs': 1e6,
            'window': 'hamming',
            'nperseg': 64,
            'noverlap': 0,
            'truncate': True,
        }
        _, _, X1 = stft(x, **kws)
        _, _, X2 = stft((scale * x).astype(np.complex128), **kws)
        assert_allclose(X2, scale * X1, rtol=1e-6)

    @settings(max_examples=50)
    @given(x=noise_waveforms(min_size=256, max_size=512), params=stft_parameters())
    def test_output_shape(self, x, params):
        """the window count follows the segment step, the bin count is nperseg, the
        dtype is the input's, and the frequency axis is fftfreq(nperseg, fs)
        (min_size=256 guarantees len(x) >= max(nperseg))"""
        freqs, times, X = stft(x, truncate=True, **params)

        step = params['nperseg'] - params['noverlap']
        assert X.shape[0] == (len(x) - params['noverlap']) // step
        assert X.shape[1] == params['nperseg']
        assert len(freqs) == params['nperseg']
        assert len(times) == X.shape[0]
        assert X.dtype == x.dtype
        assert_array_equal(freqs, fftfreq(params['nperseg'], params['fs']))

    @pytest.mark.parametrize(
        'size, noverlap, nseg',
        [(128, 0, 2), (100, 0, None), (96, 32, 2), (100, 32, None)],
    )
    def test_truncate_false_requires_whole_segments(self, size, noverlap, nseg):
        x = np.ones(size, dtype=np.complex64)
        kws = {
            'fs': 1e6,
            'window': 'hamming',
            'nperseg': 64,
            'noverlap': noverlap,
            'truncate': False,
        }
        if nseg is None:
            with pytest.raises(ValueError, match=f'size {size}'):
                stft(x, **kws)
        else:
            _, _, X = stft(x, **kws)
            assert X.shape == (nseg, 64)

    def test_array_window_matches_named_window(self):
        from scipy import signal

        x = gaussian_iq(512)
        w = signal.windows.get_window('hann', 64, fftbins=True).astype(np.complex64)

        _, _, X_named = stft(x, fs=1.0, window='hann', nperseg=64)
        _, _, X_array = stft(x, fs=1.0, window=w, nperseg=64)

        # the 2 paths differ only in the order of the window and 1/nfft multiplies
        sigma = fft_tolerance_rms(np.complex64, [64], n_elementwise=2)
        assert_close(X_array, X_named, sigma=sigma)

    def test_variants(self):
        x = np.ones(512, dtype=np.complex64)

        X = stft(x, fs=1.0, window=None, nperseg=64, return_axis_arrays=False)
        _, _, X_rect = stft(x, fs=1.0, window='rect', nperseg=64)
        assert_array_equal(X, X_rect)

        with pytest.raises(TypeError, match='norm'):
            stft(x, fs=1.0, window='hann', nperseg=64, norm='bogus')

    def test_no_overlap_writes_in_place(self):
        x = gaussian_iq(512)
        kws = {'fs': 1.0, 'window': 'hann', 'nperseg': 64}
        _, _, X_ref = stft(x, **kws)

        y = x.copy()
        X = stft(y, overwrite_x=True, return_axis_arrays=False, **kws)
        assert np.shares_memory(X, y)
        assert_array_equal(X, X_ref)

        out = np.empty((8, 64), dtype=np.complex64)
        X = stft(x, out=out, return_axis_arrays=False, **kws)
        assert np.shares_memory(X, out)
        assert_array_equal(X, X_ref)

    def test_helpers(self):
        x = np.arange(8)
        assert not _same_base_memory(x, None)
        assert _same_base_memory(x, x)
        assert _same_base_memory(x, x[2:])
        assert not _same_base_memory(x, x.copy())

        freqs, times = _get_stft_axes(1e6, 64, 10, overlap_frac=0.5)
        assert_array_equal(freqs, fftfreq(64, 1e6))
        assert_allclose(times, np.arange(10) * 32 / 1e6)


# %% istft
class TestIstft:
    NFFT = 64

    def _roundtrip_sigma(self, dtype):
        # window multiply, fft, ifft, fftshift multiply and the overlap-add
        return fft_tolerance_rms(dtype, [self.NFFT, self.NFFT], n_elementwise=3)

    @settings(max_examples=30)
    @given(
        x=iq_waveforms(min_size=8 * NFFT, max_size=16 * NFFT, multiple_of=NFFT),
        window=st.sampled_from(['hann', 'hamming']),
        overwrite_x=st.booleans(),
    )
    def test_cola_roundtrip_is_identity(self, x, window, overwrite_x):
        nfft = self.NFFT
        _, _, X = stft(x, fs=1.0, window=window, nperseg=nfft, noverlap=nfft // 2)
        y = istft(X, nfft=nfft, noverlap=nfft // 2, overwrite_x=overwrite_x)

        assert y.dtype == x.dtype
        assert y.shape == x.shape
        sigma = self._roundtrip_sigma(x.dtype)
        assert_close(interior(y, nfft), interior(x, nfft), sigma=sigma)

    @pytest.mark.xfail(
        strict=True,
        reason='the no-overlap path of stft scales the window by 1/nfft '
        '(fourier/stft.py, stft) and istft never undoes it, unlike the overlapped '
        'path whose _stack_stft_windows normalization cancels the factor, so '
        'istft(stft(x)) with noverlap=0 returns x/nfft',
    )
    @settings(max_examples=30)
    @given(x=iq_waveforms(min_size=8 * NFFT, max_size=16 * NFFT, multiple_of=NFFT))
    def test_no_overlap_roundtrip_is_identity(self, x):
        """istft(stft(x)) reproduces x with a rectangular window and no overlap."""
        nfft = self.NFFT
        _, _, X = stft(x, fs=1.0, window='rect', nperseg=nfft, noverlap=0)
        y = istft(X, nfft=nfft, noverlap=0)

        assert y.dtype == x.dtype
        assert y.shape == x.shape
        assert_close(y, x, sigma=self._roundtrip_sigma(x.dtype))

    @given(size=st.integers(min_value=NFFT, max_value=8 * NFFT))
    def test_size_trims_output(self, size):
        nfft = self.NFFT
        x = np.ones(8 * nfft, dtype=np.complex64)
        _, _, X = stft(x, fs=1.0, window='hann', nperseg=nfft, noverlap=nfft // 2)
        y = istft(X, size, nfft=nfft, noverlap=nfft // 2)
        assert y.shape == (size,)

    def test_out_buffer_is_used(self):
        nfft = self.NFFT
        x = np.ones(8 * nfft, dtype=np.complex64)
        _, _, X = stft(x, fs=1.0, window='hann', nperseg=nfft, noverlap=nfft // 2)
        out = np.empty(X.size, dtype=X.dtype)
        y = istft(X, nfft=nfft, noverlap=nfft // 2, out=out)
        assert np.shares_memory(y, out)

    @pytest.mark.parametrize('hop_divisor', [2, 4])
    def test_stack_unstack_roundtrip(self, hop_divisor):
        nfft = self.NFFT
        noverlap = nfft - nfft // hop_divisor
        x = gaussian_iq(8 * nfft)
        window = np.ones(nfft, dtype=np.complex64)

        stacked = _stack_stft_windows(x, window, nfft, noverlap, norm=None)
        assert stacked.shape == ((x.size - nfft) // (nfft - noverlap) + 1, nfft)
        y = _unstack_stft_windows(stacked, noverlap=noverlap, nperseg=nfft)

        # only the overlap-add sums round
        sigma = accum_rtol(np.float32, hop_divisor)
        assert_close(interior(y, nfft), interior(x, nfft), sigma=sigma)

    def test_stack_windows_rejects_unknown_norm(self):
        x = np.ones(256, dtype=np.complex64)
        with pytest.raises(ValueError):
            _stack_stft_windows(x, np.ones(64), 64, 32, norm='bogus')


# %% zero_stft_by_freq and downsample_stft
class TestStftFrequencyEditing:
    NFFT = 64
    NSEG = 8

    def _stft(self, fs, dtype=np.complex64):
        x = gaussian_iq(self.NFFT * self.NSEG, dtype)
        freqs, _, Y = stft(x, fs=fs, window='rect', nperseg=self.NFFT, noverlap=0)
        return freqs, Y

    @pytest.mark.parametrize(
        'edge_shift_bins',
        [0, 0.5],
        ids=['on_grid', 'off_grid'],
    )
    def test_zero_stft_by_freq_zeroes_outside_passband(self, edge_shift_bins):
        fs = 1e6
        freqs, Y = self._stft(fs)
        hi = fs / 4 + edge_shift_bins * fs / self.NFFT
        # the passband is half-open, so a bin centered on the upper cutoff is zeroed
        inside = (freqs >= -hi) & (freqs < hi)

        Yz = zero_stft_by_freq(freqs, Y.copy(), passband=(-hi, hi))
        assert_array_equal(Yz[:, ~inside], 0)
        assert_array_equal(Yz[:, inside], Y[:, inside])

    def test_downsample_stft_keeps_center_band(self):
        freqs, Y = self._stft(1e6)
        nfft_out = self.NFFT // 2
        lo = self.NFFT // 2 - nfft_out // 2

        freqs_out, Yo = downsample_stft(freqs, Y, nfft_out)
        assert Yo.shape == (self.NSEG, nfft_out)
        assert freqs_out.shape == (nfft_out,)
        assert_array_equal(Yo, Y[:, lo : lo + nfft_out])
        assert not np.shares_memory(Yo, Y)
        assert_allclose(np.diff(freqs_out), freqs[1] - freqs[0], rtol=1e-12)

    def test_downsample_stft_fast_path_returns_view(self):
        freqs, Y = self._stft(1e6)
        _, Yo = downsample_stft(freqs, Y, self.NFFT // 2, out=Y)
        assert np.shares_memory(Yo, Y)

    def test_downsample_stft_zero_pads_when_upsampling(self):
        freqs, Y = self._stft(1e6)
        nfft_out = 2 * self.NFFT
        pad = (nfft_out - self.NFFT) // 2

        _, Yo = downsample_stft(freqs, Y, nfft_out)
        assert Yo.shape == (self.NSEG, nfft_out)
        assert_array_equal(Yo[:, :pad], 0)
        assert_array_equal(Yo[:, pad + self.NFFT :], 0)
        assert_array_equal(Yo[:, pad : pad + self.NFFT], Y)

    @pytest.mark.parametrize(
        'edge_shift_bins',
        [0, -0.5],
        ids=['on_grid', 'off_grid'],
    )
    def test_downsample_stft_passband_zeroing(self, edge_shift_bins):
        fs = 1e6
        freqs, Y = self._stft(fs)
        nfft_out = self.NFFT // 2
        # a passband half as wide as the output leaves a quarter zeroed on each side.
        # Shifting both half-open edges down by less than a bin selects the same bins.
        shift = edge_shift_bins * fs / self.NFFT
        passband = (-fs / 8 + shift, fs / 8 + shift)
        _, Yo = downsample_stft(freqs, Y, nfft_out, passband=passband)
        quarter = nfft_out // 4
        assert_array_equal(Yo[:, :quarter], 0)
        assert_array_equal(Yo[:, -quarter:], 0)
        assert np.all(Yo[:, quarter:-quarter] != 0)

    def test_downsample_stft_centers_an_odd_width_passband_on_dc(self):
        fs = 1e6
        freqs, Y = self._stft(fs)
        nfft_out = self.NFFT // 2
        half_width = nfft_out // 4
        step = fs / self.NFFT
        # 2 * half_width + 1 passband bins leave an odd number of zeroed bins
        passband = (-half_width * step, (half_width + 0.5) * step)
        _, Yo = downsample_stft(freqs, Y, nfft_out, passband=passband)

        band_in = slice(self.NFFT // 2 - half_width, self.NFFT // 2 + half_width + 1)
        band_out = slice(nfft_out // 2 - half_width, nfft_out // 2 + half_width + 1)
        assert_array_equal(Yo[:, band_out], Y[:, band_in])
        assert_array_equal(Yo[:, : band_out.start], 0)
        assert_array_equal(Yo[:, band_out.stop :], 0)

    def test_downsample_stft_out_buffer(self):
        freqs, Y = self._stft(1.0)
        nfft_out = self.NFFT // 2
        _, expected = downsample_stft(freqs, Y, nfft_out, passband=(-0.125, 0.125))
        _, Yo = downsample_stft(
            freqs, Y, nfft_out, passband=(-0.125, 0.125), out=np.empty_like(Y)
        )
        assert Yo.shape == (self.NSEG, nfft_out)
        assert_array_equal(Yo, expected)

    def test_downsample_stft_writes_into_out(self):
        freqs, Y = self._stft(1.0)
        out = np.empty_like(Y)
        _, Yo = downsample_stft(
            freqs, Y, self.NFFT // 2, passband=(-0.125, 0.125), out=out
        )
        assert np.shares_memory(Yo, out)

    @given(
        nfft_in=st.integers(min_value=2, max_value=256),
        nfft_out=st.integers(min_value=1, max_value=256),
        data=st.data(),
    )
    def test_find_downsample_copy_range_invariants(self, nfft_in, nfft_out, data):
        if data.draw(st.booleans()):
            start = end = None
        else:
            start = data.draw(st.integers(min_value=0, max_value=nfft_in - 1))
            end = data.draw(st.integers(min_value=start + 1, max_value=nfft_in))

        (out0, out1), (in0, in1), _ = _find_downsample_copy_range(
            nfft_in, nfft_out, start, end
        )
        assert out1 - out0 == in1 - in0
        assert 0 <= in0 <= in1 <= nfft_in
        assert 0 <= out0 <= out1 <= nfft_out
        assert in1 - in0 <= nfft_out
        # the copied block is centered in the output
        assert abs(out0 - (nfft_out - out1)) <= 1


# %% numpy vs cupy
class TestNumpyCupyCrossComparison:
    """Tolerances come from `cross_backend_sigma`, parameterized by the FFT sizes and
    elementwise operations each function performs. Elementwise checks use an absolute
    tolerance anchored on the output rms, because the outputs are noise-like and a
    relative tolerance is meaningless in their smallest bins.
    """

    NFFT = 64

    @given(x=noise_waveforms(min_size=256, max_size=512, dtype=np.complex64))
    def test_stft(self, cupy_available, x):
        kws = {
            'fs': 1e6,
            'window': 'hamming',
            'nperseg': 64,
            'noverlap': 32,
            'truncate': True,
        }
        (freqs_np, times_np, X_np), (freqs_cp, times_cp, X_cp) = numpy_and_cupy(
            cupy_available, stft, x, **kws
        )

        assert_close(freqs_cp, freqs_np, rtol=RTOL_FLOAT64, atol=0)
        assert_close(times_cp, times_np, rtol=RTOL_FLOAT64, atol=0)

        # window/nfft and the window multiply, then fft(nperseg)
        sigma = cross_backend_sigma(x.dtype, [64], n_elementwise=2)
        assert_close(X_cp, X_np, sigma=sigma)

    @given(x=iq_waveforms(min_size=8 * NFFT, max_size=16 * NFFT, multiple_of=NFFT))
    def test_istft_roundtrip(self, cupy_available, x):
        nfft = self.NFFT

        def roundtrip(x):
            _, _, X = stft(x, fs=1.0, window='hann', nperseg=nfft, noverlap=nfft // 2)
            return istft(X, nfft=nfft, noverlap=nfft // 2)

        y_np, y_cp = numpy_and_cupy(cupy_available, roundtrip, x)
        sigma = cross_backend_sigma(x.dtype, [nfft, nfft], n_elementwise=3)
        assert_close(y_cp, y_np, sigma=sigma)

    def test_stft_frequency_editing(self, cupy_available):
        x = gaussian_iq(8 * 64)
        freqs, _, Y = stft(x, fs=1.0, window='rect', nperseg=64, noverlap=0)

        def zeroed(freqs, Y):
            return zero_stft_by_freq(freqs, Y.copy(), passband=(-0.25, 0.25))

        Yz_np, Yz_cp = numpy_and_cupy(cupy_available, zeroed, freqs, Y)
        assert_array_equal(Yz_cp, Yz_np)

        (fo_np, Yo_np), (fo_cp, Yo_cp) = numpy_and_cupy(
            cupy_available, downsample_stft, freqs, Y, 32
        )
        assert_array_equal(Yo_cp, Yo_np)
        assert_close(fo_cp, fo_np, rtol=RTOL_FLOAT64)
