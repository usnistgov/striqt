"""Property-based tests for striqt.waveform.lib.fourier using Hypothesis.

Covers windowing, fftfreq, frequency slicing, oaconvolve, resample, stft/istft and
spectrogram, STFT frequency editing, the filter and resampler designs, and the
overlap-add filters: identities, linearity, dtype and shape properties,
numpy/cupy/dask compatibility, and roundoff bounds derived from an FFT error model
(numpy vs cupy, and the floor far from a tone).
"""

from __future__ import annotations

import functools

import numpy as np
import pytest
from conftest import gaussian_iq, iq_waveforms
from hypothesis import given, settings
from hypothesis import strategies as st
from numeric_checks import (
    ATOL,
    FFT_ROUNDOFF_SAFETY,
    FIR_LEAKAGE,
    FIR_RECT_STOPBAND_DB,
    FLOAT_DTYPES,
    RTOL_FLOAT64,
    assert_close,
    assert_tone_level,
    bin_centered_tone,
    cross_backend_rms,
    dtype_id,
    elementwise_rtol,
    far_bin_floor_dBc,
    interior,
    level_error_dB,
    level_tolerance_dB,
    numpy_and_cupy,
    peak_factor,
    rms,
    single_backend_rms,
    to_numpy,
    tone_bin,
    tone_frequency,
    tone_peak_roundoff,
    unit_roundoff,
    unit_tone,
)
from numpy.testing import assert_allclose, assert_array_equal

from striqt.waveform.lib import fourier

FAR_BIN_NFFTS = [64, 256, 1024, 4096]
WINDOW_NAMES = ['hann', 'hamming', 'blackman', 'blackmanharris', 'bartlett', 'nuttall']


def fft_sizes(min_size: int = 4, max_size: int = 512):
    """powers of two, or arbitrary even sizes (resample requires an even length)"""
    powers = st.integers(min_value=2, max_value=9).map(lambda p: 2**p)
    arbitrary = st.integers(min_value=min_size, max_value=max_size).filter(
        lambda n: n % 2 == 0
    )
    return st.one_of(powers, arbitrary)


def sample_rates(min_rate: float = 1e3, max_rate: float = 100e6):
    return st.floats(
        min_value=min_rate, max_value=max_rate, allow_nan=False, allow_infinity=False
    )


def noise_waveforms(min_size=64, max_size=2048, dtype=None):
    """even-length complex noise spanning the power range of the sensor's inputs"""
    return iq_waveforms(
        min_size, max_size, multiple_of=2, dtype=dtype, log_power=(-13, 3)
    )


@st.composite
def stft_parameters(draw):
    """Window names and sizes are drawn from short lists so that get_window's
    persistent on-disk cache does not fill with one-off entries."""
    nperseg = draw(st.sampled_from([64, 128, 256]))
    return {
        'nperseg': nperseg,
        # stft supports only noverlap == 0 or nperseg // 2
        'noverlap': draw(st.sampled_from([0, nperseg // 2])),
        'fs': draw(sample_rates(min_rate=1e3, max_rate=10e6)),
        'window': draw(st.sampled_from(['hamming', 'hann', 'blackman'])),
    }


@pytest.fixture
def max_cupy_fft_chunk():
    """a setter for the cupy FFT chunk size; the prior value is restored afterwards"""
    previous = fourier.get_max_cupy_fft_chunk()
    yield fourier.set_max_cupy_fft_chunk
    fourier.set_max_cupy_fft_chunk(previous)


def test_max_cupy_fft_chunk_roundtrip(max_cupy_fft_chunk):
    previous = fourier.get_max_cupy_fft_chunk()
    max_cupy_fft_chunk(4096)
    assert fourier.get_max_cupy_fft_chunk() == 4096
    max_cupy_fft_chunk(previous)
    assert fourier.get_max_cupy_fft_chunk() == previous


class TestGetWindow:
    @pytest.mark.parametrize('dtype', FLOAT_DTYPES, ids=dtype_id)
    @given(
        name=st.sampled_from(WINDOW_NAMES),
        nwindow=st.integers(min_value=2, max_value=256),
        nzero=st.integers(min_value=0, max_value=64),
    )
    def test_length_and_dtype(self, name, nwindow, nzero, dtype):
        w = fourier.get_window(name, nwindow, nzero=nzero, dtype=dtype)
        assert w.shape == (nwindow + nzero,)
        assert w.dtype == dtype

    def test_dtype_none_keeps_float64(self):
        assert fourier.get_window('hann', 16, dtype=None).dtype == np.float64

    @given(
        name=st.sampled_from(WINDOW_NAMES),
        nwindow=st.integers(min_value=8, max_value=256),
    )
    def test_normalization(self, name, nwindow):
        """norm=True gives unit mean square; norm=False peaks at 1 and is non-negative"""
        w = fourier.get_window(name, nwindow, norm=True)
        assert_allclose(np.mean(np.abs(w) ** 2), 1.0, rtol=1e-6)

        w = fourier.get_window(name, nwindow, fftshift=False, norm=False)
        # blackman's endpoints are zero up to roundoff
        assert np.all(w >= -1e-10)
        assert np.max(w) <= 1.0 + 1e-10

    @given(
        name=st.sampled_from(WINDOW_NAMES),
        nwindow=st.integers(min_value=8, max_value=128),
    )
    def test_fftshift_preserves_energy(self, name, nwindow):
        """an odd size shifts by half a sample, which makes the window complex"""
        w0 = fourier.get_window(name, nwindow, fftshift=False, norm=False)
        w = fourier.get_window(name, nwindow, fftshift=True, norm=False)
        assert np.iscomplexobj(w) == bool(nwindow % 2)
        assert_allclose(np.sum(np.abs(w) ** 2), np.sum(w0**2), rtol=1e-6)

    @pytest.mark.parametrize('name', ['hann', 'hamming'])
    @pytest.mark.parametrize('nwindow', [8, 16, 64])
    @pytest.mark.parametrize('nzero', [2, 4, 6])
    def test_center_zeros_pads_both_sides(self, name, nwindow, nzero):
        w = fourier.get_window(
            name, nwindow, nzero=nzero, center_zeros=True, norm=False
        )
        ws = fourier.get_window(name, nwindow, norm=False)
        assert w.size == nwindow + nzero
        assert_array_equal(w[: nzero // 2], 0)
        assert_array_equal(w[nzero // 2 + nwindow :], 0)
        assert_array_equal(w[nzero // 2 : nzero // 2 + nwindow], ws)

    def test_enbw_of_rect_and_hann(self):
        assert fourier.equivalent_noise_bandwidth('rect', 64) == pytest.approx(
            1.0, abs=0
        )
        # the periodic hann window has ENBW exactly 1.5 bins
        assert_allclose(fourier.equivalent_noise_bandwidth('hann', 64), 1.5, rtol=1e-6)
        assert_allclose(
            fourier.equivalent_noise_bandwidth('hann', 64, cached=False),
            fourier.equivalent_noise_bandwidth('hann', 64),
            rtol=0,
        )

    @pytest.mark.parametrize('window', ['kaiser', 'dpss', 'chebwin'])
    def test_find_window_param_reproduces_enbw(self, window):
        target = 1.5
        param = fourier.find_window_param_from_enbw(window, target, nfft=1024)
        enbw = fourier.equivalent_noise_bandwidth((window, param), 1024)
        # bisect stops at |param error| <= atol=1e-6; the ENBW slope in the
        # parameter is of order 1, so this is loose by a factor of 100
        assert_allclose(enbw, target, rtol=1e-4)

    @pytest.mark.parametrize(
        'window, enbw, match',
        [('kaiser', 1.0, 'greater than 1'), ('hann', 1.5, 'window_name')],
        ids=['enbw_at_rect_limit', 'window_without_parameter'],
    )
    def test_find_window_param_errors(self, window, enbw, match):
        with pytest.raises(ValueError, match=match):
            fourier.find_window_param_from_enbw(window, enbw)


class TestFftfreq:
    @given(nfft=st.integers(min_value=2, max_value=512), fs=sample_rates())
    def test_grid(self, nfft, fs):
        """nfft bins spaced fs/nfft apart and symmetric about 0; an even nfft
        includes -fs/2 but not +fs/2"""
        freqs = fourier.fftfreq(nfft, fs)
        assert freqs.shape == (nfft,)
        assert_allclose(np.diff(freqs), fs / nfft, rtol=1e-10)
        if nfft % 2:
            assert freqs[nfft // 2] == 0
            assert_allclose(freqs, -freqs[::-1], rtol=1e-10)
        else:
            assert_allclose(freqs[0], -fs / 2, rtol=1e-10)
            assert_allclose(
                freqs[1 : nfft // 2], -freqs[-1 : nfft // 2 : -1], rtol=1e-10
            )

        assert fourier.fftfreq(nfft, fs, dtype='float32').dtype == np.float32
        assert fourier.fftfreq(nfft, fs, dtype='float64').dtype == np.float64

    @given(nfft=st.sampled_from([8, 9, 64, 65, 256]), fs=sample_rates())
    def test_as_index_false_matches_exact(self, nfft, fs):
        exact = fourier.fftfreq(nfft, fs)
        approx = fourier.fftfreq(nfft, fs, as_index=False)
        # linspace endpoints carry a few ulps of fs/2
        eps = np.finfo(np.float64).eps
        assert_allclose(approx, exact, rtol=0, atol=8 * eps * fs)
        assert fourier.fftfreq(nfft, fs, dtype='float32', as_index=False).dtype == (
            np.float32
        )

    @pytest.mark.parametrize('as_index', [True, False])
    def test_non_numpy_namespace(self, as_index):
        da = pytest.importorskip('dask.array')
        freqs = fourier.fftfreq(16, 1e6, as_index=as_index, xp=da)
        assert isinstance(freqs, da.Array)
        assert_allclose(np.asarray(freqs), fourier.fftfreq(16, 1e6), rtol=1e-12)


class TestFrequencySlicing:
    @given(
        nfft=st.sampled_from([32, 64, 128]),
        bw_bins=st.integers(min_value=1, max_value=16),
        offset_bins=st.integers(min_value=-8, max_value=8),
    )
    def test_slice_freqs_selects_band(self, nfft, bw_bins, offset_bins):
        fs = 1.0
        fres = fs / nfft
        bandwidth = bw_bins * fres
        offset = offset_bins * fres

        s = fourier._slice_freqs(nfft, fs, bandwidth, offset=offset)
        bins = np.arange(nfft)[s]

        assert bins.size == bw_bins
        # an even bin count cannot be centered on a bin, so the mean may sit half
        # a bin off the offset
        assert abs(bins.mean() - (nfft // 2 + offset_bins)) <= 0.5

        x = np.arange(nfft, dtype=np.float32)[np.newaxis, np.newaxis, :]
        x = np.broadcast_to(x, (2, 3, nfft)).copy()
        trimmed = fourier.truncate_freqs(x, nfft, fs, bandwidth, offset=offset, axis=2)
        assert_array_equal(trimmed, x[:, :, s])

        fourier.null_lo(x, nfft, fs, bandwidth, offset=offset, axis=2)
        assert np.isnan(x[:, :, s]).all()
        assert np.isnan(x).sum() == 2 * 3 * bw_bins

    @pytest.mark.parametrize(
        'kws, match',
        [
            ({'bandwidth': -1.0}, 'negative bandwidth'),
            ({'bandwidth': 0.5, 'offset': 0.3}, 'not a multiple'),
            ({'bandwidth': 0.5, 'offset': 0.5}, r'> fs/2'),
            ({'bandwidth': 0.5, 'offset': -0.5}, r'< fs/2'),
        ],
    )
    def test_slice_freqs_errors(self, kws, match):
        with pytest.raises(ValueError, match=match):
            fourier._slice_freqs(64, 1.0, **kws)

    def test_freq_band_edges(self):
        freqs = fourier.fftfreq(64, 1.0)
        assert fourier._freq_band_edges(64, 1.0, None, None) == (None, None)

        ilo, ihi = fourier._freq_band_edges(64, 1.0, -0.25, 0.25)
        assert ilo == np.searchsorted(freqs, -0.25)
        assert freqs[ilo] >= -0.25 and freqs[ilo - 1] < -0.25
        assert freqs[ihi] <= 0.25 and freqs[ihi + 1] > 0.25

        assert fourier._freq_band_edges(64, 1.0, -0.25, 0.6)[1] == 64


class TestTimeFftshift:
    @given(x=iq_waveforms(min_size=8, max_size=64, multiple_of=2, channels=3))
    def test_alternates_sign_along_axis(self, x):
        sign = (-1) ** np.arange(x.shape[1])[np.newaxis, :]

        assert_array_equal(fourier.time_fftshift(x, axis=1), x * sign)
        assert_array_equal(fourier.time_fftshift(x[0], axis=0), x[0] * sign[0])

        scale = np.array([1, 2, 3], dtype=x.real.dtype)
        assert_allclose(
            fourier.time_fftshift(x, scale, axis=1),
            scale[:, np.newaxis] * x * sign,
            rtol=np.finfo(x.dtype).eps,
        )

        y = x.copy()
        assert fourier.time_fftshift(y, overwrite_x=True, axis=1) is y
        assert_array_equal(y, x * sign)

        y = x.copy()
        assert fourier.time_fftshift(y, 2.0, overwrite_x=True, axis=1) is y
        assert_allclose(y, 2 * x * sign, rtol=np.finfo(x.dtype).eps)

    def test_scale_must_be_1d(self):
        x = np.ones((2, 8), dtype=np.complex64)
        with pytest.raises(ValueError, match='1-D or scalar'):
            fourier.time_fftshift(x, np.ones((2, 2)), axis=1)

    def test_ifftshift_is_fftshift(self):
        assert fourier.time_ifftshift is fourier.time_fftshift


class TestOaconvolve:
    @settings(max_examples=50)
    @given(
        x=noise_waveforms(min_size=64, max_size=512, dtype=np.float64),
        kernel_size=st.integers(min_value=1, max_value=31).filter(lambda n: n % 2 == 1),
    )
    def test_identity_kernel(self, x, kernel_size):
        kernel = np.zeros(kernel_size, dtype=x.dtype)
        kernel[kernel_size // 2] = 1.0
        result = fourier.oaconvolve(x, kernel, mode='same')
        # atol for samples near zero, where rtol is meaningless
        assert_allclose(result, x, rtol=1e-8, atol=1e-12)


class TestResample:
    def test_identity_returns_input(self):
        x = gaussian_iq(128, dtype=np.complex128, seed=42)
        assert fourier.resample(x, len(x)) is x

    @settings(max_examples=50)
    @given(scale=st.floats(min_value=0.5, max_value=2.0, allow_nan=False))
    def test_scale_parameter(self, scale):
        x = gaussian_iq(128, dtype=np.complex128, seed=42)
        num_out = len(x) // 2
        result_unscaled = fourier.resample(x, num_out, scale=1.0)
        result_scaled = fourier.resample(x, num_out, scale=scale)
        assert_allclose(result_scaled, scale * result_unscaled, rtol=1e-8, atol=1e-15)

    @settings(max_examples=30)
    @given(x=noise_waveforms(min_size=64, max_size=256, dtype=np.complex128))
    def test_frequency_domain_input(self, x):
        """domain='freq' takes the spectrum as fft(time_fftshift(x)) and matches the
        time-domain path"""
        num = x.size // 2
        X = fourier.fft(fourier.time_fftshift(x), axis=0)
        y_time = fourier.resample(x, num)
        y_freq = fourier.resample(X.copy(), num, domain='freq')
        y_freq_inplace = fourier.resample(X, num, domain='freq', overwrite_x=True)
        sigma = cross_backend_rms(x.dtype, [x.size], n_elementwise=1)
        assert_close(y_freq, y_time, sigma=sigma)
        assert_array_equal(y_freq_inplace, y_freq)

    @given(shift=st.integers(min_value=-16, max_value=16))
    def test_shift_moves_tone_when_downsampling(self, shift):
        nfft_in, nfft_out = 256, 128
        k = 26
        x = unit_tone(nfft_in, 1.0, k / nfft_in)
        y = fourier.resample(x, nfft_out, shift=shift)
        assert y.shape == (nfft_out,) and y.dtype == x.dtype
        # the band copied to the output starts `shift` bins higher, so the tone
        # lands `shift` bins lower in the output spectrum
        expected = (k - shift) / nfft_out
        assert abs(tone_frequency(y, 1.0) - expected) <= 1 / nfft_out
        # a bin-centered unit tone comes back as a unit tone; fftshift multiply,
        # fft(N), ifft(N/2), ifftshift multiply
        sigma = single_backend_rms(x.dtype, [nfft_in, nfft_out], n_elementwise=2)
        assert_close(np.abs(y), np.ones(y.size), sigma=sigma)

    @pytest.mark.parametrize(
        'kws, match',
        [
            ({'num': 128, 'domain': 'bogus'}, 'domain'),
            ({'num': 128, 'window': 'hann'}, 'window'),
            ({'num': 512, 'shift': 1}, 'downsampling'),
            ({'num': 128, 'shift': -100}, 'too small'),
            ({'num': 128, 'shift': 100}, 'too large'),
        ],
    )
    def test_errors(self, kws, match):
        x = np.ones(256, dtype=np.complex64)
        with pytest.raises(ValueError, match=match):
            fourier.resample(x, **kws)

    def test_odd_length_rejected(self):
        with pytest.raises(ValueError, match='even'):
            fourier.resample(np.ones(255, dtype=np.complex64), 128)


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
        _, _, X1 = fourier.stft(x, **kws)
        _, _, X2 = fourier.stft((scale * x).astype(np.complex128), **kws)
        assert_allclose(X2, scale * X1, rtol=1e-6)

    @settings(max_examples=50)
    @given(x=noise_waveforms(min_size=256, max_size=512), params=stft_parameters())
    def test_output_shape(self, x, params):
        """stft and spectrogram outputs have the right shape, dtype and frequency axis
        (min_size=256 guarantees len(x) >= max(nperseg))"""
        _, _, X = fourier.stft(x, truncate=True, **params)
        freqs, times, Sxx = fourier.spectrogram(x, truncate=True, **params)

        step = params['nperseg'] - params['noverlap']
        assert X.shape == Sxx.shape
        assert Sxx.shape[0] == (len(x) - params['noverlap']) // step
        assert Sxx.shape[1] == params['nperseg']
        assert len(freqs) == params['nperseg']
        assert len(times) == Sxx.shape[0]
        assert X.dtype == x.dtype
        assert Sxx.dtype == np.finfo(x.dtype).dtype
        assert_array_equal(freqs, fourier.fftfreq(params['nperseg'], params['fs']))

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
                fourier.stft(x, **kws)
        else:
            _, _, X = fourier.stft(x, **kws)
            assert X.shape == (nseg, 64)

    def test_array_window_matches_named_window(self):
        from scipy import signal

        x = gaussian_iq(512)
        w = signal.windows.get_window('hann', 64, fftbins=True).astype(np.complex64)

        _, _, X_named = fourier.stft(x, fs=1.0, window='hann', nperseg=64)
        _, _, X_array = fourier.stft(x, fs=1.0, window=w, nperseg=64)

        # the two paths differ only in the order of the window and 1/nfft multiplies
        sigma = single_backend_rms(np.complex64, [64], n_elementwise=2)
        assert_close(X_array, X_named, sigma=sigma)

    def test_variants(self):
        x = np.ones(512, dtype=np.complex64)

        X = fourier.stft(x, fs=1.0, window=None, nperseg=64, return_axis_arrays=False)
        _, _, X_rect = fourier.stft(x, fs=1.0, window='rect', nperseg=64)
        assert_array_equal(X, X_rect)

        Sxx = fourier.spectrogram(
            x, fs=1.0, window='hann', nperseg=64, return_axis_arrays=False
        )
        assert Sxx.shape == X.shape and np.isrealobj(Sxx)

        with pytest.raises(TypeError, match='norm'):
            fourier.stft(x, fs=1.0, window='hann', nperseg=64, norm='bogus')

    def test_no_overlap_writes_in_place(self):
        x = gaussian_iq(512)
        kws = {'fs': 1.0, 'window': 'hann', 'nperseg': 64}
        _, _, X_ref = fourier.stft(x, **kws)

        y = x.copy()
        X = fourier.stft(y, overwrite_x=True, return_axis_arrays=False, **kws)
        assert np.shares_memory(X, y)
        assert_array_equal(X, X_ref)

        out = np.empty((8, 64), dtype=np.complex64)
        X = fourier.stft(x, out=out, return_axis_arrays=False, **kws)
        assert np.shares_memory(X, out)
        assert_array_equal(X, X_ref)

    def test_helpers(self):
        x = np.arange(8)
        assert not fourier._same_base_memory(x, None)
        assert fourier._same_base_memory(x, x)
        assert fourier._same_base_memory(x, x[2:])
        assert not fourier._same_base_memory(x, x.copy())

        with pytest.raises(ValueError, match='1-D'):
            fourier._broadcast_onto(np.ones((2, 2)), np.ones((4, 4)), axis=0)

        freqs, times = fourier._get_stft_axes(1e6, 64, 10, overlap_frac=0.5)
        assert_array_equal(freqs, fourier.fftfreq(64, 1e6))
        assert_allclose(times, np.arange(10) * 32 / 1e6)

        primes = fourier._prime_fft_sizes(100, 1000)
        sieve = [n for n in range(101, 1000) if all(n % k for k in range(2, n))]
        assert_array_equal(primes, sieve)


class TestIstft:
    NFFT = 64

    def _roundtrip_sigma(self, dtype):
        # window multiply, fft, ifft, fftshift multiply and the overlap-add
        return single_backend_rms(dtype, [self.NFFT, self.NFFT], n_elementwise=3)

    @settings(max_examples=30)
    @given(
        x=iq_waveforms(min_size=8 * NFFT, max_size=16 * NFFT, multiple_of=NFFT),
        window=st.sampled_from(['hann', 'hamming']),
        overwrite_x=st.booleans(),
    )
    def test_cola_roundtrip_is_identity(self, x, window, overwrite_x):
        nfft = self.NFFT
        _, _, X = fourier.stft(
            x, fs=1.0, window=window, nperseg=nfft, noverlap=nfft // 2
        )
        y = fourier.istft(X, nfft=nfft, noverlap=nfft // 2, overwrite_x=overwrite_x)

        assert y.dtype == x.dtype
        assert y.shape == x.shape
        sigma = self._roundtrip_sigma(x.dtype)
        assert_close(interior(y, nfft), interior(x, nfft), sigma=sigma)

    @pytest.mark.xfail(
        strict=True,
        reason='the no-overlap path of stft scales the window by 1/nfft '
        '(fourier.py:558) and istft never undoes it, unlike the overlapped path '
        'whose _stack_stft_windows normalization (fourier.py:750) cancels the '
        'factor, so istft(stft(x)) with noverlap=0 returns x/nfft',
    )
    @settings(max_examples=30)
    @given(x=iq_waveforms(min_size=8 * NFFT, max_size=16 * NFFT, multiple_of=NFFT))
    def test_no_overlap_roundtrip_is_identity(self, x):
        """With a rectangular window and no overlap, istft(stft(x)) reproduces x."""
        nfft = self.NFFT
        _, _, X = fourier.stft(x, fs=1.0, window='rect', nperseg=nfft, noverlap=0)
        y = fourier.istft(X, nfft=nfft, noverlap=0)

        assert y.dtype == x.dtype
        assert y.shape == x.shape
        assert_close(y, x, sigma=self._roundtrip_sigma(x.dtype))

    @given(size=st.integers(min_value=NFFT, max_value=8 * NFFT))
    def test_size_trims_output(self, size):
        nfft = self.NFFT
        x = np.ones(8 * nfft, dtype=np.complex64)
        _, _, X = fourier.stft(
            x, fs=1.0, window='hann', nperseg=nfft, noverlap=nfft // 2
        )
        y = fourier.istft(X, size, nfft=nfft, noverlap=nfft // 2)
        assert y.shape == (size,)

    @pytest.mark.xfail(
        strict=True,
        reason='_truncated_buffer flattens (copies) `out`, so istft never writes into it',
    )
    def test_out_buffer_is_used(self):
        nfft = self.NFFT
        x = np.ones(8 * nfft, dtype=np.complex64)
        _, _, X = fourier.stft(
            x, fs=1.0, window='hann', nperseg=nfft, noverlap=nfft // 2
        )
        out = np.empty(X.size, dtype=X.dtype)
        y = fourier.istft(X, nfft=nfft, noverlap=nfft // 2, out=out)
        assert np.shares_memory(y, out)

    @pytest.mark.parametrize('hop_divisor', [2, 4])
    def test_stack_unstack_roundtrip(self, hop_divisor):
        nfft = self.NFFT
        noverlap = nfft - nfft // hop_divisor
        x = gaussian_iq(8 * nfft)
        window = np.ones(nfft, dtype=np.complex64)

        stacked = fourier._stack_stft_windows(x, window, nfft, noverlap, norm=None)
        assert stacked.shape == ((x.size - nfft) // (nfft - noverlap) + 1, nfft)
        y = fourier._unstack_stft_windows(stacked, noverlap=noverlap, nperseg=nfft)

        # only the overlap-add sums round
        sigma = FFT_ROUNDOFF_SAFETY * hop_divisor * unit_roundoff(np.float32)
        assert_close(interior(y, nfft), interior(x, nfft), sigma=sigma)

    def test_stack_windows_rejects_unknown_norm(self):
        x = np.ones(256, dtype=np.complex64)
        with pytest.raises(ValueError):
            fourier._stack_stft_windows(x, np.ones(64), 64, 32, norm='bogus')


class TestStftFrequencyEditing:
    NFFT = 64
    NSEG = 8

    def _stft(self, fs, dtype=np.complex64):
        x = gaussian_iq(self.NFFT * self.NSEG, dtype)
        freqs, _, Y = fourier.stft(
            x, fs=fs, window='rect', nperseg=self.NFFT, noverlap=0
        )
        return freqs, Y

    OFF_GRID_XFAIL = pytest.mark.xfail(
        strict=True,
        reason='_freq_band_edges returns the index of the last bin <= cutoff_hi as '
        'the exclusive end of the passband (fourier.py:446), so an upper cutoff '
        'between bins zeroes the last in-band bin',
    )

    @pytest.mark.parametrize(
        'edge_shift_bins',
        [0, pytest.param(0.5, marks=OFF_GRID_XFAIL)],
        ids=['on_grid', 'off_grid'],
    )
    def test_zero_stft_by_freq_zeroes_outside_passband(self, edge_shift_bins):
        fs = 1e6
        freqs, Y = self._stft(fs)
        hi = fs / 4 + edge_shift_bins * fs / self.NFFT
        # the passband is half-open: a bin centered on the upper cutoff is zeroed
        inside = (freqs >= -hi) & (freqs < hi)

        Yz = fourier.zero_stft_by_freq(freqs, Y.copy(), passband=(-hi, hi))
        assert_array_equal(Yz[:, ~inside], 0)
        assert_array_equal(Yz[:, inside], Y[:, inside])

    def test_downsample_stft_keeps_center_band(self):
        freqs, Y = self._stft(1e6)
        nfft_out = self.NFFT // 2
        lo = self.NFFT // 2 - nfft_out // 2

        freqs_out, Yo = fourier.downsample_stft(freqs, Y, nfft_out)
        assert Yo.shape == (self.NSEG, nfft_out)
        assert freqs_out.shape == (nfft_out,)
        assert_array_equal(Yo, Y[:, lo : lo + nfft_out])
        assert not np.shares_memory(Yo, Y)
        assert_allclose(np.diff(freqs_out), freqs[1] - freqs[0], rtol=1e-12)

    def test_downsample_stft_fast_path_returns_view(self):
        freqs, Y = self._stft(1e6)
        _, Yo = fourier.downsample_stft(freqs, Y, self.NFFT // 2, out=Y)
        assert np.shares_memory(Yo, Y)

    def test_downsample_stft_zero_pads_when_upsampling(self):
        freqs, Y = self._stft(1e6)
        nfft_out = 2 * self.NFFT
        pad = (nfft_out - self.NFFT) // 2

        _, Yo = fourier.downsample_stft(freqs, Y, nfft_out)
        assert Yo.shape == (self.NSEG, nfft_out)
        assert_array_equal(Yo[:, :pad], 0)
        assert_array_equal(Yo[:, pad + self.NFFT :], 0)
        assert_array_equal(Yo[:, pad : pad + self.NFFT], Y)

    @pytest.mark.parametrize(
        'edge_shift_bins',
        [0, pytest.param(-0.5, marks=OFF_GRID_XFAIL)],
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
        _, Yo = fourier.downsample_stft(freqs, Y, nfft_out, passband=passband)
        quarter = nfft_out // 4
        assert_array_equal(Yo[:, :quarter], 0)
        assert_array_equal(Yo[:, -quarter:], 0)
        assert np.all(Yo[:, quarter:-quarter] != 0)

    def test_downsample_stft_out_buffer(self):
        freqs, Y = self._stft(1.0)
        nfft_out = self.NFFT // 2
        _, expected = fourier.downsample_stft(
            freqs, Y, nfft_out, passband=(-0.125, 0.125)
        )
        _, Yo = fourier.downsample_stft(
            freqs, Y, nfft_out, passband=(-0.125, 0.125), out=np.empty_like(Y)
        )
        assert Yo.shape == (self.NSEG, nfft_out)
        assert_array_equal(Yo, expected)

    @pytest.mark.xfail(
        strict=True,
        reason='_truncated_buffer flattens `out` with ndarray.flatten(), which '
        'copies, so downsample_stft allocates a new array instead of writing '
        'into the buffer it was given',
    )
    def test_downsample_stft_writes_into_out(self):
        freqs, Y = self._stft(1.0)
        out = np.empty_like(Y)
        _, Yo = fourier.downsample_stft(
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

        (out0, out1), (in0, in1), _ = fourier._find_downsample_copy_range(
            nfft_in, nfft_out, start, end
        )
        assert out1 - out0 == in1 - in0
        assert 0 <= in0 <= in1 <= nfft_in
        assert 0 <= out0 <= out1 <= nfft_out
        assert in1 - in0 <= nfft_out
        # the copied block is centered in the output
        assert abs(out0 - (nfft_out - out1)) <= 1


class TestFilterDesign:
    @pytest.mark.parametrize(
        'window, overlap_scale',
        [
            ('hamming', 1 / 2),
            ('blackman', 2 / 3),
            ('blackmanharris', 4 / 5),
            pytest.param(
                'rect',
                1,
                marks=pytest.mark.xfail(
                    strict=True,
                    reason="the 'rect'/None branch is followed by `if` rather than "
                    "`elif`, so it falls through to the 'unexpected matching error'",
                ),
            ),
            pytest.param(
                None,
                1,
                marks=pytest.mark.xfail(
                    strict=True, reason='same fall-through as the rect window'
                ),
            ),
        ],
    )
    def test_design_oafilter_overlap(self, window, overlap_scale):
        nfft = 120
        nfft_out, noverlap, scale, pad_out = fourier.design_oafilter(
            40 * nfft, window=window, nfft_out=nfft, nfft=nfft, extend=False
        )
        assert nfft_out == nfft
        assert scale == overlap_scale
        assert noverlap == round(nfft * overlap_scale)
        assert pad_out == 0

    @pytest.mark.parametrize(
        'size, kws, exc, match',
        [
            (1200, {'window': 'hann'}, TypeError, 'window'),
            (1210, {'window': 'blackman', 'nfft_out': 121}, ValueError, '% 3'),
            (1000, {'window': 'hamming'}, ValueError, 'integer multiple of noverlap'),
        ],
        ids=['unsupported_window', 'nfft_out_off_divisor', 'size_off_noverlap'],
    )
    def test_design_oafilter_errors(self, size, kws, exc, match):
        kws = {'nfft_out': 120, 'nfft': 120, 'extend': False, **kws}
        with pytest.raises(exc, match=match):
            fourier.design_oafilter(size, **kws)

    def test_design_oafilter_options(self):
        pad_out = fourier.design_oafilter(
            1000, window='hamming', nfft_out=120, nfft=120, extend=True
        )[3]
        assert pad_out == 1000 % 60

        nfft_out = fourier.design_oafilter(
            1200, window='hamming', nfft_out=None, nfft=120, extend=False
        )[0]
        assert nfft_out == 120

    @pytest.mark.parametrize('fs_base', [122.88e6, 125e6])
    @pytest.mark.parametrize(
        'fs_target', [7.68e6, 15.36e6, 20e6, 30.72e6, 61.44e6, 107.52e6]
    )
    @pytest.mark.parametrize('window', ['hamming', 'blackman'])
    @pytest.mark.parametrize('shift', [None, 'none', 'left', 'right'])
    def test_design_cola_resampler_consistency(self, fs_base, fs_target, window, shift):
        """The rates, FFT sizes and passband of a design are mutually consistent.

        The grid omits `bw`, which only sets the passband width, and shift=False,
        which the design treats exactly like None (unlike the truthy 'none').
        """
        bw = 10e6
        design = fourier.design_cola_resampler(
            fs_base, fs_target, bw=bw, shift=shift, window=window
        )

        assert design['fs'] == design['fs_sdr'] <= fs_base
        # the SDR rate is an integer division of the base clock
        assert_allclose(fs_base / design['fs_sdr'], round(fs_base / design['fs_sdr']))
        assert_allclose(
            design['nfft'] / design['nfft_out'], design['fs_sdr'] / fs_target
        )

        divisor = fourier._COLA_WINDOW_SIZE_DIVISOR[window]
        assert design['nfft'] % divisor == 0 and design['nfft_out'] % divisor == 0
        assert design['window'] == window

        lo, hi = design['passband']
        assert_allclose(hi - lo, bw)
        assert_allclose((hi + lo) / 2, design['lo_offset'])
        sign = {'left': -1, 'right': 1}.get(shift, 0)
        assert np.sign(design['lo_offset']) == sign

    def test_design_cola_resampler_options(self):
        design = fourier.design_cola_resampler(125e6, 107.52e6)
        assert design['window'] == 'hamming'
        assert design['passband'] == (None, None) and design['lo_offset'] == 0

        forced = fourier.design_cola_resampler(125e6, 15.36e6, fs_sdr=50e6)
        assert forced['fs_sdr'] == pytest.approx(50e6)

        upsample = fourier.design_cola_resampler(10e6, 15.36e6)
        assert (
            upsample['fs_sdr'] == pytest.approx(10e6)
            and upsample['nfft'] < upsample['nfft_out']
        )

        # 8209 is the first prime above the default min_fft_size, so it is the
        # smallest rational FFT size for this ratio and the one avoid_primes rejects
        prime = 8209
        fs_target = 125e6 * prime / (prime + 1)
        pruned = fourier.design_cola_resampler(125e6, fs_target)
        with_primes = fourier.design_cola_resampler(
            125e6, fs_target, avoid_primes=False
        )
        assert with_primes['nfft_out'] == prime
        assert pruned['nfft_out'] == 2 * prime

        fs_sdr, fir = fourier.design_fir_resampler(125e6, 107.52e6)
        assert fs_sdr == pytest.approx(125e6)
        assert_allclose(fir['down'] / fir['up'], 125e6 / 107.52e6)

    @pytest.mark.parametrize(
        'kws, match',
        [
            ({'bw': np.inf, 'shift': 'left'}, 'analysis bandwidth'),
            ({'bw': 200e6}, 'exceeds Nyquist'),
            ({'bw': 10e6, 'shift': 'sideways'}, 'shift argument'),
            ({'bw': 10e6, 'shift': 'left', 'fs_base': 20e6}, 'minimum'),
            ({'fs_target': 125e6 / np.pi}, 'no rational FFT sizes'),
        ],
    )
    def test_design_cola_resampler_errors(self, kws, match):
        kws = {'fs_base': 125e6, 'fs_target': 15.36e6, **kws}
        with pytest.raises(ValueError, match=match):
            fourier.design_cola_resampler(**kws)

    @pytest.mark.parametrize('numtaps', [401, 1001, 4001])
    @pytest.mark.parametrize('dtype', FLOAT_DTYPES, ids=dtype_id)
    def test_design_fir_lpf(self, numtaps, dtype):
        b = fourier.design_fir_lpf(10e6, 50e6, numtaps=numtaps, dtype=dtype)
        assert b.shape == (numtaps,)
        assert b.dtype == np.dtype(dtype)
        # least-squares passband ripple bounds the DC gain error
        assert abs(level_error_dB(b.sum())) < level_tolerance_dB(FIR_LEAKAGE)

    def test_fir_lowpass_fft_response(self):
        size = 256
        H = fourier._fir_lowpass_fft(size, 1.0, cutoff=0.1, transition=0.05)
        freqs = fourier.fftfreq(size, 1.0)
        assert H.shape == (size,) and H.dtype == np.complex64

        passband = np.abs(freqs) < 0.08
        stopband = np.abs(freqs) > 0.1 + 0.05 + 2 / size
        assert np.abs(np.abs(H[passband]) - 1).max() < FIR_LEAKAGE
        assert np.abs(H[stopband]).max() < FIR_LEAKAGE

    @pytest.mark.parametrize('cutoff', [np.inf, 0.5])
    def test_fir_lowpass_fft_cutoff_at_or_past_nyquist_is_allpass(self, cutoff):
        H = fourier._fir_lowpass_fft(256, 1.0, cutoff=cutoff, transition=0.05)
        assert H.dtype == np.complex64
        assert_array_equal(H, 1)

    def test_fir_lowpass_fft_transition_is_cut_at_nyquist(self):
        """a transition band that would run past nyquist ends there instead"""
        size = 256
        H = fourier._fir_lowpass_fft(size, 1.0, cutoff=0.4, transition=0.3)
        freqs = fourier.fftfreq(size, 1.0)
        assert H.shape == (size,) and H.dtype == np.complex64

        passband = np.abs(freqs) < 0.38
        assert np.abs(np.abs(H[passband]) - 1).max() < FIR_LEAKAGE
        # firwin2 constrains an even-length (type II) design to zero gain at nyquist
        assert np.abs(H[np.argmin(freqs)]) < FIR_LEAKAGE


class TestOverlapAddFilters:
    FS = 1e6
    NFFT = 256
    NSEG = 40
    PASSBAND = (-0.2e6, 0.2e6)

    def _tone(self, f0, channels=None):
        x = unit_tone(self.NFFT * self.NSEG, self.FS, f0)
        return x if channels is None else np.tile(x, (channels, 1))

    def _oafilter(self, x, **kws):
        kws = {'window': 'hamming', 'passband': self.PASSBAND, **kws}
        return fourier.oafilter(x, fs=self.FS, nfft=self.NFFT, **kws)

    @settings(max_examples=20)
    @given(f0_bins=st.integers(min_value=-40, max_value=40))
    def test_oafilter_passes_in_band_tone(self, f0_bins):
        f0 = f0_bins * self.FS / self.NFFT
        x = self._tone(f0)
        y = self._oafilter(x)
        assert y.shape == x.shape and y.dtype == x.dtype
        yi = interior(y, self.NFFT)
        assert_tone_level(yi)
        assert abs(tone_frequency(yi, self.FS) - f0) <= self.FS / yi.size

    def test_oafilter_rejects_out_of_band_tone(self):
        y = self._oafilter(self._tone(0.4e6))
        yi = interior(y, self.NFFT)
        # the residual is the hamming sidelobe leakage (-53 dB); the window's
        # -22 dB edge pedestal leaves the ringing concentrated at the segment edges
        assert level_error_dB(rms(yi)) < -40
        assert level_error_dB(yi).max() < -25

    def test_oafilter_downsample_length(self):
        x = self._tone(0.05e6)
        nfft_out = self.NFFT // 2
        y = self._oafilter(x, nfft_out=nfft_out)
        assert y.shape == (round(x.size * nfft_out / self.NFFT),)

    @pytest.mark.xfail(
        strict=True,
        reason='the downsample_stft branch of oafilter doubles the tone amplitude',
    )
    def test_oafilter_downsample_preserves_level(self):
        nfft_out = self.NFFT // 2
        y = self._oafilter(self._tone(0.05e6), nfft_out=nfft_out)
        assert_tone_level(interior(y, nfft_out))

    @settings(max_examples=20)
    @given(
        f0_bins=st.integers(min_value=-20, max_value=20),
        shift_bins=st.sampled_from([0, 8, -8]),
        scale=st.sampled_from([1.0, 2.0]),
    )
    def test_oaresample_downsample_tone(self, f0_bins, shift_bins, scale):
        """As sensor.lib.compute.corrections._oaresample: 2-D (channel, sample)
        input on axis=1, a bin-grid frequency shift and a voltage scale."""
        up, down = self.NFFT // 2, self.NFFT
        fs_out = self.FS * up / down
        f0 = f0_bins * self.FS / self.NFFT
        shift = shift_bins * self.FS / down
        x = self._tone(f0, channels=2)

        kws = {'window': 'hamming', 'axis': 1, 'frequency_shift': shift, 'scale': scale}
        y = fourier.oaresample(x, up, down, self.FS, **kws)
        assert y.shape == (2, round(x.shape[1] * up / down))
        assert y.dtype == x.dtype
        yi = interior(y, up, axis=1)
        assert_tone_level(yi, scale)
        for row in yi:
            assert abs(tone_frequency(row, fs_out) - (f0 - shift)) <= fs_out / row.size

    def test_oaresample_filter_bandwidth(self):
        up, down = self.NFFT // 2, self.NFFT
        kws = {
            'window': 'hamming',
            'axis': 1,
            'filter_bandwidth': 0.2e6,
            'transition_bandwidth': 0.05e6,
        }
        y_pass = fourier.oaresample(self._tone(0.05e6, 1), up, down, self.FS, **kws)
        y_stop = fourier.oaresample(self._tone(0.2e6, 1), up, down, self.FS, **kws)
        assert_tone_level(interior(y_pass, up, axis=1))
        assert level_error_dB(interior(y_stop, up, axis=1)).max() < FIR_RECT_STOPBAND_DB

    def test_oaresample_upsample(self):
        up, down = 2 * self.NFFT, self.NFFT
        x = self._tone(0.05e6, 1)
        y = fourier.oaresample(x, up, down, self.FS, window='hamming', axis=1)
        assert y.shape == (1, 2 * x.shape[1])
        yi = interior(y, up, axis=1)
        assert_tone_level(yi)
        assert (
            abs(tone_frequency(yi[0], 2 * self.FS) - 0.05e6)
            <= 2 * self.FS / yi.shape[1]
        )

    @pytest.mark.parametrize(
        'up, down, shift, match',
        [
            (128, 256, 1e6 / 256 / 3, 'multiple of fs/up'),
            (128, 256, 100 * 1e6 / 256, 'too large'),
            (128, 256, -100 * 1e6 / 256, 'too small'),
            (512, 256, 1e6 / 256, 'only supported when downsampling'),
        ],
    )
    def test_oaresample_shift_errors(self, up, down, shift, match):
        x = self._tone(0.05e6, 1)
        with pytest.raises(ValueError, match=match):
            fourier.oaresample(x, up, down, self.FS, axis=1, frequency_shift=shift)


class TestToneFarBinFloor:
    """Roundoff in the bins away from a unit, bin-centered complex64 tone, on each
    array namespace.

    With an integer number of cycles per segment (and a rectangular window for the
    STFT) the exact spectrum occupies one bin, so every other bin measures roundoff
    alone against a float64 reference of the same float32-quantized input. For the
    time-domain outputs of resample and oaconvolve the bins are those of the error
    spectrum, taken in float64 so the analysis adds no roundoff of its own.
    """

    NSEG = 4
    KERNEL_TAPS = 400

    @staticmethod
    def _assert_far_bins(err_far, rms_bound, tone_peak, nfft, sigma):
        """`err_far`: the roundoff in the far bins; `rms_bound`: the rms error model
        in the same units; `tone_peak`: the tone's own bin value, which anchors the
        structured roundoff bound"""
        assert rms(err_far) < rms_bound, (
            f'far-bin rms roundoff above {far_bin_floor_dBc(sigma, nfft):.1f} dBc'
        )
        white = peak_factor(err_far.size) * rms_bound
        structured = tone_peak_roundoff(np.complex64) * tone_peak
        assert np.abs(err_far).max() < max(white, structured), (
            f'far-bin peak roundoff above '
            f'{far_bin_floor_dBc(sigma, nfft, err_far.size):.1f} dBc'
        )

    def _check_error_spectrum(self, out, out_ref, sigma):
        """bound the spectrum of the roundoff error in a time-domain output of a unit
        tone, excluding the tone's own bin; `sigma` is the rms error model relative to
        the unit input"""
        err = np.fft.fft(to_numpy(out).astype(np.complex128) - out_ref)
        size = err.size
        far = np.ones(size, dtype=bool)
        far[int(np.argmax(np.abs(np.fft.fft(out_ref))))] = False

        # unnormalized fft: bin rms is sqrt(size) times the sample rms, and the unit
        # tone peaks at size
        self._assert_far_bins(err[far], np.sqrt(size) * sigma, size, size, sigma)

    @staticmethod
    def _stft(x, nfft):
        return fourier.stft(x, fs=1.0, window='rect', nperseg=nfft, noverlap=0)[2]

    @given(
        nfft=st.sampled_from(FAR_BIN_NFFTS),
        bin_fraction=st.floats(min_value=0, max_value=1),
    )
    def test_stft_far_bins(self, xp, nfft, bin_fraction):
        k = tone_bin(nfft, bin_fraction)
        x = bin_centered_tone(nfft, k, self.NSEG)
        X_ref = self._stft(x.astype(np.complex128), nfft)
        X = to_numpy(self._stft(xp.asarray(x), nfft)).astype(np.complex128)

        # the bins are in ascending fftfreq order, and a unit tone fills its bin
        # with the rect window's unit gain
        peak_bin = k + nfft // 2
        assert np.all(np.argmax(np.abs(X), axis=1) == peak_bin)
        tone_peak = np.abs(X_ref[0, peak_bin])
        assert abs(tone_peak - 1) < tone_peak_roundoff(np.complex64)
        assert np.abs(np.abs(X[:, peak_bin]) - 1).max() < tone_peak_roundoff(
            np.complex64
        )

        far = np.ones(nfft, dtype=bool)
        far[peak_bin] = False

        # window/nfft and the window multiply, then fft(nfft)
        sigma = single_backend_rms(np.complex64, [nfft], n_elementwise=2)
        rms_bound = sigma * rms(X_ref)
        self._assert_far_bins((X - X_ref)[:, far], rms_bound, tone_peak, nfft, sigma)

    @given(
        nfft=st.sampled_from(FAR_BIN_NFFTS),
        bin_fraction=st.floats(min_value=0, max_value=1),
    )
    def test_resample_far_bins(self, xp, nfft, bin_fraction):
        # keep the tone inside the half band that survives downsampling by 2
        k = tone_bin(nfft, 0.3 + 0.4 * bin_fraction)
        x = bin_centered_tone(nfft, k, self.NSEG)
        num_out = x.size // 2
        # fftshift multiply, fft(N), ifft(N/2), ifftshift multiply
        sigma = single_backend_rms(np.complex64, [x.size, num_out], n_elementwise=2)
        out_ref = fourier.resample(x.astype(np.complex128), num_out)
        out = fourier.resample(xp.asarray(x), num_out)
        self._check_error_spectrum(out, out_ref, sigma)

    @given(
        nfft=st.sampled_from(FAR_BIN_NFFTS),
        bin_fraction=st.floats(min_value=0, max_value=1),
    )
    def test_oaconvolve_far_bins(self, xp, nfft, bin_fraction):
        x = bin_centered_tone(nfft, tone_bin(nfft, bin_fraction), self.NSEG)
        # a unit-gain lowpass; the tone may land in its stopband, so bounds are
        # anchored to the input tone rather than the output
        kernel = np.hanning(self.KERNEL_TAPS).astype(np.float32)
        kernel /= kernel.sum()
        # the overlap-add block is at most x.size long
        sigma = single_backend_rms(np.complex64, [x.size, x.size], n_elementwise=1)
        out_ref = fourier.oaconvolve(
            x.astype(np.complex128), kernel.astype(np.float64), mode='same'
        )
        out = fourier.oaconvolve(xp.asarray(x), xp.asarray(kernel), mode='same')
        self._check_error_spectrum(out, out_ref, sigma)


class TestNumpyCupyCrossComparison:
    """Cross-comparison tests validating numpy and cupy produce close results.

    Tolerances for the FFT-based functions come from `cross_backend_rms`, parameterized
    by the FFT sizes and elementwise operations each function performs. Elementwise
    checks use an absolute tolerance anchored on the output rms, because the outputs
    are noise-like and a relative tolerance is meaningless in their smallest bins.
    """

    NFFT = 64

    @pytest.mark.parametrize(
        'name', ['hamming', 'hann', 'blackman', 'bartlett', 'flattop']
    )
    @pytest.mark.parametrize('nwindow', [8, 64, 129, 512])
    @pytest.mark.parametrize('dtype', FLOAT_DTYPES, ids=dtype_id)
    def test_get_window(self, cupy_available, name, nwindow, dtype):
        get_window = functools.partial(fourier.get_window, name, nwindow, dtype=dtype)
        w_np, w_cp = numpy_and_cupy(cupy_available, get_window, xp_kwarg='xp')
        assert_close(w_cp, w_np, rtol=elementwise_rtol(dtype), atol=ATOL)

    @given(nfft=fft_sizes(min_size=8, max_size=512), fs=sample_rates())
    def test_fftfreq(self, cupy_available, nfft, fs):
        f_np, f_cp = numpy_and_cupy(
            cupy_available, fourier.fftfreq, nfft, fs, dtype=np.float64, xp_kwarg='xp'
        )
        assert_close(f_cp, f_np, rtol=RTOL_FLOAT64, atol=ATOL)

    def test_designs_on_cupy(self, cupy_available):
        b_np = fourier.design_fir_lpf(10e6, 50e6, numtaps=401)
        b_cp = fourier.design_fir_lpf(10e6, 50e6, numtaps=401, xp=cupy_available)
        assert isinstance(b_cp, cupy_available.ndarray)
        assert_close(b_cp, b_np)

        f_np = fourier.fftfreq(65, 1e6, as_index=False)
        f_cp = fourier.fftfreq(65, 1e6, as_index=False, xp=cupy_available)
        assert isinstance(f_cp, cupy_available.ndarray)
        assert_close(f_cp, f_np, rtol=RTOL_FLOAT64)

    @given(x=noise_waveforms(min_size=64, max_size=256, dtype=np.complex64))
    def test_resample(self, cupy_available, x):
        num_out = len(x) // 2
        y_np, y_cp = numpy_and_cupy(cupy_available, fourier.resample, x, num_out)
        # fftshift multiply, fft(N), ifft(N/2), ifftshift multiply
        sigma = cross_backend_rms(x.dtype, [len(x), num_out], n_elementwise=2)
        assert_close(y_cp, y_np, sigma=sigma)

    @pytest.mark.parametrize(
        'func, n_elementwise, power',
        [(fourier.stft, 2, False), (fourier.spectrogram, 3, True)],
        ids=['stft', 'spectrogram'],
    )
    @given(x=noise_waveforms(min_size=256, max_size=512, dtype=np.complex64))
    def test_stft_and_spectrogram(self, cupy_available, func, n_elementwise, power, x):
        kws = {
            'fs': 1e6,
            'window': 'hamming',
            'nperseg': 64,
            'noverlap': 32,
            'truncate': True,
        }
        (freqs_np, times_np, X_np), (freqs_cp, times_cp, X_cp) = numpy_and_cupy(
            cupy_available, func, x, **kws
        )

        assert_close(
            freqs_cp, freqs_np, rtol=elementwise_rtol(freqs_np.dtype), atol=ATOL
        )
        assert_close(
            times_cp, times_np, rtol=elementwise_rtol(times_np.dtype), atol=ATOL
        )

        # window/nfft and the window multiply, then fft(nperseg); the spectrogram
        # adds the |X|**2 rounding. For noise-like input the relative rms error of
        # power equals that of amplitude, but the error in one bin grows with
        # sqrt(Sxx), so rtol covers the large bins and atol the small ones.
        sigma = cross_backend_rms(x.dtype, [64], n_elementwise=n_elementwise)
        assert_close(X_cp, X_np, rtol=4 * sigma if power else 0, sigma=sigma)

    @given(x=noise_waveforms(64, 256, dtype=[np.float32, np.float64]))
    def test_oaconvolve(self, cupy_available, x):
        kernel = np.array([0.25, 0.5, 0.25], dtype=x.dtype)
        y_np, y_cp = numpy_and_cupy(
            cupy_available, fourier.oaconvolve, x, kernel, mode='same'
        )
        # overlap-add blocks are at most len(x) long; anchoring to the input rms
        # accounts for the kernel's gain
        sigma = cross_backend_rms(x.dtype, [len(x), len(x)], n_elementwise=1)
        assert_close(y_cp, y_np, sigma=sigma, scale=rms(x))

    @given(x=iq_waveforms(min_size=8 * NFFT, max_size=16 * NFFT, multiple_of=NFFT))
    def test_istft_roundtrip(self, cupy_available, x):
        nfft = self.NFFT

        def roundtrip(x):
            _, _, X = fourier.stft(
                x, fs=1.0, window='hann', nperseg=nfft, noverlap=nfft // 2
            )
            return fourier.istft(X, nfft=nfft, noverlap=nfft // 2)

        y_np, y_cp = numpy_and_cupy(cupy_available, roundtrip, x)
        sigma = cross_backend_rms(x.dtype, [nfft, nfft], n_elementwise=3)
        assert_close(y_cp, y_np, sigma=sigma)

    @given(x=iq_waveforms(min_size=8, max_size=64, multiple_of=2, channels=3))
    def test_time_fftshift(self, cupy_available, x):
        scale = np.array([1, 2, 3], dtype=x.real.dtype)
        y_np, y_cp = numpy_and_cupy(
            cupy_available, fourier.time_fftshift, x, scale, axis=1
        )
        assert_close(y_cp, y_np, rtol=np.finfo(x.dtype).eps)

    def test_frequency_slicing(self, cupy_available):
        x = np.arange(2 * 3 * 64, dtype=np.float32).reshape(2, 3, 64)

        t_np, t_cp = numpy_and_cupy(
            cupy_available, fourier.truncate_freqs, x, 64, 1.0, 0.5, axis=2
        )
        assert_array_equal(t_cp, t_np)

        def nulled(x):
            x = x.copy()
            fourier.null_lo(x, 64, 1.0, 0.25, axis=2)
            return x

        n_np, n_cp = numpy_and_cupy(cupy_available, nulled, x)
        assert_array_equal(n_cp, n_np)

    def test_stft_frequency_editing(self, cupy_available):
        x = gaussian_iq(8 * 64)
        freqs, _, Y = fourier.stft(x, fs=1.0, window='rect', nperseg=64, noverlap=0)

        def zeroed(freqs, Y):
            return fourier.zero_stft_by_freq(freqs, Y.copy(), passband=(-0.25, 0.25))

        Yz_np, Yz_cp = numpy_and_cupy(cupy_available, zeroed, freqs, Y)
        assert_array_equal(Yz_cp, Yz_np)

        (fo_np, Yo_np), (fo_cp, Yo_cp) = numpy_and_cupy(
            cupy_available, fourier.downsample_stft, freqs, Y, 32
        )
        assert_array_equal(Yo_cp, Yo_np)
        assert_close(fo_cp, fo_np, rtol=RTOL_FLOAT64)

    @settings(max_examples=10)
    @given(f0_bins=st.integers(min_value=-20, max_value=20))
    def test_oafilter_and_oaresample(self, cupy_available, f0_bins):
        fs, nfft = 1e6, 256
        x = np.tile(unit_tone(nfft * 40, fs, f0_bins * fs / nfft), (2, 1))

        kws = {'fs': fs, 'nfft': nfft, 'window': 'hamming', 'passband': (-0.2e6, 0.2e6)}
        y_np, y_cp = numpy_and_cupy(cupy_available, fourier.oafilter, x[0], **kws)
        # window multiply, fft(nfft), zeroing, ifft(nfft), window multiply, overlap add
        sigma = cross_backend_rms(x.dtype, [nfft, nfft], n_elementwise=3)
        assert_close(y_cp, y_np, sigma=sigma, err_msg='oafilter')

        shift = 8 * fs / nfft
        kws = {'window': 'hamming', 'axis': 1, 'filter_bandwidth': 0.2e6}
        resample = functools.partial(fourier.oaresample, frequency_shift=shift, **kws)
        y_np, y_cp = numpy_and_cupy(cupy_available, resample, x, nfft // 2, nfft, fs)
        # as oafilter, plus the FIR multiply and the output scaling
        sigma = cross_backend_rms(x.dtype, [nfft, nfft // 2], n_elementwise=5)
        assert_close(y_cp, y_np, sigma=sigma, err_msg='oaresample')

    @given(x=iq_waveforms(min_size=256, max_size=1024, multiple_of=64, channels=4))
    def test_fft_out_buffer_and_chunking(self, cupy_available, max_cupy_fft_chunk, x):
        X_np = fourier.fft(x, axis=1)
        x_np_back = fourier.ifft(X_np, axis=1)

        # small enough that the 4-row input is split across several calls
        max_cupy_fft_chunk(x.shape[1])
        x_cp = cupy_available.asarray(x)
        out = cupy_available.empty_like(x_cp)
        X_cp = fourier.fft(x_cp, axis=1, out=out)
        assert X_cp is out or cupy_available.shares_memory(X_cp, out)
        x_cp_back = fourier.ifft(X_cp, axis=1, out=cupy_available.empty_like(x_cp))

        n = x.shape[1]
        assert_close(X_cp, X_np, sigma=cross_backend_rms(x.dtype, [n]), err_msg='fft')
        sigma = cross_backend_rms(x.dtype, [n, n])
        assert_close(x_cp_back, x_np_back, sigma=sigma, err_msg='ifft')
