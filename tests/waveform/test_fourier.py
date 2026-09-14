"""Property-based tests for striqt.waveform.fourier using Hypothesis.

Covers get_window, fftfreq, oaconvolve, resample, stft and spectrogram: identities,
linearity, dtype and shape properties, numpy/cupy/dask compatibility, and roundoff
bounds derived from an FFT error model (numpy vs cupy, and the floor far from a tone).
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import iq_waveforms
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import array_shapes
from numpy.testing import assert_allclose, assert_array_equal


def _get_fourier():
    """Lazily import fourier module to avoid reifying scipy at test collection time."""
    from striqt.waveform import fourier

    return fourier


@pytest.fixture
def fourier_module():
    """Fixture that lazily imports the fourier module."""
    return _get_fourier()


@pytest.fixture
def cupy_available():
    from conftest import _cupy

    if _cupy is None:
        pytest.skip('cupy is not available')
    return _cupy


def window_names():
    """Strategy for valid window function names."""
    return st.sampled_from([
        'hann',
        'hamming',
        'blackman',
        'blackmanharris',
        'bartlett',
        'nuttall',
    ])


def window_sizes(min_size: int = 4, max_size: int = 1024):
    """Strategy for valid window sizes (must be positive integers)."""
    return st.integers(min_value=min_size, max_value=max_size)


def fft_sizes(min_size: int = 4, max_size: int = 512):
    """Strategy for FFT sizes: powers of two, or arbitrary even sizes (resample
    requires an even length)."""
    powers = st.integers(min_value=2, max_value=9).map(lambda p: 2**p)
    arbitrary = st.integers(min_value=min_size, max_value=max_size).filter(
        lambda n: n % 2 == 0
    )
    return st.one_of(powers, arbitrary)


def sample_rates(min_rate: float = 1e3, max_rate: float = 100e6):
    """Strategy for sample rates in Hz."""
    return st.floats(
        min_value=min_rate,
        max_value=max_rate,
        allow_nan=False,
        allow_infinity=False,
    )


def complex_waveforms(
    min_size: int = 64,
    max_size: int = 2048,
    dtype=None,
    min_dims: int = 1,
    max_dims: int = 1,
    min_log_power=-13,
    max_log_power=3,
):
    """Strategy for complex gaussian-noise waveforms.

    Hypothesis draws only the dtype, shape and rms (one of the two extremes
    10**(min_log_power/2) and 10**(max_log_power/2)); the noise samples come from
    fixed-seed generators, so they do not vary with the example or shrink.
    """
    if dtype is None:
        dtype_strategy = st.sampled_from([np.complex64, np.complex128])
    else:
        dtype_strategy = st.just(dtype)

    re_rng = np.random.default_rng(seed=42)
    im_rng = np.random.default_rng(seed=43)

    @st.composite
    def _complex_waveform(draw):
        dt = draw(dtype_strategy)
        # resample requires an even length
        size = draw(st.integers(min_value=min_size // 2, max_value=max_size // 2)) * 2

        if min_dims == max_dims == 1:
            shape = (size,)
        else:
            extra_dims = draw(
                array_shapes(
                    min_dims=min_dims - 1,
                    max_dims=max_dims - 1,
                    min_side=1,
                    max_side=4,
                )
            )
            shape = (size,) + extra_dims

        oom = draw(st.sampled_from([min_log_power, max_log_power]))
        lin_scale = np.asarray(10 ** (oom / 2), dtype=dt)

        real = re_rng.normal(loc=0.0, scale=1.0, size=shape).astype(dt)
        imag = im_rng.normal(loc=0.0, scale=1.0, size=shape).astype(dt)

        return (lin_scale * (real + 1j * imag)).astype(dt)

    return _complex_waveform()


def real_waveforms(
    min_size: int = 64,
    max_size: int = 2048,
    min_log_power=-13,
    max_log_power=3,
    dtype=None,
):
    """Strategy for real-valued waveform arrays."""
    if dtype is None:
        dtype_strategy = st.sampled_from([np.float32, np.float64])
    else:
        dtype_strategy = st.just(dtype)

    re_rng = np.random.default_rng(seed=42)

    @st.composite
    def _real_waveform(draw):
        dt = draw(dtype_strategy)
        # resample requires an even length
        size = draw(st.integers(min_value=min_size // 2, max_value=max_size // 2)) * 2

        oom = draw(st.integers(min_value=min_log_power, max_value=max_log_power))
        lin_scale = np.asarray(10 ** (oom / 2), dtype=dt)

        real = re_rng.normal(loc=0.0, scale=1.0, size=(size,)).astype(dt)

        return (lin_scale * real).astype(dt)

    return _real_waveform()


def stft_parameters():
    """Strategy for STFT parameters.

    Window names and sizes are drawn from short lists so that get_window's
    persistent on-disk cache does not fill with one-off entries.
    """

    @st.composite
    def _stft_params(draw):
        nperseg = draw(st.sampled_from([64, 128, 256]))
        # stft supports only noverlap == 0 or nperseg // 2
        noverlap = draw(st.sampled_from([0, nperseg // 2]))
        fs = draw(sample_rates(min_rate=1e3, max_rate=10e6))
        window = draw(st.sampled_from(['hamming', 'hann', 'blackman']))

        return {
            'nperseg': nperseg,
            'noverlap': noverlap,
            'fs': fs,
            'window': window,
        }

    return _stft_params()


class TestGetWindowProperties:
    """Properties: Window function generation."""

    @given(
        name=window_names(),
        nwindow=st.sampled_from([8, 16, 32, 64, 128, 256, 512]),
    )
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None
    )
    def test_window_length_no_padding(self, name, nwindow):
        """Property: Window length equals requested size (no zero padding)."""
        fourier = _get_fourier()
        w = fourier.get_window(name, nwindow, nzero=0)
        assert len(w) == nwindow

    @given(
        name=window_names(),
        nwindow=window_sizes(min_size=8, max_size=256),
        nzero=st.integers(min_value=1, max_value=64),
    )
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None
    )
    def test_window_with_zero_padding(self, name, nwindow, nzero):
        """Property: Zero-padded window has correct total length."""
        fourier = _get_fourier()
        w = fourier.get_window(name, nwindow, nzero=nzero)
        assert len(w) == nwindow + nzero

    @given(
        name=window_names(),
        nwindow=window_sizes(min_size=8, max_size=256),
    )
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None
    )
    def test_normalized_window_power(self, name, nwindow):
        """Property: Normalized window has unit mean-square value."""
        fourier = _get_fourier()
        w = fourier.get_window(name, nwindow, norm=True)
        mean_square = np.mean(np.abs(w) ** 2)
        assert_allclose(mean_square, 1.0, rtol=1e-6)

    @given(
        name=window_names(),
        nwindow=window_sizes(min_size=8, max_size=256),
    )
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None
    )
    def test_window_non_negative_no_fftshift(self, name, nwindow):
        """Property: Standard windows are non-negative (before fftshift)."""
        fourier = _get_fourier()
        w = fourier.get_window(name, nwindow, fftshift=False, norm=False)
        # blackman's endpoints are zero up to roundoff
        assert np.all(w >= -1e-10)

    @given(
        name=window_names(),
        nwindow=window_sizes(min_size=8, max_size=256),
        dtype=st.sampled_from([np.float32, np.float64]),
    )
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None
    )
    def test_window_dtype(self, name, nwindow, dtype):
        """Property: Window dtype matches requested dtype."""
        fourier = _get_fourier()
        w = fourier.get_window(name, nwindow, dtype=dtype)
        assert w.dtype == dtype

    @given(
        name=window_names(),
        nwindow=st.integers(min_value=8, max_value=128).filter(lambda n: n % 2 == 0),
    )
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None
    )
    def test_fftshift_preserves_energy(self, name, nwindow):
        """Property: fftshift=True preserves window energy."""
        fourier = _get_fourier()
        w_normal = fourier.get_window(name, nwindow, fftshift=False, norm=False)
        w_shifted = fourier.get_window(name, nwindow, fftshift=True, norm=False)

        energy_normal = np.sum(np.abs(w_normal) ** 2)
        energy_shifted = np.sum(np.abs(w_shifted) ** 2)
        assert_allclose(energy_normal, energy_shifted, rtol=1e-6)

    @given(
        name=window_names(),
        nwindow=window_sizes(min_size=8, max_size=256),
    )
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None
    )
    def test_unnormalized_window_has_max_one(self, name, nwindow):
        """Property: Unnormalized window has maximum value of 1."""
        fourier = _get_fourier()
        w = fourier.get_window(name, nwindow, norm=False, fftshift=False)
        assert np.max(w) <= 1.0 + 1e-10


class TestFftfreqProperties:
    """Properties: FFT frequency array generation."""

    @given(
        nfft=fft_sizes(min_size=4, max_size=1024),
        fs=sample_rates(min_rate=1e3, max_rate=100e6),
    )
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None
    )
    def test_fftfreq_length(self, nfft, fs):
        """Property: fftfreq returns array of length nfft."""
        fourier = _get_fourier()
        freqs = fourier.fftfreq(nfft, fs)
        assert len(freqs) == nfft

    @given(
        nfft=fft_sizes(min_size=4, max_size=1024),
        fs=sample_rates(min_rate=1e3, max_rate=100e6),
    )
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None
    )
    def test_fftfreq_spacing(self, nfft, fs):
        """Property: Frequency spacing is fs/nfft."""
        fourier = _get_fourier()
        freqs = fourier.fftfreq(nfft, fs)
        expected_spacing = fs / nfft
        actual_spacing = np.diff(freqs)
        assert_allclose(actual_spacing, expected_spacing, rtol=1e-10)

    @given(
        nfft=st.integers(min_value=4, max_value=512).filter(lambda n: n % 2 == 0),
        fs=sample_rates(min_rate=1e3, max_rate=100e6),
    )
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None
    )
    def test_fftfreq_symmetry_even(self, nfft, fs):
        """Property: For even nfft, frequencies are symmetric around 0."""
        fourier = _get_fourier()
        freqs = fourier.fftfreq(nfft, fs)
        # For even nfft, we have -fs/2 but not +fs/2
        assert_allclose(freqs[0], -fs / 2, rtol=1e-10)
        assert_allclose(freqs[1 : nfft // 2], -freqs[-1 : nfft // 2 : -1], rtol=1e-10)

    @given(
        nfft=st.integers(min_value=5, max_value=511).filter(lambda n: n % 2 == 1),
        fs=sample_rates(min_rate=1e3, max_rate=100e6),
    )
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None
    )
    def test_fftfreq_symmetry_odd(self, nfft, fs):
        fourier = _get_fourier()
        freqs = fourier.fftfreq(nfft, fs)
        assert freqs[nfft // 2] == 0
        assert_allclose(freqs, -freqs[::-1], rtol=1e-10)

    @given(
        nfft=fft_sizes(min_size=4, max_size=512),
        fs=sample_rates(min_rate=1e3, max_rate=100e6),
    )
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None
    )
    def test_fftfreq_dtype(self, nfft, fs):
        """Property: fftfreq respects dtype argument."""
        fourier = _get_fourier()
        freqs_32 = fourier.fftfreq(nfft, fs, dtype='float32')
        freqs_64 = fourier.fftfreq(nfft, fs, dtype='float64')
        assert freqs_32.dtype == np.float32
        assert freqs_64.dtype == np.float64


class TestOaconvolveProperties:
    """Properties: Overlap-add convolution."""

    @given(
        x=real_waveforms(min_size=64, max_size=512, dtype=np.float64),
        kernel_size=st.integers(min_value=1, max_value=31).filter(lambda n: n % 2 == 1),
    )
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture],
        max_examples=50,
        deadline=None,
    )
    def test_oaconvolve_identity_kernel(self, x, kernel_size):
        """Property: Convolution with delta function is identity (shifted)."""
        fourier = _get_fourier()
        kernel = np.zeros(kernel_size, dtype=x.dtype)
        kernel[kernel_size // 2] = 1.0

        result = fourier.oaconvolve(x, kernel, mode='same')

        # atol for samples near zero, where rtol is meaningless
        assert_allclose(result, x, rtol=1e-8, atol=1e-12)

    @given(
        x=real_waveforms(min_size=64, max_size=256, dtype=np.float64),
        scale=st.floats(min_value=0.1, max_value=10.0, allow_nan=False),
    )
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture],
        max_examples=50,
        deadline=None,
    )
    def test_oaconvolve_linearity_scaling(self, x, scale):
        """Property: Convolution is linear (scaling)."""
        fourier = _get_fourier()
        kernel = np.array([0.25, 0.5, 0.25], dtype=x.dtype)

        result_scaled_input = fourier.oaconvolve(scale * x, kernel, mode='same')
        result_scaled_output = scale * fourier.oaconvolve(x, kernel, mode='same')

        # atol for samples near zero, where rtol is meaningless
        assert_allclose(
            result_scaled_input, result_scaled_output, rtol=1e-6, atol=1e-12
        )

    @given(
        x1=real_waveforms(min_size=64, max_size=256, dtype=np.float64),
    )
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture],
        max_examples=50,
        deadline=None,
    )
    def test_oaconvolve_commutativity(self, x1):
        """Property: Convolution is commutative for same-sized inputs."""
        fourier = _get_fourier()
        rng = np.random.default_rng(42)
        x2 = rng.standard_normal(15).astype(x1.dtype)

        result1 = fourier.oaconvolve(x1, x2, mode='full')
        result2 = fourier.oaconvolve(x2, x1, mode='full')

        # atol for samples near zero, where rtol is meaningless
        assert_allclose(result1, result2, rtol=1e-8, atol=1e-12)

    @given(
        x=real_waveforms(min_size=64, max_size=256, dtype=np.float64),
    )
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture],
        max_examples=50,
        deadline=None,
    )
    def test_oaconvolve_output_length_full(self, x):
        """Property: Full convolution output length is len(x) + len(kernel) - 1."""
        fourier = _get_fourier()
        kernel = np.array([0.25, 0.5, 0.25], dtype=x.dtype)

        result = fourier.oaconvolve(x, kernel, mode='full')

        expected_length = len(x) + len(kernel) - 1
        assert len(result) == expected_length


class TestResampleProperties:
    """Properties: Frequency-domain resampling."""

    @given(
        x=complex_waveforms(min_size=64, max_size=512, dtype=np.complex128),
    )
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture],
        max_examples=50,
        deadline=None,
    )
    def test_resample_identity(self, x):
        """Property: Resampling to same size is identity."""
        fourier = _get_fourier()
        result = fourier.resample(x, len(x))
        assert_allclose(result, x, rtol=1e-10)

    @given(
        scale=st.floats(min_value=0.5, max_value=2.0, allow_nan=False),
    )
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture],
        max_examples=50,
        deadline=None,
    )
    def test_resample_scale_parameter(self, scale):
        """Property: Scale parameter multiplies output."""
        fourier = _get_fourier()
        rng = np.random.default_rng(42)
        x = (rng.standard_normal(128) + 1j * rng.standard_normal(128)).astype(
            np.complex128
        )
        num_out = len(x) // 2

        result_unscaled = fourier.resample(x, num_out, scale=1.0)
        result_scaled = fourier.resample(x, num_out, scale=scale)

        # atol for samples near zero, where rtol is meaningless
        assert_allclose(result_scaled, scale * result_unscaled, rtol=1e-8, atol=1e-15)

    @given(
        x=complex_waveforms(min_size=64, max_size=512),
        ratio=st.sampled_from([0.5, 2]),
    )
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture],
        max_examples=50,
        deadline=None,
    )
    def test_resample_length_and_dtype(self, x, ratio):
        """Property: Resampled output has the requested length and the input dtype."""
        fourier = _get_fourier()
        num_out = int(len(x) * ratio)
        result = fourier.resample(x, num_out)
        assert len(result) == num_out
        assert result.dtype == x.dtype


class TestStftProperties:
    """Properties: Short-time Fourier transform."""

    @given(
        scale=st.floats(min_value=0.5, max_value=2.0, allow_nan=False),
    )
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture],
        max_examples=30,
        deadline=None,
    )
    def test_stft_linearity(self, scale):
        """Property: STFT is linear (scaling)."""
        fourier = _get_fourier()
        rng = np.random.default_rng(42)
        x = (rng.standard_normal(512) + 1j * rng.standard_normal(512)).astype(
            np.complex128
        )

        _, _, X1 = fourier.stft(
            x,
            fs=1e6,
            window='hamming',
            nperseg=64,
            noverlap=0,
            truncate=True,
        )

        scaled_x = (scale * x).astype(np.complex128)
        _, _, X2 = fourier.stft(
            scaled_x,
            fs=1e6,
            window='hamming',
            nperseg=64,
            noverlap=0,
            truncate=True,
        )

        assert_allclose(X2, scale * X1, rtol=1e-6)

    @given(
        x=complex_waveforms(min_size=64, max_size=256, dtype=np.complex64),
    )
    @settings(
        suppress_health_check=[
            HealthCheck.function_scoped_fixture,
            HealthCheck.data_too_large,
        ],
        max_examples=30,
        deadline=None,
    )
    def test_stft_time_segments(self, x):
        """Property: Number of time segments is correct."""
        fourier = _get_fourier()
        nperseg = 64
        noverlap = 0

        freqs, times, X = fourier.stft(
            x,
            fs=1e6,
            window='hamming',
            nperseg=nperseg,
            noverlap=noverlap,
            truncate=True,
        )

        step = nperseg - noverlap
        expected_segments = len(x) // step
        assert X.shape[0] == expected_segments


class TestSpectrogramProperties:
    """Properties: Power spectrogram computation."""

    @given(
        x=complex_waveforms(min_size=256, max_size=512),
        params=stft_parameters(),
    )
    @settings(
        suppress_health_check=[
            HealthCheck.function_scoped_fixture,
            HealthCheck.data_too_large,
        ],
        max_examples=50,
        deadline=None,
    )
    def test_spectrogram_output_shape(self, x, params):
        """Property: stft and spectrogram outputs have the right shape, dtype and
        frequency axis."""
        fourier = _get_fourier()
        # min_size=256 guarantees len(x) >= max(nperseg)=256

        _, _, X = fourier.stft(x, truncate=True, **params)
        freqs, times, Sxx = fourier.spectrogram(x, truncate=True, **params)

        assert X.shape == Sxx.shape
        assert Sxx.shape[1] == params['nperseg']
        assert len(freqs) == params['nperseg']
        assert len(times) == Sxx.shape[0]
        assert X.dtype == x.dtype
        assert Sxx.dtype == np.finfo(x.dtype).dtype
        assert_array_equal(freqs, fourier.fftfreq(params['nperseg'], params['fs']))

    @given(
        x=complex_waveforms(min_size=128, max_size=256, dtype=np.complex64),
    )
    @settings(
        suppress_health_check=[
            HealthCheck.function_scoped_fixture,
            HealthCheck.data_too_large,
        ],
        max_examples=30,
        deadline=None,
    )
    def test_spectrogram_equals_stft_power_squared(self, x):
        """Property: Spectrogram equals |STFT with norm='power'|²."""
        fourier = _get_fourier()
        fs = 1e6
        nperseg = 64

        _, _, X = fourier.stft(
            x,
            fs=fs,
            window='hamming',
            nperseg=nperseg,
            noverlap=0,
            truncate=True,
            norm='power',
        )

        _, _, Sxx = fourier.spectrogram(
            x,
            fs=fs,
            window='hamming',
            nperseg=nperseg,
            noverlap=0,
            truncate=True,
        )

        assert np.isrealobj(Sxx)
        assert np.all(Sxx >= 0)
        expected = np.abs(X) ** 2
        assert_allclose(Sxx, expected, rtol=1e-5)


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_get_window_minimum_size(self):
        """Test get_window with minimum valid size."""
        fourier = _get_fourier()
        w = fourier.get_window('hamming', 2)
        assert len(w) == 2

    def test_fftfreq_minimum_size(self):
        """Test fftfreq with minimum valid size."""
        fourier = _get_fourier()
        freqs = fourier.fftfreq(2, 1e6)
        assert len(freqs) == 2

    def test_spectrogram_zero_input(self):
        """Test spectrogram with zero input."""
        fourier = _get_fourier()
        x = np.zeros(256, dtype=np.complex64)

        freqs, times, Sxx = fourier.spectrogram(
            x,
            fs=1e6,
            window='hamming',
            nperseg=64,
            noverlap=0,
            truncate=True,
        )

        assert_allclose(Sxx, 0, atol=1e-10)

    @pytest.mark.parametrize(
        'size, noverlap, nseg',
        [(128, 0, 2), (100, 0, None), (96, 32, 2), (100, 32, None)],
    )
    def test_stft_truncate_false_requires_whole_segments(self, size, noverlap, nseg):
        fourier = _get_fourier()
        x = np.ones(size, dtype=np.complex64)
        kws = dict(
            fs=1e6, window='hamming', nperseg=64, noverlap=noverlap, truncate=False
        )
        if nseg is None:
            with pytest.raises(ValueError, match=f'size {size}'):
                fourier.stft(x, **kws)
        else:
            _, _, X = fourier.stft(x, **kws)
            assert X.shape == (nseg, 64)

    def test_stft_with_overlap(self):
        """Test STFT with 50% overlap."""
        fourier = _get_fourier()
        rng = np.random.default_rng(42)
        x = (rng.standard_normal(256) + 1j * rng.standard_normal(256)).astype(
            np.complex64
        )

        freqs, times, X = fourier.stft(
            x,
            fs=1e6,
            window='hamming',
            nperseg=64,
            noverlap=32,
            truncate=True,
        )

        # (len(x) - nperseg) // (nperseg - noverlap) + 1 = 7 segments
        assert X.shape[0] == 7
        assert X.shape[1] == 64

    def test_resample_extreme_downsample(self):
        """Test resample with extreme downsampling."""
        fourier = _get_fourier()
        rng = np.random.default_rng(42)
        x = (rng.standard_normal(256) + 1j * rng.standard_normal(256)).astype(
            np.complex128
        )

        result = fourier.resample(x, 16)
        assert len(result) == 16
        assert result.dtype == x.dtype

    def test_resample_extreme_upsample(self):
        """Test resample with extreme upsampling."""
        fourier = _get_fourier()
        rng = np.random.default_rng(42)
        x = (rng.standard_normal(64) + 1j * rng.standard_normal(64)).astype(
            np.complex128
        )

        result = fourier.resample(x, 512)
        assert len(result) == 512
        assert result.dtype == x.dtype


# Roundoff model for the cross-backend and far-bin tests. Per FFT pass the rms error
# relative to the output rms is c*eps*sqrt(log2 N), eps the unit roundoff (Gentleman &
# Sande 1966; FFTW accuracy notes), and each elementwise rounding adds (eps/sqrt(3))**2
# of error variance. c was fitted with chores/tests/measure_fft_accuracy.py: pocketfft
# gives 0.55-0.65 (1.2 for sizes with a prime factor >= 128); cuFFT on a Jetson TX2i
# reaches 1.9 at N=512 and 2.1 for Bluestein sizes. Errors of independent backends add
# in quadrature (measured 0.83-1.0).
FFT_ROUNDOFF_C = 2.2
ROUNDOFF_SAFETY = 3
# An FFT of a tone concentrates roundoff in a few bins instead of spreading it evenly:
# measured up to 6.7 units of roundoff of the tone amplitude at the tone (cuFFT, N=512)
# and 3.9 in a far bin (cuFFT, N=1024), against ~2 for pocketfft.
TONE_PEAK_ROUNDOFF = 8


def roundoff_rms(dtype, nffts, n_elementwise=0):
    """expected rms roundoff error of one backend, relative to the output rms"""
    eps = np.finfo(dtype).eps / 2
    var = n_elementwise * (eps / np.sqrt(3)) ** 2
    var += sum((FFT_ROUNDOFF_C * eps * np.sqrt(np.log2(n))) ** 2 for n in nffts)
    return float(np.sqrt(var))


def single_backend_rms(dtype, nffts, n_elementwise=0):
    """rms tolerance on one backend's error against an exact reference, relative to
    the output rms"""
    return ROUNDOFF_SAFETY * roundoff_rms(dtype, nffts, n_elementwise)


def cross_backend_rms(dtype, nffts, n_elementwise=0):
    """rms tolerance on the difference between two backends, relative to output rms"""
    return np.sqrt(2) * single_backend_rms(dtype, nffts, n_elementwise)


def peak_factor(size):
    """max/rms ratio of `size` complex gaussian errors, with 2x margin on the tail"""
    return 2 * np.sqrt(np.log(size))


def _rms(x):
    return float(np.sqrt(np.mean(np.abs(x) ** 2)))


def rms_tolerance_dBc(sigma):
    """express an rms amplitude tolerance relative to the output rms as error power
    relative to the signal, in dBc"""
    return 20 * np.log10(sigma)


def level_tolerance_dB(sigma, power=False):
    """express a relative tolerance on an output as the uncertainty of its level in dB.

    `sigma` bounds an amplitude ratio (level 20*log10|x|) unless `power` is True
    (level 10*log10 x, e.g. spectrogram bins).
    """
    return (10 if power else 20) * np.log10(1 + sigma)


def peak_level_tolerance_dB(sigma, size, power=False):
    """express the peak tolerance implied by `sigma` over `size` outputs as a level
    uncertainty in dB"""
    return level_tolerance_dB(peak_factor(size) * sigma, power=power)


def tone_peak_roundoff(dtype):
    """bound on structured roundoff in any one bin, relative to a tone's amplitude"""
    return ROUNDOFF_SAFETY * TONE_PEAK_ROUNDOFF * np.finfo(dtype).eps / 2


def far_bin_floor_dBc(sigma, nfft, size=None, dtype=np.complex64):
    """express the roundoff tolerance in bins away from a bin-centered tone, relative
    to the tone, in dBc.

    The tone occupies one bin while roundoff spreads evenly over all `nfft` bins. With
    `size`, the result is the peak tolerance over that many far bins, which is the
    larger of the white-noise tail and the structured `tone_peak_roundoff`.
    """
    if size is None:
        return rms_tolerance_dBc(sigma / np.sqrt(nfft))
    white = peak_factor(size) * sigma / np.sqrt(nfft)
    return rms_tolerance_dBc(max(white, tone_peak_roundoff(dtype)))


def bin_centered_tone(nfft, bin_fraction, nseg, dtype=np.complex64):
    """a unit tone with an integer number of cycles per segment, over `nseg` segments"""
    k = int(round(bin_fraction * (nfft - 1))) - nfft // 2
    n = np.arange(nseg * nfft)
    return np.exp(2j * np.pi * k * n / nfft).astype(dtype)


class TestToneFarBinFloor:
    """Roundoff in the bins away from a unit, bin-centered complex64 tone.

    With an integer number of cycles per segment (and a rectangular window for the
    STFT) the exact spectrum occupies one bin, so every other bin measures roundoff
    alone against a float64 reference of the same float32-quantized input. For the
    time-domain outputs of resample and oaconvolve the bins are those of the error
    spectrum, taken in float64 so the analysis adds no roundoff of its own.
    """

    NSEG = 4
    KERNEL_TAPS = 400

    @staticmethod
    def _stft(x, nfft):
        fourier = _get_fourier()
        return fourier.stft(x, fs=1.0, window='rect', nperseg=nfft, noverlap=0)[2]

    def _check_far_bins(self, X, X_ref, nfft):
        err = np.asarray(X).astype(np.complex128) - X_ref
        peak_bin = int(np.argmax(np.abs(X_ref[0])))
        peak = np.abs(X_ref[0, peak_bin])
        far = np.ones(nfft, dtype=bool)
        far[peak_bin] = False
        err_far = err[:, far]

        # window/nfft and the window multiply, then fft(nfft)
        sigma = single_backend_rms(np.complex64, [nfft], n_elementwise=2)
        scale = _rms(X_ref)
        assert _rms(err_far) < sigma * scale, (
            f'far-bin rms roundoff above {far_bin_floor_dBc(sigma, nfft):.1f} dBc'
        )
        white = peak_factor(err_far.size) * sigma * scale
        structured = tone_peak_roundoff(np.complex64) * peak
        assert np.abs(err_far).max() < max(white, structured), (
            f'far-bin peak roundoff above '
            f'{far_bin_floor_dBc(sigma, nfft, err_far.size):.1f} dBc'
        )

    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None
    )
    @given(
        nfft=st.sampled_from([64, 256, 1024, 4096]),
        bin_fraction=st.floats(min_value=0, max_value=1),
    )
    def test_far_bins_numpy_float32(self, nfft, bin_fraction):
        x = bin_centered_tone(nfft, bin_fraction, self.NSEG)
        X_ref = self._stft(x.astype(np.complex128), nfft)
        self._check_far_bins(self._stft(x, nfft), X_ref, nfft)

    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None
    )
    @given(
        nfft=st.sampled_from([64, 256, 1024, 4096]),
        bin_fraction=st.floats(min_value=0, max_value=1),
    )
    def test_far_bins_cupy_float32(self, cupy_available, nfft, bin_fraction):
        cp = cupy_available
        x = bin_centered_tone(nfft, bin_fraction, self.NSEG)
        X_ref = self._stft(x.astype(np.complex128), nfft)
        self._check_far_bins(self._stft(cp.asarray(x), nfft).get(), X_ref, nfft)

    def _check_error_spectrum(self, out, out_ref, sigma):
        """bound the spectrum of the roundoff error in a time-domain output of a unit
        tone, excluding the tone's own bin; `sigma` is the rms error model relative to
        the unit input"""
        err = np.fft.fft(np.asarray(out).astype(np.complex128) - out_ref)
        size = err.size
        far = np.ones(size, dtype=bool)
        far[int(np.argmax(np.abs(np.fft.fft(out_ref))))] = False
        err_far = err[far]

        # unnormalized fft: bin rms is sqrt(size) times the sample rms, and the unit
        # tone peaks at size
        rms_bins = np.sqrt(size) * sigma
        assert _rms(err_far) < rms_bins, (
            f'far-bin rms roundoff above {far_bin_floor_dBc(sigma, size):.1f} dBc'
        )
        white = peak_factor(err_far.size) * rms_bins
        structured = tone_peak_roundoff(np.complex64) * size
        assert np.abs(err_far).max() < max(white, structured), (
            f'far-bin peak roundoff above '
            f'{far_bin_floor_dBc(sigma, size, err_far.size):.1f} dBc'
        )

    def _resample_case(self, nfft, bin_fraction):
        # keep the tone inside the half band that survives downsampling by 2
        x = bin_centered_tone(nfft, 0.3 + 0.4 * bin_fraction, self.NSEG)
        num_out = x.size // 2
        # fftshift multiply, fft(N), ifft(N/2), ifftshift multiply
        sigma = single_backend_rms(np.complex64, [x.size, num_out], n_elementwise=2)
        out_ref = _get_fourier().resample(x.astype(np.complex128), num_out)
        return x, num_out, out_ref, sigma

    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None
    )
    @given(
        nfft=st.sampled_from([64, 256, 1024, 4096]),
        bin_fraction=st.floats(min_value=0, max_value=1),
    )
    def test_resample_far_bins_numpy_complex64(self, nfft, bin_fraction):
        x, num_out, out_ref, sigma = self._resample_case(nfft, bin_fraction)
        out = _get_fourier().resample(x, num_out)
        self._check_error_spectrum(out, out_ref, sigma)

    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None
    )
    @given(
        nfft=st.sampled_from([64, 256, 1024, 4096]),
        bin_fraction=st.floats(min_value=0, max_value=1),
    )
    def test_resample_far_bins_cupy_complex64(self, cupy_available, nfft, bin_fraction):
        cp = cupy_available
        x, num_out, out_ref, sigma = self._resample_case(nfft, bin_fraction)
        out = _get_fourier().resample(cp.asarray(x), num_out).get()
        self._check_error_spectrum(out, out_ref, sigma)

    def _oaconvolve_case(self, nfft, bin_fraction):
        x = bin_centered_tone(nfft, bin_fraction, self.NSEG)
        # a unit-gain lowpass; the tone may land in its stopband, so bounds are
        # anchored to the input tone rather than the output
        kernel = np.hanning(self.KERNEL_TAPS).astype(np.float32)
        kernel /= kernel.sum()
        # the overlap-add block is at most x.size long
        sigma = single_backend_rms(np.complex64, [x.size, x.size], n_elementwise=1)
        out_ref = _get_fourier().oaconvolve(
            x.astype(np.complex128), kernel.astype(np.float64), mode='same'
        )
        return x, kernel, out_ref, sigma

    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None
    )
    @given(
        nfft=st.sampled_from([64, 256, 1024, 4096]),
        bin_fraction=st.floats(min_value=0, max_value=1),
    )
    def test_oaconvolve_far_bins_numpy_complex64(self, nfft, bin_fraction):
        x, kernel, out_ref, sigma = self._oaconvolve_case(nfft, bin_fraction)
        out = _get_fourier().oaconvolve(x, kernel, mode='same')
        self._check_error_spectrum(out, out_ref, sigma)

    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None
    )
    @given(
        nfft=st.sampled_from([64, 256, 1024, 4096]),
        bin_fraction=st.floats(min_value=0, max_value=1),
    )
    def test_oaconvolve_far_bins_cupy_complex64(
        self, cupy_available, nfft, bin_fraction
    ):
        cp = cupy_available
        x, kernel, out_ref, sigma = self._oaconvolve_case(nfft, bin_fraction)
        out = (
            _get_fourier()
            .oaconvolve(cp.asarray(x), cp.asarray(kernel), mode='same')
            .get()
        )
        self._check_error_spectrum(out, out_ref, sigma)


class TestNumpyCupyCrossComparison:
    """Cross-comparison tests validating numpy and cupy produce close results.

    Tolerances for the FFT-based functions come from `cross_backend_rms`, parameterized
    by the FFT sizes and elementwise operations each function performs. Elementwise
    checks use an absolute tolerance anchored on the output rms, because the outputs
    are noise-like and a relative tolerance is meaningless in their smallest bins.
    """

    RTOL_FLOAT32 = 1e-5
    RTOL_FLOAT64 = 1e-12
    ATOL = 1e-10

    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=3000
    )
    @given(
        name=st.sampled_from(['hamming', 'hann', 'blackman', 'bartlett', 'flattop']),
        nwindow=st.sampled_from([8, 64, 129, 512]),
        dtype=st.sampled_from([np.float32, np.float64]),
    )
    def test_get_window_numpy_vs_cupy(self, cupy_available, name, nwindow, dtype):
        """Test get_window produces close results for numpy vs cupy."""
        cp = cupy_available
        fourier = _get_fourier()

        result_np = fourier.get_window(name, nwindow, dtype=dtype)

        result_cp = fourier.get_window(name, nwindow, dtype=dtype, xp=cp)
        result_cp_np = result_cp.get()

        rtol = {np.float32: self.RTOL_FLOAT32, np.float64: self.RTOL_FLOAT64}[dtype]
        assert_allclose(result_cp_np, result_np, rtol=rtol, atol=self.ATOL)

    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        nfft=fft_sizes(min_size=8, max_size=512),
        fs=sample_rates(min_rate=1e3, max_rate=100e6),
    )
    def test_fftfreq_numpy_vs_cupy_float64(self, cupy_available, nfft, fs):
        """Test fftfreq produces close results for numpy vs cupy (float64)."""
        cp = cupy_available
        fourier = _get_fourier()

        result_np = fourier.fftfreq(nfft, fs, dtype=np.float64)

        result_cp = fourier.fftfreq(nfft, fs, dtype=np.float64, xp=cp)
        result_cp_np = result_cp.get()

        assert_allclose(result_cp_np, result_np, rtol=self.RTOL_FLOAT64, atol=self.ATOL)

    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=3000
    )
    @given(x=complex_waveforms(min_size=64, max_size=256, dtype=np.complex64))
    def test_resample_numpy_vs_cupy_complex64(self, cupy_available, x):
        """Test resample produces close results for numpy vs cupy (complex64)."""
        cp = cupy_available
        fourier = _get_fourier()

        num_out = len(x) // 2

        result_np = fourier.resample(x, num_out)

        x_cp = cp.asarray(x)
        result_cp = fourier.resample(x_cp, num_out)
        result_cp_np = result_cp.get()

        # fftshift multiply, fft(N), ifft(N/2), ifftshift multiply
        sigma = cross_backend_rms(x.dtype, [len(x), num_out], n_elementwise=2)
        scale = _rms(result_np)
        assert _rms(result_cp_np - result_np) < sigma * scale, (
            f'rms roundoff above {level_tolerance_dB(sigma):.1e} dB'
        )
        peak_dB = peak_level_tolerance_dB(sigma, result_np.size)
        assert_allclose(
            result_cp_np,
            result_np,
            rtol=0,
            atol=peak_factor(result_np.size) * sigma * scale,
            err_msg=f'peak above {peak_dB:.1e} dB',
        )

    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=3000
    )
    @given(x=complex_waveforms(min_size=256, max_size=512, dtype=np.complex64))
    def test_stft_numpy_vs_cupy_complex64(self, cupy_available, x):
        """Test stft produces close results for numpy vs cupy (complex64)."""
        cp = cupy_available
        fourier = _get_fourier()

        freqs_np, times_np, X_np = fourier.stft(
            x, fs=1e6, window='hamming', nperseg=64, noverlap=32, truncate=True
        )

        x_cp = cp.asarray(x)
        freqs_cp, times_cp, X_cp = fourier.stft(
            x_cp, fs=1e6, window='hamming', nperseg=64, noverlap=32, truncate=True
        )
        freqs_cp_np = freqs_cp.get()
        times_cp_np = times_cp.get()
        X_cp_np = X_cp.get()

        assert_allclose(freqs_cp_np, freqs_np, rtol=self.RTOL_FLOAT32, atol=self.ATOL)
        assert_allclose(times_cp_np, times_np, rtol=self.RTOL_FLOAT32, atol=self.ATOL)

        # window/nfft and the window multiply, then fft(nperseg)
        sigma = cross_backend_rms(x.dtype, [64], n_elementwise=2)
        scale = _rms(X_np)
        assert _rms(X_cp_np - X_np) < sigma * scale, (
            f'rms roundoff above {level_tolerance_dB(sigma):.1e} dB'
        )
        peak_dB = peak_level_tolerance_dB(sigma, X_np.size)
        assert_allclose(
            X_cp_np,
            X_np,
            rtol=0,
            atol=peak_factor(X_np.size) * sigma * scale,
            err_msg=f'peak above {peak_dB:.1e} dB',
        )

    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=3000
    )
    @given(x=complex_waveforms(min_size=256, max_size=512, dtype=np.complex64))
    def test_spectrogram_numpy_vs_cupy_complex64(self, cupy_available, x):
        """Test spectrogram produces close results for numpy vs cupy (complex64)."""
        cp = cupy_available
        fourier = _get_fourier()

        freqs_np, times_np, Sxx_np = fourier.spectrogram(
            x, fs=1e6, window='hamming', nperseg=64, noverlap=32, truncate=True
        )

        x_cp = cp.asarray(x)
        freqs_cp, times_cp, Sxx_cp = fourier.spectrogram(
            x_cp, fs=1e6, window='hamming', nperseg=64, noverlap=32, truncate=True
        )
        freqs_cp_np = freqs_cp.get()
        times_cp_np = times_cp.get()
        Sxx_cp_np = Sxx_cp.get()

        assert_allclose(freqs_cp_np, freqs_np, rtol=self.RTOL_FLOAT32, atol=self.ATOL)
        assert_allclose(times_cp_np, times_np, rtol=self.RTOL_FLOAT32, atol=self.ATOL)

        # as stft, plus the |X|**2 rounding. For noise-like input the relative rms
        # error of power equals that of amplitude, but the error in one bin grows with
        # sqrt(Sxx), so atol covers the small bins and rtol the large ones.
        sigma = cross_backend_rms(x.dtype, [64], n_elementwise=3)
        scale = _rms(Sxx_np)
        assert _rms(Sxx_cp_np - Sxx_np) < sigma * scale, (
            f'rms roundoff above {level_tolerance_dB(sigma, power=True):.1e} dB'
        )
        peak_dB = peak_level_tolerance_dB(sigma, Sxx_np.size, power=True)
        assert_allclose(
            Sxx_cp_np,
            Sxx_np,
            rtol=4 * sigma,
            atol=peak_factor(Sxx_np.size) * sigma * scale,
            err_msg=f'peak above {peak_dB:.1e} dB',
        )

    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=3000
    )
    @given(x=real_waveforms(min_size=64, max_size=256))
    def test_oaconvolve_numpy_vs_cupy(self, cupy_available, x):
        """Test oaconvolve produces close results for numpy vs cupy."""
        cp = cupy_available
        fourier = _get_fourier()

        kernel = np.array([0.25, 0.5, 0.25], dtype=x.dtype)

        result_np = fourier.oaconvolve(x, kernel, mode='same')

        x_cp = cp.asarray(x)
        kernel_cp = cp.asarray(kernel)
        result_cp = fourier.oaconvolve(x_cp, kernel_cp, mode='same')
        result_cp_np = result_cp.get()

        # overlap-add blocks are at most len(x) long; anchoring to the input rms
        # accounts for the kernel's gain
        sigma = cross_backend_rms(x.dtype, [len(x), len(x)], n_elementwise=1)
        scale = _rms(x)
        assert _rms(result_cp_np - result_np) < sigma * scale, (
            f'rms roundoff above {level_tolerance_dB(sigma):.1e} dB'
        )
        peak_dB = peak_level_tolerance_dB(sigma, x.size)
        assert_allclose(
            result_cp_np,
            result_np,
            rtol=0,
            atol=peak_factor(x.size) * sigma * scale,
            err_msg=f'peak above {peak_dB:.1e} dB',
        )


# ---------------------------------------------------------------------------
# Coverage of the remaining fourier API: private helpers are reached through
# striqt.waveform.lib.fourier, still imported lazily
# ---------------------------------------------------------------------------

PROPERTY = settings(
    suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None
)

# Overlap-add reconstruction of a tone is limited by the COLA window's passband
# ripple after spectral truncation, not by roundoff. Measured for a hamming window at
# nfft=256: the rms level is within 0.015 dB and the per-sample envelope ripple within
# 0.25 dB (0.47 dB for a tone in the FIR transition band). Engineering bounds with margin:
COLA_LEVEL_DB = 0.05
COLA_RIPPLE_DB = 0.5

# scipy.signal.firwin2 with a hamming window reaches about -53 dB in the stopband,
# and its passband ripple is far below that; bound both at -40 dB
FIR_LEAKAGE = 1e-2
# _stft_fir_lowpass designs its FIR with a rectangular window, whose Gibbs sidelobes
# limit the stopband to about -21 dB
FIR_RECT_STOPBAND_DB = -20


def _get_lib_fourier():
    from striqt.waveform.lib import fourier

    return fourier


def unit_tone(size, fs, f0, dtype=np.complex64):
    """a unit complex tone at f0 Hz sampled at fs Hz"""
    t = np.arange(size) / fs
    return np.exp(2j * np.pi * f0 * t).astype(dtype)


def tone_frequency(x, fs):
    """the frequency of the strongest spectral line of x, to within fs/x.size"""
    X = np.fft.fft(np.asarray(x).astype(np.complex128))
    return float(np.fft.fftfreq(x.size, 1 / fs)[np.argmax(np.abs(X))])


def interior(x, pad, axis=-1):
    """x with `pad` samples trimmed from each end of `axis`"""
    return np.moveaxis(np.moveaxis(np.asarray(x), axis, 0)[pad:-pad], 0, axis)


def level_error_dB(x, ref=1.0):
    return 20 * np.log10(np.abs(x) / ref)


def assert_tone_level(y, ref=1.0):
    """check the rms level and per-sample envelope of a reconstructed unit tone"""
    assert abs(level_error_dB(_rms(y), ref)) < COLA_LEVEL_DB
    assert np.abs(level_error_dB(y, ref)).max() < COLA_RIPPLE_DB


class TestWindowExtras:
    @given(
        name=st.sampled_from(['hann', 'hamming']),
        nwindow=st.sampled_from([8, 16, 64]),
        nzero=st.sampled_from([2, 4, 6]),
    )
    @PROPERTY
    def test_center_zeros_pads_both_sides(self, name, nwindow, nzero):
        fourier = _get_fourier()
        w = fourier.get_window(
            name, nwindow, nzero=nzero, center_zeros=True, norm=False
        )
        ws = fourier.get_window(name, nwindow, norm=False)
        assert w.size == nwindow + nzero
        assert_array_equal(w[: nzero // 2], 0)
        assert_array_equal(w[nzero // 2 + nwindow :], 0)
        assert_array_equal(w[nzero // 2 : nzero // 2 + nwindow], ws)

    @given(nwindow=st.sampled_from([9, 15, 33]))
    @PROPERTY
    def test_fftshift_odd_size_is_complex_and_preserves_energy(self, nwindow):
        fourier = _get_fourier()
        w = fourier.get_window('hann', nwindow, fftshift=True, norm=False)
        w0 = fourier.get_window('hann', nwindow, fftshift=False, norm=False)
        assert np.iscomplexobj(w)
        assert_allclose(np.sum(np.abs(w) ** 2), np.sum(w0**2), rtol=1e-6)

    def test_dtype_none_keeps_float64(self):
        fourier = _get_fourier()
        assert fourier.get_window('hann', 16, dtype=None).dtype == np.float64

    def test_array_window_matches_named_window_in_stft(self):
        from scipy import signal

        fourier = _get_fourier()
        rng = np.random.default_rng(0)
        x = (rng.standard_normal(512) + 1j * rng.standard_normal(512)).astype(
            np.complex64
        )
        w = signal.windows.get_window('hann', 64, fftbins=True).astype(np.complex64)

        _, _, X_named = fourier.stft(x, fs=1.0, window='hann', nperseg=64)
        _, _, X_array = fourier.stft(x, fs=1.0, window=w, nperseg=64)

        # the two paths differ only in the order of the window and 1/nfft multiplies
        sigma = single_backend_rms(np.complex64, [64], n_elementwise=2)
        assert _rms(X_array - X_named) < sigma * _rms(X_named)

    def test_enbw_of_rect_and_hann(self):
        fourier = _get_fourier()
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
        fourier = _get_lib_fourier()
        target = 1.5
        param = fourier.find_window_param_from_enbw(window, target, nfft=1024)
        enbw = fourier.equivalent_noise_bandwidth((window, param), 1024)
        # bisect stops at |param error| <= atol=1e-6; the ENBW slope in the
        # parameter is of order 1, so this is loose by a factor of 100
        assert_allclose(enbw, target, rtol=1e-4)

    def test_find_window_param_errors(self):
        fourier = _get_lib_fourier()
        with pytest.raises(ValueError, match='greater than 1'):
            fourier.find_window_param_from_enbw('kaiser', 1.0)
        with pytest.raises(ValueError, match='window_name'):
            fourier.find_window_param_from_enbw('hann', 1.5)


class TestFftfreqExtras:
    @given(nfft=st.sampled_from([8, 9, 64, 65, 256]), fs=sample_rates())
    @PROPERTY
    def test_as_index_false_matches_exact(self, nfft, fs):
        fourier = _get_fourier()
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
        from conftest import _get_dask_array

        da = _get_dask_array()
        fourier = _get_fourier()
        freqs = fourier.fftfreq(16, 1e6, as_index=as_index, xp=da)
        assert isinstance(freqs, da.Array)
        assert_allclose(np.asarray(freqs), fourier.fftfreq(16, 1e6), rtol=1e-12)


class TestFrequencySlicing:
    @given(
        nfft=st.sampled_from([32, 64, 128]),
        bw_bins=st.integers(min_value=1, max_value=16),
        offset_bins=st.integers(min_value=-8, max_value=8),
    )
    @PROPERTY
    def test_slice_freqs_selects_band(self, nfft, bw_bins, offset_bins):
        fourier = _get_lib_fourier()
        fs = 1.0
        fres = fs / nfft
        bandwidth = bw_bins * fres
        offset = offset_bins * fres

        s = fourier._slice_freqs(nfft, fs, bandwidth, offset=offset)
        freqs = fourier.fftfreq(nfft, fs)[s]

        assert freqs.size == bw_bins
        # the selected bins are centered on the offset to within one bin
        assert abs(float(freqs.mean()) - offset) <= fres

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
        fourier = _get_lib_fourier()
        with pytest.raises(ValueError, match=match):
            fourier._slice_freqs(64, 1.0, **kws)

    def test_freq_band_edges(self):
        fourier = _get_lib_fourier()
        freqs = fourier.fftfreq(64, 1.0)
        assert fourier._freq_band_edges(64, 1.0, None, None) == (None, None)

        ilo, ihi = fourier._freq_band_edges(64, 1.0, -0.25, 0.25)
        assert ilo == np.searchsorted(freqs, -0.25)
        assert freqs[ilo] >= -0.25 and freqs[ilo - 1] < -0.25
        assert freqs[ihi] <= 0.25 and freqs[ihi + 1] > 0.25

        assert fourier._freq_band_edges(64, 1.0, -0.25, 0.6)[1] == 64


class TestTimeFftshift:
    @given(x=iq_waveforms(min_size=8, max_size=64, multiple_of=2, channels=3))
    @PROPERTY
    def test_alternates_sign_along_axis(self, x):
        fourier = _get_lib_fourier()
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
        fourier = _get_lib_fourier()
        x = np.ones((2, 8), dtype=np.complex64)
        with pytest.raises(ValueError, match='1-D or scalar'):
            fourier.time_fftshift(x, np.ones((2, 2)), axis=1)

    def test_ifftshift_is_fftshift(self):
        fourier = _get_lib_fourier()
        assert fourier.time_ifftshift is fourier.time_fftshift


class TestStftIstft:
    NFFT = 64

    def _roundtrip_sigma(self, dtype):
        # window multiply, fft, ifft, fftshift multiply and the overlap-add
        return single_backend_rms(dtype, [self.NFFT, self.NFFT], n_elementwise=3)

    @given(
        x=iq_waveforms(min_size=8 * NFFT, max_size=16 * NFFT, multiple_of=NFFT),
        window=st.sampled_from(['hann', 'hamming']),
        overwrite_x=st.booleans(),
    )
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture],
        deadline=None,
        max_examples=30,
    )
    def test_cola_roundtrip_is_identity(self, x, window, overwrite_x):
        fourier = _get_fourier()
        nfft = self.NFFT
        _, _, X = fourier.stft(
            x, fs=1.0, window=window, nperseg=nfft, noverlap=nfft // 2
        )
        y = fourier.istft(X, nfft=nfft, noverlap=nfft // 2, overwrite_x=overwrite_x)

        assert y.dtype == x.dtype
        assert y.shape == x.shape
        err = interior(y, nfft) - interior(x, nfft)
        sigma = self._roundtrip_sigma(x.dtype)
        assert _rms(err) < sigma * _rms(x), (
            f'rms error above {level_tolerance_dB(sigma):.1e} dB'
        )

    @given(x=iq_waveforms(min_size=8 * NFFT, max_size=16 * NFFT, multiple_of=NFFT))
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture],
        deadline=None,
        max_examples=30,
    )
    def test_no_overlap_roundtrip_is_proportional(self, x):
        """Without overlap, istft(stft(x)) reproduces x up to a constant gain."""
        fourier = _get_fourier()
        nfft = self.NFFT
        _, _, X = fourier.stft(x, fs=1.0, window='rect', nperseg=nfft, noverlap=0)
        y = fourier.istft(X, nfft=nfft, noverlap=0)

        gain = np.vdot(x, y) / np.vdot(x, x)
        sigma = self._roundtrip_sigma(x.dtype)
        assert _rms(y - gain * x) < sigma * _rms(y), (
            f'rms error above {level_tolerance_dB(sigma):.1e} dB'
        )

    @given(size=st.integers(min_value=NFFT, max_value=8 * NFFT))
    @PROPERTY
    def test_istft_size_trims_output(self, size):
        fourier = _get_fourier()
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
    def test_istft_out_buffer_is_used(self):
        fourier = _get_fourier()
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
        fourier = _get_lib_fourier()
        nfft = self.NFFT
        noverlap = nfft - nfft // hop_divisor
        rng = np.random.default_rng(0)
        x = (rng.standard_normal(8 * nfft) + 1j * rng.standard_normal(8 * nfft)).astype(
            np.complex64
        )
        window = np.ones(nfft, dtype=np.complex64)

        stacked = fourier._stack_stft_windows(x, window, nfft, noverlap, norm=None)
        assert stacked.shape == ((x.size - nfft) // (nfft - noverlap) + 1, nfft)
        y = fourier._unstack_stft_windows(stacked, noverlap=noverlap, nperseg=nfft)

        # only the overlap-add sums round
        eps = np.finfo(np.float32).eps / 2
        sigma = ROUNDOFF_SAFETY * hop_divisor * eps
        assert _rms(interior(y, nfft) - interior(x, nfft)) < sigma * _rms(x)

    def test_stack_windows_rejects_unknown_norm(self):
        fourier = _get_lib_fourier()
        x = np.ones(256, dtype=np.complex64)
        with pytest.raises(ValueError):
            fourier._stack_stft_windows(x, np.ones(64), 64, 32, norm='bogus')

    def test_stft_variants(self):
        fourier = _get_fourier()
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

    def test_stft_no_overlap_writes_in_place(self):
        fourier = _get_fourier()
        rng = np.random.default_rng(0)
        x = (rng.standard_normal(512) + 1j * rng.standard_normal(512)).astype(
            np.complex64
        )
        _, _, X_ref = fourier.stft(x, fs=1.0, window='hann', nperseg=64)

        y = x.copy()
        X = fourier.stft(
            y,
            fs=1.0,
            window='hann',
            nperseg=64,
            overwrite_x=True,
            return_axis_arrays=False,
        )
        assert np.shares_memory(X, y)
        assert_array_equal(X, X_ref)

        out = np.empty((8, 64), dtype=np.complex64)
        X = fourier.stft(
            x, fs=1.0, window='hann', nperseg=64, out=out, return_axis_arrays=False
        )
        assert np.shares_memory(X, out)
        assert_array_equal(X, X_ref)


class TestStftFrequencyEditing:
    NFFT = 64
    NSEG = 8

    def _stft(self, fs, dtype=np.complex64):
        fourier = _get_fourier()
        rng = np.random.default_rng(0)
        size = self.NFFT * self.NSEG
        x = (rng.standard_normal(size) + 1j * rng.standard_normal(size)).astype(dtype)
        freqs, _, Y = fourier.stft(
            x, fs=fs, window='rect', nperseg=self.NFFT, noverlap=0
        )
        return freqs, Y

    def test_zero_stft_by_freq_zeroes_outside_passband(self):
        fourier = _get_fourier()
        fs = 1e6
        freqs, Y = self._stft(fs)
        # the passband is half-open: a bin centered on the upper cutoff is zeroed
        inside = (freqs >= -fs / 4) & (freqs < fs / 4)

        Yz = fourier.zero_stft_by_freq(freqs, Y.copy(), passband=(-fs / 4, fs / 4))
        assert_array_equal(Yz[:, ~inside], 0)
        assert_array_equal(Yz[:, inside], Y[:, inside])

    def test_downsample_stft_keeps_center_band(self):
        fourier = _get_fourier()
        freqs, Y = self._stft(1e6)
        nfft_out = self.NFFT // 2
        lo = self.NFFT // 2 - nfft_out // 2

        freqs_out, Yo = fourier.downsample_stft(freqs, Y, nfft_out)
        assert Yo.shape == (self.NSEG, nfft_out)
        assert freqs_out.shape == (nfft_out,)
        assert_array_equal(Yo, Y[:, lo : lo + nfft_out])
        assert not np.shares_memory(Yo, Y)

    def test_downsample_stft_fast_path_returns_view(self):
        fourier = _get_fourier()
        freqs, Y = self._stft(1e6)
        _, Yo = fourier.downsample_stft(freqs, Y, self.NFFT // 2, out=Y)
        assert np.shares_memory(Yo, Y)

    def test_downsample_stft_zero_pads_when_upsampling(self):
        fourier = _get_fourier()
        freqs, Y = self._stft(1e6)
        nfft_out = 2 * self.NFFT
        pad = (nfft_out - self.NFFT) // 2

        _, Yo = fourier.downsample_stft(freqs, Y, nfft_out)
        assert Yo.shape == (self.NSEG, nfft_out)
        assert_array_equal(Yo[:, :pad], 0)
        assert_array_equal(Yo[:, pad + self.NFFT :], 0)
        assert_array_equal(Yo[:, pad : pad + self.NFFT], Y)

    def test_downsample_stft_passband_zeroing(self):
        fourier = _get_fourier()
        fs = 1e6
        freqs, Y = self._stft(fs)
        nfft_out = self.NFFT // 2
        # a passband half as wide as the output leaves a quarter zeroed on each side
        _, Yo = fourier.downsample_stft(freqs, Y, nfft_out, passband=(-fs / 8, fs / 8))
        quarter = nfft_out // 4
        assert_array_equal(Yo[:, :quarter], 0)
        assert_array_equal(Yo[:, -quarter:], 0)
        assert np.all(Yo[:, quarter:-quarter] != 0)

    def test_downsample_stft_preserves_frequency_step(self):
        fourier = _get_fourier()
        freqs, Y = self._stft(1e6)
        freqs_out, _ = fourier.downsample_stft(freqs, Y, self.NFFT // 2)
        assert_allclose(np.diff(freqs_out), freqs[1] - freqs[0], rtol=1e-12)

    def test_downsample_stft_out_buffer(self):
        fourier = _get_fourier()
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
        fourier = _get_fourier()
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
    @PROPERTY
    def test_find_downsample_copy_range_invariants(self, nfft_in, nfft_out, data):
        fourier = _get_lib_fourier()
        if data.draw(st.booleans()):
            start = end = None
        else:
            start = data.draw(st.integers(min_value=0, max_value=nfft_in - 1))
            end = data.draw(st.integers(min_value=start + 1, max_value=nfft_in))

        (out0, out1), (in0, in1), center = fourier._find_downsample_copy_range(
            nfft_in, nfft_out, start, end
        )
        assert out1 - out0 == in1 - in0
        assert 0 <= in0 <= in1 <= nfft_in
        assert 0 <= out0 <= out1 <= nfft_out
        assert in1 - in0 <= nfft_out
        # the copied block is centered in the output
        assert abs(out0 - (nfft_out - out1)) <= 1
        assert center == ((start or 0) + (nfft_in if end is None else end)) // 2

    def test_helpers(self):
        fourier = _get_lib_fourier()
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
        fourier = _get_fourier()
        nfft = 120
        nfft_out, noverlap, scale, pad_out = fourier.design_oafilter(
            40 * nfft, window=window, nfft_out=nfft, nfft=nfft, extend=False
        )
        assert nfft_out == nfft
        assert scale == overlap_scale
        assert noverlap == round(nfft * overlap_scale)
        assert pad_out == 0

    def test_design_oafilter_errors(self):
        fourier = _get_fourier()
        with pytest.raises(TypeError, match='window'):
            fourier.design_oafilter(
                1200, window='hann', nfft_out=120, nfft=120, extend=False
            )
        with pytest.raises(ValueError, match='% 3'):
            fourier.design_oafilter(
                1210, window='blackman', nfft_out=121, nfft=120, extend=False
            )
        with pytest.raises(ValueError, match='integer multiple of noverlap'):
            fourier.design_oafilter(
                1000, window='hamming', nfft_out=120, nfft=120, extend=False
            )

        pad_out = fourier.design_oafilter(
            1000, window='hamming', nfft_out=120, nfft=120, extend=True
        )[3]
        assert pad_out == 1000 % 60

        nfft_out = fourier.design_oafilter(
            1200, window='hamming', nfft_out=None, nfft=120, extend=False
        )[0]
        assert nfft_out == 120

    @given(
        fs_base=st.sampled_from([122.88e6, 125e6]),
        fs_target=st.sampled_from([7.68e6, 15.36e6, 20e6, 30.72e6, 61.44e6, 107.52e6]),
        window=st.sampled_from(['hamming', 'blackman']),
        shift=st.sampled_from([False, 'none', None, 'left', 'right']),
        bw=st.sampled_from([5e6, 10e6, 20e6]),
    )
    @PROPERTY
    def test_design_cola_resampler_consistency(
        self, fs_base, fs_target, window, shift, bw
    ):
        fourier = _get_fourier()
        design = fourier.design_cola_resampler(
            fs_base, fs_target, bw=bw, shift=shift, window=window
        )

        assert design['fs'] == design['fs_sdr'] <= fs_base
        # the SDR rate is an integer division of the base clock
        assert_allclose(fs_base / design['fs_sdr'], round(fs_base / design['fs_sdr']))
        assert_allclose(
            design['nfft'] / design['nfft_out'], design['fs_sdr'] / fs_target
        )

        divisor = _get_lib_fourier()._COLA_WINDOW_SIZE_DIVISOR[window]
        assert design['nfft'] % divisor == 0 and design['nfft_out'] % divisor == 0
        assert design['window'] == window

        lo, hi = design['passband']
        assert_allclose(hi - lo, bw)
        assert_allclose((hi + lo) / 2, design['lo_offset'])
        sign = {'left': -1, 'right': 1}.get(shift, 0)
        assert np.sign(design['lo_offset']) == sign

    def test_design_cola_resampler_options(self):
        fourier = _get_fourier()
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

        with_primes = fourier.design_cola_resampler(125e6, 15.36e6, avoid_primes=False)
        assert with_primes['nfft_out'] <= design['nfft_out']

        fs_sdr, fir = _get_lib_fourier().design_fir_resampler(125e6, 107.52e6)
        assert fs_sdr == pytest.approx(125e6)
        assert_allclose(fir['down'] / fir['up'], 125e6 / 107.52e6)

    @pytest.mark.parametrize(
        'kws, match',
        [
            ({'bw': np.inf, 'shift': 'left'}, 'analysis bandwidth'),
            ({'bw': 200e6}, 'exceeds Nyquist'),
            ({'bw': 10e6, 'shift': 'sideways'}, 'shift argument'),
        ],
    )
    def test_design_cola_resampler_errors(self, kws, match):
        fourier = _get_fourier()
        with pytest.raises(ValueError, match=match):
            fourier.design_cola_resampler(125e6, 15.36e6, **kws)

    def test_design_cola_resampler_rejects_irrational_ratio(self):
        fourier = _get_fourier()
        with pytest.raises(ValueError, match='no rational FFT sizes'):
            fourier.design_cola_resampler(125e6, 125e6 / np.pi)

    def test_design_cola_resampler_shift_needs_headroom(self):
        fourier = _get_fourier()
        with pytest.raises(ValueError, match='minimum'):
            fourier.design_cola_resampler(20e6, 15.36e6, bw=10e6, shift='left')

    @given(
        numtaps=st.sampled_from([401, 1001, 4001]),
        dtype=st.sampled_from(['float32', 'float64']),
    )
    @PROPERTY
    def test_design_fir_lpf(self, numtaps, dtype):
        fourier = _get_fourier()
        b = fourier.design_fir_lpf(10e6, 50e6, numtaps=numtaps, dtype=dtype)
        assert b.shape == (numtaps,)
        assert b.dtype == np.dtype(dtype)
        # least-squares passband ripple bounds the DC gain error
        assert abs(level_error_dB(b.sum())) < level_tolerance_dB(FIR_LEAKAGE)

    def test_fir_lowpass_fft_response(self):
        fourier = _get_lib_fourier()
        size = 256
        H = fourier._fir_lowpass_fft(size, 1.0, cutoff=0.1, transition=0.05)
        freqs = fourier.fftfreq(size, 1.0)
        assert H.shape == (size,) and H.dtype == np.complex64

        passband = np.abs(freqs) < 0.08
        stopband = np.abs(freqs) > 0.1 + 0.05 + 2 / size
        assert np.abs(np.abs(H[passband]) - 1).max() < FIR_LEAKAGE
        assert np.abs(H[stopband]).max() < FIR_LEAKAGE

    @pytest.mark.xfail(
        strict=True,
        reason='the cutoff=inf branch sets the impulse response to ones rather than '
        'an impulse, so the "all-pass" response is a single spectral line',
    )
    def test_fir_lowpass_fft_infinite_cutoff_is_allpass(self):
        fourier = _get_lib_fourier()
        H = fourier._fir_lowpass_fft(256, 1.0, cutoff=np.inf, transition=0.05)
        assert_allclose(np.abs(H), 1)


class TestOverlapAddFilters:
    FS = 1e6
    NFFT = 256
    NSEG = 40

    def _tone(self, f0, channels=None):
        x = unit_tone(self.NFFT * self.NSEG, self.FS, f0)
        return x if channels is None else np.tile(x, (channels, 1))

    @given(f0_bins=st.integers(min_value=-40, max_value=40))
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture],
        deadline=None,
        max_examples=20,
    )
    def test_oafilter_passes_in_band_tone(self, f0_bins):
        fourier = _get_fourier()
        f0 = f0_bins * self.FS / self.NFFT
        x = self._tone(f0)
        y = fourier.oafilter(
            x, fs=self.FS, nfft=self.NFFT, window='hamming', passband=(-0.2e6, 0.2e6)
        )
        assert y.shape == x.shape and y.dtype == x.dtype
        yi = interior(y, self.NFFT)
        assert_tone_level(yi)
        assert abs(tone_frequency(yi, self.FS) - f0) <= self.FS / yi.size

    def test_oafilter_rejects_out_of_band_tone(self):
        fourier = _get_fourier()
        x = self._tone(0.4e6)
        y = fourier.oafilter(
            x, fs=self.FS, nfft=self.NFFT, window='hamming', passband=(-0.2e6, 0.2e6)
        )
        yi = interior(y, self.NFFT)
        # the residual is the hamming sidelobe leakage (-53 dB); the window's
        # -22 dB edge pedestal leaves the ringing concentrated at the segment edges
        assert level_error_dB(_rms(yi)) < -40
        assert level_error_dB(yi).max() < -25

    def test_oafilter_downsample_length(self):
        fourier = _get_fourier()
        x = self._tone(0.05e6)
        nfft_out = self.NFFT // 2
        y = fourier.oafilter(
            x,
            fs=self.FS,
            nfft=self.NFFT,
            nfft_out=nfft_out,
            window='hamming',
            passband=(-0.2e6, 0.2e6),
        )
        assert y.shape == (round(x.size * nfft_out / self.NFFT),)

    @pytest.mark.xfail(
        strict=True,
        reason='the downsample_stft branch of oafilter doubles the tone amplitude',
    )
    def test_oafilter_downsample_preserves_level(self):
        fourier = _get_fourier()
        x = self._tone(0.05e6)
        nfft_out = self.NFFT // 2
        y = fourier.oafilter(
            x,
            fs=self.FS,
            nfft=self.NFFT,
            nfft_out=nfft_out,
            window='hamming',
            passband=(-0.2e6, 0.2e6),
        )
        assert_tone_level(interior(y, nfft_out))

    @given(
        f0_bins=st.integers(min_value=-20, max_value=20),
        shift_bins=st.sampled_from([0, 8, -8]),
        scale=st.sampled_from([1.0, 2.0]),
    )
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture],
        deadline=None,
        max_examples=20,
    )
    def test_oaresample_downsample_tone(self, f0_bins, shift_bins, scale):
        """As sensor.lib.compute.corrections._oaresample: 2-D (channel, sample)
        input on axis=1, a bin-grid frequency shift and a voltage scale."""
        fourier = _get_fourier()
        up, down = self.NFFT // 2, self.NFFT
        fs_out = self.FS * up / down
        f0 = f0_bins * self.FS / self.NFFT
        shift = shift_bins * self.FS / down
        x = self._tone(f0, channels=2)

        y = fourier.oaresample(
            x,
            up,
            down,
            self.FS,
            window='hamming',
            axis=1,
            frequency_shift=shift,
            scale=scale,
        )
        assert y.shape == (2, round(x.shape[1] * up / down))
        assert y.dtype == x.dtype
        yi = interior(y, up, axis=1)
        assert_tone_level(yi, scale)
        for row in yi:
            assert abs(tone_frequency(row, fs_out) - (f0 - shift)) <= fs_out / row.size

    def test_oaresample_filter_bandwidth(self):
        fourier = _get_fourier()
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
        fourier = _get_fourier()
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
        fourier = _get_fourier()
        x = self._tone(0.05e6, 1)
        with pytest.raises(ValueError, match=match):
            fourier.oaresample(x, up, down, self.FS, axis=1, frequency_shift=shift)


class TestResampleExtras:
    @given(x=complex_waveforms(min_size=64, max_size=256, dtype=np.complex128))
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture],
        deadline=None,
        max_examples=30,
    )
    def test_frequency_domain_input(self, x):
        """domain='freq' takes the spectrum as fft(time_fftshift(x)) and matches the
        time-domain path"""
        fourier = _get_lib_fourier()
        num = x.size // 2
        X = fourier.fft(fourier.time_fftshift(x), axis=0)
        y_time = fourier.resample(x, num)
        y_freq = fourier.resample(X.copy(), num, domain='freq')
        y_freq_inplace = fourier.resample(X, num, domain='freq', overwrite_x=True)
        sigma = cross_backend_rms(x.dtype, [x.size], n_elementwise=1)
        assert _rms(y_freq - y_time) < sigma * _rms(y_time)
        assert_array_equal(y_freq_inplace, y_freq)

    @given(shift=st.integers(min_value=-16, max_value=16))
    @PROPERTY
    def test_shift_moves_tone_when_downsampling(self, shift):
        fourier = _get_fourier()
        nfft_in, nfft_out = 256, 128
        k = 26
        x = unit_tone(nfft_in, 1.0, k / nfft_in)
        y = fourier.resample(x, nfft_out, shift=shift)
        # the band copied to the output starts `shift` bins higher, so the tone
        # lands `shift` bins lower in the output spectrum
        expected = (k - shift) / nfft_out
        assert abs(tone_frequency(y, 1.0) - expected) <= 1 / nfft_out

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
        fourier = _get_fourier()
        x = np.ones(256, dtype=np.complex64)
        with pytest.raises(ValueError, match=match):
            fourier.resample(x, **kws)

    def test_odd_length_rejected(self):
        fourier = _get_fourier()
        with pytest.raises(ValueError, match='even'):
            fourier.resample(np.ones(255, dtype=np.complex64), 128)

    @pytest.mark.xfail(
        strict=True,
        reason='upfirdn builds its keyword arguments from locals() after importing '
        '`signal`, which is then passed on to scipy',
    )
    def test_upfirdn_matches_scipy(self):
        from scipy import signal

        fourier = _get_lib_fourier()
        rng = np.random.default_rng(0)
        x = rng.standard_normal(64)
        h = np.array([0.5, 1.0, 0.5])
        assert_allclose(
            fourier.upfirdn(h, x, up=2, down=3), signal.upfirdn(h, x, up=2, down=3)
        )

    def test_max_cupy_fft_chunk_roundtrip(self):
        fourier = _get_fourier()
        previous = fourier.get_max_cupy_fft_chunk()
        try:
            _get_lib_fourier().set_max_cupy_fft_chunk(4096)
            assert fourier.get_max_cupy_fft_chunk() == 4096
        finally:
            _get_lib_fourier().set_max_cupy_fft_chunk(previous)
        assert fourier.get_max_cupy_fft_chunk() == previous


class TestNumpyCupyCrossComparisonExtras:
    """numpy-vs-cupy agreement for the STFT editing, filtering and resampling paths."""

    NFFT = 64

    def _check(self, result_cp, result_np, sigma, err_msg=''):
        result_cp_np = result_cp.get()
        scale = _rms(result_np)
        assert _rms(result_cp_np - result_np) < sigma * scale, (
            f'{err_msg} rms roundoff above {level_tolerance_dB(sigma):.1e} dB'
        )
        assert_allclose(
            result_cp_np,
            result_np,
            rtol=0,
            atol=peak_factor(result_np.size) * sigma * scale,
            err_msg=f'{err_msg} peak above '
            f'{peak_level_tolerance_dB(sigma, result_np.size):.1e} dB',
        )

    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None
    )
    @given(x=iq_waveforms(min_size=8 * NFFT, max_size=16 * NFFT, multiple_of=NFFT))
    def test_istft_roundtrip(self, cupy_available, x):
        cp = cupy_available
        fourier = _get_fourier()
        nfft = self.NFFT

        def roundtrip(x):
            _, _, X = fourier.stft(
                x, fs=1.0, window='hann', nperseg=nfft, noverlap=nfft // 2
            )
            return fourier.istft(X, nfft=nfft, noverlap=nfft // 2)

        sigma = cross_backend_rms(x.dtype, [nfft, nfft], n_elementwise=3)
        self._check(roundtrip(cp.asarray(x)), roundtrip(x), sigma)

    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None
    )
    @given(x=iq_waveforms(min_size=8, max_size=64, multiple_of=2, channels=3))
    def test_time_fftshift(self, cupy_available, x):
        cp = cupy_available
        fourier = _get_lib_fourier()
        scale = np.array([1, 2, 3], dtype=x.real.dtype)
        y_np = fourier.time_fftshift(x, scale, axis=1)
        y_cp = fourier.time_fftshift(cp.asarray(x), cp.asarray(scale), axis=1)
        assert_allclose(y_cp.get(), y_np, rtol=np.finfo(x.dtype).eps)

    def test_frequency_slicing(self, cupy_available):
        cp = cupy_available
        fourier = _get_fourier()
        x = np.arange(2 * 3 * 64, dtype=np.float32).reshape(2, 3, 64)

        t_np = fourier.truncate_freqs(x, 64, 1.0, 0.5, axis=2)
        t_cp = fourier.truncate_freqs(cp.asarray(x), 64, 1.0, 0.5, axis=2)
        assert_array_equal(t_cp.get(), t_np)

        x_cp = cp.asarray(x)
        fourier.null_lo(x, 64, 1.0, 0.25, axis=2)
        fourier.null_lo(x_cp, 64, 1.0, 0.25, axis=2)
        assert_array_equal(x_cp.get(), x)

    def test_stft_frequency_editing(self, cupy_available):
        cp = cupy_available
        fourier = _get_fourier()
        rng = np.random.default_rng(0)
        x = (rng.standard_normal(8 * 64) + 1j * rng.standard_normal(8 * 64)).astype(
            np.complex64
        )
        freqs, _, Y = fourier.stft(x, fs=1.0, window='rect', nperseg=64, noverlap=0)
        Y_cp = cp.asarray(Y)
        freqs_cp = cp.asarray(freqs)

        Yz_np = fourier.zero_stft_by_freq(freqs, Y.copy(), passband=(-0.25, 0.25))
        Yz_cp = fourier.zero_stft_by_freq(freqs_cp, Y_cp.copy(), passband=(-0.25, 0.25))
        assert_array_equal(Yz_cp.get(), Yz_np)

        fo_np, Yo_np = fourier.downsample_stft(freqs, Y, 32)
        fo_cp, Yo_cp = fourier.downsample_stft(freqs_cp, Y_cp, 32)
        assert_array_equal(Yo_cp.get(), Yo_np)
        assert_allclose(fo_cp.get(), fo_np, rtol=1e-12)

    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture],
        deadline=None,
        max_examples=10,
    )
    @given(f0_bins=st.integers(min_value=-20, max_value=20))
    def test_oafilter_and_oaresample(self, cupy_available, f0_bins):
        cp = cupy_available
        fourier = _get_fourier()
        fs, nfft = 1e6, 256
        x = np.tile(unit_tone(nfft * 40, fs, f0_bins * fs / nfft), (2, 1))

        y_np = fourier.oafilter(
            x[0], fs=fs, nfft=nfft, window='hamming', passband=(-0.2e6, 0.2e6)
        )
        y_cp = fourier.oafilter(
            cp.asarray(x[0]),
            fs=fs,
            nfft=nfft,
            window='hamming',
            passband=(-0.2e6, 0.2e6),
        )
        # window multiply, fft(nfft), zeroing, ifft(nfft), window multiply, overlap add
        sigma = cross_backend_rms(x.dtype, [nfft, nfft], n_elementwise=3)
        self._check(y_cp, y_np, sigma, 'oafilter')

        kws = {
            'window': 'hamming',
            'axis': 1,
            'frequency_shift': 8 * fs / nfft,
            'filter_bandwidth': 0.2e6,
        }
        y_np = fourier.oaresample(x, nfft // 2, nfft, fs, **kws)
        y_cp = fourier.oaresample(cp.asarray(x), nfft // 2, nfft, fs, **kws)
        # as oafilter, plus the FIR multiply and the output scaling
        sigma = cross_backend_rms(x.dtype, [nfft, nfft // 2], n_elementwise=5)
        self._check(y_cp, y_np, sigma, 'oaresample')

    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None
    )
    @given(x=iq_waveforms(min_size=256, max_size=1024, multiple_of=64, channels=4))
    def test_fft_out_buffer_and_chunking(self, cupy_available, x):
        cp = cupy_available
        fourier = _get_fourier()
        X_np = fourier.fft(x, axis=1)
        x_np_back = fourier.ifft(X_np, axis=1)

        previous = fourier.get_max_cupy_fft_chunk()
        try:
            # small enough that the 4-row input is split across several calls
            _get_lib_fourier().set_max_cupy_fft_chunk(x.shape[1])
            x_cp = cp.asarray(x)
            out = cp.empty_like(x_cp)
            X_cp = fourier.fft(x_cp, axis=1, out=out)
            assert X_cp is out or cp.shares_memory(X_cp, out)
            x_cp_back = fourier.ifft(X_cp, axis=1, out=cp.empty_like(x_cp))
        finally:
            _get_lib_fourier().set_max_cupy_fft_chunk(previous)

        n = x.shape[1]
        self._check(X_cp, X_np, cross_backend_rms(x.dtype, [n]), 'fft')
        self._check(x_cp_back, x_np_back, cross_backend_rms(x.dtype, [n, n]), 'ifft')

    def test_designs_on_cupy(self, cupy_available):
        cp = cupy_available
        fourier = _get_fourier()

        b_np = fourier.design_fir_lpf(10e6, 50e6, numtaps=401)
        b_cp = fourier.design_fir_lpf(10e6, 50e6, numtaps=401, xp=cp)
        assert isinstance(b_cp, cp.ndarray)
        assert_array_equal(b_cp.get(), b_np)

        f_np = fourier.fftfreq(65, 1e6, as_index=False)
        f_cp = fourier.fftfreq(65, 1e6, as_index=False, xp=cp)
        assert isinstance(f_cp, cp.ndarray)
        assert_allclose(f_cp.get(), f_np, rtol=1e-12)
