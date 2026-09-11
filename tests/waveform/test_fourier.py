"""Property-based tests for striqt.waveform.fourier using Hypothesis.

Covers get_window, fftfreq, oaconvolve, resample, stft and spectrogram: identities,
linearity, dtype and shape properties, numpy/cupy/dask compatibility, and roundoff
bounds derived from an FFT error model (numpy vs cupy, and the floor far from a tone).
"""

from __future__ import annotations

import numpy as np
import pytest
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

    @pytest.mark.parametrize('size, ok', [(128, True), (100, False)])
    def test_stft_truncate_false_requires_whole_segments(self, size, ok):
        fourier = _get_fourier()
        x = np.ones(size, dtype=np.complex64)
        kws = dict(fs=1e6, window='hamming', nperseg=64, noverlap=0, truncate=False)
        if ok:
            _, _, X = fourier.stft(x, **kws)
            assert X.shape == (size // 64, 64)
        else:
            with pytest.raises(ValueError, match='not a factor'):
                fourier.stft(x, **kws)

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
