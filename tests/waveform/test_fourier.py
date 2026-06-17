"""Property-based tests for Fourier transform functions using Hypothesis.

This module tests the following functions from striqt.waveform.fourier:
- get_window: Window function generation with optional zero-padding
- fftfreq: FFT frequency array generation with high precision
- oaconvolve: Overlap-add convolution
- resample: Frequency-domain resampling
- spectrogram: Power spectrogram computation
- stft: Short-time Fourier transform

Test Categories:
- Mathematical identities (Parseval's theorem, energy conservation)
- Algebraic properties (linearity, symmetry)
- Dtype preservation
- Shape consistency
- Multi-backend compatibility (numpy, cupy)
"""

from __future__ import annotations

import typing

import numpy as np
import pytest
from hypothesis import assume, given, settings, HealthCheck
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays, array_shapes
from numpy.testing import assert_allclose, assert_array_equal

# Import shared strategies and utilities from conftest
from conftest import (
    to_numpy,
    convert_array,
)

if typing.TYPE_CHECKING:
    from striqt.waveform.fourier import (
        fftfreq,
        get_window,
        oaconvolve,
        resample,
        spectrogram,
        stft,
    )



def _get_fourier():
    """Lazily import fourier module to avoid reifying scipy at test collection time."""
    from striqt.waveform import fourier
    return fourier


@pytest.fixture
def fourier_module():
    """Fixture that lazily imports the fourier module."""
    return _get_fourier()


# =============================================================================
# Hypothesis Strategies for Fourier Functions
# =============================================================================


def window_names():
    """Strategy for valid window function names.

    Note: 'rect' is a custom window registered by striqt, not scipy's 'boxcar'.
    """
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
    """Strategy for FFT sizes (prefer powers of 2 for efficiency)."""
    # Generate powers of 2 for efficient FFTs
    powers = st.integers(min_value=2, max_value=9).map(lambda p: 2**p)
    # Also allow some non-power-of-2 sizes
    arbitrary = st.integers(min_value=min_size, max_value=max_size).filter(
        lambda n: n % 2 == 0  # Must be even for resample
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
    allow_subnormal: bool = True,
):
    """Strategy for complex-valued waveform arrays.

    Specification:
        - Complex64 or complex128 dtype
        - Controlled magnitude to avoid overflow
        - 1-D or 2-D arrays
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
        # Ensure size is even for FFT operations
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

        # Generate real and imaginary parts separately for better control
        real_dtype = np.float32 if dt == np.complex64 else np.float64
        float_width = 32 if dt == np.complex64 else 64

        oom = draw(
            st.sampled_from([min_log_power, max_log_power])
        )
        lin_scale = np.asarray(10**(oom/2), dtype=dt)
        
        real = re_rng.normal(loc=0.0, scale=1.0, size=shape).astype(dt)
        imag = im_rng.normal(loc=0.0, scale=1.0, size=shape).astype(dt)

        return lin_scale * (real + 1j * imag)

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
        # Ensure size is even for FFT operations
        size = draw(st.integers(min_value=min_size // 2, max_value=max_size // 2)) * 2

        oom = draw(
            st.integers(min_value=min_log_power, max_value=max_log_power)
        )
        lin_scale = np.asarray(10**(oom/2), dtype=dt)

        real = re_rng.normal(loc=0.0, scale=1.0, size=(size,)).astype(dt)

        return lin_scale * (real + 1j * real).astype(dt)

    return _real_waveform()


def stft_parameters():
    """Strategy for valid STFT parameter combinations.

    Uses a limited set of windows to avoid cache eviction issues with the
    persistent LRU cache in get_window.
    """

    @st.composite
    def _stft_params(draw):
        # nperseg must be even and reasonable
        nperseg = draw(st.sampled_from([64, 128, 256]))
        # noverlap must be less than nperseg and compatible with COLA
        noverlap = draw(st.sampled_from([0, nperseg // 2]))
        # Sample rate
        fs = draw(sample_rates(min_rate=1e3, max_rate=10e6))
        # Use only common windows to avoid cache eviction issues
        window = draw(st.sampled_from(['hamming', 'hann', 'blackman']))

        return {
            'nperseg': nperseg,
            'noverlap': noverlap,
            'fs': fs,
            'window': window,
        }

    return _stft_params()


def resample_ratios():
    """Strategy for valid resample ratios (output_size / input_size)."""
    return st.sampled_from([
        0.5,    # Downsample by 2
        0.25,   # Downsample by 4
        1.0,    # No change
        2.0,    # Upsample by 2
        4.0,    # Upsample by 4
    ])


# =============================================================================
# get_window Tests
# =============================================================================


class TestGetWindowProperties:
    """Properties: Window function generation."""

    @given(
        name=window_names(),
        nwindow=st.sampled_from([8, 16, 32, 64, 128, 256, 512]),  # Use power-of-2 sizes
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
    def test_window_length_no_padding(self, name, nwindow):
        """Property: Window length equals requested size (no zero padding).

        Note: get_window is cached, so we use power-of-2 sizes to avoid cache
        collisions with nearby sizes.
        """
        fourier = _get_fourier()
        w = fourier.get_window(name, nwindow, nzero=0)
        assert len(w) == nwindow

    @given(
        name=window_names(),
        nwindow=window_sizes(min_size=8, max_size=256),
        nzero=st.integers(min_value=1, max_value=64),
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
    def test_window_with_zero_padding(self, name, nwindow, nzero):
        """Property: Zero-padded window has correct total length."""
        fourier = _get_fourier()
        w = fourier.get_window(name, nwindow, nzero=nzero)
        assert len(w) == nwindow + nzero

    @given(
        name=window_names(),
        nwindow=window_sizes(min_size=8, max_size=256),
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
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
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
    def test_window_non_negative_no_fftshift(self, name, nwindow):
        """Property: Standard windows are non-negative (before fftshift)."""
        fourier = _get_fourier()
        w = fourier.get_window(name, nwindow, fftshift=False, norm=False)
        # Most windows are non-negative
        assert np.all(w >= -1e-10)

    @given(
        name=window_names(),
        nwindow=window_sizes(min_size=8, max_size=256),
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
    def test_window_dtype_float32(self, name, nwindow):
        """Property: Window dtype matches requested dtype."""
        fourier = _get_fourier()
        w = fourier.get_window(name, nwindow, dtype='float32')
        assert w.dtype == np.float32

    @given(
        name=window_names(),
        nwindow=window_sizes(min_size=8, max_size=256),
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
    def test_window_dtype_float64(self, name, nwindow):
        """Property: Window dtype matches requested dtype."""
        fourier = _get_fourier()
        w = fourier.get_window(name, nwindow, dtype='float64')
        assert w.dtype == np.float64

    @given(
        name=window_names(),
        nwindow=st.integers(min_value=8, max_value=128).filter(lambda n: n % 2 == 0),
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
    def test_fftshift_preserves_energy(self, name, nwindow):
        """Property: fftshift=True preserves window energy."""
        fourier = _get_fourier()
        w_normal = fourier.get_window(name, nwindow, fftshift=False, norm=False)
        w_shifted = fourier.get_window(name, nwindow, fftshift=True, norm=False)

        # Energy should be preserved
        energy_normal = np.sum(np.abs(w_normal) ** 2)
        energy_shifted = np.sum(np.abs(w_shifted) ** 2)
        assert_allclose(energy_normal, energy_shifted, rtol=1e-6)

    @given(
        name=window_names(),
        nwindow=window_sizes(min_size=8, max_size=256),
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
    def test_unnormalized_window_has_max_one(self, name, nwindow):
        """Property: Unnormalized window has maximum value of 1."""
        fourier = _get_fourier()
        w = fourier.get_window(name, nwindow, norm=False, fftshift=False)
        assert np.max(w) <= 1.0 + 1e-10


# =============================================================================
# fftfreq Tests
# =============================================================================


class TestFftfreqProperties:
    """Properties: FFT frequency array generation."""

    @given(
        nfft=fft_sizes(min_size=4, max_size=1024),
        fs=sample_rates(min_rate=1e3, max_rate=100e6),
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
    def test_fftfreq_length(self, nfft, fs):
        """Property: fftfreq returns array of length nfft."""
        fourier = _get_fourier()
        freqs = fourier.fftfreq(nfft, fs)
        assert len(freqs) == nfft

    @given(
        nfft=fft_sizes(min_size=4, max_size=1024),
        fs=sample_rates(min_rate=1e3, max_rate=100e6),
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
    def test_fftfreq_range(self, nfft, fs):
        """Property: Frequencies are within [-fs/2, fs/2)."""
        fourier = _get_fourier()
        freqs = fourier.fftfreq(nfft, fs)
        assert np.all(freqs >= -fs / 2 - 1e-6)
        assert np.all(freqs < fs / 2 + 1e-6)

    @given(
        nfft=fft_sizes(min_size=4, max_size=1024),
        fs=sample_rates(min_rate=1e3, max_rate=100e6),
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
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
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
    def test_fftfreq_symmetry_even(self, nfft, fs):
        """Property: For even nfft, frequencies are symmetric around 0."""
        fourier = _get_fourier()
        freqs = fourier.fftfreq(nfft, fs)
        # For even nfft, we have -fs/2 but not +fs/2
        assert_allclose(freqs[0], -fs / 2, rtol=1e-10)
        # Check symmetry of interior points
        assert_allclose(freqs[1:nfft // 2], -freqs[-1:nfft // 2:-1], rtol=1e-10)

    @given(
        nfft=fft_sizes(min_size=4, max_size=512),
        fs=sample_rates(min_rate=1e3, max_rate=100e6),
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
    def test_fftfreq_dtype(self, nfft, fs):
        """Property: fftfreq respects dtype argument."""
        fourier = _get_fourier()
        freqs_32 = fourier.fftfreq(nfft, fs, dtype='float32')
        freqs_64 = fourier.fftfreq(nfft, fs, dtype='float64')
        assert freqs_32.dtype == np.float32
        assert freqs_64.dtype == np.float64

    @given(
        nfft=fft_sizes(min_size=4, max_size=256),
        fs=sample_rates(min_rate=1e3, max_rate=10e6),
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
    def test_fftfreq_contains_zero(self, nfft, fs):
        """Property: fftfreq contains zero frequency (DC component)."""
        fourier = _get_fourier()
        freqs = fourier.fftfreq(nfft, fs, dtype='float64')
        # Zero should be present in the frequency array
        assert np.any(np.abs(freqs) < fs / nfft / 2)


# =============================================================================
# oaconvolve Tests
# =============================================================================


class TestOaconvolveProperties:
    """Properties: Overlap-add convolution."""

    @given(
        x=real_waveforms(min_size=64, max_size=512, dtype=np.float64),
        kernel_size=st.integers(min_value=3, max_value=31).filter(lambda n: n % 2 == 1),
    )
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture],
        max_examples=50,
        deadline=None,
    )
    def test_oaconvolve_identity_kernel(self, x, kernel_size):
        """Property: Convolution with delta function is identity (shifted)."""
        fourier = _get_fourier()
        # Create delta function kernel
        kernel = np.zeros(kernel_size, dtype=x.dtype)
        kernel[kernel_size // 2] = 1.0

        result = fourier.oaconvolve(x, kernel, mode='same')

        # Result should equal input (delta convolution is identity)
        # Use larger atol for floating point precision in FFT-based convolution
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

        # Use atol for values near zero where rtol is meaningless
        assert_allclose(result_scaled_input, result_scaled_output, rtol=1e-6, atol=1e-12)

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
        # Use a small kernel for this test
        rng = np.random.default_rng(42)
        x2 = rng.standard_normal(15).astype(x1.dtype)

        result1 = fourier.oaconvolve(x1, x2, mode='full')
        result2 = fourier.oaconvolve(x2, x1, mode='full')

        # Use atol for values near zero
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

    @given(
        x=real_waveforms(min_size=64, max_size=256, dtype=np.float64),
    )
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture],
        max_examples=50,
        deadline=None,
    )
    def test_oaconvolve_output_length_same(self, x):
        """Property: Same-mode convolution preserves input length."""
        fourier = _get_fourier()
        kernel = np.array([0.25, 0.5, 0.25], dtype=x.dtype)

        result = fourier.oaconvolve(x, kernel, mode='same')

        assert len(result) == len(x)


# =============================================================================
# resample Tests
# =============================================================================


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
        x=complex_waveforms(min_size=128, max_size=512, dtype=np.complex128),
    )
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture],
        max_examples=50,
        deadline=None,
    )
    def test_resample_output_length(self, x):
        """Property: Resampled output has requested length."""
        fourier = _get_fourier()
        # Downsample by 2
        num_out = len(x) // 2
        result = fourier.resample(x, num_out)
        assert len(result) == num_out

    @given(
        x=complex_waveforms(min_size=64, max_size=256, dtype=np.complex128),
    )
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture],
        max_examples=50,
        deadline=None,
    )
    def test_resample_upsample_output_length(self, x):
        """Property: Upsampled output has requested length."""
        fourier = _get_fourier()
        # Upsample by 2
        num_out = len(x) * 2
        result = fourier.resample(x, num_out)
        assert len(result) == num_out

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
        # Use fixed input to avoid edge cases with sparse signals
        rng = np.random.default_rng(42)
        x = (rng.standard_normal(128) + 1j * rng.standard_normal(128)).astype(np.complex128)
        num_out = len(x) // 2

        result_unscaled = fourier.resample(x, num_out, scale=1.0)
        result_scaled = fourier.resample(x, num_out, scale=scale)

        # Use atol for values near zero
        assert_allclose(result_scaled, scale * result_unscaled, rtol=1e-8, atol=1e-15)

    @given(
        x=complex_waveforms(min_size=64, max_size=256, allow_subnormal=False),
    )
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture],
        max_examples=50,
        deadline=None,
    )
    def test_resample_dtype_preservation(self, x):
        """Property: Resample preserves complex dtype."""
        fourier = _get_fourier()
        num_out = len(x) // 2
        result = fourier.resample(x, num_out)
        assert result.dtype == x.dtype


# =============================================================================
# stft Tests
# =============================================================================


class TestStftProperties:
    """Properties: Short-time Fourier transform."""

    @given(
        x=complex_waveforms(min_size=256, max_size=512, dtype=np.complex64),
        params=stft_parameters(),
    )
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.data_too_large],
        max_examples=50,
        deadline=None,
    )
    def test_stft_output_shape(self, x, params):
        """Property: STFT output has correct shape."""
        fourier = _get_fourier()
        # min_size=256 guarantees len(x) >= max(nperseg)=256

        freqs, times, X = fourier.stft(
            x,
            fs=params['fs'],
            window=params['window'],
            nperseg=params['nperseg'],
            noverlap=params['noverlap'],
            truncate=True,
        )

        # Output shape is (time, freq) based on implementation
        # Frequency axis (axis 1) should have nperseg bins
        assert X.shape[1] == params['nperseg']
        # Frequency array should match
        assert len(freqs) == params['nperseg']
        # Time array should match first dimension
        assert len(times) == X.shape[0]

    @given(
        x=complex_waveforms(min_size=128, max_size=256, dtype=np.complex64),
    )
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.data_too_large],
        max_examples=30,
        deadline=None,
    )
    def test_stft_frequency_range(self, x):
        """Property: STFT frequencies span [-fs/2, fs/2)."""
        fourier = _get_fourier()
        fs = 1e6
        nperseg = 128

        freqs, times, X = fourier.stft(
            x,
            fs=fs,
            window='hamming',
            nperseg=nperseg,
            noverlap=0,
            truncate=True,
        )

        assert np.min(freqs) >= -fs / 2 - 1
        assert np.max(freqs) < fs / 2 + 1

    @given(
        x=complex_waveforms(min_size=128, max_size=256, dtype=np.complex64),
    )
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.data_too_large],
        max_examples=30,
        deadline=None,
    )
    def test_stft_dtype_preservation(self, x):
        """Property: STFT preserves input dtype."""
        fourier = _get_fourier()
        freqs, times, X = fourier.stft(
            x,
            fs=1e6,
            window='hamming',
            nperseg=64,
            noverlap=0,
            truncate=True,
        )

        assert X.dtype == x.dtype

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
        x = (rng.standard_normal(512) + 1j * rng.standard_normal(512)).astype(np.complex128)

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
        x=complex_waveforms(min_size=128, max_size=256, dtype=np.complex64),
    )
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.data_too_large],
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

        # Number of segments should be floor(len(x) / (nperseg - noverlap))
        step = nperseg - noverlap
        expected_segments = len(x) // step
        assert X.shape[0] == expected_segments


# =============================================================================
# spectrogram Tests
# =============================================================================


class TestSpectrogramProperties:
    """Properties: Power spectrogram computation."""

    @given(
        x=complex_waveforms(min_size=256, max_size=512, dtype=np.complex64),
        params=stft_parameters(),
    )
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.data_too_large],
        max_examples=50,
        deadline=None,
    )
    def test_spectrogram_output_shape(self, x, params):
        """Property: Spectrogram output has correct shape."""
        fourier = _get_fourier()
        # min_size=256 guarantees len(x) >= max(nperseg)=256

        freqs, times, Sxx = fourier.spectrogram(
            x,
            fs=params['fs'],
            window=params['window'],
            nperseg=params['nperseg'],
            noverlap=params['noverlap'],
            truncate=True,
        )

        # Same shape requirements as STFT
        assert Sxx.shape[1] == params['nperseg']
        assert len(freqs) == params['nperseg']
        assert len(times) == Sxx.shape[0]

    @given(
        x=complex_waveforms(min_size=128, max_size=256, dtype=np.complex64),
    )
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.data_too_large],
        max_examples=30,
        deadline=None,
    )
    def test_spectrogram_non_negative(self, x):
        """Property: Spectrogram (power) values are non-negative."""
        fourier = _get_fourier()
        freqs, times, Sxx = fourier.spectrogram(
            x,
            fs=1e6,
            window='hamming',
            nperseg=64,
            noverlap=0,
            truncate=True,
        )

        assert np.all(Sxx >= 0)

    @given(
        x=complex_waveforms(min_size=128, max_size=256, dtype=np.complex64),
    )
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.data_too_large],
        max_examples=30,
        deadline=None,
    )
    def test_spectrogram_real_output(self, x):
        """Property: Spectrogram output is real (power values)."""
        fourier = _get_fourier()
        freqs, times, Sxx = fourier.spectrogram(
            x,
            fs=1e6,
            window='hamming',
            nperseg=64,
            noverlap=0,
            truncate=True,
        )

        assert np.isrealobj(Sxx)

    @given(
        scale=st.floats(min_value=0.5, max_value=2.0, allow_nan=False),
    )
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture],
        max_examples=30,
        deadline=None,
    )
    def test_spectrogram_power_scaling(self, scale):
        """Property: Spectrogram scales as square of amplitude.

        If input is scaled by k, power spectrogram scales by k².
        """
        fourier = _get_fourier()
        rng = np.random.default_rng(42)
        x = (rng.standard_normal(512) + 1j * rng.standard_normal(512)).astype(np.complex128)

        _, _, Sxx1 = fourier.spectrogram(
            x,
            fs=1e6,
            window='hamming',
            nperseg=64,
            noverlap=0,
            truncate=True,
        )

        scaled_x = (scale * x).astype(np.complex128)
        _, _, Sxx2 = fourier.spectrogram(
            scaled_x,
            fs=1e6,
            window='hamming',
            nperseg=64,
            noverlap=0,
            truncate=True,
        )

        assert_allclose(Sxx2, scale**2 * Sxx1, rtol=1e-5)

    @given(
        x=complex_waveforms(min_size=128, max_size=256, dtype=np.complex64),
    )
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.data_too_large],
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

        expected = np.abs(X) ** 2
        assert_allclose(Sxx, expected, rtol=1e-5)


# =============================================================================
# Multi-backend Tests
# =============================================================================


class TestMultiBackendCompatibility:
    """Tests for numpy/cupy compatibility."""

    @pytest.fixture(params=['numpy', 'cupy', 'dask'])
    def xp_name_and_module(self, request):
        """Lazily provide array namespace to avoid importing dask at collection time."""
        name = request.param
        if name == 'numpy':
            return name, np
        elif name == 'dask':
            from conftest import _get_dask_array
            return name, _get_dask_array()
        elif name == 'cupy':
            from conftest import _get_cupy
            return name, _get_cupy()
        else:
            raise ValueError(f'invalid namespace {name}')

    def test_get_window_backend(self, xp_name_and_module):
        """Test get_window works with different backends."""
        fourier = _get_fourier()
        xp_name, xp = xp_name_and_module
        if xp_name == 'dask':
            pytest.skip('get_window does not support dask')
        if xp is None:
            pytest.skip(f'{xp_name} is not available')

        w = fourier.get_window('hamming', 64, xp=xp)
        w_np = to_numpy(w)

        assert len(w_np) == 64
        assert_allclose(np.mean(np.abs(w_np) ** 2), 1.0, rtol=1e-6)

    def test_fftfreq_backend(self, xp_name_and_module):
        """Test fftfreq works with different backends."""
        fourier = _get_fourier()
        xp_name, xp = xp_name_and_module
        if xp_name == 'dask':
            pytest.skip('fftfreq does not support dask')
        if xp is None:
            pytest.skip(f'{xp_name} is not available')

        freqs = fourier.fftfreq(64, 1e6, xp=xp)
        freqs_np = to_numpy(freqs)

        assert len(freqs_np) == 64
        assert np.all(freqs_np >= -0.5e6 - 1)
        assert np.all(freqs_np < 0.5e6 + 1)


# =============================================================================
# Edge Case Tests
# =============================================================================


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

    def test_resample_no_change(self):
        """Test resample when output size equals input size."""
        fourier = _get_fourier()
        rng = np.random.default_rng(42)
        x = (rng.standard_normal(64) + 1j * rng.standard_normal(64)).astype(np.complex64)

        result = fourier.resample(x, 64)
        assert_allclose(result, x, rtol=1e-6)

    def test_stft_single_segment(self):
        """Test STFT with input exactly one segment long."""
        fourier = _get_fourier()
        rng = np.random.default_rng(42)
        x = (rng.standard_normal(64) + 1j * rng.standard_normal(64)).astype(np.complex64)

        freqs, times, X = fourier.stft(
            x,
            fs=1e6,
            window='hamming',
            nperseg=64,
            noverlap=0,
            truncate=True,
        )

        assert X.shape[0] == 1  # One time segment
        assert X.shape[1] == 64  # 64 frequency bins

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

    def test_oaconvolve_single_element_kernel(self):
        """Test oaconvolve with single-element kernel."""
        fourier = _get_fourier()
        rng = np.random.default_rng(42)
        x = rng.standard_normal(64)
        kernel = np.array([2.0])

        result = fourier.oaconvolve(x, kernel, mode='same')
        assert_allclose(result, 2.0 * x, rtol=1e-10)

    def test_stft_with_overlap(self):
        """Test STFT with 50% overlap."""
        fourier = _get_fourier()
        rng = np.random.default_rng(42)
        x = (rng.standard_normal(256) + 1j * rng.standard_normal(256)).astype(np.complex64)

        freqs, times, X = fourier.stft(
            x,
            fs=1e6,
            window='hamming',
            nperseg=64,
            noverlap=32,
            truncate=True,
        )

        # With 50% overlap using sliding_window_view:
        # hop_size = nperseg - noverlap = 64 - 32 = 32
        # num_segments = (len(x) - nperseg) // hop_size + 1 = (256 - 64) // 32 + 1 = 7
        assert X.shape[0] == 7
        assert X.shape[1] == 64

    def test_resample_extreme_downsample(self):
        """Test resample with extreme downsampling."""
        fourier = _get_fourier()
        rng = np.random.default_rng(42)
        x = (rng.standard_normal(256) + 1j * rng.standard_normal(256)).astype(np.complex128)

        result = fourier.resample(x, 16)
        assert len(result) == 16
        assert result.dtype == x.dtype

    def test_resample_extreme_upsample(self):
        """Test resample with extreme upsampling."""
        fourier = _get_fourier()
        rng = np.random.default_rng(42)
        x = (rng.standard_normal(64) + 1j * rng.standard_normal(64)).astype(np.complex128)

        result = fourier.resample(x, 512)
        assert len(result) == 512
        assert result.dtype == x.dtype


# =============================================================================
# NumPy vs CuPy Cross-Comparison Tests
# =============================================================================


class TestNumpyCupyCrossComparison:
    """Cross-comparison tests validating numpy and cupy produce close results.

    These tests ensure that the fourier module functions produce numerically
    close results when operating on numpy arrays vs cupy arrays. Tolerances
    are set assuming IEEE fast-math level precision for cupy operations.
    """

    # Tolerances for IEEE fast-math precision
    RTOL_FLOAT32 = 2e-5
    RTOL_FLOAT64 = 1e-12
    ATOL = 1e-7

    @pytest.fixture
    def cupy_available(self):
        from conftest import _cupy

        if _cupy is None:
            pytest.skip('cupy is not available')
        return _cupy

    # -------------------------------------------------------------------------
    # get_window tests
    # -------------------------------------------------------------------------

    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        name=st.sampled_from(['hamming', 'hann', 'blackman', 'bartlett', 'flattop']),
        nwindow=st.integers(min_value=8, max_value=512),
    )
    def test_get_window_numpy_vs_cupy_float64(self, cupy_available, name, nwindow):
        """Test get_window produces close results for numpy vs cupy (float64)."""
        cp = cupy_available
        fourier = _get_fourier()

        # NumPy result
        result_np = fourier.get_window(name, nwindow, dtype=np.float64)

        # CuPy result
        result_cp = fourier.get_window(name, nwindow, dtype=np.float64, xp=cp)
        result_cp_np = result_cp.get()

        assert_allclose(result_cp_np, result_np, rtol=self.RTOL_FLOAT64, atol=self.ATOL)

    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        name=st.sampled_from(['hamming', 'hann', 'blackman', 'bartlett', 'flattop']),
        nwindow=st.integers(min_value=8, max_value=512),
    )
    def test_get_window_numpy_vs_cupy_float32(self, cupy_available, name, nwindow):
        """Test get_window produces close results for numpy vs cupy (float32)."""
        cp = cupy_available
        fourier = _get_fourier()

        # NumPy result
        result_np = fourier.get_window(name, nwindow, dtype=np.float32)

        # CuPy result
        result_cp = fourier.get_window(name, nwindow, dtype=np.float32, xp=cp)
        result_cp_np = result_cp.get()

        assert_allclose(result_cp_np, result_np, rtol=self.RTOL_FLOAT32, atol=self.ATOL)

    # -------------------------------------------------------------------------
    # fftfreq tests
    # -------------------------------------------------------------------------

    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        nfft=fft_sizes(min_size=8, max_size=512),
        fs=sample_rates(min_rate=1e3, max_rate=100e6),
    )
    def test_fftfreq_numpy_vs_cupy_float64(self, cupy_available, nfft, fs):
        """Test fftfreq produces close results for numpy vs cupy (float64)."""
        cp = cupy_available
        fourier = _get_fourier()

        # NumPy result
        result_np = fourier.fftfreq(nfft, fs, dtype=np.float64)

        # CuPy result
        result_cp = fourier.fftfreq(nfft, fs, dtype=np.float64, xp=cp)
        result_cp_np = result_cp.get()

        assert_allclose(result_cp_np, result_np, rtol=self.RTOL_FLOAT64, atol=self.ATOL)

    # -------------------------------------------------------------------------
    # resample tests
    # -------------------------------------------------------------------------

    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=3000)
    @given(x=complex_waveforms(min_size=64, max_size=256, dtype=np.complex64, allow_subnormal=False))
    def test_resample_numpy_vs_cupy_complex64(self, cupy_available, x):
        """Test resample produces close results for numpy vs cupy (complex64)."""
        cp = cupy_available
        fourier = _get_fourier()

        # Resample to half the size
        num_out = len(x) // 2

        # NumPy result
        result_np = fourier.resample(x, num_out)

        # CuPy result
        x_cp = cp.asarray(x)
        result_cp = fourier.resample(x_cp, num_out)
        result_cp_np = result_cp.get()

        assert_allclose(result_cp_np, result_np, rtol=self.RTOL_FLOAT32, atol=self.ATOL)

    # -------------------------------------------------------------------------
    # stft tests
    # -------------------------------------------------------------------------

    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=3000)
    @given(x=complex_waveforms(min_size=256, max_size=512, dtype=np.complex64, allow_subnormal=False))
    def test_stft_numpy_vs_cupy_complex64(self, cupy_available, x):
        """Test stft produces close results for numpy vs cupy (complex64)."""
        cp = cupy_available
        fourier = _get_fourier()

        # NumPy result
        freqs_np, times_np, X_np = fourier.stft(
            x, fs=1e6, window='hamming', nperseg=64, noverlap=32, truncate=True
        )

        # CuPy result
        x_cp = cp.asarray(x)
        freqs_cp, times_cp, X_cp = fourier.stft(
            x_cp, fs=1e6, window='hamming', nperseg=64, noverlap=32, truncate=True
        )
        freqs_cp_np = freqs_cp.get()
        times_cp_np = times_cp.get()
        X_cp_np = X_cp.get()

        assert_allclose(freqs_cp_np, freqs_np, rtol=self.RTOL_FLOAT32, atol=self.ATOL)
        assert_allclose(times_cp_np, times_np, rtol=self.RTOL_FLOAT32, atol=self.ATOL)
        assert_allclose(X_cp_np, X_np, rtol=self.RTOL_FLOAT32, atol=self.ATOL)

    # -------------------------------------------------------------------------
    # spectrogram tests
    # -------------------------------------------------------------------------

    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=3000)
    @given(x=complex_waveforms(min_size=256, max_size=512, dtype=np.complex64, allow_subnormal=False))
    def test_spectrogram_numpy_vs_cupy_complex64(self, cupy_available, x):
        """Test spectrogram produces close results for numpy vs cupy (complex64)."""
        cp = cupy_available
        fourier = _get_fourier()

        # NumPy result
        freqs_np, times_np, Sxx_np = fourier.spectrogram(
            x, fs=1e6, window='hamming', nperseg=64, noverlap=32, truncate=True
        )

        # CuPy result
        x_cp = cp.asarray(x)
        freqs_cp, times_cp, Sxx_cp = fourier.spectrogram(
            x_cp, fs=1e6, window='hamming', nperseg=64, noverlap=32, truncate=True
        )
        freqs_cp_np = freqs_cp.get()
        times_cp_np = times_cp.get()
        Sxx_cp_np = Sxx_cp.get()

        assert_allclose(freqs_cp_np, freqs_np, rtol=self.RTOL_FLOAT32, atol=self.ATOL)
        assert_allclose(times_cp_np, times_np, rtol=self.RTOL_FLOAT32, atol=self.ATOL)
        assert_allclose(Sxx_cp_np, Sxx_np, rtol=self.RTOL_FLOAT32, atol=self.ATOL)

    # -------------------------------------------------------------------------
    # oaconvolve tests
    # -------------------------------------------------------------------------

    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=3000)
    @given(x=real_waveforms(min_size=64, max_size=256, dtype=np.float64))
    def test_oaconvolve_numpy_vs_cupy_float64(self, cupy_available, x):
        """Test oaconvolve produces close results for numpy vs cupy (float64)."""
        cp = cupy_available
        fourier = _get_fourier()

        # Create a simple kernel
        kernel = np.array([0.25, 0.5, 0.25], dtype=np.float64)

        # NumPy result
        result_np = fourier.oaconvolve(x, kernel, mode='same')

        # CuPy result
        x_cp = cp.asarray(x)
        kernel_cp = cp.asarray(kernel)
        result_cp = fourier.oaconvolve(x_cp, kernel_cp, mode='same')
        result_cp_np = result_cp.get()

        assert_allclose(result_cp_np, result_np, rtol=self.RTOL_FLOAT64, atol=self.ATOL)

    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=3000)
    @given(x=real_waveforms(min_size=64, max_size=256, dtype=np.float32))
    def test_oaconvolve_numpy_vs_cupy_float32(self, cupy_available, x):
        """Test oaconvolve produces close results for numpy vs cupy (float32)."""
        cp = cupy_available
        fourier = _get_fourier()

        # Create a simple kernel
        kernel = np.array([0.25, 0.5, 0.25], dtype=np.float32)

        # NumPy result
        result_np = fourier.oaconvolve(x, kernel, mode='same')

        # CuPy result
        x_cp = cp.asarray(x)
        kernel_cp = cp.asarray(kernel)
        result_cp = fourier.oaconvolve(x_cp, kernel_cp, mode='same')
        result_cp_np = result_cp.get()

        assert_allclose(result_cp_np, result_np, rtol=self.RTOL_FLOAT32, atol=self.ATOL)
