"""striqt.waveform.lib.fourier.spectrum: the power spectrogram as the real power of the
STFT on the same axes, on numpy and cupy.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from numeric_checks import ATOL, assert_close, elementwise_rtol, numpy_and_cupy
from numpy.testing import assert_array_equal
from waveform_strategies import cross_backend_sigma, noise_waveforms, stft_parameters

from striqt.waveform.lib.fourier.spectrum import spectrogram
from striqt.waveform.lib.fourier.stft import stft
from striqt.waveform.lib.power import envelope_power_rtol


# %% spectrogram
class TestSpectrogram:
    @pytest.mark.parametrize('return_axis_arrays', [True, False], ids=['axes', 'array'])
    # 25 waveforms per row, 50 in all as the single shape test ran
    @settings(max_examples=25)
    @given(x=noise_waveforms(min_size=256, max_size=512), params=stft_parameters())
    def test_is_the_power_of_the_normalized_stft(self, return_axis_arrays, x, params):
        """``Sxx == |X|**2`` for the STFT with ``norm='power'``, in the real dtype of
        the input and on the frequency and time axes of the STFT"""
        freqs, times, X = stft(x, truncate=True, norm='power', **params)
        if return_axis_arrays:
            f, t, Sxx = spectrogram(x, truncate=True, **params)
            assert_array_equal(f, freqs)
            assert_array_equal(t, times)
        else:
            Sxx = spectrogram(x, truncate=True, return_axis_arrays=False, **params)

        assert Sxx.dtype == np.finfo(x.dtype).dtype
        rtol = envelope_power_rtol(Sxx.dtype, complex_input=True)
        assert_close(Sxx, np.abs(X.astype(np.complex128)) ** 2, rtol=rtol)


# %% numpy vs cupy
class TestNumpyCupyCrossComparison:
    @given(x=noise_waveforms(min_size=256, max_size=512, dtype=np.complex64))
    def test_spectrogram(self, cupy_available, x):
        kws = {
            'fs': 1e6,
            'window': 'hamming',
            'nperseg': 64,
            'noverlap': 32,
            'truncate': True,
        }
        (freqs_np, times_np, S_np), (freqs_cp, times_cp, S_cp) = numpy_and_cupy(
            cupy_available, spectrogram, x, **kws
        )

        assert_close(
            freqs_cp, freqs_np, rtol=elementwise_rtol(freqs_np.dtype), atol=ATOL
        )
        assert_close(
            times_cp, times_np, rtol=elementwise_rtol(times_np.dtype), atol=ATOL
        )

        # window/nfft and the window multiply, then fft(nperseg), then the |X|**2
        # rounding. For noise-like input the relative rms error of power equals
        # that of amplitude, but the error in one bin grows with sqrt(Sxx), so rtol
        # covers the large bins and atol the small ones.
        sigma = cross_backend_sigma(x.dtype, [64], n_elementwise=3)
        assert_close(S_cp, S_np, rtol=4 * sigma, sigma=sigma)
