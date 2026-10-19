"""striqt.waveform.lib.fourier.spectrum: the power spectrogram as the real power of the
STFT on the same axes, on numpy and cupy.
"""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from numeric_checks import ATOL, assert_close, elementwise_rtol, numpy_and_cupy
from numpy.testing import assert_array_equal
from waveform_strategies import cross_backend_sigma, noise_waveforms, stft_parameters

from striqt.waveform.lib.fourier.fft import fftfreq
from striqt.waveform.lib.fourier.spectrum import spectrogram
from striqt.waveform.lib.fourier.stft import stft


# %% spectrogram
class TestSpectrogram:
    @settings(max_examples=50)
    @given(x=noise_waveforms(min_size=256, max_size=512), params=stft_parameters())
    def test_output_shape(self, x, params):
        """the spectrogram has the shape and axes of the stft and the real dtype of
        its input (min_size=256 guarantees len(x) >= max(nperseg))"""
        _, _, X = stft(x, truncate=True, **params)
        freqs, times, Sxx = spectrogram(x, truncate=True, **params)

        step = params['nperseg'] - params['noverlap']
        assert Sxx.shape == X.shape
        assert Sxx.shape[0] == (len(x) - params['noverlap']) // step
        assert Sxx.shape[1] == params['nperseg']
        assert len(freqs) == params['nperseg']
        assert len(times) == Sxx.shape[0]
        assert Sxx.dtype == np.finfo(x.dtype).dtype
        assert_array_equal(freqs, fftfreq(params['nperseg'], params['fs']))

    def test_return_axis_arrays_false(self):
        x = np.ones(512, dtype=np.complex64)
        X = stft(x, fs=1.0, window='hann', nperseg=64, return_axis_arrays=False)
        Sxx = spectrogram(
            x, fs=1.0, window='hann', nperseg=64, return_axis_arrays=False
        )
        assert Sxx.shape == X.shape and np.isrealobj(Sxx)


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
