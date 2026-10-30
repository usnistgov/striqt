"""striqt.waveform.lib.fourier.filters: the COLA overlap design, the FIR low-pass
designs and the STFT-domain response of each, the overlap-add bandpass filter on tones,
and overlap-add convolution, on numpy and cupy.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
from numeric_checks import (
    FIR_LEAKAGE,
    FLOAT_DTYPES,
    assert_close,
    assert_tone_level,
    dtype_id,
    interior,
    level_error_dB,
    numpy_and_cupy,
    rms,
    tone_frequency,
)
from numpy.testing import assert_allclose, assert_array_equal
from waveform_strategies import cross_backend_sigma, noise_waveforms

from striqt.analysis import testing
from striqt.waveform.lib.fourier.fft import fftfreq
from striqt.waveform.lib.fourier.filters import (
    _fir_lowpass_fft,
    design_fir_lpf,
    design_oafilter,
    oaconvolve,
    oafilter,
)
from striqt.waveform.power_analysis import level_tolerance_dB


# %% design_oafilter, design_fir_lpf, _fir_lowpass_fft
class TestFilterDesign:
    @pytest.mark.parametrize(
        'window, overlap_scale',
        [
            ('hamming', 1 / 2),
            ('hann', 1 / 2),
            ('blackman', 2 / 3),
            ('blackmanharris', 4 / 5),
            ('rect', 1),
            (None, 1),
        ],
    )
    def test_design_oafilter_overlap(self, window, overlap_scale):
        nfft = 120
        nfft_out, noverlap, scale, pad_out = design_oafilter(
            40 * nfft, window=window, nfft_out=nfft, nfft=nfft, extend=False
        )
        assert nfft_out == nfft
        assert scale == overlap_scale
        assert noverlap == round(nfft * overlap_scale)
        assert pad_out == 0

    @pytest.mark.parametrize(
        'size, kws, exc, match',
        [
            (
                1200,
                {'window': 'flattop'},
                TypeError,
                "'blackmanharris'.*, not 'flattop'",
            ),
            (1210, {'window': 'blackman', 'nfft_out': 121}, ValueError, '% 3'),
            (1000, {'window': 'hamming'}, ValueError, 'integer multiple of noverlap'),
            (1230, {'window': 'blackman', 'nfft_out': 123}, ValueError, 'must be even'),
            (1200, {'window': 'rect', 'nfft': 125}, ValueError, 'must be even'),
        ],
        ids=[
            'unsupported_window',
            'nfft_out_off_divisor',
            'size_off_noverlap',
            'odd_nfft_out',
            'odd_nfft',
        ],
    )
    def test_design_oafilter_errors(self, size, kws, exc, match):
        kws = {'nfft_out': 120, 'nfft': 120, 'extend': False, **kws}
        with pytest.raises(exc, match=match):
            design_oafilter(size, **kws)

    def test_design_oafilter_options(self):
        pad_out = design_oafilter(
            1000, window='hamming', nfft_out=120, nfft=120, extend=True
        )[3]
        assert pad_out == 1000 % 60

        nfft_out = design_oafilter(
            1200, window='hamming', nfft_out=None, nfft=120, extend=False
        )[0]
        assert nfft_out == 120

    @pytest.mark.parametrize('numtaps', [401, 1001, 4001])
    @pytest.mark.parametrize('dtype', FLOAT_DTYPES, ids=dtype_id)
    def test_design_fir_lpf(self, numtaps, dtype):
        b = design_fir_lpf(10e6, 50e6, numtaps=numtaps, dtype=dtype)
        assert b.shape == (numtaps,)
        assert b.dtype == np.dtype(dtype)
        # least-squares passband ripple bounds the DC gain error
        assert abs(level_error_dB(b.sum())) < level_tolerance_dB(FIR_LEAKAGE)

    def test_fir_lowpass_fft_response(self):
        """the transition band is centered on the cutoff, as in `design_fir_lpf`: unity
        below ``cutoff - transition/2``, zero above ``cutoff + transition/2``, and
        half gain (-6 dB) at the cutoff"""
        size = 256
        cutoff, transition = 0.1, 0.05
        H = _fir_lowpass_fft(size, 1.0, cutoff=cutoff, transition=transition)
        freqs = fftfreq(size, 1.0)
        assert H.shape == (size,) and H.dtype == np.complex64

        passband = np.abs(freqs) < cutoff - transition / 2 - 2 / size
        stopband = np.abs(freqs) > cutoff + transition / 2 + 2 / size
        assert np.abs(np.abs(H[passband]) - 1).max() < FIR_LEAKAGE
        assert np.abs(H[stopband]).max() < FIR_LEAKAGE
        # the ramp is linear over the transition, so the gain at the cutoff is 1/2
        # within the slope times half a bin
        at_cutoff = np.abs(H[np.argmin(np.abs(freqs - cutoff))])
        assert abs(at_cutoff - 0.5) < (1 / size) / transition / 2 + FIR_LEAKAGE

    @pytest.mark.parametrize('cutoff', [np.inf, 0.5])
    def test_fir_lowpass_fft_cutoff_at_or_past_nyquist_is_allpass(self, cutoff):
        H = _fir_lowpass_fft(256, 1.0, cutoff=cutoff, transition=0.05)
        assert H.dtype == np.complex64
        assert_array_equal(H, 1)

    def test_fir_lowpass_fft_transition_is_cut_at_nyquist(self):
        """a transition band that runs past Nyquist ends at Nyquist"""
        size = 256
        H = _fir_lowpass_fft(size, 1.0, cutoff=0.4, transition=0.3)
        freqs = fftfreq(size, 1.0)
        assert H.shape == (size,) and H.dtype == np.complex64

        passband = np.abs(freqs) < 0.4 - 0.15 - 2 / size
        assert np.abs(np.abs(H[passband]) - 1).max() < FIR_LEAKAGE
        # firwin2 constrains an even-length (type II) design to zero gain at nyquist
        assert np.abs(H[np.argmin(freqs)]) < FIR_LEAKAGE


# %% oaconvolve
class TestOaconvolve:
    @settings(max_examples=50)
    @given(
        x=noise_waveforms(min_size=64, max_size=512, dtype=np.float64),
        kernel_size=st.integers(min_value=1, max_value=31).filter(lambda n: n % 2 == 1),
    )
    def test_identity_kernel(self, x, kernel_size):
        kernel = np.zeros(kernel_size, dtype=x.dtype)
        kernel[kernel_size // 2] = 1.0
        result = oaconvolve(x, kernel, mode='same')
        # atol for samples near zero, where rtol is meaningless
        assert_allclose(result, x, rtol=1e-8, atol=1e-12)


# %% oafilter
class TestOafilter:
    FS = 1e6
    NFFT = 256
    NSEG = 40
    PASSBAND = (-0.2e6, 0.2e6)

    def _tone(self, f0):
        return testing.tone(None, self.FS, frequency=f0, count=self.NFFT * self.NSEG)[0]

    def _oafilter(self, x, **kws):
        kws = {'window': 'hamming', 'passband': self.PASSBAND, **kws}
        return oafilter(x, fs=self.FS, nfft=self.NFFT, **kws)

    @settings(max_examples=20)
    @given(f0_bins=st.integers(min_value=-40, max_value=40))
    def test_passes_in_band_tone(self, f0_bins):
        f0 = f0_bins * self.FS / self.NFFT
        x = self._tone(f0)
        y = self._oafilter(x)
        assert y.shape == x.shape and y.dtype == x.dtype
        yi = interior(y, self.NFFT)
        assert_tone_level(yi)
        assert abs(tone_frequency(yi, self.FS) - f0) <= self.FS / yi.size

    def test_rejects_out_of_band_tone(self):
        y = self._oafilter(self._tone(0.4e6))
        yi = interior(y, self.NFFT)
        # the residual is the hamming sidelobe leakage (-53 dB). The -22 dB edge
        # pedestal of the window leaves the ringing concentrated at the segment edges
        assert level_error_dB(rms(yi)) < -40
        assert level_error_dB(yi).max() < -25

    def test_downsample_length_and_level(self):
        x = self._tone(0.05e6)
        nfft_out = self.NFFT // 2
        y = self._oafilter(x, nfft_out=nfft_out)
        assert y.shape == (round(x.size * nfft_out / self.NFFT),)
        assert_tone_level(interior(y, nfft_out))


# %% numpy vs cupy
class TestNumpyCupyCrossComparison:
    """tolerances come from `cross_backend_sigma`, parameterized by the FFT sizes and
    elementwise operations each function performs
    """

    def test_design_fir_lpf_on_cupy(self, cupy_available):
        b_np = design_fir_lpf(10e6, 50e6, numtaps=401)
        b_cp = design_fir_lpf(10e6, 50e6, numtaps=401, xp=cupy_available)
        assert isinstance(b_cp, cupy_available.ndarray)
        assert_close(b_cp, b_np)

    @given(x=noise_waveforms(64, 256, dtype=[np.float32, np.float64]))
    def test_oaconvolve(self, cupy_available, x):
        kernel = np.array([0.25, 0.5, 0.25], dtype=x.dtype)
        y_np, y_cp = numpy_and_cupy(cupy_available, oaconvolve, x, kernel, mode='same')
        # overlap-add blocks are at most len(x) long. Anchoring to the input rms
        # accounts for the gain of the kernel
        sigma = cross_backend_sigma(x.dtype, [len(x), len(x)], n_elementwise=1)
        assert_close(y_cp, y_np, sigma=sigma, scale=rms(x))

    @settings(max_examples=10)
    @given(f0_bins=st.integers(min_value=-20, max_value=20))
    def test_oafilter(self, cupy_available, f0_bins):
        fs, nfft = 1e6, 256
        x = testing.tone(None, fs, frequency=f0_bins * fs / nfft, count=nfft * 40)[0]

        kws = {'fs': fs, 'nfft': nfft, 'window': 'hamming', 'passband': (-0.2e6, 0.2e6)}
        y_np, y_cp = numpy_and_cupy(cupy_available, oafilter, x, **kws)
        # window multiply, fft(nfft), zeroing, ifft(nfft), window multiply, overlap add
        sigma = cross_backend_sigma(x.dtype, [nfft, nfft], n_elementwise=3)
        assert_close(y_cp, y_np, sigma=sigma)
