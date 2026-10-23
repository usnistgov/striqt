"""striqt.waveform.lib.fourier.fft: the FFT roundoff model, the exact `fftfreq` grid and
its band selectors, the time-domain fftshift, and the cupy FFT `out=` and chunking
paths, on numpy and cupy.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import iq_waveforms
from hypothesis import given
from hypothesis import strategies as st
from numeric_checks import (
    ATOL,
    RTOL_FLOAT64,
    assert_close,
    cross_backend,
    numpy_and_cupy,
    rms,
    to_numpy,
    tone_bin,
)
from numpy.testing import assert_allclose, assert_array_equal
from waveform_strategies import cross_backend_sigma, sample_rates

from striqt.analysis import testing
from striqt.waveform.lib.arrays import accum_rtol, unit_roundoff
from striqt.waveform.lib.fourier.fft import (
    _broadcast_onto,
    _freq_band_edges,
    fft,
    fft_roundoff_rms,
    fft_tolerance_rms,
    fftfreq,
    get_max_cupy_fft_chunk,
    ifft,
    null_lo,
    off_peak_floor_dBc,
    on_peak_roundoff,
    set_max_cupy_fft_chunk,
    slice_freqs,
    time_fftshift,
    truncate_freqs,
)
from striqt.waveform.lib.fourier.filters import oaconvolve
from striqt.waveform.lib.fourier.resamplers import resample
from striqt.waveform.lib.fourier.stft import stft

OFF_PEAK_NFFTS = [64, 256, 1024, 4096]


def fft_sizes(min_size: int = 4, max_size: int = 512):
    """powers of two, or arbitrary even sizes (resample requires an even length)"""
    powers = st.integers(min_value=2, max_value=9).map(lambda p: 2**p)
    arbitrary = st.integers(min_value=min_size, max_value=max_size).filter(
        lambda n: n % 2 == 0
    )
    return st.one_of(powers, arbitrary)


@pytest.fixture
def max_cupy_fft_chunk():
    """a setter for the cupy FFT chunk size; the prior value is restored afterwards"""
    previous = get_max_cupy_fft_chunk()
    yield set_max_cupy_fft_chunk
    set_max_cupy_fft_chunk(previous)


# %% fft_roundoff_rms, fft_tolerance_rms, on_peak_roundoff, off_peak_floor_dBc
class TestRoundoffModels:
    ARGS = (np.complex64, [256, 256], 2)

    @pytest.mark.parametrize(
        'smaller, larger',
        [
            ((np.complex64, [256]), (np.complex64, [4096])),
            ((np.complex64, [256]), (np.complex64, [256, 256])),
            ((np.complex64, [256], 1), (np.complex64, [256], 4)),
            ((np.complex64, [256]), (np.complex64, [256], 1)),
        ],
    )
    def test_fft_roundoff_rms_monotonic(self, smaller, larger):
        assert fft_roundoff_rms(*smaller) < fft_roundoff_rms(*larger)

    def test_fft_roundoff_rms_scales_with_unit_roundoff(self):
        _, *shape_args = self.ARGS
        ratio = fft_roundoff_rms(np.complex64, *shape_args) / fft_roundoff_rms(
            np.complex128, *shape_args
        )
        assert ratio == pytest.approx(
            unit_roundoff(np.float32) / unit_roundoff(np.float64)
        )

    def test_cupy_bound_is_looser_than_numpy(self):
        by_numpy = fft_roundoff_rms(*self.ARGS, array_backend='numpy')
        by_cupy = fft_roundoff_rms(*self.ARGS, array_backend='cupy')
        assert by_cupy > by_numpy

    def test_off_peak_floor_peak_covers_rms(self):
        sigma = fft_tolerance_rms(np.complex64, [1024], 2)
        assert off_peak_floor_dBc(sigma, 1024, size=1023) >= off_peak_floor_dBc(
            sigma, 1024
        )

    def test_on_peak_roundoff_ignores_complexness(self):
        assert on_peak_roundoff(np.complex64) == on_peak_roundoff(np.float32)
        assert on_peak_roundoff(np.complex128) < on_peak_roundoff(np.complex64)


# %% fftfreq
class TestFftfreq:
    @given(nfft=st.integers(min_value=2, max_value=512), fs=sample_rates())
    def test_grid(self, nfft, fs):
        """nfft bins spaced fs/nfft apart and symmetric about 0; an even nfft
        includes -fs/2 but not +fs/2"""
        freqs = fftfreq(nfft, fs)
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

        assert fftfreq(nfft, fs, dtype='float32').dtype == np.float32
        assert fftfreq(nfft, fs, dtype='float64').dtype == np.float64

    @given(nfft=st.sampled_from([8, 9, 64, 65, 256]), fs=sample_rates())
    def test_as_index_false_matches_exact(self, nfft, fs):
        exact = fftfreq(nfft, fs)
        approx = fftfreq(nfft, fs, as_index=False)
        # linspace endpoints are off by a few ulps of fs/2
        eps = np.finfo(np.float64).eps
        assert_allclose(approx, exact, rtol=0, atol=8 * eps * fs)
        assert fftfreq(nfft, fs, dtype='float32', as_index=False).dtype == (np.float32)

    @pytest.mark.parametrize('as_index', [True, False])
    def test_non_numpy_namespace(self, as_index):
        da = pytest.importorskip('dask.array')
        freqs = fftfreq(16, 1e6, as_index=as_index, xp=da)
        assert isinstance(freqs, da.Array)
        assert_allclose(np.asarray(freqs), fftfreq(16, 1e6), rtol=1e-12)


# %% slice_freqs, truncate_freqs, null_lo, _freq_band_edges
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

        s = slice_freqs(nfft, fs, bandwidth, offset=offset)
        bins = np.arange(nfft)[s]

        assert bins.size == bw_bins
        # an even bin count cannot be centered on a bin, so the mean may sit half
        # a bin off the offset
        assert abs(bins.mean() - (nfft // 2 + offset_bins)) <= 0.5

        x = np.arange(nfft, dtype=np.float32)[np.newaxis, np.newaxis, :]
        x = np.broadcast_to(x, (2, 3, nfft)).copy()
        trimmed = truncate_freqs(x, nfft, fs, bandwidth, offset=offset, axis=2)
        assert_array_equal(trimmed, x[:, :, s])

        null_lo(x, nfft, fs, bandwidth, offset=offset, axis=2)
        assert np.isnan(x[:, :, s]).all()
        assert np.isnan(x).sum() == 2 * 3 * bw_bins

    def test_slice_freqs_half_band(self):
        """half of a 64-bin axis centered on DC is the 32 bins from 16 to 48"""
        assert slice_freqs(64, 1.0, 0.5) == slice(16, 48)

    @pytest.mark.parametrize(
        'nfft, kws, match',
        [
            (64, {'bandwidth': -1.0}, 'negative bandwidth'),
            (64, {'bandwidth': 0.5, 'offset': 0.3}, 'not a multiple'),
            (63, {'bandwidth': 0.5}, 'not a multiple'),
            (64, {'bandwidth': 0.5, 'offset': 0.5}, r'> fs/2'),
            (64, {'bandwidth': 0.5, 'offset': -0.5}, r'< fs/2'),
            (64, {'bandwidth': 1.5}, r'fs/2'),
        ],
        ids=[
            'negative_bandwidth',
            'offset_off_grid',
            'odd_nfft_puts_dc_off_grid',
            'offset_too_high',
            'offset_too_low',
            'bandwidth_exceeds_fs',
        ],
    )
    def test_slice_freqs_errors(self, nfft, kws, match):
        with pytest.raises(ValueError, match=match):
            slice_freqs(nfft, 1.0, **kws)

    def test_freq_band_edges(self):
        freqs = fftfreq(64, 1.0)
        assert _freq_band_edges(64, 1.0, None, None) == (None, None)

        ilo, ihi = _freq_band_edges(64, 1.0, -0.25, 0.25)
        assert isinstance(ilo, int) and isinstance(ihi, int)
        assert ilo == np.searchsorted(freqs, -0.25)
        assert freqs[ilo] >= -0.25 and freqs[ilo - 1] < -0.25
        assert freqs[ihi] <= 0.25 and freqs[ihi + 1] > 0.25

        assert _freq_band_edges(64, 1.0, -0.25, 0.6)[1] == 64


# %% time_fftshift and _broadcast_onto
class TestTimeFftshift:
    @given(x=iq_waveforms(min_size=8, max_size=64, multiple_of=2, channels=3))
    def test_alternates_sign_along_axis(self, x):
        sign = (-1) ** np.arange(x.shape[1])[np.newaxis, :]

        assert_array_equal(time_fftshift(x, axis=1), x * sign)
        assert_array_equal(time_fftshift(x[0], axis=0), x[0] * sign[0])

        scale = np.array([1, 2, 3], dtype=x.real.dtype)
        one_rounding = accum_rtol(x.dtype, 1)
        assert_allclose(
            time_fftshift(x, scale, axis=1),
            scale[:, np.newaxis] * x * sign,
            rtol=one_rounding,
        )

        y = x.copy()
        assert time_fftshift(y, overwrite_x=True, axis=1) is y
        assert_array_equal(y, x * sign)

        y = x.copy()
        assert time_fftshift(y, 2.0, overwrite_x=True, axis=1) is y
        assert_allclose(y, 2 * x * sign, rtol=one_rounding)

    def test_scale_must_be_1d(self):
        x = np.ones((2, 8), dtype=np.complex64)
        with pytest.raises(ValueError, match='1-D or scalar'):
            time_fftshift(x, np.ones((2, 2)), axis=1)

    def test_broadcast_onto_requires_1d(self):
        with pytest.raises(ValueError, match='1-D'):
            _broadcast_onto(np.ones((2, 2)), np.ones((4, 4)), axis=0)


# %% the roundoff model against the kernels
class TestOffPeakFloor:
    """Roundoff in the off-peak bins of a unit, bin-centered complex64 tone, on each
    array namespace.

    With an integer number of cycles per segment (and a rectangular window for the
    STFT) the exact spectrum occupies the on-peak bin alone, so every other bin
    measures roundoff alone against a float64 reference of the same float32-quantized
    input. For the time-domain outputs of resample and oaconvolve the bins are those
    of the error spectrum, taken in float64 so the analysis adds no roundoff of its
    own.
    """

    NSEG = 4
    KERNEL_TAPS = 400

    @staticmethod
    def _assert_off_peak_bins(err_off_peak, tone_peak, nfft, sigma):
        """`err_off_peak`: the roundoff in the off-peak bins of an `nfft`-point
        spectrum; `tone_peak`: the tone's own on-peak bin value, which the dBc floor
        is relative to; `sigma`: the rms error model relative to the output rms"""
        with np.errstate(divide='ignore'):
            rms_dBc = 20 * np.log10(rms(err_off_peak) / tone_peak)
            peak_dBc = 20 * np.log10(np.abs(err_off_peak).max() / tone_peak)

        rms_floor = off_peak_floor_dBc(sigma, nfft)
        assert rms_dBc < rms_floor, (
            f'off-peak rms roundoff {rms_dBc:.1f} dBc above the {rms_floor:.1f} dBc floor'
        )
        peak_floor = off_peak_floor_dBc(sigma, nfft, err_off_peak.size)
        assert peak_dBc < peak_floor, (
            f'off-peak peak roundoff {peak_dBc:.1f} dBc above the {peak_floor:.1f} dBc '
            'floor'
        )

    def _assert_off_peak_error_spectrum(self, out, out_ref, sigma):
        """bound the spectrum of the roundoff error in a time-domain output of a unit
        tone, excluding the tone's own on-peak bin; `sigma` is the rms error model
        relative to the unit input"""
        err = np.fft.fft(to_numpy(out).astype(np.complex128) - out_ref)
        size = err.size
        off_peak = np.ones(size, dtype=bool)
        off_peak[int(np.argmax(np.abs(np.fft.fft(out_ref))))] = False

        # the fft is unnormalized, so the unit tone peaks at size
        self._assert_off_peak_bins(err[off_peak], size, size, sigma)

    @staticmethod
    def _stft(x, nfft):
        return stft(x, fs=1.0, window='rect', nperseg=nfft, noverlap=0)[2]

    def _bin_centered_tone(self, nfft: int, k: int) -> np.ndarray:
        """a unit tone with `k` cycles per `nfft`-sample segment, over NSEG segments"""
        return testing.tone(None, nfft, frequency=k, count=self.NSEG * nfft)[0]

    @given(
        nfft=st.sampled_from(OFF_PEAK_NFFTS),
        bin_fraction=st.floats(min_value=0, max_value=1),
    )
    def test_stft_off_peak_bins(self, xp, array_backend, nfft, bin_fraction):
        k = tone_bin(nfft, bin_fraction)
        x = self._bin_centered_tone(nfft, k)
        X_ref = self._stft(x.astype(np.complex128), nfft)
        X = to_numpy(self._stft(xp.asarray(x), nfft)).astype(np.complex128)

        # the bins are in ascending fftfreq order, and a unit tone fills its bin
        # with the rect window's unit gain
        peak_bin = k + nfft // 2
        assert np.all(np.argmax(np.abs(X), axis=1) == peak_bin)
        tone_peak = np.abs(X_ref[0, peak_bin])
        assert abs(tone_peak - 1) < on_peak_roundoff(np.complex64)
        assert np.abs(np.abs(X[:, peak_bin]) - 1).max() < on_peak_roundoff(np.complex64)

        off_peak = np.ones(nfft, dtype=bool)
        off_peak[peak_bin] = False

        # window/nfft and the window multiply, then fft(nfft)
        sigma = fft_tolerance_rms(
            np.complex64, [nfft], n_elementwise=2, array_backend=array_backend
        )
        self._assert_off_peak_bins((X - X_ref)[:, off_peak], tone_peak, nfft, sigma)

    @given(
        nfft=st.sampled_from(OFF_PEAK_NFFTS),
        bin_fraction=st.floats(min_value=0, max_value=1),
    )
    def test_resample_off_peak_bins(self, xp, array_backend, nfft, bin_fraction):
        # keep the tone inside the half band that survives downsampling by 2
        k = tone_bin(nfft, 0.3 + 0.4 * bin_fraction)
        x = self._bin_centered_tone(nfft, k)
        num_out = x.size // 2
        # fftshift multiply, fft(N), ifft(N/2), ifftshift multiply
        sigma = fft_tolerance_rms(
            np.complex64,
            [x.size, num_out],
            n_elementwise=2,
            array_backend=array_backend,
        )
        out_ref = resample(x.astype(np.complex128), num_out)
        out = resample(xp.asarray(x), num_out)
        self._assert_off_peak_error_spectrum(out, out_ref, sigma)

    @given(
        nfft=st.sampled_from(OFF_PEAK_NFFTS),
        bin_fraction=st.floats(min_value=0, max_value=1),
    )
    def test_oaconvolve_off_peak_bins(self, xp, array_backend, nfft, bin_fraction):
        x = self._bin_centered_tone(nfft, tone_bin(nfft, bin_fraction))
        # a unit-gain lowpass; the tone may fall in its stopband, so bounds are
        # anchored to the input tone rather than the output
        kernel = np.hanning(self.KERNEL_TAPS).astype(np.float32)
        kernel /= kernel.sum()
        # the overlap-add block is at most x.size long
        sigma = fft_tolerance_rms(
            np.complex64, [x.size, x.size], n_elementwise=1, array_backend=array_backend
        )
        out_ref = oaconvolve(
            x.astype(np.complex128), kernel.astype(np.float64), mode='same'
        )
        out = oaconvolve(xp.asarray(x), xp.asarray(kernel), mode='same')
        self._assert_off_peak_error_spectrum(out, out_ref, sigma)


# %% numpy vs cupy
class TestNumpyCupyCrossComparison:
    """Tolerances for the FFT-based functions come from `cross_backend_sigma`,
    parameterized by the FFT sizes and elementwise operations each function performs.
    """

    @given(nfft=fft_sizes(min_size=8, max_size=512), fs=sample_rates())
    def test_fftfreq(self, cupy_available, nfft, fs):
        f_np, f_cp = numpy_and_cupy(
            cupy_available, fftfreq, nfft, fs, dtype=np.float64, xp_kwarg='xp'
        )
        assert_close(f_cp, f_np, rtol=RTOL_FLOAT64, atol=ATOL)

    def test_fftfreq_as_index_false_on_cupy(self, cupy_available):
        f_np = fftfreq(65, 1e6, as_index=False)
        f_cp = fftfreq(65, 1e6, as_index=False, xp=cupy_available)
        assert isinstance(f_cp, cupy_available.ndarray)
        assert_close(f_cp, f_np, rtol=RTOL_FLOAT64)

    @given(x=iq_waveforms(min_size=8, max_size=64, multiple_of=2, channels=3))
    def test_time_fftshift(self, cupy_available, x):
        scale = np.array([1, 2, 3], dtype=x.real.dtype)
        y_np, y_cp = numpy_and_cupy(cupy_available, time_fftshift, x, scale, axis=1)
        one_rounding = accum_rtol(x.dtype, 1)
        assert_close(y_cp, y_np, rtol=cross_backend(one_rounding, one_rounding))

    def test_frequency_slicing(self, cupy_available):
        x = np.arange(2 * 3 * 64, dtype=np.float32).reshape(2, 3, 64)

        t_np, t_cp = numpy_and_cupy(
            cupy_available, truncate_freqs, x, 64, 1.0, 0.5, axis=2
        )
        assert_array_equal(t_cp, t_np)

        def nulled(x):
            x = x.copy()
            null_lo(x, 64, 1.0, 0.25, axis=2)
            return x

        n_np, n_cp = numpy_and_cupy(cupy_available, nulled, x)
        assert_array_equal(n_cp, n_np)

    @given(x=iq_waveforms(min_size=256, max_size=1024, multiple_of=64, channels=4))
    def test_fft_out_buffer_and_chunking(self, cupy_available, max_cupy_fft_chunk, x):
        X_np = fft(x, axis=1)
        x_np_back = ifft(X_np, axis=1)

        # small enough that the 4-row input is split across several calls
        max_cupy_fft_chunk(x.shape[1])
        x_cp = cupy_available.asarray(x)
        out = cupy_available.empty_like(x_cp)
        X_cp = fft(x_cp, axis=1, out=out)
        assert X_cp is out or cupy_available.shares_memory(X_cp, out)
        x_cp_back = ifft(X_cp, axis=1, out=cupy_available.empty_like(x_cp))

        n = x.shape[1]
        sigma = cross_backend_sigma(x.dtype, [n])
        assert_close(X_cp, X_np, sigma=sigma, err_msg='fft')
        sigma = cross_backend_sigma(x.dtype, [n, n])
        assert_close(x_cp_back, x_np_back, sigma=sigma, err_msg='ifft')
