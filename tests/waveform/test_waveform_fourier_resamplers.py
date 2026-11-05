"""striqt.waveform.lib.fourier.resamplers: the COLA resampler design, the FFT
resampler against scaling, band-shift and domain identities, and the overlap-add
resampler on tones, on numpy and cupy.
"""

from __future__ import annotations

import functools
from fractions import Fraction

import numpy as np
import pytest
from conftest import gaussian_iq
from hypothesis import given, settings, strategies as st
from numeric_checks import (
    FIR_RECT_STOPBAND_DB,
    assert_close,
    assert_tone_level,
    interior,
    level_error_dB,
    numpy_and_cupy,
    tone_frequency,
)
from numpy.testing import assert_allclose, assert_array_equal
from waveform_strategies import cross_backend_sigma, msps_id, noise_waveforms

from striqt.analysis import testing
from striqt.waveform.lib.fourier.fft import (
    _broadcast_onto,
    fft,
    fft_tolerance_rms,
    time_fftshift,
)
from striqt.waveform.lib.fourier.filters import (
    _COLA_WINDOW_SIZE_DIVISOR,
    design_oafilter,
)
from striqt.waveform.lib.fourier.resamplers import (
    OLA_MAX_FFT_SIZE,
    _copy_bins_to_fftfreq_order,
    design_cola_resampler,
    design_fir_resampler,
    oaresample,
    resample,
    resample_edges,
)
from striqt.waveform.lib.fourier.stft import _find_downsample_copy_range

# %% design_cola_resampler and design_fir_resampler

# the window enters the design only through its COLA divisor. The hann rows span the
# rates and shifts; the blackman rows cover the other divisor.
COLA_DESIGN_CASES = [
    pytest.param(
        fs_base,
        fs_target,
        'hann',
        shift,
        id=f'{msps_id(fs_base)}-{msps_id(fs_target)}-{shift}',
    )
    for fs_base in [122.88e6, 125e6]
    for fs_target in [7.68e6, 15.36e6, 20e6, 30.72e6, 61.44e6, 107.52e6]
    for shift in [None, 'left', 'right']
] + [
    pytest.param(125e6, 15.36e6, 'blackman', None, id='125MSps-15.36MSps-blackman'),
    # 25/24 at 6 MS/s: the smallest rational pair is odd, and the sizing must grow
    # it to even multiples of 3
    pytest.param(125e6, 6e6, 'blackman', None, id='125MSps-6MSps-blackman'),
]


def largest_prime_factor(n: int) -> int:
    factor = 1
    for candidate in range(2, n + 1):
        while n % candidate == 0:
            n //= candidate
            factor = candidate
        if n == 1:
            return factor
    return factor


def reduced_denominator(fs_sdr: float, fs_target: float) -> int:
    return Fraction(fs_sdr / fs_target).limit_denominator(OLA_MAX_FFT_SIZE).denominator


class TestColaResamplerDesign:
    @pytest.mark.parametrize('fs_base, fs_target, window, shift', COLA_DESIGN_CASES)
    def test_consistency(self, fs_base, fs_target, window, shift):
        """the rates, FFT sizes, and passband of a design are mutually consistent.

        The grid omits `bw` and the shift spellings False and 'none'. `bw` only sets
        the passband width. False and 'none' design the same centered passband as
        None.
        """
        bw = 10e6
        design = design_cola_resampler(
            fs_base, fs_target, bw=bw, shift=shift, window=window
        )

        assert design['fs'] == design['fs_sdr'] <= fs_base
        # the SDR rate is an integer division of the base clock
        assert_allclose(fs_base / design['fs_sdr'], round(fs_base / design['fs_sdr']))
        # the FFT sizes give the rate ratio exactly, so the resampled rate is fs_target
        assert design['nfft'] * fs_target == pytest.approx(
            design['nfft_out'] * design['fs_sdr'], rel=1e-12
        )
        # both sizes are even multiples of the COLA divisor that the window imposes
        assert design['nfft'] % 2 == 0 and design['nfft_out'] % 2 == 0
        # the multiplier on the reduced ratio is 2·3·5·7-smooth
        q = reduced_denominator(design['fs_sdr'], fs_target)
        assert design['nfft_out'] % q == 0
        assert largest_prime_factor(design['nfft_out'] // q) <= 7
        for nfft, nfft_out in [
            (design['nfft'], design['nfft_out']),
            (design['nfft_out'], design['nfft']),
        ]:
            design_oafilter(
                nfft, window=window, nfft_out=nfft_out, nfft=nfft, extend=True
            )
        assert design['window'] == window

        lo, hi = design['passband']
        assert_allclose(hi - lo, bw)
        assert_allclose((hi + lo) / 2, design['lo_offset'])
        sign = {'left': -1, 'right': 1}.get(shift, 0)
        assert np.sign(design['lo_offset']) == sign
        # the shift must keep consecutive STFT frames in phase in oaresample
        bins = round(design['lo_offset'] / (design['fs_sdr'] / design['nfft']))
        assert bins % _COLA_WINDOW_SIZE_DIVISOR[window] == 0

    def test_options(self):
        design = design_cola_resampler(125e6, 107.52e6)
        assert design['window'] == 'hann'
        assert design['passband'] == (None, None) and design['lo_offset'] == 0

        forced = design_cola_resampler(125e6, 15.36e6, fs_sdr=50e6)
        assert forced['fs_sdr'] == pytest.approx(50e6)

        upsample = design_cola_resampler(10e6, 15.36e6)
        assert upsample['fs_sdr'] == pytest.approx(10e6)
        assert upsample['nfft'] < upsample['nfft_out']

        # 25/24 at 6 MS/s with the blackman divisor 6: the smallest even multiple
        # above min_fft_size is k = 684 = 2^2 * 3^2 * 19, and the design grows it
        # to the 7-smooth k = 720
        smooth = design_cola_resampler(
            125e6, 6e6, min_fft_size=16383, window='blackman'
        )
        assert smooth['fs_sdr'] == pytest.approx(6.25e6)
        q = reduced_denominator(smooth['fs_sdr'], 6e6)
        assert q == 24
        assert largest_prime_factor(smooth['nfft_out'] // q) <= 7
        assert (smooth['nfft'], smooth['nfft_out']) == (18000, 17280)

        fs_sdr, fir = design_fir_resampler(125e6, 107.52e6)
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
            ({'window': 'flattop'}, "'blackmanharris'.*, not 'flattop'"),
        ],
        ids=[
            'shift_without_bandwidth',
            'bandwidth_exceeds_nyquist',
            'bad_shift',
            'shift_needs_faster_radio',
            'irrational_ratio',
            'window_without_cola_divisor',
        ],
    )
    def test_errors(self, kws, match):
        kws = {'fs_base': 125e6, 'fs_target': 15.36e6, **kws}
        with pytest.raises(ValueError, match=match):
            design_cola_resampler(**kws)


# %% resample and resample_edges
class TestResample:
    def test_identity_returns_input(self):
        x = gaussian_iq(128, dtype=np.complex128, seed=42)
        assert resample(x, len(x)) is x

    @settings(max_examples=50)
    @given(scale=st.floats(min_value=0.5, max_value=2.0, allow_nan=False))
    def test_scale_parameter(self, scale):
        x = gaussian_iq(128, dtype=np.complex128, seed=42)
        num_out = len(x) // 2
        result_unscaled = resample(x, num_out, scale=1.0)
        result_scaled = resample(x, num_out, scale=scale)
        assert_allclose(result_scaled, scale * result_unscaled, rtol=1e-8, atol=1e-15)

    @settings(max_examples=30)
    @given(x=noise_waveforms(min_size=64, max_size=256, dtype=np.complex128))
    def test_frequency_domain_input(self, x):
        """domain='freq' takes the spectrum as fft(time_fftshift(x)) and matches the
        time-domain path"""
        num = x.size // 2
        X = fft(time_fftshift(x), axis=0)
        y_time = resample(x, num)
        y_freq = resample(X.copy(), num, domain='freq')
        y_frequency = resample(X.copy(), num, domain='frequency')
        y_freq_inplace = resample(X, num, domain='freq', overwrite_x=True)
        sigma = cross_backend_sigma(x.dtype, [x.size], n_elementwise=1)
        assert_close(y_freq, y_time, sigma=sigma)
        assert_array_equal(y_frequency, y_freq)
        assert_array_equal(y_freq_inplace, y_freq)

    @given(shift=st.integers(min_value=-16, max_value=16))
    def test_shift_moves_tone_when_downsampling(self, shift):
        nfft_in, nfft_out = 256, 128
        k = 26
        x = testing.tone(None, 1.0, frequency=k / nfft_in, count=nfft_in)[0]
        y = resample(x, nfft_out, shift=shift)
        assert y.shape == (nfft_out,) and y.dtype == x.dtype
        # the band copied to the output starts `shift` bins higher, so the tone
        # falls `shift` bins lower in the output spectrum
        expected = (k - shift) / nfft_out
        assert abs(tone_frequency(y, 1.0) - expected) <= 1 / nfft_out
        # a bin-centered unit tone comes back as a unit tone; fftshift multiply,
        # fft(N), ifft(N/2), ifftshift multiply
        sigma = fft_tolerance_rms(x.dtype, [nfft_in, nfft_out], n_elementwise=2)
        assert_close(np.abs(y), np.ones(y.size), sigma=sigma)

    @pytest.mark.parametrize(
        'kws, match',
        [
            ({'num': 128, 'domain': 'bogus'}, 'domain'),
            ({'num': 128, 'window': 'flattop'}, 'window'),
            ({'num': 512, 'shift': 1}, 'downsampling'),
            ({'num': 128, 'shift': -100}, 'too small'),
            ({'num': 128, 'shift': 100}, 'too large'),
        ],
    )
    def test_errors(self, kws, match):
        x = np.ones(256, dtype=np.complex64)
        with pytest.raises(ValueError, match=match):
            resample(x, **kws)

    def test_odd_length_rejected(self):
        with pytest.raises(ValueError, match='even'):
            resample(np.ones(255, dtype=np.complex64), 128)

    def test_resample_edges(self):
        """downsampling 256 to 128 bins keeps the middle 128, shifted by `shift`"""
        assert resample_edges(256, 128) is None
        assert resample_edges(256, 512) is None
        assert resample_edges(256, 128, shift=10) == (74, 202)
        assert resample_edges(256, 128, shift=-64) == (0, 128)
        assert resample_edges(256, 128, shift=64) == (128, 256)

    @pytest.mark.parametrize(
        'nfft_in, nfft_out, shift, match',
        [
            (255, 128, 0, 'even, not 255'),
            (256, 512, 1, 'downsampling'),
            (256, 128, -65, 'too small'),
            (256, 128, 65, 'too large'),
        ],
        ids=[
            'odd_input',
            'shift_while_upsampling',
            'shift_too_small',
            'shift_too_large',
        ],
    )
    def test_resample_edges_errors(self, nfft_in, nfft_out, shift, match):
        with pytest.raises(ValueError, match=match):
            resample_edges(nfft_in, nfft_out, shift)


# %% oaresample
class TestOaresample:
    FS = 1e6
    NFFT = 256
    NSEG = 40

    def _tone(self, f0, channels=None, count=None):
        if count is None:
            count = self.NFFT * self.NSEG
        x = testing.tone(None, self.FS, frequency=f0, count=count)[0]
        return x if channels is None else np.tile(x, (channels, 1))

    @settings(max_examples=20)
    @given(
        f0_bins=st.integers(min_value=-20, max_value=20),
        shift_bins=st.sampled_from([0, 8, -8]),
        scale=st.sampled_from([1.0, 2.0]),
    )
    def test_downsample_tone(self, f0_bins, shift_bins, scale):
        """as `sensor.lib.compute.corrections._oaresample` calls `oaresample`: 2-D
        (channel, sample) input on axis=1, a bin-grid frequency shift, and a voltage
        scale"""
        up, down = self.NFFT // 2, self.NFFT
        fs_out = self.FS * up / down
        f0 = f0_bins * self.FS / self.NFFT
        shift = shift_bins * self.FS / down
        x = self._tone(f0, channels=2)

        kws = {'window': 'hann', 'axis': 1, 'frequency_shift': shift, 'scale': scale}
        y = oaresample(x, up, down, self.FS, **kws)
        assert y.shape == (2, round(x.shape[1] * up / down))
        assert y.dtype == x.dtype
        yi = interior(y, up, axis=1)
        assert_tone_level(yi, scale)
        for row in yi:
            assert abs(tone_frequency(row, fs_out) - (f0 - shift)) <= fs_out / row.size

    def test_downsample_tone_with_a_scale_per_channel(self):
        """a 1-D `scale` applies one gain per channel, as `resample` does with the
        per-port `voltage_scale` of a calibrated acquisition"""
        up, down = self.NFFT // 2, self.NFFT
        x = self._tone(self.FS / self.NFFT * 4, channels=2)
        scale = np.array([1.0, 2.0])
        y = oaresample(x, up, down, self.FS, window='hann', axis=1, scale=scale)
        yi = interior(y, up, axis=1)
        for row, gain in zip(yi, scale):
            assert_tone_level(row[np.newaxis], gain)

    @pytest.mark.parametrize('window', ['hann', 'blackman'])
    def test_filter_bandwidth(self, window):
        """a tone inside the passband keeps its level; a tone past the transition band
        (``bw/2 + transition/2``) is rejected below the block-size floor"""
        divisor = _COLA_WINDOW_SIZE_DIVISOR[window]
        down = self.NFFT - self.NFFT % (2 * divisor)
        up = down // 2
        kws = {
            'window': window,
            'axis': 1,
            'filter_bandwidth': 0.2e6,
            'transition_bandwidth': 0.05e6,
        }
        # the STFT needs a whole number of hops, and `down` is a multiple of the hop
        count = down * self.NSEG
        y_pass = oaresample(self._tone(0.05e6, 1, count), up, down, self.FS, **kws)
        y_stop = oaresample(self._tone(0.15e6, 1, count), up, down, self.FS, **kws)
        assert_tone_level(interior(y_pass, up, axis=1))
        assert level_error_dB(interior(y_stop, up, axis=1)).max() < FIR_RECT_STOPBAND_DB

    def test_filter_bandwidth_within_the_transition_is_rejected(self):
        """a passband no wider than the default 250 kHz transition band leaves no
        passband, as `sensor.lib.compute.corrections.validate_fir_band` rejects it"""
        up, down = self.NFFT // 2, self.NFFT
        kws = {'window': 'hann', 'axis': 1, 'filter_bandwidth': 0.2e6}
        with pytest.raises(ValueError, match='no passband'):
            oaresample(self._tone(0.0, 1), up, down, self.FS, **kws)

    def test_upsample(self):
        up, down = 2 * self.NFFT, self.NFFT
        x = self._tone(0.05e6, 1)
        y = oaresample(x, up, down, self.FS, window='hann', axis=1)
        assert y.shape == (1, 2 * x.shape[1])
        yi = interior(y, up, axis=1)
        assert_tone_level(yi)
        f_err = abs(tone_frequency(yi[0], 2 * self.FS) - 0.05e6)
        assert f_err <= 2 * self.FS / yi.shape[1]

    def test_out_is_x_under_overwrite_x(self):
        """with an overlapping window the STFT leaves `x` free, so the output is
        written into the buffer of `x` and equals the output into a fresh array"""
        up, down = self.NFFT // 2, self.NFFT
        x = self._tone(0.05e6, channels=2)
        expected = oaresample(x.copy(), up, down, self.FS, window='hann', axis=1)

        y = oaresample(
            x, up, down, self.FS, window='hann', axis=1, overwrite_x=True, out=x
        )

        assert np.shares_memory(y, x)
        assert y.shape == expected.shape
        assert_array_equal(y, expected)

    def test_out_sharing_x_requires_overwrite_x(self):
        """without `overwrite_x` the input is kept, so it may not receive the output"""
        up, down = self.NFFT // 2, self.NFFT
        x = self._tone(0.05e6, channels=2)
        with pytest.raises(ValueError, match='overwrite_x'):
            oaresample(x, up, down, self.FS, window='hann', axis=1, out=x)

    @pytest.mark.parametrize(
        'make_out, match',
        [
            (lambda x: x[:, ::2], 'C-contiguous'),
            (lambda x: np.empty(x.shape, dtype=np.complex128), 'dtype'),
            (lambda x: np.empty(x.size // 4, dtype=x.dtype), 'fewer than'),
        ],
        ids=['strided', 'dtype', 'too_small'],
    )
    def test_out_buffer_rejected(self, make_out, match):
        up, down = self.NFFT // 2, self.NFFT
        x = self._tone(0.05e6, channels=2)
        with pytest.raises(ValueError, match=match):
            oaresample(x, up, down, self.FS, window='hann', axis=1, out=make_out(x))

    def test_out_is_x_rejected_when_upsampling(self):
        up, down = 2 * self.NFFT, self.NFFT
        x = self._tone(0.05e6, channels=1)
        with pytest.raises(ValueError, match='fewer than'):
            oaresample(
                x, up, down, self.FS, window='hann', axis=1, overwrite_x=True, out=x
            )

    @pytest.mark.parametrize(
        'up, down, shift, match',
        [
            (128, 256, 1e6 / 256 / 3, 'multiple of fs/down'),
            (128, 256, 3 * 1e6 / 256, 'multiple of 2'),
            (128, 256, 100 * 1e6 / 256, 'too large'),
            (128, 256, -100 * 1e6 / 256, 'too small'),
            (512, 256, 1e6 / 256, 'only supported when downsampling'),
        ],
    )
    def test_shift_errors(self, up, down, shift, match):
        x = self._tone(0.05e6, 1)
        with pytest.raises(ValueError, match=match):
            oaresample(x, up, down, self.FS, axis=1, frequency_shift=shift)


# %% _copy_bins_to_fftfreq_order
def reorder_case(
    xp,
    nfft: int,
    nfft_out: int,
    *,
    ports: int = 2,
    segments: int = 40,
    edges: tuple[int | None, int | None] = (None, None),
    per_port_gain: bool = False,
):
    """return ``(y, bounds_in, bounds_out, gain)`` for a reorder of `nfft` bins to
    `nfft_out` with the bounds that `oaresample` derives, and a gain distinct per bin
    (and per port when `per_port_gain`)."""
    y = xp.asarray(gaussian_iq((ports, segments, nfft)))
    if nfft_out <= nfft:
        bounds_out, bounds_in, _ = _find_downsample_copy_range(nfft, nfft_out, *edges)
    else:
        pad_left = (nfft_out - nfft) // 2
        bounds_in, bounds_out = (0, nfft), (pad_left, pad_left + nfft)
    gain = _broadcast_onto(xp.linspace(0.5, 1.5, nfft_out), y, axis=2)
    if per_port_gain:
        gain = gain * xp.arange(1, ports + 1).reshape(ports, 1, 1)
    return y, bounds_in, bounds_out, gain.astype(y.dtype)


def reorder(y, case, **kws):
    _, bounds_in, bounds_out, gain = case
    return _copy_bins_to_fftfreq_order(
        y, gain.shape[-1], bounds_in, bounds_out, gain=gain, axis=1, **kws
    )


class TestCopyBinsToFftfreqOrder:
    @pytest.mark.namespaces('numpy', 'cupy')
    @pytest.mark.parametrize(
        'nfft,nfft_out,edges',
        [(100, 86, (None, None)), (100, 86, (20, 100)), (64, 64, (None, None))],
        ids=['downsample', 'zeros_both_sides_of_dc', 'equal_sizes'],
    )
    def test_matches_ifftshift_of_cropped_bins(self, xp, nfft, nfft_out, edges):
        """the output is the gained passband placed at `bounds_out` of an otherwise
        zero ascending-order spectrum, rotated into `fftfreq` order"""
        case = reorder_case(xp, nfft, nfft_out, edges=edges)
        y, (bi0, bi1), (bo0, bo1), gain = case
        if edges != (None, None):
            assert bo0 > 0 and bo1 < nfft_out, 'zero bins on both sides of DC'

        expected = xp.zeros(y.shape[:2] + (nfft_out,), dtype=y.dtype)
        expected[..., bo0:bo1] = y[..., bi0:bi1] * gain[..., bo0:bo1]
        expected = xp.fft.ifftshift(expected, axes=-1)

        xp.testing.assert_array_equal(reorder(y, case, overwrite_x=False), expected)

    @pytest.mark.namespaces('numpy', 'cupy')
    @pytest.mark.parametrize(
        'per_port_gain', [False, True], ids=['one_gain', 'per_port']
    )
    @pytest.mark.parametrize(
        'nfft,nfft_out,segments,block_rows',
        [(100, 86, 40, 3), (100, 100, 40, 3), (64, 32, 17, 5), (100, 86, 1, 1)],
        ids=[
            'scratch_then_direct_with_partial_block',
            'equal_sizes_all_scratch',
            'halving_one_scratch_block',
            'single_segment',
        ],
    )
    def test_in_place_equals_out_of_place(
        self, xp, nfft, nfft_out, segments, block_rows, per_port_gain
    ):
        """the in-place reorder writes the same values as the out-of-place one into
        the head of the buffer of `y`. Small blocks force both the scratch and the
        direct regime and a partial final block; the cupy case is the one an ordering
        error fails, since numpy buffers overlapping ufunc operands on its own."""
        case = reorder_case(
            xp, nfft, nfft_out, segments=segments, per_port_gain=per_port_gain
        )
        y = case[0]
        reference = reorder(y.copy(), case, overwrite_x=False)

        result = reorder(y, case, overwrite_x=True, block_rows=block_rows)

        assert result.shape == (2, segments, nfft_out)
        assert result.flags.c_contiguous
        assert xp.shares_memory(result, y)
        xp.testing.assert_array_equal(result, reference)

    @pytest.mark.namespaces('numpy', 'cupy')
    @pytest.mark.parametrize(
        'reason',
        ['upsampling', 'non_contiguous', 'bin_axis_not_last', 'overwrite_x_false'],
    )
    def test_falls_back_to_a_copy_when_unsafe(self, xp, reason):
        """an output wider than the input, a strided `y`, or bins off the last axis
        take the out-of-place path"""
        overwrite_x = reason != 'overwrite_x_false'
        if reason == 'upsampling':
            case = reorder_case(xp, 64, 86)
            y = case[0]
        elif reason == 'non_contiguous':
            case = reorder_case(xp, 100, 86)
            wide = xp.asarray(gaussian_iq((2, 40, 120), seed=1))
            y = wide[:, :, :100]
        elif reason == 'bin_axis_not_last':
            case = reorder_case(xp, 100, 86)
            y = xp.asarray(gaussian_iq((2, 40, 100, 3), seed=1))
        else:
            case = reorder_case(xp, 100, 86)
            y = case[0]
        _, bounds_in, bounds_out, gain = case
        if reason == 'bin_axis_not_last':
            gain = gain[..., xp.newaxis]
        kws = {'gain': gain, 'axis': 1}

        reference = _copy_bins_to_fftfreq_order(
            y.copy(), gain.shape[2], bounds_in, bounds_out, **kws
        )
        result = _copy_bins_to_fftfreq_order(
            y, gain.shape[2], bounds_in, bounds_out, overwrite_x=overwrite_x, **kws
        )

        assert not xp.shares_memory(result, y)
        xp.testing.assert_array_equal(result, reference)

    @given(
        half_out=st.integers(min_value=2, max_value=24),
        extra=st.integers(min_value=0, max_value=12),
        segments=st.integers(min_value=1, max_value=12),
        block_rows=st.integers(min_value=1, max_value=6),
        ports=st.integers(min_value=1, max_value=2),
    )
    @settings(max_examples=30)
    def test_in_place_property(self, half_out, extra, segments, block_rows, ports):
        """every even output size at or below the input size reorders in place to
        the out-of-place values, at any block size"""
        nfft_out = 2 * half_out
        nfft = nfft_out + 2 * extra
        case = reorder_case(np, nfft, nfft_out, ports=ports, segments=segments)
        y = case[0]
        reference = reorder(y.copy(), case, overwrite_x=False)

        result = reorder(y, case, overwrite_x=True, block_rows=block_rows)

        assert np.shares_memory(result, y)
        assert_array_equal(result, reference)


# %% numpy vs cupy
class TestNumpyCupyCrossComparison:
    """tolerances come from `cross_backend_sigma`, parameterized by the FFT sizes and
    elementwise operations each function performs
    """

    @given(x=noise_waveforms(min_size=64, max_size=256, dtype=np.complex64))
    def test_resample(self, cupy_available, x):
        num_out = len(x) // 2
        y_np, y_cp = numpy_and_cupy(cupy_available, resample, x, num_out)
        # fftshift multiply, fft(N), ifft(N/2), ifftshift multiply
        sigma = cross_backend_sigma(x.dtype, [len(x), num_out], n_elementwise=2)
        assert_close(y_cp, y_np, sigma=sigma)

    @settings(max_examples=10)
    @given(f0_bins=st.integers(min_value=-20, max_value=20))
    def test_oaresample(self, cupy_available, f0_bins):
        fs, nfft = 1e6, 256
        x = testing.tone(
            None, fs, frequency=f0_bins * fs / nfft, count=nfft * 40, ports=2
        )

        shift = 8 * fs / nfft
        kws = {
            'window': 'hann',
            'axis': 1,
            'filter_bandwidth': 0.2e6,
            'transition_bandwidth': 0.05e6,
        }
        resampler = functools.partial(oaresample, frequency_shift=shift, **kws)
        y_np, y_cp = numpy_and_cupy(cupy_available, resampler, x, nfft // 2, nfft, fs)
        # window multiply, fft(nfft), zeroing, ifft(nfft), window multiply, overlap
        # add, plus the FIR multiply and the output scaling
        sigma = cross_backend_sigma(x.dtype, [nfft, nfft // 2], n_elementwise=5)
        assert_close(y_cp, y_np, sigma=sigma)

    def test_oaresample_out_is_x(self, cupy_available):
        """the output written into the buffer of `x` on cupy equals the output into
        a fresh array"""
        cp = cupy_available
        fs, nfft = 1e6, 256
        x = cp.asarray(
            testing.tone(None, fs, frequency=4 * fs / nfft, count=nfft * 40, ports=2)
        )
        kws = {
            'window': 'hann',
            'axis': 1,
            'filter_bandwidth': 0.2e6,
            'transition_bandwidth': 0.05e6,
        }
        expected = oaresample(x.copy(), nfft // 2, nfft, fs, **kws)

        y = oaresample(x, nfft // 2, nfft, fs, overwrite_x=True, out=x, **kws)

        assert cp.shares_memory(y, x)
        cp.testing.assert_array_equal(y, expected)
