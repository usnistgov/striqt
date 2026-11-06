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
from numpy.testing import assert_allclose
from waveform_strategies import (
    cola_ripple_rtol,
    cola_sum,
    cross_backend_sigma,
    msps_id,
    noise_waveforms,
    tiled_tone,
)

from striqt.analysis import testing
from striqt.waveform.lib.fourier.fft import (
    _broadcast_onto,
    fft,
    fft_tolerance_rms,
    time_fftshift,
)
from striqt.waveform.lib.fourier.filters import design_oafilter
from striqt.waveform.lib.fourier.resamplers import (
    _copy_bins_to_fftfreq_order,
    design_cola_resampler,
    design_fir_resampler,
    oaresample,
    resample,
    resample_edges,
)
from striqt.waveform.lib.fourier.stft import _find_downsample_copy_range

# %% design_cola_resampler and design_fir_resampler

# the window enters the design only through its COLA hop count, and the shift only
# through the passband placement, so the hann rows span the rates at one shift, 2
# rows add the shifts at one rate pair, and the blackman rows cover the other hop
# count
COLA_DESIGN_CASES = [
    pytest.param(
        fs_base,
        fs_target,
        'hann',
        None,
        id=f'{msps_id(fs_base)}-{msps_id(fs_target)}-None',
    )
    for fs_base in [122.88e6, 125e6]
    for fs_target in [7.68e6, 15.36e6, 20e6, 30.72e6, 61.44e6, 107.52e6]
] + [
    pytest.param(125e6, 15.36e6, 'hann', 'left', id='125MSps-15.36MSps-left'),
    pytest.param(125e6, 15.36e6, 'hann', 'right', id='125MSps-15.36MSps-right'),
    pytest.param(125e6, 15.36e6, 'blackman', None, id='125MSps-15.36MSps-blackman'),
    pytest.param(
        125e6, 15.36e6, 'blackman', 'left', id='125MSps-15.36MSps-blackman-left'
    ),
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


def assert_cola_at_hop(window, nfft, overlap_scale):
    """assert that `window` of `nfft` samples satisfies COLA at the hop
    ``nfft - round(nfft * overlap_scale)``, and return the hop."""
    hop = nfft - round(nfft * overlap_scale)
    assert nfft % hop == 0
    total = cola_sum(window, nfft, hop)
    assert_close(total, total.mean(), rtol=cola_ripple_rtol(nfft, hop))
    return hop


class TestColaResamplerDesign:
    @pytest.mark.parametrize('fs_base, fs_target, window, shift', COLA_DESIGN_CASES)
    def test_consistency(self, fs_base, fs_target, window, shift):
        """the rates, FFT sizes, and passband of a design are mutually consistent.

        `bw` sets only the passband width, and the shift spellings False and 'none'
        design the same centered passband as None, so the grid omits them.
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
        # the half-band shift of the STFT is self-inverse only on an even block
        assert design['nfft'] % 2 == 0 and design['nfft_out'] % 2 == 0
        # the multiplier on the reduced ratio is 2·3·5·7-smooth
        q = Fraction(design['nfft'], design['nfft_out']).denominator
        assert largest_prime_factor(design['nfft_out'] // q) <= 7
        # the window satisfies COLA at the hop of both block sizes
        _, _, overlap_scale, _ = design_oafilter(
            design['nfft'],
            window=window,
            nfft_out=design['nfft_out'],
            nfft=design['nfft'],
            extend=True,
        )
        hop = assert_cola_at_hop(window, design['nfft'], overlap_scale)
        assert_cola_at_hop(window, design['nfft_out'], overlap_scale)
        assert design['window'] == window

        lo, hi = design['passband']
        assert_allclose(hi - lo, bw)
        assert_allclose((hi + lo) / 2, design['lo_offset'])
        sign = {'left': -1, 'right': 1}.get(shift, 0)
        assert np.sign(design['lo_offset']) == sign
        # a shift of `bins` bins rotates the phase by bins * hop / nfft cycles per
        # hop, which must be whole to keep consecutive STFT frames in phase
        bins = round(design['lo_offset'] / (design['fs_sdr'] / design['nfft']))
        assert (bins * hop) % design['nfft'] == 0

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
        q = Fraction(smooth['nfft'], smooth['nfft_out']).denominator
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

    @pytest.mark.parametrize(
        'domain, overwrite_x',
        [('time', False), ('freq', False), ('frequency', False), ('freq', True)],
        ids=['time', 'freq', 'frequency', 'freq_in_place'],
    )
    # 30 examples per row: the shift takes 33 values and the scale enters linearly
    @settings(max_examples=30)
    @given(
        shift=st.integers(min_value=-16, max_value=16),
        scale=st.floats(min_value=0.5, max_value=2.0),
    )
    def test_downsampled_tone_is_shifted_and_scaled(
        self, domain, overwrite_x, shift, scale
    ):
        """a bin-centered unit tone comes back as a tone at amplitude `scale`, `shift`
        bins lower in the output spectrum, from the time-domain input or from its
        spectrum ``fft(time_fftshift(x))`` under either spelling of `domain`"""
        nfft_in, nfft_out = 256, 128
        k = 26
        x = testing.tone(None, 1.0, frequency=k / nfft_in, count=nfft_in)[0]
        kws = {'shift': shift, 'scale': scale, 'overwrite_x': overwrite_x}
        if domain == 'time':
            y = resample(x, nfft_out, **kws)
        else:
            X = fft(time_fftshift(x), axis=0)
            y = resample(X, nfft_out, domain=domain, **kws)
        assert y.shape == (nfft_out,) and y.dtype == x.dtype
        # the band copied to the output starts `shift` bins higher, so the tone
        # falls `shift` bins lower in the output spectrum
        expected = (k - shift) / nfft_out
        assert abs(tone_frequency(y, 1.0) - expected) <= 1 / nfft_out
        # fftshift multiply, fft(N), ifft(N/2), ifftshift multiply, and the scale
        sigma = fft_tolerance_rms(x.dtype, [nfft_in, nfft_out], n_elementwise=3)
        assert_close(np.abs(y), np.full(y.size, scale), sigma=sigma)

    @pytest.mark.parametrize(
        'kws, match',
        [({'domain': 'bogus'}, 'domain'), ({'window': 'flattop'}, 'window')],
        ids=['domain', 'window'],
    )
    def test_errors(self, kws, match):
        x = np.ones(256, dtype=np.complex64)
        with pytest.raises(ValueError, match=match):
            resample(x, 128, **kws)

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
FILTER_KWS = {
    'window': 'hann',
    'axis': 1,
    'filter_bandwidth': 0.2e6,
    'transition_bandwidth': 0.05e6,
}


class TestOaresample:
    FS = 1e6
    NFFT = 256
    NSEG = 40
    COUNT = NFFT * NSEG

    @settings(max_examples=20)
    @given(
        f0_bins=st.integers(min_value=-20, max_value=20),
        shift_bins=st.sampled_from([0, 8, -8]),
        scale=st.sampled_from([1.0, 2.0]),
    )
    def test_downsample_tone(self, f0_bins, shift_bins, scale):
        """a tone in 2-D (channel, sample) input on ``axis=1`` is moved by a bin-grid
        frequency shift and scaled by the voltage `scale` at the output rate"""
        up, down = self.NFFT // 2, self.NFFT
        fs_out = self.FS * up / down
        f0 = f0_bins * self.FS / self.NFFT
        shift = shift_bins * self.FS / down
        x = tiled_tone(self.FS, f0, self.COUNT, 2)

        kws = {'window': 'hann', 'axis': 1, 'frequency_shift': shift, 'scale': scale}
        y = oaresample(x, up, down, self.FS, **kws)
        assert y.shape == (2, round(x.shape[1] * up / down))
        assert y.dtype == x.dtype
        yi = interior(y, up, axis=1)
        assert_tone_level(yi, scale)
        for row in yi:
            assert abs(tone_frequency(row, fs_out) - (f0 - shift)) <= fs_out / row.size

    def test_downsample_tone_with_a_scale_per_channel(self):
        """a 1-D `scale` applies one gain per channel"""
        up, down = self.NFFT // 2, self.NFFT
        x = tiled_tone(self.FS, self.FS / self.NFFT * 4, self.COUNT, 2)
        scale = np.array([1.0, 2.0])
        y = oaresample(x, up, down, self.FS, window='hann', axis=1, scale=scale)
        yi = interior(y, up, axis=1)
        for row, gain in zip(yi, scale):
            assert_tone_level(row[np.newaxis], gain)

    # the input block is the largest even multiple of the COLA hop count of the
    # window at or below NFFT, which `assert_cola_at_hop` verifies
    @pytest.mark.parametrize(
        'window, down', [('hann', 256), ('blackman', 252)], ids=['hann', 'blackman']
    )
    def test_filter_bandwidth(self, window, down):
        """a tone inside the passband keeps its level, and a tone past the transition
        band at ``bw/2 + transition/2`` is rejected below the block-size floor"""
        up = down // 2
        kws = {**FILTER_KWS, 'window': window}
        # the STFT needs a whole number of hops, and `down` is a multiple of the hop
        count = down * self.NSEG
        _, _, overlap_scale, _ = design_oafilter(
            count, window=window, nfft_out=up, nfft=down, extend=True
        )
        assert_cola_at_hop(window, down, overlap_scale)
        assert_cola_at_hop(window, up, overlap_scale)
        x_pass = tiled_tone(self.FS, 0.05e6, count, 1)
        x_stop = tiled_tone(self.FS, 0.15e6, count, 1)
        y_pass = oaresample(x_pass, up, down, self.FS, **kws)
        y_stop = oaresample(x_stop, up, down, self.FS, **kws)
        assert_tone_level(interior(y_pass, up, axis=1))
        assert level_error_dB(interior(y_stop, up, axis=1)).max() < FIR_RECT_STOPBAND_DB

    def test_upsample(self):
        up, down = 2 * self.NFFT, self.NFFT
        x = tiled_tone(self.FS, 0.05e6, self.COUNT, 1)
        y = oaresample(x, up, down, self.FS, window='hann', axis=1)
        assert y.shape == (1, 2 * x.shape[1])
        yi = interior(y, up, axis=1)
        assert_tone_level(yi)
        f_err = abs(tone_frequency(yi[0], 2 * self.FS) - 0.05e6)
        assert f_err <= 2 * self.FS / yi.shape[1]

    @pytest.mark.namespaces('numpy', 'cupy')
    def test_out_is_x_under_overwrite_x(self, xp):
        """with ``out=x`` under `overwrite_x`, the filtered downsample is written into
        the buffer of `x` and equals the output into a fresh array"""
        up, down = self.NFFT // 2, self.NFFT
        x = xp.asarray(tiled_tone(self.FS, 0.05e6, self.COUNT, 2))
        expected = oaresample(x.copy(), up, down, self.FS, **FILTER_KWS)

        y = oaresample(x, up, down, self.FS, overwrite_x=True, out=x, **FILTER_KWS)

        assert xp.shares_memory(y, x)
        assert y.shape == expected.shape
        xp.testing.assert_array_equal(y, expected)

    def test_out_sharing_x_requires_overwrite_x(self):
        """``out=x`` without `overwrite_x` raises `ValueError`"""
        up, down = self.NFFT // 2, self.NFFT
        x = tiled_tone(self.FS, 0.05e6, self.COUNT, 2)
        with pytest.raises(ValueError, match='overwrite_x'):
            oaresample(x, up, down, self.FS, window='hann', axis=1, out=x)

    @pytest.mark.parametrize(
        'make_out, kws, match',
        [
            (lambda x: x[:, ::2], {}, 'C-contiguous'),
            (lambda x: np.empty(x.shape, dtype=np.complex128), {}, 'dtype'),
            (lambda x: np.empty(x.size // 4, dtype=x.dtype), {}, 'fewer than'),
            (lambda x: x, {'up': 2 * NFFT, 'overwrite_x': True}, 'fewer than'),
        ],
        ids=['strided', 'dtype', 'too_small', 'x_when_upsampling'],
    )
    def test_out_buffer_rejected(self, make_out, kws, match):
        """a strided, mistyped, or undersized `out` is rejected, and `x` is undersized
        when upsampling"""
        kws = {'up': self.NFFT // 2, 'down': self.NFFT, 'axis': 1, **kws}
        x = tiled_tone(self.FS, 0.05e6, self.COUNT, 2)
        with pytest.raises(ValueError, match=match):
            oaresample(x, fs=self.FS, window='hann', out=make_out(x), **kws)

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
        x = tiled_tone(self.FS, 0.05e6, self.COUNT, 1)
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
    `nfft_out` with the bounds that `oaresample` derives, and a gain distinct per bin,
    and per port when `per_port_gain`."""
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
        direct regime and a partial final block."""
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
        """an output wider than the input, a strided `y`, bins off the last axis, or
        ``overwrite_x=False`` take the out-of-place path"""
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
        x = tiled_tone(fs, f0_bins * fs / nfft, nfft * 40, 2)

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
