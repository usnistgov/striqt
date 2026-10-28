"""striqt.waveform.lib.comms.sync_5gnr: the 5G NR synchronization used by the cellular
measurements.

The checks compare the PSS/SSS sequences, the SSB cases of TS 38.213 Section 4.1, the
sync parameters, SSB IQ extraction, the correlation grid, and the offset choice against
an independent restatement of the standard or a synthetic waveform with a known sync
sequence. The numpy and cupy paths are compared against each other.
"""

from __future__ import annotations

import dataclasses
from fractions import Fraction

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from numeric_checks import (
    assert_close,
    interior,
    numpy_and_cupy,
    reference_power,
    rms,
    tone_frequency,
)
from numpy.testing import assert_array_equal
from waveform_strategies import (
    FS,
    embed_pss_ports,
    embed_symbol,
    get_ssb_iq,
    phy_5g,
    pss_setup,
    silent_block,
    sync_params,
)

from striqt.analysis import testing
from striqt.waveform.lib.comms import sync_5gnr
from striqt.waveform.lib.comms.sync_5gnr import (
    _pss_m_sequence,
    _sss_m_sequence,
)

SC_COUNT = 127


def sync_bins(nfft, offset_bins=0):
    """return the FFT bin indices (in fftfreq order) occupied by a 5G NR sync sequence
    shifted by `offset_bins`.

    The PSS/SSS occupy subcarriers 56..182 of the 240-subcarrier SSB (TS 38.211 Table
    7.4.3.1-1), i.e., -64..62 relative to the SSB center.
    """
    k = np.arange(-64, 63) + offset_bins
    return k % nfft


class TestSyncSequences:
    @pytest.mark.parametrize('n_id2', [0, 1, 2])
    def test_pss_m_sequence(self, n_id2):
        """TS 38.211 7.4.2.2.1: d(n) = 1 - 2 x((n + 43 N_id2) mod 127), where
        x(i+7) = (x(i+4) + x(i)) mod 2 from x(6)...x(0) = 1 1 1 0 1 1 0"""
        x = [0, 1, 1, 0, 1, 1, 1]
        for i in range(SC_COUNT - 7):
            x.append((x[i + 4] + x[i]) % 2)
        expected = [1 - 2 * x[(n + 43 * n_id2) % SC_COUNT] for n in range(SC_COUNT)]

        assert_array_equal(_pss_m_sequence(n_id2), expected)

    def test_sss_m_sequences_are_distinct(self):
        sss = np.array([_sss_m_sequence(n) for n in range(1008)])
        assert sss.shape == (1008, SC_COUNT)
        assert set(sss.ravel()) <= {-1, 1}
        assert len({row.tobytes() for row in sss}) == 1008

    @given(n_id=st.integers(0, 1007))
    def test_sss_is_a_gold_sequence(self, n_id):
        """TS 38.211 7.4.2.3.1: the product of 2 shifted m-sequences"""
        n_id1, n_id2 = divmod(n_id, 3)
        m0 = 15 * (n_id1 // 112) + 5 * n_id2
        m1 = n_id1 % 112

        x0 = [1, 0, 0, 0, 0, 0, 0]
        x1 = [1, 0, 0, 0, 0, 0, 0]
        for i in range(7, SC_COUNT):
            x0.append((x0[i - 3] + x0[i - 7]) % 2)
            x1.append((x1[i - 6] + x1[i - 7]) % 2)
        s0 = 1 - 2 * np.roll(np.array(x0), -m0)
        s1 = 1 - 2 * np.roll(np.array(x1), -m1)

        assert_array_equal(_sss_m_sequence(n_id), s0 * s1)

    @pytest.mark.parametrize(
        'fs, scs', [(FS, 15e3), (FS, 30e3), (FS, 60e3), (15.36e6, 30e3)]
    )
    def test_pss_time_domain(self, fs, scs):
        phy = phy_5g(scs, fs)
        nfft = int(phy.nfft)
        cp = int(phy.cp_sizes.min())

        pss = sync_5gnr.pss_5g_nr(fs, scs)
        assert pss.shape == (3, nfft + cp)
        assert pss.dtype == np.complex64

        # the shortest cyclic prefix duration is zero-filled
        assert_array_equal(pss[:, :cp], 0)

        # the body is the inverse FFT of the m-sequence on the PSS subcarriers of the
        # SSB, normalized to unit energy in the frequency domain
        X = np.fft.fft(pss[:, cp:].astype(np.complex128), axis=1)
        expected = np.zeros((3, nfft), dtype=np.complex128)
        for n_id2 in range(3):
            expected[n_id2, sync_bins(nfft)] = _pss_m_sequence(n_id2)
        expected /= np.sqrt(SC_COUNT)
        assert_close(X, expected, atol=1e-5)
        assert_close(np.sum(np.abs(pss) ** 2, axis=1), 1 / nfft, rtol=1e-4)

    def test_pss_center_frequency_shifts_the_subcarriers(self):
        phy = phy_5g(30e3)
        nfft, cp = int(phy.nfft), int(phy.cp_sizes.min())

        pss = sync_5gnr.pss_5g_nr(FS, 30e3, 5 * 30e3, dtype='complex128')
        assert pss.dtype == np.complex128
        X = np.fft.fft(pss[0, cp:])
        occupied = np.abs(X) > 1e-6
        assert_array_equal(np.sort(np.where(occupied)[0]), np.sort(sync_bins(nfft, 5)))

    def test_sss_time_domain(self, subtests):
        phy = phy_5g(30e3)
        nfft, cp = int(phy.nfft), int(phy.cp_sizes.min())

        sss = sync_5gnr.sss_5g_nr(FS, 30e3)
        assert sss.shape == (1008, nfft + cp)
        assert sss.dtype == np.complex64
        assert_array_equal(sss[:, :cp], 0)

        for n_id in (0, 1, 500, 1007):
            with subtests.test(n_id=n_id):
                X = np.fft.fft(sss[n_id, cp:].astype(np.complex128))
                expected = np.zeros(nfft, dtype=np.complex128)
                expected[sync_bins(nfft)] = _sss_m_sequence(n_id)
                assert_close(X, expected / np.sqrt(SC_COUNT), atol=1e-5)

    @pytest.mark.parametrize(
        'args, match',
        [
            ((FS, 20e3), 'subcarrier_spacing'),
            ((1e6, 30e3), 'sample_rate'),
            ((FS + 15e3, 30e3), 'sample_rate'),
            ((FS, 30e3, 15e3), 'center_frequency'),
            ((FS, 30e3, 100 * 30e3), 'center_frequency'),
        ],
        ids=[
            'scs',
            'sample_rate_below_sequence',
            'sample_rate_off_grid',
            'center_frequency_off_grid',
            'center_frequency_beyond_nyquist',
        ],
    )
    def test_argument_errors(self, args, match):
        with pytest.raises(ValueError, match=match):
            sync_5gnr.pss_5g_nr(*args)


# TS 38.213 Section 4.1 lists the values of n for cases D and E explicitly
CASE_D_N = (0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 17, 18)
CASE_E_N = (0, 1, 2, 3, 5, 6, 7, 8)

# (subcarrier spacing, shared spectrum, symbol_indexes, center_frequency) ->
# (offsets, symbols per repetition, values of n) from TS 38.213 Section 4.1
SSB_CASES = [
    ((15e3, False, 'auto', None), ([2, 8], 14, range(4))),
    ((15e3, False, 'auto', 2e9), ([2, 8], 14, range(2))),
    ((15e3, False, 'auto', (1e9, 4e9)), ([2, 8], 14, range(4))),
    ((15e3, True, 'auto', None), ([2, 8], 14, range(5))),
    ((15e3, False, 'A', None), ([2, 8], 14, range(4))),
    ((30e3, True, 'auto', None), ([2, 8], 14, range(10))),
    ((30e3, False, 'b', None), ([4, 8, 16, 20], 28, range(2))),
    ((30e3, False, 'b', 2e9), ([4, 8, 16, 20], 28, range(1))),
    ((30e3, False, 'c', None), ([2, 8], 14, range(4))),
    ((30e3, False, 'c', 1.5e9), ([2, 8], 14, range(2))),
    ((120e3, False, 'auto', None), ([4, 8, 16, 20], 28, CASE_D_N)),
    ((240e3, False, 'auto', None), ([8, 12, 16, 20, 32, 36, 40, 44], 56, CASE_E_N)),
    ((480e3, False, 'auto', None), ([2, 9], 14, range(32))),
    ((960e3, False, 'auto', None), ([2, 9], 14, range(32))),
]


class TestIndexPssSymbols:
    @pytest.mark.parametrize('args, expected', SSB_CASES)
    def test_cell_search_cases(self, args, expected):
        offsets, period, n_values = expected
        table = tuple(o + period * n for n in n_values for o in offsets)
        assert sync_5gnr.index_pss_symbols(*args) == table

    def test_explicit_indexes_pass_through(self):
        assert sync_5gnr.index_pss_symbols(30e3, symbol_indexes=(3, 9)) == (3, 9)
        assert sync_5gnr.index_pss_symbols(30e3, symbol_indexes=[3, 9]) == [3, 9]

    @pytest.mark.parametrize(
        'kws, exc, match',
        [
            ({}, ValueError, 'choose case'),
            ({'subcarrier_spacing': 45e3}, ValueError, 'do not exist'),
            ({'symbol_indexes': 'z'}, ValueError, 'symbol_indexes'),
            (
                {'shared_spectrum': True, 'symbol_indexes': 'b'},
                ValueError,
                'shared_spectrum',
            ),
            ({'symbol_indexes': 5}, TypeError, None),
        ],
        ids=[
            '30khz_case_ambiguous',
            'scs_without_case',
            'unknown_case',
            'shared_spectrum_case_b',
            'symbol_indexes_type',
        ],
    )
    def test_argument_errors(self, kws, exc, match):
        with pytest.raises(exc, match=match):
            sync_5gnr.index_pss_symbols(**{'subcarrier_spacing': 30e3, **kws})


class TestSyncParams:
    def test_pss_params_structure(self):
        """the production configuration: 30 kHz Case C shared spectrum at 7.68 MS/s.

        TS 38.213 4.1 Case C with shared spectrum puts the PSS in symbols {2, 8} + 14n
        for n = 0..9. The closest pair is 6 symbols apart, so 5 lag symbols are
        searched, and the last candidate (134) plus those lags rounds up to 10 slots.
        A 30 kHz slot at 7.68 MS/s is 3840 samples, so a short symbol is
        3840 // 14 = 274 samples; the normal CP is 144 * 256 / 2048 = 18 samples and
        the excess 16 kappa in the first symbol is 7.68e6 / 1.92e6 = 4 samples.
        """
        params = sync_params()

        assert params.symbol_indexes == [
            *(2, 8, 16, 22, 30, 36, 44, 50, 58, 64),
            *(72, 78, 86, 92, 100, 106, 114, 120, 128, 134),
        ]
        assert params.max_lag_symbols == 5
        assert params.slot_count == 10
        assert params.frame_size == 76800
        assert params.frames_per_sync == 2
        assert params.duration == pytest.approx(5e-3)
        assert params.corr_size == 38400
        assert params.short_symbol_size == 274
        assert params.lag_count == 1370
        assert params.min_cp_size == 18
        assert params.cp_offsets == [4] * 14
        assert params.sample_rate == FS
        assert params.subcarrier_spacing == pytest.approx(30e3)
        assert params.shared_spectrum is True
        assert params.discovery_periodicity == pytest.approx(20e-3)

        assert sync_params(discovery_periodicity=40e-3).frames_per_sync == 4

    def test_explicit_max_lag_symbols(self):
        params = sync_params(max_lag_symbols=3)
        assert params.max_lag_symbols == 3
        assert params.lag_count == 3 * params.short_symbol_size

    def test_adjacent_symbol_indexes_need_explicit_max_lag_symbols(self):
        """the automatic lag search spans at least the 4 symbols of one SSB, and
        symbols 1 apart cannot separate that span"""
        with pytest.raises(ValueError, match=r'\(2, 3\).*max_lag_symbols'):
            sync_params(symbol_indexes=(2, 3))

        params = sync_params(symbol_indexes=(2, 3), max_lag_symbols=1)
        assert params.symbol_indexes == [2, 3]
        assert params.lag_count == params.short_symbol_size

    def test_sss_params_follow_pss_by_two_symbols(self, subtests):
        pss = sync_params()
        sss = sync_5gnr.sss_params(
            sample_rate=FS, subcarrier_spacing=30e3, shared_spectrum=True
        )
        assert sss.symbol_indexes == [i + 2 for i in pss.symbol_indexes]
        assert sss.max_lag_symbols == 2
        assert sss.lag_count == 2 * sss.short_symbol_size
        for name in (
            'frame_size',
            'frames_per_sync',
            'short_symbol_size',
            'min_cp_size',
        ):
            with subtests.test(field=name):
                assert getattr(sss, name) == getattr(pss, name)

        explicit = sync_5gnr.sss_params(
            sample_rate=FS, subcarrier_spacing=30e3, symbol_indexes=(4, 10)
        )
        assert explicit.symbol_indexes == [4, 10]

    @pytest.mark.parametrize(
        'kws, match',
        [
            ({'scs': 20e3}, 'subcarrier_spacing'),
            ({'fs': 8e6}, 'sample_rate'),
            ({'discovery_periodicity': 15e-3}, 'discovery_periodicity'),
        ],
        ids=['scs', 'sample_rate', 'discovery_periodicity'],
    )
    def test_argument_errors(self, kws, match):
        with pytest.raises(ValueError, match=match):
            sync_params(**kws)


class TestGet5gSsbIq:
    FS_IN = 15.36e6
    SIZE = round(4e-3 * 15.36e6)

    def _tones(self, f, scales=(1.0, 2.0), dtype=np.complex64):
        x = testing.tone(None, self.FS_IN, frequency=f, count=self.SIZE, dtype=dtype)[0]
        return np.stack([s * x for s in scales]).astype(dtype)

    def _ssb_iq(self, iq, **kws):
        return get_ssb_iq(iq, fs_in=self.FS_IN, **kws)

    @given(
        oaresample=st.booleans(),
        f0_bins=st.integers(-8, 8),
        offset_bins=st.sampled_from([0, 3, -3]),
    )
    @settings(max_examples=20)
    def test_recenters_and_downsamples(self, oaresample, f0_bins, offset_bins):
        """a tone at frequency_offset + f0 in the capture comes out at f0"""
        # the overlap-add path needs the offset on the grid fs_in/(3 * up)
        grid = 160e3 if oaresample else self.FS_IN / self.SIZE
        f0 = f0_bins * 100e3
        frequency_offset = offset_bins * grid

        iq = self._tones(f0 + frequency_offset)
        out = self._ssb_iq(iq, frequency_offset=frequency_offset, oaresample=oaresample)

        assert out.shape == (2, round(self.SIZE * FS / self.FS_IN))
        assert out.dtype == iq.dtype
        mid = interior(out, 2000, axis=1)
        f_est = tone_frequency(mid[0], FS)
        assert f_est == pytest.approx(f0, abs=FS / mid.shape[1])
        assert_close(rms(mid, axis=1), [1.0, 2.0], rtol=1e-3)

    def test_off_grid_frequency_offset(self):
        """a frequency_offset between input FFT bins still recenters the tone"""
        grid = self.FS_IN / self.SIZE
        f0 = 300e3
        frequency_offset = 3.4 * grid

        iq = self._tones(f0 + frequency_offset)
        out = self._ssb_iq(iq, frequency_offset=frequency_offset)

        mid = interior(out, 2000, axis=1)
        # the shift is applied in whole input bins, so the bound adds half a bin of
        # quantization to the resolution of the estimator
        f_est = tone_frequency(mid[0], FS)
        assert f_est == pytest.approx(f0, abs=grid / 2 + FS / mid.shape[1])
        assert_close(rms(mid, axis=1), [1.0, 2.0], rtol=1e-3)

    @pytest.mark.parametrize('oaresample', [False, True])
    def test_block_count_and_delay_crop_the_input(self, oaresample):
        iq = self._tones(1e6)
        blocks = {'discovery_periodicity': 1e-3, 'max_block_count': 2}
        delay = 0.5e-3
        out = self._ssb_iq(iq, delay=delay, oaresample=oaresample, **blocks)

        offs = round(delay * self.FS_IN)
        crop_duration = blocks['max_block_count'] * blocks['discovery_periodicity']
        size_in = round(crop_duration * self.FS_IN)
        expected = self._ssb_iq(iq[:, offs : offs + size_in], oaresample=oaresample)
        assert out.shape == (2, round(size_in * FS / self.FS_IN))
        assert_array_equal(out, expected)


class TestCorrelateSyncSequence:
    """a PSS symbol embedded in an otherwise silent synchronization block"""

    @given(
        n_id2=st.integers(0, 2),
        beam=st.integers(0, 19),
        delay=st.integers(0, 1000),
        scale=st.sampled_from([0.5, 1.0, 3.0]),
    )
    @settings(max_examples=25)
    def test_peak_locates_cell_beam_and_delay(self, n_id2, beam, delay, scale):
        params, phy, pss, bodies = pss_setup()
        iq = silent_block(params)
        symbol = params.symbol_indexes[beam]
        embed_symbol(iq, phy, bodies[n_id2], symbol, delay, scale=scale)

        R = sync_5gnr.correlate_sync_sequence(iq, pss, params=params)
        assert R.shape == (1, 3, 1, len(params.symbol_indexes), params.lag_count)
        assert R.dtype == np.complex64

        mag = np.abs(R)
        assert np.unravel_index(mag.argmax(), mag.shape) == (0, n_id2, 0, beam, delay)
        energy = reference_power(bodies[n_id2]).sum()
        assert mag.max() == pytest.approx(scale * energy, rel=1e-3)

    def test_sync_block_index(self):
        params, phy, pss, bodies = pss_setup()
        iq = silent_block(params, n_blocks=3)
        block_size = params.frame_size * params.frames_per_sync
        start = embed_symbol(
            iq[:, 2 * block_size :], phy, bodies[0], params.symbol_indexes[4], 7
        )
        assert start < params.corr_size

        R = sync_5gnr.correlate_sync_sequence(iq, pss, params=params)
        assert R.shape[2] == 3
        mag = np.abs(R)
        assert np.unravel_index(mag.argmax(), mag.shape) == (0, 0, 2, 4, 7)

    def test_multiple_ports(self, subtests):
        params, pss, iq = embed_pss_ports([(1, 2, 11), (2, 6, 23)])

        R = sync_5gnr.correlate_sync_sequence(iq, pss, params=params)
        assert R.shape[0] == 2
        for port, expected in enumerate([(1, 0, 2, 11), (2, 0, 6, 23)]):
            with subtests.test(port=port):
                mag = np.abs(R[port])
                assert np.unravel_index(mag.argmax(), mag.shape) == expected

    def test_mixed_excess_cp_rejected(self):
        params, _, pss, _ = pss_setup()
        bad = dataclasses.replace(
            params, cp_offsets=[0] + [4] * 13, symbol_indexes=[0, 2]
        )
        with pytest.raises(ValueError, match='same excess CP'):
            sync_5gnr.correlate_sync_sequence(silent_block(params), pss, params=bad)

    def test_sync_frame_count(self):
        params = sync_params()
        assert sync_5gnr.sync_frame_count(2 * params.frame_size, params) == 2

    def test_sync_frame_count_rejects_a_partial_frame(self):
        params, _, pss, _ = pss_setup()
        match = 'whole number of 10 ms frames'
        with pytest.raises(ValueError, match=match):
            sync_5gnr.sync_frame_count(params.frame_size + 1, params)
        with pytest.raises(ValueError, match=match):
            sync_5gnr.sync_frame_count(0, params)
        with pytest.raises(ValueError, match=match):
            sync_5gnr.correlate_sync_sequence(
                silent_block(params)[:, :-1], pss, params=params
            )

    def test_sync_frame_count_rejects_a_span_beyond_the_frame(self):
        """symbol 280 is past the 20 slots (280 symbols) of a 30 kHz frame, so the
        21-slot correlation span of that symbol cannot come from one frame"""
        params, _, pss, _ = pss_setup()
        beyond = sync_params(symbol_indexes=(280,))
        assert beyond.slot_count == 21
        with pytest.raises(ValueError, match=r'21-slot.*\(280,\).*does not fit'):
            sync_5gnr.correlate_sync_sequence(silent_block(params), pss, params=beyond)

    def test_sync_frame_count_rejects_too_few_lags(self):
        """one slot of 3840 samples less the 4 excess CP samples cannot fit a
        3840-sample lag search"""
        params = dataclasses.replace(
            sync_params(), slot_count=1, corr_size=3840, lag_count=3840
        )
        with pytest.raises(ValueError, match='3836 correlation samples, fewer than'):
            sync_5gnr.sync_frame_count(params.frame_size, params)


class TestChooseSsbOffset:
    def _correlate(self, delays, scales=None, beam=3, n_id2=1):
        embeds = [(n_id2, beam, delay) for delay in delays]
        params, pss, iq = embed_pss_ports(embeds, scales=scales)
        return params, sync_5gnr.correlate_sync_sequence(iq, pss, params=params)

    @given(
        delay=st.integers(0, 1000),
        window_fill=st.sampled_from([1, Fraction(1, 2), Fraction(3, 4), 0.9, 0.25]),
    )
    @settings(max_examples=20)
    def test_recovers_the_delay(self, delay, window_fill):
        params, R = self._correlate([delay])
        offset = sync_5gnr.choose_ssb_offset(R, params, window_fill=window_fill)
        assert offset.shape == (1,)
        assert offset[0] == delay

        # the port axis is optional
        assert_array_equal(sync_5gnr.choose_ssb_offset(R[0], params), offset)

    def test_last_fine_lag_wraps(self):
        delay = 3 * sync_params().short_symbol_size - 1
        params, R = self._correlate([delay])
        assert sync_5gnr.choose_ssb_offset(R, params)[0] == delay

    def test_weighted_detect_shape(self):
        params, R = self._correlate([300])
        weights = sync_5gnr.weighted_ssb_detect(R, params)
        assert weights.shape == (1, params.max_lag_symbols, params.short_symbol_size)

    def test_per_port(self):
        params, R = self._correlate([40, 700], scales=[1.0, 3.0])
        assert_array_equal(
            sync_5gnr.choose_ssb_offset(R, params, per_port=True), [40, 700]
        )
        # the ports are averaged in power, so the stronger one decides
        assert_array_equal(
            sync_5gnr.choose_ssb_offset(R, params, per_port=False), [700]
        )

    def test_max_beams(self):
        """only the first max_beams SSB candidates are searched"""
        params, phy, pss, bodies = pss_setup()
        iq = silent_block(params)
        embed_symbol(iq, phy, bodies[1], params.symbol_indexes[1], 55)
        embed_symbol(iq, phy, bodies[1], params.symbol_indexes[6], 400, scale=3.0)
        R = sync_5gnr.correlate_sync_sequence(iq, pss, params=params)

        assert sync_5gnr.choose_ssb_offset(R, params)[0] == 400
        assert sync_5gnr.choose_ssb_offset(R, params, max_beams=4)[0] == 55

    def test_rejects_low_dimensional_input(self):
        params, R = self._correlate([0])
        with pytest.raises(TypeError, match='5 dimensions'):
            sync_5gnr.weighted_ssb_detect(R[0, 0], params)

    def test_from_capture_rate_waveform(self):
        """the measurement path: capture at 15.36 MS/s, resample to 7.68 MS/s, detect"""
        fs_in = 15.36e6
        params, _, pss, _ = pss_setup()
        _, phy_in, _, bodies_in = pss_setup(fs_in)
        delay_in = 2 * 123

        iq = silent_block(params, n_blocks=2)
        embed_symbol(iq, phy_in, bodies_in[2], params.symbol_indexes[5], delay_in)

        ssb_iq = get_ssb_iq(iq, fs_in=fs_in)
        R = sync_5gnr.correlate_sync_sequence(ssb_iq, pss, params=params)
        mag = np.abs(R[0])
        n_id2, _, beam, lag = np.unravel_index(mag.argmax(), mag.shape)
        assert (n_id2, beam) == (2, 5)
        assert abs(lag - delay_in // 2) <= 1
        assert abs(sync_5gnr.choose_ssb_offset(R, params)[0] - delay_in // 2) <= 1


class TestCupy:
    def test_sync_sequences(self, cupy_available):
        pss_np, pss_cp = numpy_and_cupy(
            cupy_available, sync_5gnr.pss_5g_nr, FS, 30e3, xp_kwarg='xp'
        )
        assert_close(pss_cp, pss_np, atol=1e-6)
        sss_np, sss_cp = numpy_and_cupy(
            cupy_available, sync_5gnr.sss_5g_nr, FS, 30e3, xp_kwarg='xp'
        )
        assert_close(sss_cp, sss_np, atol=1e-6)

    def test_detection_pipeline(self, cupy_available):
        params, pss, iq = embed_pss_ports([(1, 3, 77)])

        ssb_np, ssb_cp = numpy_and_cupy(cupy_available, get_ssb_iq, iq)
        assert_close(ssb_cp, ssb_np, atol=1e-5)

        R_np, R_cp = numpy_and_cupy(
            cupy_available,
            sync_5gnr.correlate_sync_sequence,
            ssb_np,
            pss,
            params=params,
        )
        assert_close(R_cp, R_np, atol=1e-5)

        offset_np, offset_cp = numpy_and_cupy(
            cupy_available, sync_5gnr.choose_ssb_offset, R_np, params
        )
        assert offset_np[0] == offset_cp[0] == 77
