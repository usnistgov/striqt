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
from hypothesis import given, settings, strategies as st
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
    cross_backend_sigma,
    embed_pss_ports,
    embed_symbol,
    get_ssb_spans,
    phy_5g,
    pss_setup,
    silent_block,
    sync_params,
    tiled_tone,
    window_cut,
)

from striqt.analysis import testing
from striqt.waveform.fourier import on_peak_roundoff
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


class TestGet5gSsbSpans:
    """the correlated span of each synchronization block, cut from a capture at
    15.36 MS/s into the 7.68 MS/s synchronization parameters of `sync_params`"""

    FS_IN = 15.36e6

    def _tones(self, params, f, scales=(1.0, 2.0), n_blocks=2):
        block_in = params.frame_size * params.frames_per_sync * round(self.FS_IN / FS)
        x = tiled_tone(self.FS_IN, f, n_blocks * block_in, len(scales))
        return (x * np.asarray(scales)[:, np.newaxis]).astype(np.complex64)

    @pytest.mark.parametrize(
        'f0, frequency_offset',
        [(0.0, 0.0), (500e3, 480e3), (-700e3, -480e3)],
        ids=['centered', 'above', 'below'],
    )
    def test_recenters_every_span(self, f0, frequency_offset):
        """a tone at frequency_offset + f0 in the capture comes out at f0, at its
        level, in every span"""
        params = sync_params()
        iq = self._tones(params, f0 + frequency_offset)
        spans = sync_5gnr.get_5g_ssb_spans(
            iq, params=params, fs_in=self.FS_IN, frequency_offset=frequency_offset
        )
        assert spans.shape == (2, 2, params.corr_size)
        assert spans.dtype == iq.dtype
        for block in range(2):
            mid = interior(spans[:, block], 500, axis=1)
            f_est = tone_frequency(mid[0], FS)
            assert f_est == pytest.approx(f0, abs=FS / mid.shape[1])
            assert_close(rms(mid, axis=1), [1.0, 2.0], rtol=1e-3)

    def test_delay_moves_the_block_grid(self):
        """a delay of one block makes the first span the second span of the
        undelayed call, margins included"""
        params = sync_params()
        iq = self._tones(params, 1e6)
        block_in = params.frame_size * params.frames_per_sync * round(self.FS_IN / FS)
        spans = sync_5gnr.get_5g_ssb_spans(iq, params=params, fs_in=self.FS_IN)
        delayed = sync_5gnr.get_5g_ssb_spans(
            iq,
            params=params,
            fs_in=self.FS_IN,
            delay=block_in / self.FS_IN,
            max_block_count=1,
        )
        assert delayed.shape == (2, 1, params.corr_size)
        assert_array_equal(delayed[:, 0], spans[:, 1])

    def test_rejects_a_fractional_block(self):
        params = sync_params()
        iq = self._tones(params, 1e6, n_blocks=1)
        with pytest.raises(ValueError, match='whole number of samples'):
            sync_5gnr.get_5g_ssb_spans(iq, params=params, fs_in=FS * 11 / 7)

    def test_one_frame_is_one_block(self):
        """a record of one 10 ms frame with a 20 ms discovery period yields the span
        of one block, as the whole-record path does"""
        params = sync_params()
        frame_in = params.frame_size * round(self.FS_IN / FS)
        iq = self._tones(params, 1e6, n_blocks=1)[:, :frame_in]
        spans = sync_5gnr.get_5g_ssb_spans(iq, params=params, fs_in=self.FS_IN)
        assert spans.shape == (2, 1, params.corr_size)

    def test_rejects_a_partial_frame(self):
        params = sync_params()
        iq = self._tones(params, 1e6, n_blocks=1)[:, :-1]
        with pytest.raises(ValueError, match='whole number of 10 ms frames'):
            sync_5gnr.get_5g_ssb_spans(iq, params=params, fs_in=self.FS_IN)


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

    def test_spans_equal_the_whole_record(self):
        """`correlate_sync_spans` on the spans that the whole-record call cuts from
        each block returns the same grid"""
        params, pss, iq = embed_pss_ports([(1, 2, 11), (2, 6, 23)], n_blocks=2)
        blocks = iq.reshape(iq.shape[0], -1, params.frame_size)
        spans = blocks[:, :: params.frames_per_sync, : params.corr_size]

        R = sync_5gnr.correlate_sync_sequence(iq, pss, params=params)
        assert_array_equal(sync_5gnr.correlate_sync_spans(spans, pss, params=params), R)

    @pytest.mark.parametrize('cells', [0, 2], ids=['one_byte', 'two_cells'])
    def test_cell_id_chunks_do_not_change_the_result(self, cells):
        """the result does not depend on the chunking of the cell IDs"""
        params, pss, iq = embed_pss_ports([(1, 2, 11), (2, 6, 23)])
        spans = iq.reshape(2, -1, params.frame_size)[:, :: params.frames_per_sync]
        spans = spans[..., : params.corr_size]
        ports, sync_count, _ = spans.shape
        nfft = sync_5gnr._correlation_fft_size(params.corr_size + pss.shape[1] - 1, np)
        # 1 byte splits the 3 cell IDs into chunks of one; a product buffer of
        # exactly 2 cells splits the cell IDs into chunks of 2 and 1
        one_cell = ports * sync_count * nfft * np.dtype(np.complex64).itemsize
        max_chunk_bytes = max(1, cells * one_cell)

        whole = sync_5gnr.correlate_sync_spans(spans, pss, params=params)
        chunked = sync_5gnr.correlate_sync_spans(
            spans, pss, params=params, max_chunk_bytes=max_chunk_bytes
        )
        assert_array_equal(chunked, whole)

    def test_rejects_a_span_of_the_wrong_length(self):
        params, pss, iq = embed_pss_ports([(1, 2, 11)])
        spans = iq[:, np.newaxis, : params.corr_size - 1]
        with pytest.raises(ValueError, match='corr_size'):
            sync_5gnr.correlate_sync_spans(spans, pss, params=params)

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

        spans = get_ssb_spans(iq, params, fs_in=fs_in)
        R = sync_5gnr.correlate_sync_spans(spans, pss, params=params)
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

        ssb_np, ssb_cp = numpy_and_cupy(cupy_available, get_ssb_spans, iq, params)
        assert_close(ssb_cp, ssb_np, atol=1e-5)

        R_np, R_cp = numpy_and_cupy(
            cupy_available,
            sync_5gnr.correlate_sync_spans,
            ssb_np,
            pss,
            params=params,
        )
        assert_close(R_cp, R_np, atol=1e-5)

        offset_np, offset_cp = numpy_and_cupy(
            cupy_available, sync_5gnr.choose_ssb_offset, R_np, params
        )
        assert offset_np[0] == offset_cp[0] == 77

    def test_multiport_multiblock_correlation(self, cupy_available):
        """2 ports and 5 blocks of noise at 15.36 MS/s: a sliding window over the lags
        of the correlation grid would span more than 2**31 elements, and the cupy
        result must match numpy on both ports"""
        params, phy, pss, bodies = pss_setup(fs=15.36e6)
        iq = silent_block(params, n_blocks=5, ports=2)
        iq += testing.circular_awgn(
            None,
            params.sample_rate,
            power=1e-2 * reference_power(bodies[0]).mean(),
            ports=2,
            count=iq.shape[1],
        )
        embeds = [(0, 2, 100), (1, 4, 300)]
        for port, (n_id2, beam, delay) in enumerate(embeds):
            symbol = params.symbol_indexes[beam]
            embed_symbol(iq, phy, bodies[n_id2], symbol, delay, port=port)

        R_np, R_cp = numpy_and_cupy(
            cupy_available, sync_5gnr.correlate_sync_sequence, iq, pss, params=params
        )
        full_size = params.corr_size + pss.shape[1] - 1
        sigma = cross_backend_sigma(iq.dtype, (full_size, full_size), n_elementwise=2)
        # the rows with the embedded symbol carry the structured roundoff of a matched
        # filter, relative to the peak rather than to the rms of the noise rows
        peak_atol = on_peak_roundoff(iq.dtype) * np.abs(R_np).max()
        assert_close(R_cp, R_np, sigma=sigma, atol=peak_atol)
        for port, (n_id2, beam, delay) in enumerate(embeds):
            mag = np.abs(R_cp[port])
            assert np.unravel_index(mag.argmax(), mag.shape) == (n_id2, 0, beam, delay)


# %% _lag_window_pieces
def _window_params():
    """return the 3 lag window shapes as parametrize cases. In the default 10-slot
    span each of the 20 windows lies inside one slot. With `symbol_indexes=(12, 23)`
    in a 3-slot span, the window of symbol 12 crosses the slot boundary at 3836
    samples and the window of symbol 23 ends on the next boundary. In a 1-slot span,
    the window of symbol 9 crosses the roll boundary at ``head = 3821`` with 19 lags
    left."""
    single = sync_params()
    across_slot = sync_params(symbol_indexes=(12, 23))
    across_roll = dataclasses.replace(
        single, slot_count=1, corr_size=3840, symbol_indexes=[9], lag_count=5 * 274
    )
    return [
        pytest.param(single, id='within_one_slot'),
        pytest.param(across_slot, id='across_slot_boundary'),
        pytest.param(across_roll, id='across_roll_boundary'),
    ]


@pytest.mark.parametrize('params', _window_params())
def test_pieces_match_the_rolled_and_trimmed_index_map(params):
    """assembling the pieces from an index ramp equals the lag window cut from the
    ramp rolled by -offs, split into slots, trimmed of the excess CP and merged; every
    source slice lies inside the inverse FFT"""
    head, offs, nfft = window_cut(params)
    pieces = sync_5gnr._lag_window_pieces(params, head, offs)

    y = np.arange(nfft)
    rolled = np.empty(params.corr_size, dtype=int)
    rolled[:head] = y[offs : offs + head]
    rolled[head:] = y[: params.corr_size - head]
    excess = params.cp_offsets[params.symbol_indexes[0] % 14]
    trimmed = rolled.reshape(params.slot_count, -1)[:, excess:].reshape(-1)

    for symbol, window in zip(params.symbol_indexes, pieces):
        start = symbol * params.short_symbol_size
        assembled = np.empty(params.lag_count, dtype=int)
        for dst, src in window:
            assert 0 <= src.start < src.stop <= nfft
            assembled[dst] = y[src]
        assert_array_equal(assembled, trimmed[start : start + params.lag_count])


def test_pieces_reject_a_window_past_the_span():
    """symbol 9 at 274 samples per symbol needs lags 2466 to 3836 of a 3836-sample
    trimmed slot; one more lag ends past the span"""
    params = dataclasses.replace(
        sync_params(), slot_count=1, corr_size=3840, symbol_indexes=[9], lag_count=1371
    )
    head, offs, _ = window_cut(params)
    with pytest.raises(ValueError, match='ends at 3837, past the 3836'):
        sync_5gnr._lag_window_pieces(params, head, offs)


def test_pieces_reject_a_corr_size_off_the_slot_grid():
    params = dataclasses.replace(sync_params(), corr_size=38401)
    head, offs, _ = window_cut(params)
    with pytest.raises(ValueError, match='not a multiple of the slot_count'):
        sync_5gnr._lag_window_pieces(params, head, offs)
