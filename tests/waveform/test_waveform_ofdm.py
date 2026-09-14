"""Tests for the parts of striqt.waveform.lib.ofdm that striqt.analysis uses.

The cellular measurements call `get_3gpp_phy`/`Phy3GPP.index_cyclic_prefix` and
`corr_at_indices` (cyclic autocorrelation), `pss_params`/`sss_params`,
`pss_5g_nr`/`sss_5g_nr`, `get_5g_ssb_iq`, `correlate_sync_sequence` and
`choose_ssb_offset` (5G NR synchronization). Each is checked against an independent
restatement of the 3GPP structure it encodes, and against a synthetic waveform that
contains a known cyclic prefix or synchronization sequence.
"""

from __future__ import annotations

import dataclasses
from fractions import Fraction
from math import ceil

import numpy as np
import pytest
from conftest import to_numpy
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose, assert_array_equal
from test_fourier import tone_frequency
from test_waveform_jit import corr_atol

from striqt.waveform.lib import ofdm

# the analysis default synchronization block sample rate (15.36e6 / 2)
FS = 7.68e6
SCS_5G = (15e3, 30e3, 60e3)
SC_COUNT = 127


def phy_5g(scs, fs=FS, xp=None):
    return ofdm.get_3gpp_phy(
        1, subcarrier_spacing=scs, sample_rate=fs, generation='5G', xp=xp
    )


def sync_params(scs=30e3, fs=FS, **kws):
    """the parameters of the sensor sweep configurations: 30 kHz shared spectrum"""
    kws.setdefault('shared_spectrum', round(scs) == 30_000)
    return ofdm.pss_params(sample_rate=fs, subcarrier_spacing=scs, **kws)


def cp_index_reference(phy, frames, symbols, slots):
    """straightforward loop construction of the cyclic prefix index tensor.

    Axis order (symbol, slot, frame, cp sample), with singleton axes removed as the
    library does.
    """
    slots_per_frame = phy.SCS_TO_SLOTS_PER_FRAME[phy.subcarrier_spacing]
    symbols = range(phy.FFT_PER_SLOT) if symbols == 'all' else symbols
    slots = range(slots_per_frame) if slots == 'all' else slots
    ncp = int(phy.cp_sizes[1])

    inds = np.empty((len(symbols), len(slots), len(frames), ncp), dtype=int)
    for a, symbol in enumerate(symbols):
        for b, slot in enumerate(slots):
            for c, frame in enumerate(frames):
                start = (
                    frame * phy.frame_size
                    + slot * phy.contiguous_size
                    + phy.cp_start_idx[symbol]
                )
                inds[a, b, c] = start + np.arange(ncp)
    return inds.squeeze()


def ofdm_slots(phy, n_slots, seed=0, dtype=np.complex64):
    """`n_slots` contiguous slots of unit-power QPSK OFDM symbols with cyclic prefixes"""
    rng = np.random.default_rng(seed)
    nfft = int(phy.nfft)
    qpsk = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]) / np.sqrt(2)

    parts = []
    for _ in range(n_slots):
        for cp in phy.cp_sizes:
            body = np.fft.ifft(rng.choice(qpsk, size=nfft)) * np.sqrt(nfft)
            parts += [body[-int(cp) :], body]
    return np.concatenate(parts).astype(dtype)


def sync_bins(nfft, offset_bins=0):
    """fft bin indices (in fftfreq order) occupied by a 5G NR sync sequence.

    The PSS/SSS occupy subcarriers 56..182 of the 240-subcarrier SSB, i.e. -64..62
    relative to the SSB center.
    """
    k = np.arange(-64, 63) + offset_bins
    return k % nfft


def embed_symbol(iq, phy, body, symbol_index, delay, port=0, scale=1.0):
    """write an OFDM symbol (with its cyclic prefix) into `iq` at the nominal position
    of `symbol_index` in the frame structure, offset by `delay` samples"""
    slot, symbol = divmod(symbol_index, phy.FFT_PER_SLOT)
    start = slot * phy.contiguous_size + phy.cp_start_idx[symbol] + delay
    cp = int(phy.cp_sizes[symbol])
    wave = scale * np.concatenate([body[-cp:], body])
    iq[port, start : start + wave.size] = wave
    return start


def pss_setup(fs=FS):
    """(params, phy, pss, bodies) for 30 kHz PSS detection at `fs`, where `bodies`
    are the PSS symbols without their cyclic prefix"""
    params = sync_params(fs=fs)
    phy = phy_5g(30e3, fs)
    pss = ofdm.pss_5g_nr(fs, 30e3)
    return params, phy, pss, pss[:, int(phy.cp_sizes.min()) :]


def silent_block(params, n_blocks=1, ports=1):
    size = params.frame_size * params.frames_per_sync * n_blocks
    return np.zeros((ports, size), dtype=np.complex64)


def get_ssb_iq(iq, fs_in=FS, **kws):
    """get_5g_ssb_iq with the 30 kHz, 20 ms discovery defaults of the measurements"""
    kws = {
        'discovery_periodicity': 20e-3,
        'fs_out': FS,
        'fs_in': fs_in,
        'subcarrier_spacing': 30e3,
        **kws,
    }
    return ofdm.get_5g_ssb_iq(iq, **kws)


class TestPhy3GPP:
    @pytest.mark.xfail(
        strict=True,
        reason='the LTE table has two extended CPs per 14 symbols (LTE_MIN_CP_SIZES) '
        'but the 5G computation produces one; TS 38.211 5.3.1 agrees with the LTE '
        'table at 15 kHz',
    )
    @pytest.mark.parametrize('sample_rate', [1.92e6, 3.84e6, 7.68e6, 15.36e6, 30.72e6])
    def test_lte_and_5g_agree_at_15khz(self, sample_rate):
        lte = ofdm.Phy3GPP(1, sample_rate=sample_rate, generation='4G')
        nr = ofdm.Phy3GPP(1, sample_rate=sample_rate, generation='5G')
        assert_array_equal(lte.cp_sizes, nr.cp_sizes)
        assert lte.nfft == nr.nfft == round(sample_rate / 15e3)

    @given(
        sample_rate=st.sampled_from([3.84e6, 7.68e6, 15.36e6, 30.72e6, 61.44e6]),
        scs=st.sampled_from(SCS_5G),
    )
    def test_5g_slot_structure(self, sample_rate, scs):
        """3GPP TS 38.211 5.3.1: normal CP is 144*kappa*2**-mu samples, with an
        extra 16*kappa in the first symbol"""
        nfft = round(sample_rate / scs)
        mu = round(np.log2(scs / 15e3))

        if (144 * nfft) % 2048:
            with pytest.raises(ValueError, match='non-integer cyclic prefix'):
                phy_5g(scs, sample_rate)
            return
        phy = phy_5g(scs, sample_rate)

        assert phy.nfft == nfft
        assert phy.subcarrier_spacing == pytest.approx(scs)
        assert phy.frame_size == round(10e-3 * sample_rate)
        assert len(phy.cp_sizes) == phy.FFT_PER_SLOT == 14

        normal_cp = 144 * nfft // 2048
        assert_array_equal(phy.cp_sizes[1:], normal_cp)
        assert phy.cp_sizes[0] == normal_cp + 16 * nfft * 2**mu // 2048
        assert phy.cp_sizes[0] - phy.cp_sizes[1] == round(sample_rate / 1.92e6)

        # the contiguous block is one slot of symbols; the index attributes tile it
        assert phy.contiguous_size == int(phy.cp_sizes.sum() + 14 * nfft)
        assert phy.cp_start_idx[0] == 0
        assert_array_equal(np.diff(phy.cp_start_idx), phy.cp_sizes[:-1] + nfft)
        assert phy.cp_idx.size == phy.cp_sizes.sum()
        assert_array_equal(
            np.sort(np.concatenate([phy.cp_idx, phy.symbol_idx])),
            np.arange(phy.contiguous_size),
        )
        if nfft in phy.FFT_SIZE_TO_SUBCARRIERS:
            assert phy.subcarriers == phy.FFT_SIZE_TO_SUBCARRIERS[nfft]

    @pytest.mark.parametrize(
        'scs',
        [
            pytest.param(
                15e3,
                marks=pytest.mark.xfail(
                    strict=True,
                    reason='TS 38.211 5.3.1 adds the extra 16*kappa CP samples at '
                    'symbols l=0 and l=7*2**mu of each 1 ms subframe; Phy3GPP adds '
                    'them once per 14 symbols, which is one too few per slot at '
                    '15 kHz',
                ),
            ),
            30e3,
            pytest.param(
                60e3,
                marks=pytest.mark.xfail(
                    strict=True,
                    reason='at 60 kHz the extra 16*kappa CP samples belong to one '
                    'slot in two; Phy3GPP adds them to every slot',
                ),
            ),
        ],
    )
    def test_slots_tile_the_frame(self, scs):
        phy = phy_5g(scs, 15.36e6)
        slots = phy.SCS_TO_SLOTS_PER_FRAME[scs]
        assert slots * phy.contiguous_size == phy.frame_size

    def test_argument_errors(self):
        with pytest.raises(ValueError, match='subcarrier_spacing'):
            ofdm.Phy3GPP(1, subcarrier_spacing=45e3, sample_rate=FS)
        with pytest.raises(ValueError, match='counting number'):
            ofdm.Phy3GPP(1, subcarrier_spacing=30e3, sample_rate=FS + 1)
        with pytest.raises(ValueError, match='generation'):
            ofdm.Phy3GPP(1, sample_rate=FS, generation='LTE')
        with pytest.raises(ValueError, match='non-integer cyclic prefix'):
            ofdm.Phy3GPP(
                1, subcarrier_spacing=60e3, sample_rate=1.92e6, generation='5G'
            )

    def test_default_sample_rate_from_bandwidth(self):
        for bw, fs in ofdm.Phy3GPP.BW_TO_SAMPLE_RATE.items():
            assert ofdm.Phy3GPP(bw).sample_rate == fs

    def test_get_3gpp_phy_caches(self):
        a = ofdm.get_3gpp_phy(
            1, subcarrier_spacing=30e3, sample_rate=FS, generation='5G'
        )
        b = ofdm.get_3gpp_phy(
            1, subcarrier_spacing=30e3, sample_rate=FS, generation='5G'
        )
        assert a is b


class TestIndexCyclicPrefix:
    @given(
        scs=st.sampled_from(SCS_5G),
        frames=st.sampled_from([(0,), (0, 1), (2,), (1, 3)]),
        data=st.data(),
    )
    def test_matches_loop_reference(self, scs, frames, data):
        phy = phy_5g(scs)
        slots_per_frame = phy.SCS_TO_SLOTS_PER_FRAME[scs]

        symbols = data.draw(
            st.one_of(
                st.just('all'),
                st.lists(st.integers(0, 13), min_size=1, max_size=4, unique=True).map(
                    tuple
                ),
            )
        )
        slots = data.draw(
            st.one_of(
                st.just('all'),
                st.lists(
                    st.integers(0, slots_per_frame - 1),
                    min_size=1,
                    max_size=4,
                    unique=True,
                ).map(tuple),
            )
        )

        inds = phy.index_cyclic_prefix(frames=frames, symbols=symbols, slots=slots)
        assert_array_equal(inds, cp_index_reference(phy, frames, symbols, slots))

    @pytest.mark.parametrize('scs', SCS_5G)
    def test_default_call_shape(self, scs):
        """the cyclic autocorrelation measurement: frame 0, all symbols, given slots"""
        phy = phy_5g(scs)
        slots_per_frame = phy.SCS_TO_SLOTS_PER_FRAME[scs]
        inds = phy.index_cyclic_prefix(frames=(0,), symbols='all', slots='all')
        assert inds.shape == (14, slots_per_frame, phy.cp_sizes[1])
        assert inds.min() == 0
        assert inds.max() < slots_per_frame * phy.contiguous_size

        # indices are cyclic prefix samples of the slot structure
        assert np.isin(inds % phy.contiguous_size, phy.cp_idx).all()

    def test_single_selection_keeps_its_offset(self):
        phy = phy_5g(30e3)
        base = phy.index_cyclic_prefix(slots=(0,))
        assert base.shape == (14, phy.cp_sizes[1])
        assert_array_equal(
            phy.index_cyclic_prefix(slots=(7,)), base + 7 * phy.contiguous_size
        )
        assert_array_equal(
            phy.index_cyclic_prefix(frames=(2,), slots=(0,)), base + 2 * phy.frame_size
        )
        assert_array_equal(
            phy.index_cyclic_prefix(symbols=(5,), slots=(0,)),
            phy.cp_start_idx[5] + np.arange(phy.cp_sizes[1]),
        )

    def test_is_cached(self):
        phy = phy_5g(30e3)
        assert phy.index_cyclic_prefix() is phy.index_cyclic_prefix()

    def test_argument_errors(self):
        phy = phy_5g(30e3)
        slots_per_frame = phy.SCS_TO_SLOTS_PER_FRAME[30e3]

        with pytest.raises(ValueError, match='indices or "all"'):
            phy.index_cyclic_prefix(slots='bogus')
        with pytest.raises(ValueError, match='exceeds the maximum'):
            phy.index_cyclic_prefix(slots=(slots_per_frame,))
        with pytest.raises(ValueError, match='exceeds the maximum'):
            phy.index_cyclic_prefix(symbols=(14,))
        with pytest.raises(ValueError, match='below the minimum'):
            phy.index_cyclic_prefix(symbols=(-15,))
        with pytest.raises(ValueError, match='sequence of indices'):
            phy.index_cyclic_prefix(slots=((0, 1), (2, 3)))
        with pytest.raises(ValueError, match='"all"'):
            ofdm._index_or_all('all', 'x', size=None)


class TestCorrAtIndices:
    """cyclic prefix correlation on a synthetic OFDM waveform, indexed as the
    cellular cyclic autocorrelation measurement does"""

    @pytest.mark.parametrize('scs', SCS_5G)
    def test_normalized_correlation_peaks_at_zero_lag(self, scs):
        phy = phy_5g(scs)
        slots_per_frame = phy.SCS_TO_SLOTS_PER_FRAME[scs]
        x = ofdm_slots(phy, slots_per_frame)
        inds = phy.index_cyclic_prefix()
        ncp = int(phy.cp_sizes[1])

        R = np.abs(ofdm.corr_at_indices(inds, x, phy.nfft, norm=True))
        assert R.shape == (phy.nfft + ncp,)

        # the cyclic prefix is an exact copy of the symbol tail
        assert R[0] == pytest.approx(1, abs=corr_atol(x, inds.size, norm=True))

        # partial overlap decays linearly over the prefix length; the remaining
        # pairs are independent QPSK products that average out
        lags = np.arange(ncp)
        assert_allclose(R[:ncp], 1 - lags / ncp, atol=0.1)
        assert R[ncp : phy.nfft].max() < 0.1

    @pytest.mark.parametrize('scs', SCS_5G)
    def test_unnormalized_zero_lag_is_prefix_power(self, scs):
        phy = phy_5g(scs)
        x = ofdm_slots(phy, phy.SCS_TO_SLOTS_PER_FRAME[scs])
        inds = phy.index_cyclic_prefix()

        R = ofdm.corr_at_indices(inds, x, phy.nfft, norm=False)
        power = np.mean(np.abs(x[inds.ravel()].astype(np.complex128)) ** 2)
        assert R[0] == pytest.approx(power, abs=corr_atol(x, inds.size, norm=False))
        assert R.dtype == x.dtype

    def test_out_buffer_and_ncp_from_indices(self):
        phy = phy_5g(30e3)
        x = ofdm_slots(phy, 2)
        inds = phy.index_cyclic_prefix(slots=(0, 1))
        out = np.empty(phy.nfft + inds.shape[-1], dtype=x.dtype)
        R = ofdm.corr_at_indices(inds, x, phy.nfft, out=out)
        assert R is out


class TestSyncSequences:
    @pytest.mark.parametrize('n_id2', [0, 1, 2])
    def test_pss_m_sequence(self, n_id2):
        """TS 38.211 7.4.2.2: d(n) = 1 - 2 x((n + 43 N_id2) mod 127)"""
        pss = np.array(ofdm._pss_m_sequence(n_id2))
        base = np.array(ofdm._pss_m_sequence(0))

        assert pss.shape == (SC_COUNT,)
        assert set(pss) <= {-1, 1}
        assert pss.sum() == -1  # the m-sequence is balanced: 64 ones, 63 zeros
        assert_array_equal(pss, np.roll(base, -43 * n_id2))

    def test_sss_m_sequences_are_distinct(self):
        sss = np.array([ofdm._sss_m_sequence(n) for n in range(1008)])
        assert sss.shape == (1008, SC_COUNT)
        assert set(sss.ravel()) <= {-1, 1}
        assert len({row.tobytes() for row in sss}) == 1008

    @given(n_id=st.integers(0, 1007))
    def test_sss_is_a_gold_sequence(self, n_id):
        """TS 38.211 7.4.2.3.1: the product of two shifted m-sequences"""
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

        assert_array_equal(ofdm._sss_m_sequence(n_id), s0 * s1)

    @pytest.mark.parametrize(
        'fs, scs', [(FS, 15e3), (FS, 30e3), (FS, 60e3), (15.36e6, 30e3)]
    )
    def test_pss_time_domain(self, fs, scs):
        phy = phy_5g(scs, fs)
        nfft = int(phy.nfft)
        cp = int(phy.cp_sizes.min())

        pss = ofdm.pss_5g_nr(fs, scs)
        assert pss.shape == (3, nfft + cp)
        assert pss.dtype == np.complex64

        # the shortest cyclic prefix duration is zero-filled
        assert_array_equal(pss[:, :cp], 0)

        # the body is the inverse FFT of the m-sequence on the SSB's PSS
        # subcarriers, normalized to unit energy in the frequency domain
        X = np.fft.fft(pss[:, cp:].astype(np.complex128), axis=1)
        expected = np.zeros((3, nfft), dtype=np.complex128)
        for n_id2 in range(3):
            expected[n_id2, sync_bins(nfft)] = ofdm._pss_m_sequence(n_id2)
        expected /= np.sqrt(SC_COUNT)
        assert_allclose(X, expected, atol=1e-5)
        assert_allclose(np.sum(np.abs(pss) ** 2, axis=1), 1 / nfft, rtol=1e-4)

    def test_pss_center_frequency_shifts_the_subcarriers(self):
        phy = phy_5g(30e3)
        nfft, cp = int(phy.nfft), int(phy.cp_sizes.min())

        pss = ofdm.pss_5g_nr(FS, 30e3, 5 * 30e3, dtype='complex128')
        assert pss.dtype == np.complex128
        X = np.fft.fft(pss[0, cp:])
        occupied = np.abs(X) > 1e-6
        assert_array_equal(np.sort(np.where(occupied)[0]), np.sort(sync_bins(nfft, 5)))

    def test_sss_time_domain(self):
        phy = phy_5g(30e3)
        nfft, cp = int(phy.nfft), int(phy.cp_sizes.min())

        sss = ofdm.sss_5g_nr(FS, 30e3)
        assert sss.shape == (1008, nfft + cp)
        assert sss.dtype == np.complex64
        assert_array_equal(sss[:, :cp], 0)

        for n_id in (0, 1, 500, 1007):
            X = np.fft.fft(sss[n_id, cp:].astype(np.complex128))
            expected = np.zeros(nfft, dtype=np.complex128)
            expected[sync_bins(nfft)] = ofdm._sss_m_sequence(n_id)
            assert_allclose(X, expected / np.sqrt(SC_COUNT), atol=1e-5)

    def test_argument_errors(self):
        with pytest.raises(ValueError, match='multiple of 15000'):
            ofdm.pss_5g_nr(FS, 20e3)
        with pytest.raises(ValueError, match='at least'):
            ofdm.pss_5g_nr(1e6, 30e3)
        with pytest.raises(ValueError, match='multiple of subcarrier spacing'):
            ofdm.pss_5g_nr(FS + 15e3, 30e3)
        with pytest.raises(ValueError, match='whole multiple'):
            ofdm.pss_5g_nr(FS, 30e3, 15e3)
        with pytest.raises(ValueError, match='outside of Nyquist'):
            ofdm.pss_5g_nr(FS, 30e3, 100 * 30e3)


# (subcarrier spacing, shared spectrum, symbol_indexes, center_frequency) ->
# (offsets, symbols per repetition, repetitions) from TS 38.213 Section 4.1
SSB_CASES = [
    ((15e3, False, 'auto', None), ([2, 8], 14, 4)),
    ((15e3, False, 'auto', 2e9), ([2, 8], 14, 2)),
    ((15e3, False, 'auto', (1e9, 4e9)), ([2, 8], 14, 4)),
    ((15e3, True, 'auto', None), ([2, 8], 14, 5)),
    ((15e3, False, 'A', None), ([2, 8], 14, 4)),
    ((30e3, True, 'auto', None), ([2, 8], 14, 10)),
    ((30e3, False, 'b', None), ([4, 8, 16, 20], 28, 2)),
    ((30e3, False, 'b', 2e9), ([4, 8, 16, 20], 28, 1)),
    ((30e3, False, 'c', None), ([2, 8], 14, 4)),
    ((30e3, False, 'c', 1.5e9), ([2, 8], 14, 2)),
    ((120e3, False, 'auto', None), ([4, 8, 16, 20], 28, 19)),
    ((240e3, False, 'auto', None), ([8, 12, 16, 20, 32, 36, 40, 44], 56, 9)),
    ((480e3, False, 'auto', None), ([2, 9], 14, 32)),
    ((960e3, False, 'auto', None), ([2, 9], 14, 32)),
]


class TestIndexPssSymbols:
    @pytest.mark.parametrize('args, expected', SSB_CASES)
    def test_cell_search_cases(self, args, expected):
        offsets, period, repetitions = expected
        table = tuple(o + period * n for n in range(repetitions) for o in offsets)
        assert ofdm.index_pss_symbols(*args) == table

    def test_explicit_indexes_pass_through(self):
        assert ofdm.index_pss_symbols(30e3, symbol_indexes=(3, 9)) == (3, 9)
        assert ofdm.index_pss_symbols(30e3, symbol_indexes=[3, 9]) == [3, 9]

    def test_argument_errors(self):
        with pytest.raises(ValueError, match='"b" or "c"'):
            ofdm.index_pss_symbols(30e3)
        with pytest.raises(ValueError, match='do not exist'):
            ofdm.index_pss_symbols(45e3)
        with pytest.raises(ValueError, match='invalid str'):
            ofdm.index_pss_symbols(30e3, symbol_indexes='z')
        with pytest.raises(ValueError, match='shared_spectrum unsupported'):
            ofdm.index_pss_symbols(30e3, shared_spectrum=True, symbol_indexes='b')
        with pytest.raises(TypeError):
            ofdm.index_pss_symbols(30e3, symbol_indexes=5)


class TestSyncParams:
    def test_min_diff(self):
        assert ofdm._min_diff([]) is None
        assert ofdm._min_diff([5]) is None
        assert ofdm._min_diff([2, 8, 16]) == 6

    @pytest.mark.parametrize(
        'scs, kws',
        [
            (30e3, {'shared_spectrum': True}),
            (15e3, {}),
            (15e3, {'center_frequency': 2e9}),
            (30e3, {'symbol_indexes': 'b'}),
            (30e3, {'symbol_indexes': (4,)}),
            (60e3, {'symbol_indexes': (2, 8), 'discovery_periodicity': 40e-3}),
        ],
    )
    def test_pss_params_structure(self, scs, kws):
        params = ofdm.pss_params(sample_rate=FS, subcarrier_spacing=scs, **kws)
        phy = phy_5g(scs)
        slot_duration = 1e-3 * 15e3 / scs
        symbols = ofdm.index_pss_symbols(
            scs,
            kws.get('shared_spectrum', False),
            kws.get('symbol_indexes', 'auto'),
            kws.get('center_frequency'),
        )

        assert params.symbol_indexes == list(symbols)
        assert params.frame_size == round(10e-3 * FS)
        assert params.frames_per_sync == round(
            kws.get('discovery_periodicity', 20e-3) / 10e-3
        )

        spacing = ofdm._min_diff(sorted(symbols))
        assert params.max_lag_symbols == (
            5 if spacing is not None and spacing >= 5 else 4
        )

        assert params.slot_count == ceil(
            (symbols[-1] + params.max_lag_symbols + 1) / 14
        )
        assert params.duration == pytest.approx(params.slot_count * slot_duration)
        assert params.corr_size == round(params.duration * FS)
        assert params.short_symbol_size == round(slot_duration * FS) // 14
        assert params.lag_count == params.short_symbol_size * params.max_lag_symbols

        assert params.min_cp_size == phy.cp_sizes.min()
        assert_array_equal(
            params.cp_offsets, np.cumsum(phy.cp_sizes - phy.cp_sizes.min())
        )
        assert params.sample_rate == FS
        assert params.subcarrier_spacing == scs

    def test_explicit_max_lag_symbols(self):
        params = sync_params(max_lag_symbols=3)
        assert params.max_lag_symbols == 3
        assert params.lag_count == 3 * params.short_symbol_size

    def test_sss_params_follow_pss_by_two_symbols(self):
        pss = sync_params()
        sss = ofdm.sss_params(
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
            assert getattr(sss, name) == getattr(pss, name)

        explicit = ofdm.sss_params(
            sample_rate=FS, subcarrier_spacing=30e3, symbol_indexes=(4, 10)
        )
        assert explicit.symbol_indexes == [4, 10]

    def test_argument_errors(self):
        with pytest.raises(ValueError, match='multiple of 15000'):
            ofdm.pss_params(sample_rate=FS, subcarrier_spacing=20e3)
        with pytest.raises(ValueError, match='sample_rate must be a multiple'):
            ofdm.pss_params(
                sample_rate=8e6, subcarrier_spacing=30e3, shared_spectrum=True
            )
        with pytest.raises(ValueError, match='discovery_periodicity'):
            sync_params(discovery_periodicity=15e-3)


class TestGet5gSsbIq:
    FS_IN = 15.36e6
    SIZE = round(4e-3 * 15.36e6)

    def _tones(self, f, scales=(1.0, 2.0), dtype=np.complex64):
        n = np.arange(self.SIZE)
        x = np.exp(2j * np.pi * f / self.FS_IN * n)
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
        interior = out[:, 2000:-2000]
        assert tone_frequency(interior[0], FS) == pytest.approx(
            f0, abs=FS / interior.shape[1]
        )
        rms = np.sqrt(np.mean(np.abs(interior) ** 2, axis=1))
        assert_allclose(rms, [1.0, 2.0], rtol=1e-3)

    @pytest.mark.parametrize('oaresample', [False, True])
    def test_block_count_and_delay_crop_the_input(self, oaresample):
        iq = self._tones(1e6)
        discovery_periodicity = 1e-3
        max_block_count = 2
        delay = 0.5e-3

        out = self._ssb_iq(
            iq,
            discovery_periodicity=discovery_periodicity,
            max_block_count=max_block_count,
            delay=delay,
            oaresample=oaresample,
        )
        offs = round(delay * self.FS_IN)
        size_in = round(max_block_count * discovery_periodicity * self.FS_IN)
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
        embed_symbol(
            iq, phy, bodies[n_id2], params.symbol_indexes[beam], delay, scale=scale
        )

        R = ofdm.correlate_sync_sequence(iq, pss, params=params)
        assert R.shape == (1, 3, 1, len(params.symbol_indexes), params.lag_count)
        assert R.dtype == np.complex64

        mag = np.abs(R)
        assert np.unravel_index(mag.argmax(), mag.shape) == (0, n_id2, 0, beam, delay)
        energy = np.sum(np.abs(bodies[n_id2].astype(np.complex128)) ** 2)
        assert mag.max() == pytest.approx(scale * energy, rel=1e-3)

    def test_sync_block_index(self):
        params, phy, pss, bodies = pss_setup()
        iq = silent_block(params, n_blocks=3)
        block_size = params.frame_size * params.frames_per_sync
        start = embed_symbol(
            iq[:, 2 * block_size :], phy, bodies[0], params.symbol_indexes[4], 7
        )
        assert start < params.corr_size

        R = ofdm.correlate_sync_sequence(iq, pss, params=params)
        assert R.shape[2] == 3
        mag = np.abs(R)
        assert np.unravel_index(mag.argmax(), mag.shape) == (0, 0, 2, 4, 7)

    def test_multiple_ports(self):
        params, phy, pss, bodies = pss_setup()
        iq = silent_block(params, ports=2)
        embed_symbol(iq, phy, bodies[1], params.symbol_indexes[2], 11, port=0)
        embed_symbol(iq, phy, bodies[2], params.symbol_indexes[6], 23, port=1)

        R = ofdm.correlate_sync_sequence(iq, pss, params=params)
        assert R.shape[0] == 2
        for port, expected in enumerate([(1, 0, 2, 11), (2, 0, 6, 23)]):
            mag = np.abs(R[port])
            assert np.unravel_index(mag.argmax(), mag.shape) == expected

    def test_mixed_excess_cp_rejected(self):
        params, _, pss, _ = pss_setup()
        bad = dataclasses.replace(
            params, cp_offsets=[0] + [4] * 13, symbol_indexes=[0, 2]
        )
        with pytest.raises(ValueError, match='same excess CP'):
            ofdm.correlate_sync_sequence(silent_block(params), pss, params=bad)


class TestChooseSsbOffset:
    def _correlate(self, delays, scales=None, beam=3, n_id2=1):
        params, phy, pss, bodies = pss_setup()
        scales = scales or [1.0] * len(delays)

        iq = silent_block(params, ports=len(delays))
        for port, (delay, scale) in enumerate(zip(delays, scales)):
            embed_symbol(
                iq,
                phy,
                bodies[n_id2],
                params.symbol_indexes[beam],
                delay,
                port=port,
                scale=scale,
            )
        return params, ofdm.correlate_sync_sequence(iq, pss, params=params)

    @given(
        delay=st.integers(0, 1000),
        window_fill=st.sampled_from([1, Fraction(1, 2), Fraction(3, 4), 0.9, 0.25]),
    )
    @settings(max_examples=20)
    def test_recovers_the_delay(self, delay, window_fill):
        params, R = self._correlate([delay])
        offset = ofdm.choose_ssb_offset(R, params, window_fill=window_fill)
        assert offset.shape == (1,)
        assert offset[0] == delay

        # the port axis is optional
        assert_array_equal(ofdm.choose_ssb_offset(R[0], params), offset)

    def test_last_fine_lag_wraps(self):
        delay = 3 * sync_params().short_symbol_size - 1
        params, R = self._correlate([delay])
        assert ofdm.choose_ssb_offset(R, params)[0] == delay

    def test_weighted_detect_shape(self):
        params, R = self._correlate([300])
        weights = ofdm.weighted_ssb_detect(R, params)
        assert weights.shape == (1, params.max_lag_symbols, params.short_symbol_size)
        symbol, sample = np.unravel_index(weights[0].argmax(), weights[0].shape)
        assert symbol * params.short_symbol_size + sample == 300

    def test_per_port(self):
        params, R = self._correlate([40, 700], scales=[1.0, 3.0])
        assert_array_equal(ofdm.choose_ssb_offset(R, params, per_port=True), [40, 700])
        # the ports are averaged in power, so the stronger one decides
        assert_array_equal(ofdm.choose_ssb_offset(R, params, per_port=False), [700])

    def test_max_beams(self):
        params, R = self._correlate([55], beam=3)
        assert ofdm.choose_ssb_offset(R, params, max_beams=4)[0] == 55
        assert (
            ofdm.choose_ssb_offset(R, params, max_beams=len(params.symbol_indexes) + 5)[
                0
            ]
            == 55
        )

    def test_rejects_low_dimensional_input(self):
        params, R = self._correlate([0])
        with pytest.raises(TypeError, match='5 dimensions'):
            ofdm.weighted_ssb_detect(R[0, 0], params)

    def test_from_capture_rate_waveform(self):
        """the measurement path: capture at 15.36 MS/s, resample to 7.68 MS/s, detect"""
        fs_in = 15.36e6
        params, _, pss, _ = pss_setup()
        _, phy_in, _, bodies_in = pss_setup(fs_in)
        delay_in = 2 * 123

        iq = silent_block(params, n_blocks=2)
        embed_symbol(iq, phy_in, bodies_in[2], params.symbol_indexes[5], delay_in)

        ssb_iq = get_ssb_iq(iq, fs_in=fs_in)
        R = ofdm.correlate_sync_sequence(ssb_iq, pss, params=params)
        mag = np.abs(R[0])
        n_id2, _, beam, lag = np.unravel_index(mag.argmax(), mag.shape)
        assert (n_id2, beam) == (2, 5)
        assert abs(lag - delay_in // 2) <= 1
        assert abs(ofdm.choose_ssb_offset(R, params)[0] - delay_in // 2) <= 1


class TestCupy:
    def test_phy_and_index(self, cupy_available):
        cp = cupy_available
        phy = phy_5g(30e3, xp=cp)
        ref = phy_5g(30e3)
        assert_array_equal(to_numpy(phy.cp_sizes), ref.cp_sizes)
        inds = phy.index_cyclic_prefix(slots=(0, 3))
        assert isinstance(inds, cp.ndarray)
        assert_array_equal(to_numpy(inds), ref.index_cyclic_prefix(slots=(0, 3)))

    def test_corr_at_indices(self, cupy_available):
        cp = cupy_available
        ref = phy_5g(30e3)
        phy = phy_5g(30e3, xp=cp)
        x = ofdm_slots(ref, 4)
        inds_np = ref.index_cyclic_prefix(slots=(0, 1, 2, 3))

        R_np = ofdm.corr_at_indices(inds_np, x, ref.nfft, norm=False)
        R_cp = ofdm.corr_at_indices(
            phy.index_cyclic_prefix(slots=(0, 1, 2, 3)),
            cp.asarray(x),
            phy.nfft,
            norm=False,
        )
        atol = corr_atol(x, inds_np.size, norm=False, n_impl=2)
        assert_allclose(to_numpy(R_cp), R_np, atol=atol)

    def test_sync_sequences(self, cupy_available):
        cp = cupy_available
        pss = ofdm.pss_5g_nr(FS, 30e3, xp=cp)
        assert isinstance(pss, cp.ndarray)
        assert_allclose(to_numpy(pss), ofdm.pss_5g_nr(FS, 30e3), atol=1e-6)
        sss = ofdm.sss_5g_nr(FS, 30e3, xp=cp)
        assert_allclose(to_numpy(sss[:8]), ofdm.sss_5g_nr(FS, 30e3)[:8], atol=1e-6)

    def test_detection_pipeline(self, cupy_available):
        cp = cupy_available
        params, phy, pss, bodies = pss_setup()
        iq = silent_block(params)
        embed_symbol(iq, phy, bodies[1], params.symbol_indexes[3], 77)

        ssb_np = get_ssb_iq(iq)
        ssb_cp = get_ssb_iq(cp.asarray(iq))
        assert_allclose(to_numpy(ssb_cp), ssb_np, atol=1e-5)

        R_np = ofdm.correlate_sync_sequence(ssb_np, pss, params=params)
        R_cp = ofdm.correlate_sync_sequence(
            ssb_cp, ofdm.pss_5g_nr(FS, 30e3, xp=cp), params=params
        )
        assert_allclose(to_numpy(R_cp), R_np, atol=1e-5)
        assert to_numpy(ofdm.choose_ssb_offset(R_cp, params))[0] == 77
