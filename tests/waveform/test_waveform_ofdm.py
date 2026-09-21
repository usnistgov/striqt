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
import functools
from fractions import Fraction

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from numeric_checks import (
    assert_close,
    corr_atol,
    interior,
    numpy_and_cupy,
    reference_power,
    rms,
    tone_frequency,
    unit_tone,
)
from numpy.testing import assert_array_equal

from striqt.waveform.lib import ofdm

# the analysis default synchronization block sample rate (15.36e6 / 2)
FS = 7.68e6
SCS_5G = (15e3, 30e3, 60e3)
SC_COUNT = 127


def khz_id(scs):
    return f'{scs / 1e3:g}kHz'


def msps_id(fs):
    return f'{fs / 1e6:g}MSps'


def scs_cp_layout(reason_15khz, reason_60khz):
    """SCS_5G as parametrize values, with the 15 and 60 kHz cyclic prefix layout
    defects marked as strict xfails under the given reasons"""
    xfail = functools.partial(pytest.mark.xfail, strict=True)
    return [
        pytest.param(15e3, marks=xfail(reason=reason_15khz)),
        30e3,
        pytest.param(60e3, marks=xfail(reason=reason_60khz)),
    ]


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
                slot_start = frame * phy.frame_size + slot * phy.contiguous_size
                start = slot_start + phy.cp_start_idx[symbol]
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


def cp_case(n_slots=None):
    """(phy, waveform, cyclic prefix indexes) at 30 kHz over `n_slots` slots, or over
    every slot of a frame when `n_slots` is None"""
    phy = phy_5g(30e3)
    slots = 'all' if n_slots is None else tuple(range(n_slots))
    n_slots = n_slots or phy.SCS_TO_SLOTS_PER_FRAME[30e3]
    return phy, ofdm_slots(phy, n_slots), phy.index_cyclic_prefix(slots=slots)


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


def embed_pss_ports(embeds, scales=None, n_blocks=1):
    """(params, pss, iq) for a silent synchronization block carrying one PSS symbol
    per port, each given as an (n_id2, beam, delay) triple"""
    params, phy, pss, bodies = pss_setup()
    scales = scales or [1.0] * len(embeds)

    iq = silent_block(params, n_blocks=n_blocks, ports=len(embeds))
    for port, ((n_id2, beam, delay), scale) in enumerate(zip(embeds, scales)):
        symbol = params.symbol_indexes[beam]
        embed_symbol(iq, phy, bodies[n_id2], symbol, delay, port=port, scale=scale)
    return params, pss, iq


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

    @pytest.mark.parametrize(
        'sample_rate', [3.84e6, 7.68e6, 15.36e6, 30.72e6, 61.44e6], ids=msps_id
    )
    @pytest.mark.parametrize('scs', SCS_5G, ids=khz_id)
    def test_5g_slot_structure(self, sample_rate, scs):
        """3GPP TS 38.211 5.3.1: normal CP is 144*kappa*2**-mu samples, with an
        extra 16*kappa in the first symbol of a subframe.

        Symbol 7 is left to `test_subframe_cp_layout`: at 15 kHz it carries the
        subframe's second extra CP, which Phy3GPP omits.
        """
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
        assert_array_equal(phy.cp_sizes[1:7], normal_cp)
        assert_array_equal(phy.cp_sizes[8:], normal_cp)
        assert phy.cp_sizes[0] == normal_cp + 16 * nfft * 2**mu // 2048
        assert phy.cp_sizes[0] - phy.cp_sizes[1] == round(sample_rate / 1.92e6)

        # the contiguous block is one slot of symbols; the index attributes tile it
        assert phy.contiguous_size == int(phy.cp_sizes.sum() + 14 * nfft)
        assert phy.cp_start_idx[0] == 0
        assert_array_equal(np.diff(phy.cp_start_idx), phy.cp_sizes[:-1] + nfft)
        assert phy.cp_idx.size == phy.cp_sizes.sum()
        all_idx = np.sort(np.concatenate([phy.cp_idx, phy.symbol_idx]))
        assert_array_equal(all_idx, np.arange(phy.contiguous_size))
        # 6 and 100 LTE resource blocks of 12 subcarriers, plus DC
        if nfft == 128:
            assert phy.subcarriers == 73
        elif nfft == 2048:
            assert phy.subcarriers == 1201

    @pytest.mark.parametrize(
        'scs',
        scs_cp_layout(
            reason_15khz='TS 38.211 5.3.1 puts the extra 16*kappa CP samples at '
            'symbols l=0 and l=7 of each 14-symbol subframe at 15 kHz; '
            'Phy3GPP.__init__ (ofdm.py:1063) lengthens only l=0, so the '
            'slot has one long CP instead of two',
            reason_60khz='TS 38.211 5.3.1 puts the extra 16*kappa CP samples at '
            'symbols l=0 and l=28 of each 56-symbol subframe at 60 kHz; '
            'Phy3GPP.__init__ (ofdm.py:1063) lengthens the first symbol '
            'of every 14-symbol slot, which is twice too often',
        ),
        ids=khz_id,
    )
    def test_subframe_cp_layout(self, scs):
        """TS 38.211 5.3.1: of the 14*2**mu symbols in a 1 ms subframe, only l=0 and
        l=7*2**mu carry the extra 16*kappa cyclic prefix samples"""
        sample_rate = 15.36e6
        phy = phy_5g(scs, sample_rate)
        mu = round(np.log2(scs / 15e3))
        nfft = round(sample_rate / scs)

        expected = np.full(14 * 2**mu, 144 * nfft // 2048)
        expected[[0, 7 * 2**mu]] += round(sample_rate / 1.92e6)

        slots_per_subframe = 2**mu
        assert_array_equal(np.tile(phy.cp_sizes, slots_per_subframe), expected)

    @pytest.mark.parametrize(
        'scs',
        scs_cp_layout(
            reason_15khz='TS 38.211 5.3.1 adds the extra 16*kappa CP samples at '
            'symbols l=0 and l=7*2**mu of each 1 ms subframe; Phy3GPP adds '
            'them once per 14 symbols, which is one too few per slot at '
            '15 kHz',
            reason_60khz='at 60 kHz the extra 16*kappa CP samples belong to one '
            'slot in two; Phy3GPP adds them to every slot',
        ),
        ids=khz_id,
    )
    def test_slots_tile_the_frame(self, scs):
        phy = phy_5g(scs, 15.36e6)
        slots = phy.SCS_TO_SLOTS_PER_FRAME[scs]
        assert slots * phy.contiguous_size == phy.frame_size

    @pytest.mark.parametrize(
        'kws, match',
        [
            ({'subcarrier_spacing': 45e3}, 'subcarrier_spacing'),
            ({'subcarrier_spacing': 30e3, 'sample_rate': FS + 1}, 'sample_rate'),
            ({'generation': 'LTE'}, 'generation'),
            (
                {'subcarrier_spacing': 60e3, 'sample_rate': 1.92e6, 'generation': '5G'},
                'non-integer cyclic prefix',
            ),
        ],
        ids=['scs', 'sample_rate', 'generation', 'fractional_cp'],
    )
    def test_argument_errors(self, kws, match):
        with pytest.raises(ValueError, match=match):
            ofdm.Phy3GPP(1, **{'sample_rate': FS, **kws})

    def test_default_sample_rate_from_bandwidth(self):
        assert ofdm.Phy3GPP(20e6).sample_rate == pytest.approx(30.72e6)
        assert ofdm.Phy3GPP(100e6).sample_rate == pytest.approx(153.6e6)


def all_or_some(max_index):
    """'all' or a tuple of up to four distinct indexes in [0, max_index]"""
    indexes = st.integers(0, max_index)
    some = st.lists(indexes, min_size=1, max_size=4, unique=True).map(tuple)
    return st.one_of(st.just('all'), some)


class TestIndexCyclicPrefix:
    @given(
        scs=st.sampled_from(SCS_5G),
        frames=st.sampled_from([(0,), (0, 1), (2,), (1, 3)]),
        data=st.data(),
    )
    def test_matches_loop_reference(self, scs, frames, data):
        phy = phy_5g(scs)
        slots_per_frame = phy.SCS_TO_SLOTS_PER_FRAME[scs]
        symbols = data.draw(all_or_some(13))
        slots = data.draw(all_or_some(slots_per_frame - 1))

        inds = phy.index_cyclic_prefix(frames=frames, symbols=symbols, slots=slots)
        assert_array_equal(inds, cp_index_reference(phy, frames, symbols, slots))

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

    @pytest.mark.parametrize(
        'kws, match',
        [
            ({'slots': 'bogus'}, 'slots'),
            ({'slots': (ofdm.Phy3GPP.SCS_TO_SLOTS_PER_FRAME[30e3],)}, 'slots'),
            ({'symbols': (14,)}, 'symbols'),
            ({'symbols': (-15,)}, 'symbols'),
            ({'slots': ((0, 1), (2, 3))}, 'slots'),
        ],
        ids=[
            'slots_string',
            'slot_past_frame',
            'symbol_past_slot',
            'symbol_below_slot',
            'slots_nested',
        ],
    )
    def test_argument_errors(self, kws, match):
        with pytest.raises(ValueError, match=match):
            phy_5g(30e3).index_cyclic_prefix(**kws)


class TestCorrAtIndices:
    """cyclic prefix correlation on a synthetic OFDM waveform, indexed as the
    cellular cyclic autocorrelation measurement does"""

    def test_normalized_correlation_peaks_at_zero_lag(self):
        phy, x, inds = cp_case()
        ncp = int(phy.cp_sizes[1])

        R = np.abs(ofdm.corr_at_indices(inds, x, phy.nfft, norm=True))
        assert R.shape == (phy.nfft + ncp,)

        # the cyclic prefix is an exact copy of the symbol tail
        assert R[0] == pytest.approx(1, abs=corr_atol(x, inds.size, norm=True))

        # partial overlap decays linearly over the prefix length; the remaining
        # pairs are independent QPSK products that average out
        lags = np.arange(ncp)
        assert_close(R[:ncp], 1 - lags / ncp, atol=0.1)
        assert R[ncp : phy.nfft].max() < 0.1

    def test_unnormalized_zero_lag_is_prefix_power(self):
        phy, x, inds = cp_case()

        R = ofdm.corr_at_indices(inds, x, phy.nfft, norm=False)
        power = reference_power(x[inds.ravel()]).mean()
        assert R[0] == pytest.approx(power, abs=corr_atol(x, inds.size, norm=False))
        assert R.dtype == x.dtype


class TestSyncSequences:
    @pytest.mark.parametrize('n_id2', [0, 1, 2])
    def test_pss_m_sequence(self, n_id2):
        """TS 38.211 7.4.2.2.1: d(n) = 1 - 2 x((n + 43 N_id2) mod 127), where
        x(i+7) = (x(i+4) + x(i)) mod 2 from x(6)...x(0) = 1 1 1 0 1 1 0"""
        x = [0, 1, 1, 0, 1, 1, 1]
        for i in range(SC_COUNT - 7):
            x.append((x[i + 4] + x[i]) % 2)
        expected = [1 - 2 * x[(n + 43 * n_id2) % SC_COUNT] for n in range(SC_COUNT)]

        assert_array_equal(ofdm._pss_m_sequence(n_id2), expected)

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
        assert_close(X, expected, atol=1e-5)
        assert_close(np.sum(np.abs(pss) ** 2, axis=1), 1 / nfft, rtol=1e-4)

    def test_pss_center_frequency_shifts_the_subcarriers(self):
        phy = phy_5g(30e3)
        nfft, cp = int(phy.nfft), int(phy.cp_sizes.min())

        pss = ofdm.pss_5g_nr(FS, 30e3, 5 * 30e3, dtype='complex128')
        assert pss.dtype == np.complex128
        X = np.fft.fft(pss[0, cp:])
        occupied = np.abs(X) > 1e-6
        assert_array_equal(np.sort(np.where(occupied)[0]), np.sort(sync_bins(nfft, 5)))

    def test_sss_time_domain(self, subtests):
        phy = phy_5g(30e3)
        nfft, cp = int(phy.nfft), int(phy.cp_sizes.min())

        sss = ofdm.sss_5g_nr(FS, 30e3)
        assert sss.shape == (1008, nfft + cp)
        assert sss.dtype == np.complex64
        assert_array_equal(sss[:, :cp], 0)

        for n_id in (0, 1, 500, 1007):
            with subtests.test(n_id=n_id):
                X = np.fft.fft(sss[n_id, cp:].astype(np.complex128))
                expected = np.zeros(nfft, dtype=np.complex128)
                expected[sync_bins(nfft)] = ofdm._sss_m_sequence(n_id)
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
            ofdm.pss_5g_nr(*args)


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
        assert ofdm.index_pss_symbols(*args) == table

    @pytest.mark.parametrize('scs', [120e3, 240e3, 480e3, 960e3])
    def test_fr2_cases_have_64_candidates(self, scs):
        """TS 38.213 Section 4.1: L_max = 64 for cases D through G"""
        assert len(ofdm.index_pss_symbols(scs)) == 64

    def test_explicit_indexes_pass_through(self):
        assert ofdm.index_pss_symbols(30e3, symbol_indexes=(3, 9)) == (3, 9)
        assert ofdm.index_pss_symbols(30e3, symbol_indexes=[3, 9]) == [3, 9]

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
            ofdm.index_pss_symbols(**{'subcarrier_spacing': 30e3, **kws})


class TestSyncParams:
    def test_min_diff(self):
        assert ofdm._min_diff([]) is None
        assert ofdm._min_diff([5]) is None
        assert ofdm._min_diff([2, 8, 16]) == 6

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

    def test_sss_params_follow_pss_by_two_symbols(self, subtests):
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
            with subtests.test(field=name):
                assert getattr(sss, name) == getattr(pss, name)

        explicit = ofdm.sss_params(
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
        x = unit_tone(self.SIZE, self.FS_IN, f, dtype=dtype)
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
        # the shift is applied in whole input bins, so allow half a bin of
        # quantization beyond the estimator's resolution
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

        R = ofdm.correlate_sync_sequence(iq, pss, params=params)
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

        R = ofdm.correlate_sync_sequence(iq, pss, params=params)
        assert R.shape[2] == 3
        mag = np.abs(R)
        assert np.unravel_index(mag.argmax(), mag.shape) == (0, 0, 2, 4, 7)

    def test_multiple_ports(self, subtests):
        params, pss, iq = embed_pss_ports([(1, 2, 11), (2, 6, 23)])

        R = ofdm.correlate_sync_sequence(iq, pss, params=params)
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
            ofdm.correlate_sync_sequence(silent_block(params), pss, params=bad)


class TestChooseSsbOffset:
    def _correlate(self, delays, scales=None, beam=3, n_id2=1):
        embeds = [(n_id2, beam, delay) for delay in delays]
        params, pss, iq = embed_pss_ports(embeds, scales=scales)
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

    def test_per_port(self):
        params, R = self._correlate([40, 700], scales=[1.0, 3.0])
        assert_array_equal(ofdm.choose_ssb_offset(R, params, per_port=True), [40, 700])
        # the ports are averaged in power, so the stronger one decides
        assert_array_equal(ofdm.choose_ssb_offset(R, params, per_port=False), [700])

    def test_max_beams(self):
        """only the first max_beams SSB candidates are searched"""
        params, phy, pss, bodies = pss_setup()
        iq = silent_block(params)
        embed_symbol(iq, phy, bodies[1], params.symbol_indexes[1], 55)
        embed_symbol(iq, phy, bodies[1], params.symbol_indexes[6], 400, scale=3.0)
        R = ofdm.correlate_sync_sequence(iq, pss, params=params)

        assert ofdm.choose_ssb_offset(R, params)[0] == 400
        assert ofdm.choose_ssb_offset(R, params, max_beams=4)[0] == 55

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
        ref = phy_5g(30e3)
        phy = phy_5g(30e3, xp=cupy_available)
        assert_close(phy.cp_sizes, ref.cp_sizes)
        inds = phy.index_cyclic_prefix(slots=(0, 3))
        assert isinstance(inds, cupy_available.ndarray)
        assert_close(inds, ref.index_cyclic_prefix(slots=(0, 3)))

    def test_corr_at_indices(self, cupy_available):
        phy, x, inds = cp_case(4)
        R_np, R_cp = numpy_and_cupy(
            cupy_available, ofdm.corr_at_indices, inds, x, phy.nfft, norm=False
        )
        assert_close(R_cp, R_np, atol=corr_atol(x, inds.size, norm=False, n_impl=2))

    def test_sync_sequences(self, cupy_available):
        pss_np, pss_cp = numpy_and_cupy(
            cupy_available, ofdm.pss_5g_nr, FS, 30e3, xp_kwarg='xp'
        )
        assert_close(pss_cp, pss_np, atol=1e-6)
        sss_np, sss_cp = numpy_and_cupy(
            cupy_available, ofdm.sss_5g_nr, FS, 30e3, xp_kwarg='xp'
        )
        assert_close(sss_cp, sss_np, atol=1e-6)

    def test_detection_pipeline(self, cupy_available):
        params, pss, iq = embed_pss_ports([(1, 3, 77)])

        ssb_np, ssb_cp = numpy_and_cupy(cupy_available, get_ssb_iq, iq)
        assert_close(ssb_cp, ssb_np, atol=1e-5)

        R_np, R_cp = numpy_and_cupy(
            cupy_available, ofdm.correlate_sync_sequence, ssb_np, pss, params=params
        )
        assert_close(R_cp, R_np, atol=1e-5)

        offset_np, offset_cp = numpy_and_cupy(
            cupy_available, ofdm.choose_ssb_offset, R_np, params
        )
        assert offset_np[0] == offset_cp[0] == 77
