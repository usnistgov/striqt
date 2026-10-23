"""striqt.waveform.lib.comms.phy_3gpp: the 3GPP frame model (FFT size, cyclic prefix
sizes and start indices, slot period) checked against an independent restatement of
TS 38.211 5.3.1 and a loop construction of the cyclic prefix index tensor.
"""

from __future__ import annotations

import functools

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numeric_checks import assert_close
from numpy.testing import assert_array_equal
from waveform_strategies import FS, SCS_5G, khz_id, msps_id, phy_5g

from striqt.waveform.lib.comms import phy_3gpp

# 3GPP TS 38.211 Table 4.3.2-1: 10 * subcarrier_spacing/15e3 slots per 10 ms frame
SLOT_PERIODS = {15e3: 1e-3, 30e3: 0.5e-3, 60e3: 0.25e-3, 120e3: 125e-6, 240e3: 62.5e-6}


def scs_cp_layout(reason_15khz, reason_60khz):
    """SCS_5G as parametrize values, with the 15 and 60 kHz cyclic prefix layout
    defects marked as strict xfails under the given reasons"""
    xfail = functools.partial(pytest.mark.xfail, strict=True)
    return [
        pytest.param(15e3, marks=xfail(reason=reason_15khz)),
        30e3,
        pytest.param(60e3, marks=xfail(reason=reason_60khz)),
    ]


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


class TestPhy3GPP:
    @pytest.mark.xfail(
        strict=True,
        reason='the LTE table has two extended CPs per 14 symbols (LTE_MIN_CP_SIZES) '
        'but the 5G computation produces one; TS 38.211 5.3.1 agrees with the LTE '
        'table at 15 kHz',
    )
    @pytest.mark.parametrize('sample_rate', [1.92e6, 3.84e6, 7.68e6, 15.36e6, 30.72e6])
    def test_lte_and_5g_agree_at_15khz(self, sample_rate):
        lte = phy_3gpp.Phy3GPP(1, sample_rate=sample_rate, generation='4G')
        nr = phy_3gpp.Phy3GPP(1, sample_rate=sample_rate, generation='5G')
        assert_array_equal(lte.cp_sizes, nr.cp_sizes)
        assert lte.nfft == nr.nfft == round(sample_rate / 15e3)

    @pytest.mark.parametrize(
        'sample_rate', [3.84e6, 7.68e6, 15.36e6, 30.72e6, 61.44e6], ids=msps_id
    )
    @pytest.mark.parametrize('scs', SCS_5G, ids=khz_id)
    def test_5g_slot_structure(self, sample_rate, scs):
        """3GPP TS 38.211 5.3.1: normal CP is 144*kappa*2**-mu samples, with an
        extra 16*kappa in the first symbol of a subframe.

        Symbol 7 is left to `test_subframe_cp_layout`. At 15 kHz it carries the
        subframe's second extra CP, which Phy3GPP omits.
        """
        nfft = round(sample_rate / scs)
        mu = round(np.log2(scs / 15e3))

        if (144 * nfft) % 2048:
            with pytest.raises(ValueError, match='non-integer cyclic prefix') as info:
                phy_5g(scs, sample_rate)
            assert repr(sample_rate) in str(info.value)
            assert repr(scs) in str(info.value)
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
            'Phy3GPP.__init__ (comms/phy_3gpp.py:176) lengthens only l=0, so the '
            'slot has one long CP instead of two',
            reason_60khz='TS 38.211 5.3.1 puts the extra 16*kappa CP samples at '
            'symbols l=0 and l=28 of each 56-symbol subframe at 60 kHz; '
            'Phy3GPP.__init__ (comms/phy_3gpp.py:176) lengthens the first symbol '
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
            phy_3gpp.Phy3GPP(1, **{'sample_rate': FS, **kws})

    def test_default_sample_rate_from_bandwidth(self):
        assert phy_3gpp.Phy3GPP(20e6).sample_rate == pytest.approx(30.72e6)
        assert phy_3gpp.Phy3GPP(100e6).sample_rate == pytest.approx(153.6e6)


def all_or_some(max_index):
    """'all' or a tuple of up to 4 distinct indexes in [0, max_index]"""
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

    @pytest.mark.parametrize(
        'kws, match',
        [
            ({'slots': 'bogus'}, 'slots'),
            ({'slots': (phy_3gpp.Phy3GPP.SCS_TO_SLOTS_PER_FRAME[30e3],)}, 'slots'),
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


class TestSlotPeriod:
    @pytest.mark.parametrize('scs', list(SLOT_PERIODS), ids=khz_id)
    def test_numerology_values_are_exact(self, scs):
        # sample counts are rounded from this, so a last-bit difference moves a shape
        assert phy_3gpp.slot_period(scs) == SLOT_PERIODS[scs]


class TestCupy:
    """the phy and its index tensor on cupy against the numpy result"""

    def test_phy_and_index(self, cupy_available):
        ref = phy_5g(30e3)
        phy = phy_5g(30e3, xp=cupy_available)
        assert_close(phy.cp_sizes, ref.cp_sizes)
        inds = phy.index_cyclic_prefix(slots=(0, 3))
        assert isinstance(inds, cupy_available.ndarray)
        assert_close(inds, ref.index_cyclic_prefix(slots=(0, 3)))


# %% nr_sample_rate

# (subcarrier spacing, 3GPP TS 38.104 Table 5.3.2-1 channel bandwidth) -> the smallest
# multiple of 896 * scs at or above the channel bandwidth plus a 250 kHz transition
# band. The step is 13.44e6 S/s at 15 kHz, 26.88e6 at 30 kHz, and 53.76e6 at 60 kHz.
# The rows near a step boundary: 50.25e6 exceeds 3 steps of 13.44e6 (40.32e6) and
# takes 4; 80.25e6 fits 3 steps of 26.88e6 (80.64e6); 60.25e6 exceeds 1 step of
# 53.76e6 and takes 2.
NR_RATES = {
    (15e3, 5e6): 13.44e6,
    (15e3, 20e6): 26.88e6,
    (15e3, 50e6): 53.76e6,
    (30e3, 20e6): 26.88e6,
    (30e3, 40e6): 53.76e6,
    (30e3, 60e6): 80.64e6,
    (30e3, 80e6): 80.64e6,
    (30e3, 100e6): 107.52e6,
    (60e3, 10e6): 53.76e6,
    (60e3, 60e6): 107.52e6,
    (60e3, 100e6): 107.52e6,
}

# one channel per subcarrier spacing. Every rate is a multiple of 896 * scs, so the
# sample grid properties below hold for one bandwidth iff they hold for all.
NR_ONE_PER_SCS = [
    pytest.param(scs, bw, id=khz_id(scs))
    for scs, bw in [(15e3, 20e6), (30e3, 100e6), (60e3, 40e6)]
]

NR_FR1_PAIRS = [
    pytest.param(scs, bw, id=f'{khz_id(scs)}-{bw / 1e6:g}MHz')
    for scs, row in phy_3gpp.NR_FR1_RESOURCE_BLOCKS.items()
    for bw in row
]


class TestNRSampleRate:
    @pytest.mark.parametrize(
        'scs, bw, expected',
        [(scs, bw, rate) for (scs, bw), rate in NR_RATES.items()],
        ids=[f'{khz_id(scs)}-{bw / 1e6:g}MHz' for scs, bw in NR_RATES],
    )
    def test_rates(self, scs, bw, expected):
        assert phy_3gpp.nr_sample_rate(bw, scs) == expected

    def test_min_sample_rate_is_a_lower_bound(self):
        rate = phy_3gpp.nr_sample_rate(60e6, 30e3, min_sample_rate=107.52e6)
        assert rate == pytest.approx(107.52e6)
        rate = phy_3gpp.nr_sample_rate(60e6, 30e3, min_sample_rate=1e6)
        assert rate == NR_RATES[(30e3, 60e6)]

    @pytest.mark.parametrize('scs, bw', NR_ONE_PER_SCS)
    def test_rate_gives_whole_sample_symbols_and_cyclic_prefixes(self, scs, bw):
        rate = phy_3gpp.nr_sample_rate(bw, scs)
        phy = phy_3gpp.Phy3GPP(bw, scs, generation='5G', sample_rate=rate)
        assert phy.nfft == rate / scs
        assert phy.cp_sizes.dtype.kind == 'i'

    @pytest.mark.parametrize('scs, bw', NR_ONE_PER_SCS)
    def test_rate_fits_the_symbol_grid_at_half_spacing(self, scs, bw):
        """the cellular spectrograms at SCS/2 resolution fill 15/28 of each hop.
        A whole-sample window needs a bin period that is a multiple of 28 samples."""
        rate = phy_3gpp.nr_sample_rate(bw, scs)
        assert (rate / (scs / 2)) % 28 == 0

    @pytest.mark.parametrize(
        'kws',
        [{'channel_bandwidth': 7e6}, {'subcarrier_spacing': 120e3}],
        ids=['bandwidth', 'spacing'],
    )
    def test_unknown_pair_raises(self, kws):
        kws = {'channel_bandwidth': 20e6, 'subcarrier_spacing': 30e3, **kws}
        with pytest.raises(ValueError, match=r'TS 38\.104'):
            phy_3gpp.nr_occupied_bandwidth(**kws)
        with pytest.raises(ValueError, match=r'TS 38\.104'):
            phy_3gpp.nr_sample_rate(**kws)


class TestNROccupiedBandwidth:
    def test_100mhz_at_30khz(self):
        """3GPP TS 38.104 Table 5.3.2-1: N_RB = 273"""
        n_rb = 273
        assert phy_3gpp.nr_occupied_bandwidth(100e6, 30e3) == pytest.approx(
            n_rb * 12 * 30e3
        )

    def test_resource_block_table_spot_checks(self):
        """3GPP TS 38.104 Table 5.3.2-1"""
        assert phy_3gpp.NR_FR1_RESOURCE_BLOCKS[15e3][20e6] == 106
        assert phy_3gpp.NR_FR1_RESOURCE_BLOCKS[60e3][100e6] == 135

    @pytest.mark.parametrize('scs, bw', NR_FR1_PAIRS)
    def test_occupied_bandwidth_leaves_a_guard_band(self, scs, bw):
        """3GPP TS 38.104 Table 5.3.3-1 lists a positive guard band for every
        entry: the transmission bandwidth is below the channel bandwidth"""
        occupied = phy_3gpp.nr_occupied_bandwidth(bw, scs)
        assert 0.75 * bw < occupied < bw
