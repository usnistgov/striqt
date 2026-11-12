"""3GPP (LTE and 5G NR) frame model.

Cyclic prefix sizes and start indices follow 3GPP TS 38.211 §5.3.1, and slot timing
follows §4.3.2.
"""

from __future__ import annotations as __

import math
import typing
from fractions import Fraction

from .. import util
from ..arrays import array_namespace, isroundmod
from ..util import np
from .core import PhyOFDM, _index_or_all


def slot_period(subcarrier_spacing: float) -> float:
    """return the duration (in s) of one 3GPP slot at `subcarrier_spacing`.

    A 10 ms radio frame has 10 subframes of ``subcarrier_spacing/15e3`` slots each
    (3GPP TS 38.211 §4.3.2).

    References:
        3GPP TS 38.211, "NR; Physical channels and modulation".
    """
    return 10e-3 / (10 * subcarrier_spacing / 15e3)


# 3GPP TS 38.104 Table 5.3.2-1: transmission bandwidth configuration N_RB for FR1,
# keyed by subcarrier spacing (Hz) and then channel bandwidth (Hz)
NR_FR1_RESOURCE_BLOCKS: dict[float, dict[float, int]] = {
    15e3: {
        5e6: 25,
        10e6: 52,
        15e6: 79,
        20e6: 106,
        25e6: 133,
        30e6: 160,
        35e6: 188,
        40e6: 216,
        45e6: 242,
        50e6: 270,
    },
    30e3: {
        5e6: 11,
        10e6: 24,
        15e6: 38,
        20e6: 51,
        25e6: 65,
        30e6: 78,
        35e6: 92,
        40e6: 106,
        45e6: 119,
        50e6: 133,
        60e6: 162,
        70e6: 189,
        80e6: 217,
        90e6: 245,
        100e6: 273,
    },
    60e3: {
        10e6: 11,
        15e6: 18,
        20e6: 24,
        25e6: 31,
        30e6: 38,
        35e6: 44,
        40e6: 51,
        45e6: 58,
        50e6: 65,
        60e6: 79,
        70e6: 93,
        80e6: 107,
        90e6: 121,
        100e6: 135,
    },
}

# lcm(128, 14): a multiple of 128 subcarriers makes the 5G NR cyclic prefixes of
# `Phy3GPP` whole samples, and a multiple of 14 makes the 15/28 window fill of the
# SCS/2 spectrograms whole samples
NR_SAMPLE_RATE_STEP_SUBCARRIERS = 896


def nr_occupied_bandwidth(channel_bandwidth: float, subcarrier_spacing: float) -> float:
    """return the transmission bandwidth (in Hz) of a 5G NR FR1 channel.

    The value is ``N_RB * 12 * subcarrier_spacing`` with ``N_RB`` from 3GPP TS 38.104
    Table 5.3.2-1.

    Args:
        channel_bandwidth: the nominal channel bandwidth (Hz), a key of
            `NR_FR1_RESOURCE_BLOCKS`
        subcarrier_spacing: 15e3, 30e3, or 60e3 (Hz)

    Raises:
        ValueError: if `NR_FR1_RESOURCE_BLOCKS` has no entry for the pair

    References:
        3GPP TS 38.104, "NR; Base Station (BS) radio transmission and reception".
    """
    try:
        n_rb = NR_FR1_RESOURCE_BLOCKS[subcarrier_spacing][channel_bandwidth]
    except KeyError:
        raise ValueError(
            f'no 3GPP TS 38.104 FR1 configuration for channel bandwidth '
            f'{channel_bandwidth!r} Hz at subcarrier spacing {subcarrier_spacing!r} Hz'
        ) from None
    return n_rb * 12 * subcarrier_spacing


def nr_sample_rate(
    channel_bandwidth: float,
    subcarrier_spacing: float,
    *,
    min_sample_rate: float | None = None,
    transition_bandwidth: float = 250e3,
) -> float:
    """return the smallest 5G NR analysis sample rate (in S/s) for a channel.

    The rate is a multiple of ``896 * subcarrier_spacing`` (26.88 MS/s at 30 kHz) at
    or above ``channel_bandwidth + transition_bandwidth`` and `min_sample_rate`.

    Args:
        channel_bandwidth: the nominal channel bandwidth (Hz), a key of
            `NR_FR1_RESOURCE_BLOCKS`
        subcarrier_spacing: 15e3, 30e3, or 60e3 (Hz)
        min_sample_rate: a lower bound (S/s), or None for none
        transition_bandwidth: the analysis filter transition band (Hz) that must fit
            between the channel edge and the Nyquist frequency

    Raises:
        ValueError: if `NR_FR1_RESOURCE_BLOCKS` has no entry for the pair
    """
    nr_occupied_bandwidth(channel_bandwidth, subcarrier_spacing)
    step = NR_SAMPLE_RATE_STEP_SUBCARRIERS * subcarrier_spacing
    needed = channel_bandwidth + transition_bandwidth
    if min_sample_rate is not None:
        needed = max(needed, min_sample_rate)
    return math.ceil(needed / step) * step


@util.lru_cache(4)
def get_3gpp_phy(
    channel_bandwidth: float,
    subcarrier_spacing=15e3,
    generation: typing.Literal['4G', '5G'] = '4G',
    sample_rate=None,
    xp=None,
) -> Phy3GPP:
    return Phy3GPP(**locals())


@util.lru_cache(4)
def _get_3gpp_index_cyclic_prefix(
    phy: Phy3GPP,
    *,
    frames: tuple[int, ...] = (0,),
    symbols: tuple[int, ...] | typing.Literal['all'] = 'all',
    slots: tuple[int, ...] | typing.Literal['all'] = 'all',
):
    """return the cyclic-prefix sample indices of `phy` in the selected frames, slots, and symbols.

    Raises:
        ValueError: if `symbols` or `slots` is neither ``'all'`` nor a sequence of
            indices within one slot or frame
    """
    xp = array_namespace(phy.cp_sizes)

    frames = xp.array(frames)
    frame_size = round(phy.sample_rate * 10e-3)

    slot_inds = _index_or_all(
        slots,
        '"slots" argument',
        size=phy.SCS_TO_SLOTS_PER_FRAME[phy.subcarrier_spacing],
        xp=xp,
    )
    symbol_inds = _index_or_all(
        symbols, '"symbols" argument', size=phy.FFT_PER_SLOT, xp=xp
    )

    grid = [
        phy.cp_start_idx[symbol_inds],
        phy.contiguous_size * slot_inds,
        frames * frame_size,
        xp.ogrid[0 : phy.cp_sizes[1]],
    ]

    inds, *offsets = xp.meshgrid(*grid, indexing='ij', copy=False)

    inds = inds.copy()
    for offset in offsets:
        inds += offset

    # the squeeze follows the sum so that a single nonzero frame, slot, or symbol
    # selection keeps its offset
    return inds.squeeze()


class Phy3GPP(PhyOFDM):
    """an OFDM frame model of one 3GPP slot: FFT size, sample rate, and cyclic prefixes.

    ``generation='4G'`` selects the LTE normal cyclic prefix at 15 kHz subcarrier
    spacing. ``generation='5G'`` scales the 5G NR cyclic prefix (3GPP TS 38.211
    §5.3.1) to `subcarrier_spacing`, with the longer first prefix in every slot.

    References:
        3GPP TS 38.211, "NR; Physical channels and modulation".
    """

    # a slot is 15 FFT periods: 14 FFTs and cyclic prefixes that sum to 1 FFT period
    FFT_PER_SLOT = 14

    FFT_SIZE_TO_SUBCARRIERS: typing.ClassVar[dict[int, int]] = {
        128: 73,
        256: 181,
        512: 301,
        1024: 601,
        1536: 901,
        2048: 1201,
    }

    # the sample rates of the LTE channel bandwidths, 1.536 times the bandwidth from
    # 5 MHz up
    BW_TO_SAMPLE_RATE: typing.ClassVar[dict[float, float]] = {
        1.4e6: 1.92e6,
        3e6: 3.84e6,
        5e6: 7.68e6,
        10e6: 15.36e6,
        15e6: 23.04e6,
        20e6: 30.72e6,
        25e6: 38.40e6,
        30e6: 46.08e6,
        40e6: 61.44e6,
        60e6: 92.16e6,
        80e6: 122.88e6,
        100e6: 153.6e6,
    }

    # the sizes (in samples) of the 14 cyclic prefixes of one 15 kHz slot at FFT size
    # 128 (3GPP TS 38.211 §5.3.1). The sizes scale with the FFT size
    LTE_MIN_CP_SIZES = (10, 9, 9, 9, 9, 9, 9, 10, 9, 9, 9, 9, 9, 9)

    SCS_TO_SLOTS_PER_FRAME: typing.ClassVar[dict[float, int]] = {
        15e3: 10,
        30e3: 20,
        60e3: 40,
    }

    # FR1 only: 120 kHz and above (FR2) are not modeled
    SUBCARRIER_SPACINGS: typing.ClassVar[set[float]] = {15e3, 30e3, 60e3}

    def __init__(
        self,
        channel_bandwidth: float,
        subcarrier_spacing=15e3,
        generation: typing.Literal['4G', '5G'] = '4G',
        sample_rate=None,
        xp=None,
    ):
        if xp is None:
            xp = np

        if subcarrier_spacing not in self.SUBCARRIER_SPACINGS:
            raise ValueError(
                f'subcarrier_spacing must be one of {self.SUBCARRIER_SPACINGS}'
            )

        if sample_rate is None:
            sample_rate = self.BW_TO_SAMPLE_RATE[channel_bandwidth]

        if isroundmod(sample_rate, subcarrier_spacing):
            nfft = round(sample_rate / subcarrier_spacing)
        else:
            raise ValueError('sample_rate / subcarrier_spacing must be counting number')

        if nfft in self.FFT_SIZE_TO_SUBCARRIERS:
            self.subcarriers = self.FFT_SIZE_TO_SUBCARRIERS[nfft]

        if generation.upper() == '4G':
            cp_sizes = nfft * xp.array(self.LTE_MIN_CP_SIZES, dtype=int) // 128
        elif generation.upper() == '5G':
            Tc = Fraction(1, 4096 * 480) * 1000  # T_c of 3GPP TS 38.211 §4.1, in µs
            kappa = 64
            fs_MHz = Fraction(round(sample_rate), 1_000_000)

            scs_norm = Fraction(round(subcarrier_spacing), 15000)
            Tcp_us = [kappa * Tc * (144 / scs_norm) + 16 * kappa * Tc] + 13 * [
                kappa * Tc * (144 / scs_norm)
            ]
            cp_fractions = [T * fs_MHz for T in Tcp_us]
            if any(cp.denominator != 1 for cp in cp_fractions):
                raise ValueError(
                    f'sample_rate {sample_rate!r} with subcarrier_spacing '
                    f'{subcarrier_spacing!r} produces non-integer cyclic prefixes'
                )
            cp_sizes = xp.array([cp.numerator for cp in cp_fractions], dtype=int)
        else:
            raise ValueError('generation must be "4G" or "5G"')

        super().__init__(
            channel_bandwidth=channel_bandwidth,
            nfft=nfft,
            sample_rate=sample_rate,
            frame_duration=10e-3,
            cp_sizes=cp_sizes,
        )

    def index_cyclic_prefix(
        self,
        *,
        frames: tuple[int, ...] = (0,),
        symbols: tuple[int, ...] | typing.Literal['all'] = 'all',
        slots: tuple[int, ...] | typing.Literal['all'] = 'all',
    ):
        """return the sample indices of the cyclic prefixes in the selected frames, slots, and symbols.

        Args:
            frames: the 10 ms frame numbers, counted from the start of the waveform
            symbols: the symbol numbers within each slot, or ``'all'``
            slots: the slot numbers within each frame, or ``'all'``

        Returns:
            integer indices of shape ``(symbol, slot, frame, prefix sample)`` with
            singleton axes removed, in the array namespace of `cp_sizes`; the last
            axis has the length of the shortest cyclic prefix

        Raises:
            ValueError: if `symbols` or `slots` is neither ``'all'`` nor a sequence of
                indices within one slot or frame
        """
        return _get_3gpp_index_cyclic_prefix(
            self, frames=frames, symbols=symbols, slots=slots
        )
