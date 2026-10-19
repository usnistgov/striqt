"""3GPP (LTE and 5G NR) frame model.

Cyclic prefix sizes and start indices follow TS 38.211 §5.3.1; slot timing follows
§4.3.2.
"""

from __future__ import annotations as __

import typing
from fractions import Fraction

from .. import util
from ..arrays import array_namespace, isroundmod
from ..util import np
from .core import PhyOFDM, _index_or_all


def slot_period(subcarrier_spacing: float) -> float:
    """return the duration (in s) of one 3GPP slot at `subcarrier_spacing`.

    A 10 ms radio frame holds 10 subframes of ``subcarrier_spacing/15e3`` slots each
    (3GPP TS 38.211 Section 4.3.2).
    """
    return 10e-3 / (10 * subcarrier_spacing / 15e3)


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
    """build the index tensor for cyclic prefix correlation over the selected frames, slots, and symbols"""
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

    # build each grid axis separately
    grid = []

    # axis 0: symbol number within each slot
    grid.append(phy.cp_start_idx[symbol_inds])

    # axis 1: slot number
    grid.append(phy.contiguous_size * slot_inds)

    # axis 2: frame number
    grid.append(frames * frame_size)

    # axis 3: cp index
    grid.append(xp.ogrid[0 : phy.cp_sizes[1]])

    # pad the axis dimensions so they can be broadcast together
    inds, *offsets = xp.meshgrid(*grid, indexing='ij', copy=False)

    # sum all of the index offsets
    inds = inds.copy()
    for offset in offsets:
        inds += offset

    # drop singleton axes only after the sum, so that a single non-zero
    # frame, slot or symbol selection keeps its offset
    return inds.squeeze()


class Phy3GPP(PhyOFDM):
    """Sampling and index parameters and lookup tables for 3GPP 5G-NR.

    These are equivalent to LTE if subcarrier_spacing is fixed to 15 kHz
    and slot length is redefined to match the period of 14 symbols including
    cyclic prefix.

    References:
        3GPP TS 38.211.
    """

    # a slot is 15 symbol periods of samples: 14 FFTs and 1 period of cyclic prefixes
    FFT_PER_SLOT = 14

    FFT_SIZE_TO_SUBCARRIERS = {
        128: 73,
        256: 181,
        512: 301,
        1024: 601,
        1536: 901,
        2048: 1201,
    }

    # "default" sample rates from LTE
    BW_TO_SAMPLE_RATE = {
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

    # Slot structure and cyclic prefix (CP) sizes follow 3GPP TS 38.211, Section
    # 5.3.1. These are the sizes (in samples) of the 14 CPs in 1 slot for FFT size
    # 128 at 15 kHz SCS; CP size scales proportionally to FFT size. 1 slot is the
    # minimum number of contiguous symbols in a sequence.
    LTE_MIN_CP_SIZES = (10, 9, 9, 9, 9, 9, 9, 10, 9, 9, 9, 9, 9, 9)

    SCS_TO_SLOTS_PER_FRAME = {15e3: 10, 30e3: 20, 60e3: 40}

    # FR1 only; the FR2 spacings (120 kHz and above) are not modeled
    SUBCARRIER_SPACINGS = {15e3, 30e3, 60e3}

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
            # assert subcarrier_spacing == 15e3, '4G LTE only supports 15 kHz subcarrier spacing'
            cp_sizes = nfft * xp.array(self.LTE_MIN_CP_SIZES, dtype=int) // 128
        elif generation.upper() == '5G':
            Tc = Fraction(1, 4096 * 480) * 1000  # microsec
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
        """build the index tensor for cyclic prefix correlation over the selected frames, slots, and symbols"""
        return _get_3gpp_index_cyclic_prefix(
            self, frames=frames, symbols=symbols, slots=slots
        )
