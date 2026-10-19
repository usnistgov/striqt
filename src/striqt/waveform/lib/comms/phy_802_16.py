"""IEEE 802.16-2017 OFDMA frame model."""

from __future__ import annotations as __

import typing
from math import isclose
from numbers import Number

from .. import util
from ..arrays import array_namespace
from ..util import np
from .core import PhyOFDM, _index_or_all

if typing.TYPE_CHECKING:
    from ..typing import Array


def _isclosetoint(v, atol=1e-6):
    xp = array_namespace(v)
    return xp.isclose(v % 1, (0, 1), atol=atol).any()


@util.lru_cache(4)
def get_802_16_phy(
    channel_bandwidth: float,
    *,
    alt_sample_rate: float | None = None,
    frame_duration: float = 5e-3,
    nfft: int = 2048,
    cp_ratio: float = 1 / 8,
    xp=None,
) -> Phy802_16:
    return Phy802_16(**locals())


class Phy802_16(PhyOFDM):
    """Sampling and index parameters and lookup tables for IEEE 802.16-2017 OFDMA"""

    VALID_CP_RATIOS = {1 / 32, 1 / 16, 1 / 8, 1 / 4}
    VALID_FFT_SIZES = {128, 512, 1024, 2048}
    VALID_FRAME_DURATIONS = {
        2e-3,
        2.5e-3,
        4e-3,
        5e-3,
        8e-3,
        10e-3,
        12.5e-3,
        20e-3,
        25e-3,
        40e-3,
        50e-3,
    }

    SAMPLING_FACTOR_BY_FREQUENCY_DIV = {
        1.25: 28 / 25,
        1.5: 28 / 25,
        1.75e6: 8 / 7,
        2: 28 / 25,
        2.75: 28 / 25,
    }

    def __init__(
        self,
        channel_bandwidth: float,
        *,
        alt_sample_rate: float | None = None,
        frame_duration: float = 5e-3,
        nfft: int = 2048,
        cp_ratio: float = 1 / 8,
        xp=None,
    ):
        """
        Args:
            channel_bandwidth: Channel bandwidth as defined by 802.16-2017
            alt_sample_rate: sample rate (in S/s) of the recorded data, which overrides the 802.16-2017 value when specified.
            frame_duration: frame duration (in s), one of the 802.16-2017 values. Defaults to 5e-3.
            nfft: the fft size corresponding to the length of the useful portion of each symbol. Defaults to 2048.
            cp_ratio: the size of the cyclic prefix as a fraction of nfft. Defaults to 1/8.
        """
        if xp is None:
            xp = np

        if not isinstance(channel_bandwidth, Number):
            raise TypeError('expected numeric value for channel_bandwidth')
        elif channel_bandwidth < 1.25e6:
            raise ValueError(
                'standardized values for channel_bandwidth not supported yet'
            )
        elif not isclose(channel_bandwidth % 125e3, 0, abs_tol=1e-6):
            raise ValueError('channel bandwidth must be set in increments of 125 kHz')

        if nfft not in self.VALID_FFT_SIZES:
            raise ValueError(f'nfft must be one of {self.VALID_FFT_SIZES}')

        if cp_ratio in self.VALID_CP_RATIOS:
            self.cp_ratio = cp_ratio
        else:
            raise ValueError(f'cp_ratio must be one of {self.VALID_CP_RATIOS}')

        if frame_duration not in self.VALID_FRAME_DURATIONS:
            raise ValueError(
                f'frame_duration must be one of {self.VALID_FRAME_DURATIONS}'
            )

        for freq_divisor, n in self.SAMPLING_FACTOR_BY_FREQUENCY_DIV.items():
            if isclose(channel_bandwidth % freq_divisor, 0, abs_tol=1e-6):
                sampling_factor = self.sampling_factor = n
                break
        else:
            # no match with the table - standardized default
            sampling_factor = self.sampling_factor = 8 / 7

        std_sample_rate = np.floor(sampling_factor * channel_bandwidth / 8000) * 8000
        cp_size = int(np.rint(cp_ratio * nfft))
        self.total_symbol_duration = (
            int(np.rint((1 + cp_ratio) * nfft)) / std_sample_rate
        )
        self.symbols_per_frame = int(
            np.floor(frame_duration / self.total_symbol_duration)
        )

        if alt_sample_rate is None:
            sample_rate = std_sample_rate
        else:
            scale = alt_sample_rate / std_sample_rate

            if not (_isclosetoint(scale) or _isclosetoint(1 / scale)):
                raise ValueError(
                    'alt_sample_rate must be integer multiple or divisor of ofdm sample_rate'
                )
            if not _isclosetoint(cp_size * scale):
                raise ValueError(
                    'alt_sample_rate is too small to capture any cyclic prefixes'
                )

            nfft = round(nfft * scale)
            cp_size = round(cp_size * scale)
            sample_rate = alt_sample_rate

        super().__init__(
            channel_bandwidth=channel_bandwidth,
            nfft=nfft,
            sample_rate=sample_rate,
            frame_duration=frame_duration,
            cp_sizes=xp.full(self.symbols_per_frame, cp_size),
            contiguous_size=round(frame_duration * sample_rate),
        )

    def index_cyclic_prefix(
        self, *, frames=(0,), symbols: tuple[int, ...] | typing.Literal['all'] = 'all'
    ) -> Array:
        """build the index tensor for cyclic prefix correlation over the selected frames and symbols"""
        return _802_16_index_cyclic_prefix(self, frames=frames, symbols=symbols)


@util.lru_cache(4)
def _802_16_index_cyclic_prefix(
    phy: Phy802_16,
    *,
    frames=(0,),
    symbols: tuple[int, ...] | typing.Literal['all'] = 'all',
) -> Array:
    """build the index tensor for cyclic prefix correlation over the selected frames and symbols"""

    xp = array_namespace(phy.cp_sizes)
    frames = xp.array(frames)

    symbol_inds = _index_or_all(
        symbols, '"symbols" argument', size=phy.symbols_per_frame, xp=xp
    )

    # build each grid axis separately
    grid = []

    # axis 0: symbol number in each frame
    grid.append(phy.cp_start_idx[symbol_inds])

    # axis 1: frame number
    grid.append(frames * phy.frame_size)

    # axis 2: cp index
    grid.append(xp.ogrid[0 : phy.cp_sizes[1]])

    # pad the axis dimensions so they can be broadcast together
    a = xp.meshgrid(*grid, indexing='ij', copy=False)

    # sum all of the index offsets
    inds = a[0].copy()
    for sub in a[1:]:
        inds += sub

    return inds
