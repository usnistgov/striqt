"""OFDM frame-model base class shared by the standard-specific PHY models.

`PhyOFDM` turns a sequence of cyclic prefix sizes into the sample indices of the
cyclic prefixes and symbols in one contiguous block; the subclasses in `phy_3gpp`
and `phy_802_16` supply the sizes and the index tensors for correlation.
`_index_or_all` validates the symbol/slot selection arguments they share.
"""

from __future__ import annotations as __

import typing

from ..arrays import array_namespace
from ..util import np

if typing.TYPE_CHECKING:
    from ..typing import Array


def _index_or_all(inds: tuple[int, ...] | typing.Literal['all'], name, size, xp=None):
    if xp is None:
        xp = np

    if isinstance(inds, (tuple, list)):
        x = xp.array(inds)
    elif isinstance(inds, str) and inds == 'all':
        if size is None:
            raise ValueError('must set max to allow "all" value')
        x = xp.arange(size)
    else:
        raise ValueError(f'{name} argument must be an array of indices or "all"')

    if x.ndim not in (0, 1):
        raise ValueError(f'{name} argument must be a sequence of indices')

    if x.max() >= size:
        raise ValueError(f'{name} value {inds} exceeds the maximum {size - 1}')
    if (-x).max() > size:
        raise ValueError(f'{name} value {inds} is below the minimum {-size}')

    return x


class PhyOFDM:
    def __init__(
        self,
        *,
        channel_bandwidth: float,
        sample_rate: float,
        nfft: int,
        cp_sizes: Array,
        frame_duration: float | None = None,
        contiguous_size: int | None = None,
    ):
        xp = array_namespace(cp_sizes)

        self.channel_bandwidth = channel_bandwidth
        self.sample_rate = sample_rate

        self.nfft: int = nfft
        self.frame_duration = frame_duration

        self.subcarrier_spacing: float = self.sample_rate / nfft
        if frame_duration is None:
            self.frame_size = None
        else:
            self.frame_size = round(sample_rate * frame_duration)

        self.cp_sizes = cp_sizes

        if contiguous_size is not None:
            self.contiguous_size = contiguous_size
        else:
            # the default contiguous block is the symbols and their prefixes with no padding
            self.contiguous_size = int(np.sum(cp_sizes) + len(cp_sizes) * nfft)

        # build a (start_idx, size) pair for each CP
        pair_sizes = xp.concatenate((xp.array((0,)), self.cp_sizes + self.nfft))
        self.cp_start_idx = (pair_sizes.cumsum()).astype(int)[:-1]
        start_and_size = zip(self.cp_start_idx, self.cp_sizes)

        idx_range = xp.arange(self.contiguous_size).astype(int)

        # indices in the contiguous range that are CP
        self.cp_idx = xp.concatenate([
            idx_range[start : start + size] for start, size in start_and_size
        ])

        # indices in the contiguous range that are not CP
        self.symbol_idx = xp.setdiff1d(idx_range, self.cp_idx)

    def index_cyclic_prefix(self) -> Array:
        raise NotImplementedError
