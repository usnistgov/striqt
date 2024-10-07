from __future__ import annotations
import dataclasses
import functools
import numbers
import typing

import numpy as np
from xarray_dataclasses import AsDataArray, Coordof, Data, Attr
import iqwaveform

from ._common import as_registered_channel_analysis
from .._api import structs, type_stubs


if typing.TYPE_CHECKING:
    from iqwaveform import ofdm
    import pandas as pd
else:
    ofdm = iqwaveform.util.lazy_import('iqwaveform.ofdm')
    pd = iqwaveform.util.lazy_import('pandas')


### Time elapsed dimension and coordinates
CyclicSampleLagAxis = typing.Literal['cyclic_sample_lag']


@dataclasses.dataclass
class CyclicSampleLagCoords:
    data: Data[CyclicSampleLagAxis, np.float32]
    standard_name: Attr[str] = 'Cyclic sample lag'
    units: Attr[str] = 's'

    @staticmethod
    @functools.lru_cache
    def factory(
        capture: structs.Capture, *, subcarrier_spacings: tuple[float, ...], **_
    ) -> dict[str, np.ndarray]:
        max_len = _get_max_corr_size(capture, subcarrier_spacings=subcarrier_spacings)
        axis_name = typing.get_args(CyclicSampleLagAxis)[0]
        return pd.RangeIndex(0, max_len, name=axis_name) / capture.sample_rate


### Subcarrier spacing label axis
SubcarrierSpacingAxis = typing.Literal['subcarrier_spacing']


@dataclasses.dataclass
class SubcarrierSpacingCoords:
    data: Data[SubcarrierSpacingAxis, np.float32]
    standard_name: Attr[str] = 'Subcarrier spacing'
    units: Attr[str] = 'Hz'

    @staticmethod
    @functools.lru_cache
    def factory(
        capture: structs.Capture, *, subcarrier_spacings: tuple[float, ...], **_
    ):
        return tuple(subcarrier_spacings)


### Dataarray definition
@dataclasses.dataclass
class CellularCyclicAutocorrelation(AsDataArray):
    power_time_series: Data[
        tuple[SubcarrierSpacingAxis, CyclicSampleLagAxis], np.float32
    ]

    subcarrier_spacing: Coordof[SubcarrierSpacingCoords]
    cyclic_sample_lag: Coordof[CyclicSampleLagCoords]


### iqwaveform wrapper
@as_registered_channel_analysis(CellularCyclicAutocorrelation)
def cellular_cyclic_autocorrelation(
    iq: type_stubs.ArrayLike,
    capture: structs.Capture,
    *,
    subcarrier_spacings: typing.Union[float, tuple[float, ...]] = (15e3, 30e3, 60e3),
    frame_range: typing.Union[int, tuple[int, typing.Optional[int]]] = (0, 1),
    slot_range: typing.Union[int, tuple[int, typing.Optional[int]]] = (0, None),
    symbol_range: typing.Union[int, tuple[int, typing.Optional[int]]] = (0, None),
    normalize: bool = True,
) -> type_stubs.ArrayLike:
    """evaluate the cyclic autocorrelation of the IQ sequence based on 4G or 5G cellular
    cyclic prefix sample lag offsets.

    The correlation can be configured to evaluate across specified ranges of frame
    indices, slot indices (across the frames), and symbol indices (across the slots).
    Each range may be specified as a single number ("first $N$ indices") or as a
    tuple that is passed to the python builtin `range`.

    Args:
        iq: the input waveform
        capture: the waveform capture specification
        subcarrier_spacings: cellular SCS to evaluate (currently supports 15e3, 30e3, or 60e3)
        frame_range: the frame indices to evaluate
        slot_range: the slots to evaluate within all indexed frames
        symbol_range: the symbols to evaluate within all indexed slots
        normalize: if True, results are normalized as autocorrelation (0 to 1);
            otherwise, autocovariance (power)

    Returns:
        an float32-valued array with matching the array type of `iq`
    """

    RANGE_MAP = {'frames': frame_range, 'slots': slot_range, 'symbols': symbol_range}

    xp = iqwaveform.util.array_namespace(iq)
    subcarrier_spacings = tuple(subcarrier_spacings)
    phy_scs = _get_phy_mappings(
        capture.analysis_bandwidth, capture.sample_rate, subcarrier_spacings, xp=xp
    )
    metadata = {}

    if isinstance(subcarrier_spacings, numbers.Number):
        subcarrier_spacings = tuple(
            subcarrier_spacings,
        )

    # transform the indexing arguments into the form expected by phy.index_cyclic_prefix
    idx_kws = {}
    for name, field_range in RANGE_MAP.items():
        if isinstance(field_range, numbers.Number):
            field_range = (field_range,)
        else:
            field_range = tuple(field_range)
        metadata[name] = field_range

        if field_range in ((0,), (None, None), (0, None)):
            idx_kws[name] = 'all'
        else:
            idx_kws[name] = tuple(range(*field_range))

    max_len = _get_max_corr_size(capture, subcarrier_spacings=subcarrier_spacings)

    result = xp.full((len(subcarrier_spacings), max_len), np.nan, dtype=np.float32)
    for i, phy in enumerate(phy_scs.values()):
        # R = _correlate_cyclic_prefixes(iq, phy, **kws)
        cp_inds = phy.index_cyclic_prefix(**idx_kws)
        R = ofdm.corr_at_indices(cp_inds, iq, phy.nfft, norm=normalize)

        result[i][: R.size] = xp.abs(R)

    if normalize:
        metadata.update(standard_name='Cyclic Autocorrelation')
    else:
        metadata.update(standard_name='Cyclic Autocovariance', units='mW')

    return result, metadata


@functools.lru_cache
def _get_phy_mappings(
    channel_bandwidth: float,
    sample_rate: float,
    subcarrier_spacings: tuple[float, ...],
    xp=np,
) -> dict[str]:
    return {
        scs: ofdm.Phy3GPP(channel_bandwidth, scs, sample_rate=sample_rate, xp=xp)
        for scs in subcarrier_spacings
    }


@functools.lru_cache
def _get_max_corr_size(
    capture: structs.Capture, *, subcarrier_spacings: tuple[float, ...]
):
    phy_scs = _get_phy_mappings(
        capture.analysis_bandwidth, capture.sample_rate, subcarrier_spacings
    )
    return max([np.diff(phy.cp_start_idx).min() for phy in phy_scs.values()])
