from __future__ import annotations
import dataclasses
import functools
import numbers
import typing

from xarray_dataclasses import AsDataArray, Coordof, Data, Attr

from ..api.registry import register_analysis_to_xarray
from ..api import structs, util


if typing.TYPE_CHECKING:
    import iqwaveform
    import numpy as np
    import pandas as pd
else:
    iqwaveform = util.lazy_import('iqwaveform')
    np = util.lazy_import('numpy')
    pd = util.lazy_import('pandas')


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
        return list(subcarrier_spacings)


### Up/down link category
LinkDirectionAxis = typing.Literal['link_direction']


@dataclasses.dataclass
class LinkDirectionCoords:
    data: Data[LinkDirectionAxis, str]
    standard_name: Attr[str] = 'Link direction'

    @staticmethod
    @functools.lru_cache
    def factory(
        capture: structs.Capture, *, downlink_slots='all', uplink_slots=tuple(), **_
    ):
        return ['downlink', 'uplink'], {
            'downlink_slots': downlink_slots,
            'uplink_slots': uplink_slots,
        }


### Dataarray definition
@dataclasses.dataclass
class CellularCyclicAutocorrelation(AsDataArray):
    power_time_series: Data[
        tuple[LinkDirectionAxis, SubcarrierSpacingAxis, CyclicSampleLagAxis], np.float32
    ]

    link_direction: Coordof[LinkDirectionCoords]
    subcarrier_spacing: Coordof[SubcarrierSpacingCoords]
    cyclic_sample_lag: Coordof[CyclicSampleLagCoords]


### iqwaveform wrapper
@register_analysis_to_xarray(CellularCyclicAutocorrelation)
def cellular_cyclic_autocorrelation(
    iq: 'iqwaveform.util.Array',
    capture: structs.Capture,
    *,
    subcarrier_spacings: typing.Union[float, tuple[float, ...]] = (15e3, 30e3, 60e3),
    frame_range: typing.Union[int, tuple[int, typing.Optional[int]]] = (0, 1),
    downlink_slots: typing.Union[None, tuple[int, ...]] = None,
    uplink_slots: typing.Union[tuple[int, ...]] = tuple(),
    symbol_range: typing.Union[int, tuple[int, typing.Optional[int]]] = (0, None),
    normalize: bool = True,
) -> 'iqwaveform.util.Array':
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
        downlink_slots: slot indexes to attribute to the downlink (or all, if None)
        uplink_slots: slot indexes to attribute to the uplink
        symbol_range: the symbols to evaluate within all indexed slots
        normalize: if True, results are normalized as autocorrelation (0 to 1);
            otherwise, autocovariance (power)

    Returns:
        an float32-valued array with matching the array type of `iq`
    """

    RANGE_MAP = {'frames': frame_range, 'symbols': symbol_range}

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

    if downlink_slots is None:
        downlink_slots = 'all'
    else:
        downlink_slots = tuple(downlink_slots)

    uplink_slots = tuple(uplink_slots)

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

    result = xp.full(
        (iq.shape[0], 2, len(subcarrier_spacings), max_len), np.nan, dtype=np.float32
    )
    for chan in range(iq.shape[0]):
        for iscs, phy in enumerate(phy_scs.values()):
            # R = _correlate_cyclic_prefixes(iq, phy, **kws)
            cp_inds = phy.index_cyclic_prefix(**idx_kws, slots=downlink_slots)
            R = iqwaveform.ofdm.corr_at_indices(
                cp_inds, iq[chan], phy.nfft, norm=normalize
            )
            result[chan][0][iscs][: R.size] = xp.abs(R)

            if len(uplink_slots) > 0:
                cp_inds = phy.index_cyclic_prefix(**idx_kws, slots=uplink_slots)
                R = iqwaveform.ofdm.corr_at_indices(
                    cp_inds, iq[chan], phy.nfft, norm=normalize
                )
                result[chan][1][iscs][: R.size] = xp.abs(R)

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
        scs: iqwaveform.ofdm.Phy3GPP(
            channel_bandwidth, scs, sample_rate=sample_rate, xp=xp
        )
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
