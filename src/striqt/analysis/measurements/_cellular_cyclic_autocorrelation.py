from __future__ import annotations
import numbers
import typing

from ..lib import registry, specs, util


if typing.TYPE_CHECKING:
    import iqwaveform
    import numpy as np
    import pandas as pd
else:
    iqwaveform = util.lazy_import('iqwaveform')
    np = util.lazy_import('numpy')
    pd = util.lazy_import('pandas')


class CellularCyclicAutocorrelationSpec(
    specs.Measurement,
    forbid_unknown_fields=True,
    cache_hash=True,
    kw_only=True,
    frozen=True,
):
    subcarrier_spacings: typing.Union[float, tuple[float, ...]] = (15e3, 30e3, 60e3)
    frame_range: typing.Union[int, tuple[int, typing.Optional[int]]] = (0, 1)
    downlink_slots: typing.Union[None, tuple[int, ...]] = None
    uplink_slots: typing.Union[tuple[int, ...]] = tuple()
    symbol_range: typing.Union[int, tuple[int, typing.Optional[int]]] = (0, None)


class CellularCyclicAutocorrelationKeywords(specs.AnalysisKeywords, total=False):
    subcarrier_spacings: typing.Union[float, tuple[float, ...]]
    frame_range: typing.Union[int, tuple[int, typing.Optional[int]]]
    downlink_slots: typing.Union[None, tuple[int, ...]]
    uplink_slots: typing.Union[tuple[int, ...]]
    symbol_range: typing.Union[int, tuple[int, typing.Optional[int]]]


@registry.coordinate_factory(
    dtype='float32', attrs={'standard_name': 'Cyclic sample lag', 'units': 's'}
)
@util.lru_cache()
def cyclic_sample_lag(
    capture: specs.Capture, spec: CellularCyclicAutocorrelationSpec
) -> dict[str, np.ndarray]:
    max_len = _get_max_corr_size(capture, subcarrier_spacings=spec.subcarrier_spacings)
    name = cyclic_sample_lag.__name__
    return pd.RangeIndex(0, max_len, name=name) / capture.sample_rate


### Subcarrier spacing label axis
SubcarrierSpacingAxis = typing.Literal['subcarrier_spacing']


@registry.coordinate_factory(
    dtype='float32', attrs={'standard_name': 'Subcarrier spacing', 'units': 'Hz'}
)
@util.lru_cache()
def subcarrier_spacing(capture: specs.Capture, spec: CellularCyclicAutocorrelationSpec):
    return list(spec.subcarrier_spacings)


@registry.coordinate_factory(dtype='str', attrs={'standard_name': 'Link direction'})
@util.lru_cache()
def link_direction(capture: specs.Capture, spec: CellularCyclicAutocorrelationSpec):
    values = np.array(['downlink', 'uplink'], dtype='U8')
    return values, {}


@util.lru_cache()
def _get_phy_mapping(
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


@util.lru_cache()
def _get_max_corr_size(
    capture: specs.Capture, *, subcarrier_spacings: tuple[float, ...]
):
    phy_scs = _get_phy_mapping(
        capture.analysis_bandwidth, capture.sample_rate, subcarrier_spacings
    )
    return max([np.diff(phy.cp_start_idx).min() for phy in phy_scs.values()])


@registry.measurement(
    coord_funcs=[link_direction, subcarrier_spacing, cyclic_sample_lag],
    dtype='float32',
    spec_type=CellularCyclicAutocorrelationSpec,
    attrs={'units': 'dBm', 'standard_name': 'Cyclic Autocovariance'},
)
def cellular_cyclic_autocorrelation(
    iq: 'iqwaveform.util.Array',
    capture: specs.Capture,
    **kwargs: typing.Unpack[CellularCyclicAutocorrelationKeywords],
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

    Returns:
        an float32-valued array with matching the array type of `iq`
    """

    spec = CellularCyclicAutocorrelationSpec.fromdict(kwargs)

    RANGE_MAP = {'frames': spec.frame_range, 'symbols': spec.symbol_range}

    xp = iqwaveform.util.array_namespace(iq)
    subcarrier_spacings = tuple(spec.subcarrier_spacings)
    phy_scs = _get_phy_mapping(
        capture.analysis_bandwidth, capture.sample_rate, subcarrier_spacings, xp=xp
    )
    metadata = {}

    if isinstance(subcarrier_spacings, numbers.Number):
        subcarrier_spacings = tuple(
            subcarrier_spacings,
        )

    if spec.downlink_slots is None:
        downlink_slots = 'all'
    else:
        downlink_slots = tuple(downlink_slots)

    uplink_slots = tuple(spec.uplink_slots)

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
            cp_inds = phy.index_cyclic_prefix(**idx_kws, slots=downlink_slots)

            # shift index to the symbol boundary rather than the CP
            cyclic_shift = -phy.cp_sizes[0] * 2 // cp_inds.shape[1]

            R = iqwaveform.ofdm.corr_at_indices(cp_inds, iq[chan], phy.nfft, norm=False)
            R = xp.roll(R, cyclic_shift)
            result[chan][0][iscs][: R.size] = xp.abs(R)

            if len(uplink_slots) > 0:
                cp_inds = phy.index_cyclic_prefix(**idx_kws, slots=uplink_slots)
                R = iqwaveform.ofdm.corr_at_indices(
                    cp_inds, iq[chan], phy.nfft, norm=False
                )
                R = xp.roll(R, cyclic_shift)
                result[chan][1][iscs][: R.size] = xp.abs(R)

    return result, metadata
