from __future__ import annotations as __

import numbers
import typing

from ..lib import specs, util
from .shared import registry

if typing.TYPE_CHECKING:
    import numpy as np
    import pandas as pd

    import striqt.waveform as iqwaveform
else:
    iqwaveform = util.lazy_import('striqt.waveform')
    np = util.lazy_import('numpy')
    pd = util.lazy_import('pandas')


class CellularCyclicAutocorrelationSpec(
    specs.Measurement,
    kw_only=True,
    frozen=True,
    dict=True,
):
    subcarrier_spacings: typing.Union[float, tuple[float, ...]] = (15e3, 30e3, 60e3)
    frame_range: typing.Union[int, tuple[int, typing.Optional[int]]] = (0, 1)
    frame_slots: typing.Union[str, dict[float, str], None] = None
    symbol_range: typing.Union[int, tuple[int, typing.Optional[int]]] = (0, None)
    generation: typing.Literal['4G','5G'] = '5G'


class CellularCyclicAutocorrelationKeywords(specs.AnalysisKeywords, total=False):
    subcarrier_spacings: typing.Union[float, tuple[float, ...]]
    frame_range: typing.Union[int, tuple[int, typing.Optional[int]]]
    frame_slots: typing.Optional[str]
    symbol_range: typing.Union[int, tuple[int, typing.Optional[int]]]
    generation: typing.Literal['4G','5G']


class NormalizedTDDSlotConfig(typing.NamedTuple):
    frame_slots: str
    special_symbols: str
    code_maps: dict[str, float]
    slot_by_symbol: dict[str, float]
    downlink_slot_indexes: tuple[int, ...]
    uplink_slot_indexes: tuple[int, ...]
    frame_by_symbol: str | None


@util.lru_cache()
def tdd_config_from_str(
    subcarrier_spacing: float,
    frame_slots: str | None,
    special_symbols: str | None = None,
    *,
    normal_cp=True,
    flex_as=None,
) -> NormalizedTDDSlotConfig:
    """generate a symbol-by-symbol sequence of masking arrays for uplink and downlink.

    The number of slots given in the frame match the appropriate number for a given
    5G NR or LTE subcarrier spacing.

    Arguments:
        frame_slots: a string composed of the characters {'d', 'u', 's'} that
            indicate the sequence of slots in 1 cellular frame (or None for all downlink)
        special_symbols: the a string composed of the characters {'d', 'u', 'f'} that
            indicate the sequence of symbol types in the special slot (or None for all downlink)

    Returns:
        a NormalizedTDDSlotConfig object containing more detailed parameters
    """

    expect_slot_count = round(10 * subcarrier_spacing / 15e3)

    if flex_as is not None:
        flex_as = flex_as.lower()

    if frame_slots is None:
        frame_slots = expect_slot_count * 'd'
    elif len(frame_slots) == 1:
        frame_slots = expect_slot_count * frame_slots
    elif len(frame_slots) != expect_slot_count:
        raise ValueError(
            f'frame_slots must have length {expect_slot_count} to match the slot count at {round(subcarrier_spacing / 1e3)} kHz'
        )
    else:
        frame_slots = frame_slots.lower()

    if special_symbols is not None:
        special_symbols = special_symbols.lower()

    if len(frame_slots.strip('dus')) > 0:
        allowed = set('dus')
        raise ValueError(f'frame_slots string may only contain {allowed}')

    if special_symbols is None:
        pass
    elif len(special_symbols.strip('duf')) > 0:
        allowed = set('duf')
        raise ValueError(f'special_symbols string may only contain {allowed}')

    if normal_cp:
        symbols_per_slot = 14
    else:
        symbols_per_slot = 12

    downlink_code_to_value = {
        'd': 1,
        'u': float('nan'),
        'f': 1 if flex_as == 'd' else float('nan'),
    }
    uplink_code_to_value = {
        'd': float('nan'),
        'u': 1,
        'f': 1 if flex_as == 'u' else float('nan'),
    }

    code_mapping = {'downlink': downlink_code_to_value, 'uplink': uplink_code_to_value}

    slot_by_symbol = {
        'd': symbols_per_slot * 'd',
        'u': symbols_per_slot * 'u',
        's': special_symbols,
    }

    downlink_slots = [i for i, s in enumerate(frame_slots) if s == 'd']
    uplink_slots = [i for i, s in enumerate(frame_slots) if s == 'u']

    if 's' not in frame_slots or special_symbols is not None:
        frame_by_symbol = ''.join([slot_by_symbol[k] for k in frame_slots])
    else:
        frame_by_symbol = None

    return NormalizedTDDSlotConfig(
        frame_slots=frame_slots,
        special_symbols=special_symbols,
        code_maps=code_mapping,
        slot_by_symbol=slot_by_symbol,
        uplink_slot_indexes=tuple(uplink_slots),
        downlink_slot_indexes=tuple(downlink_slots),
        frame_by_symbol=frame_by_symbol,
    )


@registry.coordinates(
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


@registry.coordinates(
    dtype='float32', attrs={'standard_name': 'Subcarrier spacing', 'units': 'Hz'}
)
@util.lru_cache()
def subcarrier_spacing(capture: specs.Capture, spec: CellularCyclicAutocorrelationSpec):
    return list(spec.subcarrier_spacings)


@registry.coordinates(dtype='str', attrs={'standard_name': 'Link direction'})
@util.lru_cache()
def link_direction(capture: specs.Capture, spec: CellularCyclicAutocorrelationSpec):
    values = np.array(['downlink', 'uplink'], dtype='U8')
    return values, {}


@util.lru_cache()
def _get_phy_mapping(
    channel_bandwidth: float,
    sample_rate: float,
    subcarrier_spacings: tuple[float, ...],
    generation: typing.Literal['4G','5G']='4G',
    xp=np,
) -> dict[float, iqwaveform.ofdm.Phy3GPP]:
    kws = dict(
        channel_bandwidth=channel_bandwidth, generation=generation, sample_rate=sample_rate
    )
    return {
        scs: iqwaveform.ofdm.Phy3GPP(subcarrier_spacing=scs, xp=xp, **kws)
        for scs in subcarrier_spacings
    }


@util.lru_cache()
def _get_max_corr_size(
    capture: specs.Capture, *, subcarrier_spacings: tuple[float, ...], generation: typing.Literal['4G','5G']='4G'
):
    phy_scs = _get_phy_mapping(
        capture.analysis_bandwidth, capture.sample_rate, subcarrier_spacings, generation=generation
    )
    return max([np.diff(phy.cp_start_idx).min() for phy in phy_scs.values()])


@registry.measurement(
    coord_factories=[link_direction, subcarrier_spacing, cyclic_sample_lag],
    dtype='float32',
    prefer_unaligned_input=True,
    spec_type=CellularCyclicAutocorrelationSpec,
    attrs={'units': 'mW', 'standard_name': 'Cyclic Autocovariance'},
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
        frame_slots: string composed of {'d', 'u', 's'} that specify the sequence
            of link direction of each slot in 1 TDD cellular frame (or None to fill with downlink)
        symbol_range: the symbols to evaluate within all indexed slots

    Returns:
        an float32-valued array with matching the array type of `iq`
    """

    spec = CellularCyclicAutocorrelationSpec.fromdict(kwargs)

    RANGE_MAP = {'frames': spec.frame_range, 'symbols': spec.symbol_range}

    xp = iqwaveform.util.array_namespace(iq)
    if isinstance(spec.subcarrier_spacings, tuple):
        scs = spec.subcarrier_spacings
    else:
        scs = (spec.subcarrier_spacings,)

    phy_scs = _get_phy_mapping(
        capture.analysis_bandwidth, capture.sample_rate, scs, generation=spec.generation, xp=xp
    )
    metadata = {}

    frame_slots = specs.maybe_lookup_with_capture_key(
        capture, spec.frame_slots, 'center_frequency', 'frame_slots', default='d'
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

    max_len = _get_max_corr_size(capture, subcarrier_spacings=scs, generation=spec.generation)

    result = xp.full(
        (iq.shape[0], 2, len(scs), max_len), np.nan, dtype=np.float32
    )
    for chan in range(iq.shape[0]):
        for iscs, phy in enumerate(phy_scs.values()):
            tdd_config = tdd_config_from_str(
                subcarrier_spacing=phy.subcarrier_spacing, frame_slots=frame_slots
            )

            cp_inds = phy.index_cyclic_prefix(
                **idx_kws, slots=tdd_config.downlink_slot_indexes
            )

            # shift index to the symbol boundary rather than the CP
            cyclic_shift = -phy.cp_sizes[0] * 2 // cp_inds.shape[1]

            R = iqwaveform.ofdm.corr_at_indices(cp_inds, iq[chan], phy.nfft, norm=False)
            R = xp.roll(R, cyclic_shift)
            result[chan][0][iscs][: R.size] = xp.abs(R)

            if len(tdd_config.uplink_slot_indexes) > 0:
                cp_inds = phy.index_cyclic_prefix(
                    **idx_kws, slots=tdd_config.uplink_slot_indexes
                )
                R = iqwaveform.ofdm.corr_at_indices(
                    cp_inds, iq[chan], phy.nfft, norm=False
                )
                R = xp.roll(R, cyclic_shift)
                result[chan][1][iscs][: R.size] = xp.abs(R)

    return result, metadata
