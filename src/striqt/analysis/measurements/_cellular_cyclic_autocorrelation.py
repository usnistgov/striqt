from __future__ import annotations as __

import numbers
import typing

from .. import specs

from ..lib import util
from .shared import registry, hint_keywords

if typing.TYPE_CHECKING:
    import numpy as np
    import pandas as pd

    import striqt.waveform as sw

else:
    sw = util.lazy_import('striqt.waveform')
    np = util.lazy_import('numpy')
    pd = util.lazy_import('pandas')


class SlotBySymbol(typing.TypedDict):
    d: str
    u: str
    s: str | None


class NormalizedTDDSlotConfig(typing.NamedTuple):
    frame_slots: str
    special_symbols: str | None
    code_maps: dict[str, dict[str, float]]
    slot_by_symbol: SlotBySymbol
    downlink_slot_indexes: tuple[int, ...]
    uplink_slot_indexes: tuple[int, ...]
    frame_by_symbol: str


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
        'd': 1.0,
        'u': float('nan'),
        'f': 1.0 if flex_as == 'd' else float('nan'),
    }
    uplink_code_to_value = {
        'd': float('nan'),
        'u': 1.0,
        'f': 1.0 if flex_as == 'u' else float('nan'),
    }

    code_mapping = {'downlink': downlink_code_to_value, 'uplink': uplink_code_to_value}

    slot_by_symbol = SlotBySymbol(
        d=symbols_per_slot * 'd',
        u=symbols_per_slot * 'u',
        s=special_symbols,
    )

    downlink_slots = [i for i, s in enumerate(frame_slots) if s == 'd']
    uplink_slots = [i for i, s in enumerate(frame_slots) if s == 'u']

    if 's' not in frame_slots or special_symbols is not None:
        frame_by_symbol = ''.join([slot_by_symbol[k] for k in frame_slots])
    else:
        frame_by_symbol = 'd' * len(frame_slots)

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
    capture: specs.Capture, spec: specs.CellularCyclicAutocorrelator
) -> 'pd.Index':
    max_len = _get_max_corr_size(capture, subcarrier_spacings=spec.subcarrier_spacings)
    name = cyclic_sample_lag.__name__
    return pd.RangeIndex(0, max_len, name=name) / capture.sample_rate


### Subcarrier spacing label axis
SubcarrierSpacingAxis = typing.Literal['subcarrier_spacing']


@registry.coordinates(
    dtype='float32', attrs={'standard_name': 'Subcarrier spacing', 'units': 'Hz'}
)
@util.lru_cache()
def subcarrier_spacing(
    capture: specs.Capture, spec: specs.CellularCyclicAutocorrelator
):
    if isinstance(spec.subcarrier_spacings, tuple):
        return list(spec.subcarrier_spacings)
    else:
        return [spec.subcarrier_spacings]


@registry.coordinates(dtype='str', attrs={'standard_name': 'Link direction'})
@util.lru_cache()
def link_direction(capture: specs.Capture, spec: specs.CellularCyclicAutocorrelator):
    values = np.array(['downlink', 'uplink'], dtype='U8')
    return values, {}


@util.lru_cache()
def _get_phy_mapping(
    channel_bandwidth: float,
    sample_rate: float,
    subcarrier_spacings: float | tuple[float, ...],
    generation: typing.Literal['4G', '5G'] = '4G',
    xp=np,
) -> dict[float, sw.ofdm._Phy3GPP]:
    seq = (
        subcarrier_spacings
        if isinstance(subcarrier_spacings, tuple)
        else [subcarrier_spacings]
    )

    phy = {}

    for scs in seq:
        phy[scs] = sw.ofdm.get_3gpp_phy(
            subcarrier_spacing=scs,
            channel_bandwidth=channel_bandwidth,
            generation=generation,
            sample_rate=sample_rate,
            xp=xp,
        )

    return phy


@util.lru_cache()
def _get_max_corr_size(
    capture: specs.Capture,
    *,
    subcarrier_spacings: float | tuple[float, ...],
    generation: typing.Literal['4G', '5G'] = '4G',
):
    phy_scs = _get_phy_mapping(
        capture.analysis_bandwidth,
        capture.sample_rate,
        subcarrier_spacings,
        generation=generation,
    )
    sizes = [np.diff(phy.cp_start_idx).min() for phy in phy_scs.values()]
    return max(sizes)


@typing.overload
def _get_spec_range(field_range: tuple[int, None], name) -> typing.Literal['all']:
    pass


@typing.overload
def _get_spec_range(
    field_range: typing.Union[int, tuple[int, int]], name
) -> tuple[int, ...]:
    pass


def _get_spec_range(
    field_range: typing.Union[int, tuple[int, None], tuple[int, int]], name
) -> tuple[int, ...] | typing.Literal['all']:
    if field_range in ((0,), (None, None), (0, None)):
        return 'all'

    elif not isinstance(field_range, tuple):
        return (field_range,)

    start, stop = field_range

    if stop is None:
        raise TypeError(
            f'{name!r} field [start, stop] indices must have two integers unless start is 0'
        )

    return tuple(range(start, stop))


@hint_keywords(specs.CellularCyclicAutocorrelator)
@registry.measurement(
    coord_factories=[link_direction, subcarrier_spacing, cyclic_sample_lag],
    dtype='float32',
    prefer_unaligned_input=True,
    spec_type=specs.CellularCyclicAutocorrelator,
    attrs={'units': 'mW', 'standard_name': 'Cyclic Autocovariance'},
)
def cellular_cyclic_autocorrelation(
    iq: 'sw.util.ArrayType',
    capture: specs.Capture,
    **kwargs,
):
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

    spec = specs.CellularCyclicAutocorrelator.from_dict(kwargs)

    xp = sw.util.array_namespace(iq)
    if isinstance(spec.subcarrier_spacings, tuple):
        scs = spec.subcarrier_spacings
    else:
        scs = (spec.subcarrier_spacings,)

    phy_scs = _get_phy_mapping(
        capture.analysis_bandwidth,
        capture.sample_rate,
        scs,
        generation=spec.generation,
        xp=xp,
    )
    metadata = {}

    metadata['frames'] = spec.frame_range
    metadata['symbols'] = spec.symbol_range

    frame_range = _get_spec_range(spec.frame_range, 'frame_range')
    symbol_range = _get_spec_range(spec.symbol_range, 'symbol_range')

    def index_cp_for_slot(slots):
        return phy.index_cyclic_prefix(
            frames=frame_range, symbols=symbol_range, slots=slots
        )

    max_len = _get_max_corr_size(
        capture, subcarrier_spacings=scs, generation=spec.generation
    )

    result = xp.full((iq.shape[0], 2, len(scs), max_len), np.nan, dtype=np.float32)
    for chan in range(iq.shape[0]):
        for iscs, phy in enumerate(phy_scs.values()):
            tdd_config = tdd_config_from_str(
                subcarrier_spacing=phy.subcarrier_spacing, frame_slots=spec.frame_slots
            )

            cp_inds = index_cp_for_slot(tdd_config.downlink_slot_indexes)
            R = sw.ofdm.corr_at_indices(cp_inds, iq[chan], phy.nfft, norm=False)
            result[chan][0][iscs][: R.size] = xp.abs(R)

            if len(tdd_config.uplink_slot_indexes) == 0:
                continue

            cp_inds = index_cp_for_slot(tdd_config.uplink_slot_indexes)
            R = sw.ofdm.corr_at_indices(cp_inds, iq[chan], phy.nfft, norm=False)
            result[chan][1][iscs][: R.size] = xp.abs(R)

    return result, metadata
