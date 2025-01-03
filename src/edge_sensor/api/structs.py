"""data structures for configuration of radio hardware, captures, and sweeps"""

from __future__ import annotations
import functools
import msgspec
import typing
from typing import Annotated, Optional, Literal, Any

from . import util

import channel_analysis
import channel_analysis.api.filters
from channel_analysis.api.structs import (
    meta,
    ChannelAnalysis,  # noqa: F401
    struct_to_builtins,  # noqa: F401
    builtins_to_struct,  # noqa: F401
    copy_struct,  # noqa: F401
)

if typing.TYPE_CHECKING:
    import pandas as pd
else:
    # this is needed to resolve the 'pd.Timestamp' stub at runtime
    pd = util.lazy_import('pandas')

_TShift = Literal['left', 'right', 'none']


def _dict_hash(d):
    key_hash = frozenset(d.keys())
    value_hash = tuple(
        [_dict_hash(v) if isinstance(v, dict) else v for v in d.values()]
    )
    return hash(key_hash) ^ hash(value_hash)


def _make_default_analysis():
    return channel_analysis.as_registered_channel_analysis.spec_type()()


class WaveformCapture(channel_analysis.Capture, forbid_unknown_fields=True):
    """Capture specification structure for a generic waveform.

    This subset of RadioCapture is broken out here to simplify the evaluation of
    sampling parameters that are independent from other radio parameters.
    """

    # acquisition
    duration: Annotated[float, meta('Acquisition duration', 's', gt=0)] = 0.1
    sample_rate: Annotated[float, meta('Sample rate', 'S/s', gt=0)] = 15.36e6

    # filtering and resampling
    analysis_bandwidth: Annotated[
        float, meta('Bandwidth of the analysis filter (or inf to disable)', 'Hz', gt=0)
    ] = float('inf')
    lo_shift: Annotated[_TShift, meta('LO shift direction')] = 'none'
    host_resample: bool = True


SingleChannelType = Annotated[int, meta('Input port index', ge=0)]
SingleGainType = Annotated[float, meta('Gain setting', 'dB')]


class RadioCapture(WaveformCapture, forbid_unknown_fields=True):
    """Capture specification for a single radio waveform"""

    # RF and leveling
    center_frequency: Annotated[float, meta('RF center frequency', 'Hz', gt=0)] = 3710e6
    channels: Annotated[tuple[SingleChannelType, ...], meta('Input port indices')] = (
        0,
    )
    gains: Annotated[
        tuple[SingleGainType, ...], meta('Gain setting for each channel', 'dB')
    ] = (-10,)

    delay: Optional[
        Annotated[float, meta('Delay in acquisition start time', 's', gt=0)]
    ] = None
    start_time: Optional[Annotated['pd.Timestamp', meta('Acquisition start time')]] = (
        None
    )


class RadioSetup(msgspec.Struct, forbid_unknown_fields=True):
    """run-time characteristics of the radio that are left invariant during a sweep"""

    driver: str = 'AirT7x01B'
    resource: Any = None
    time_source: Literal['host', 'internal', 'external', 'gps'] = 'host'
    continuous_trigger: Annotated[
        bool,
        meta('Whether to trigger immediately after each call to acquire() when armed'),
    ] = True
    periodic_trigger: Optional[float] = None
    calibration: Optional[str] = None
    _transient_holdoff_time: Optional[float] = None
    gapless_repeats: Annotated[
        bool,
        meta('whether to raise an exception on overflows between identical captures'),
    ] = False
    time_sync_every_capture: Annotated[
        bool, meta('whether to sync to PPS before each capture in a sweep')
    ] = False
    warmup_sweep: Annotated[
        bool,
        meta(
            'whether to run the GPU compute on empty buffers before sweeping for more even run time'
        ),
    ] = True


class Description(msgspec.Struct, forbid_unknown_fields=True):
    summary: Optional[str] = None
    location: Optional[tuple[float, float, float]] = None
    signal_chain: Optional[tuple[str, ...]] = tuple()
    version: str = 'unversioned'


class Output(msgspec.Struct, forbid_unknown_fields=True, frozen=True, cache_hash=True):
    path: Optional[str] = '{yaml_name}-{start_time}'
    store: typing.Union[Literal['zip'], Literal['directory']] = 'zip'
    coord_aliases: dict[str, dict[str, dict[str, Any]]] = {}

    def __hash__(self):
        # hashing coordinate aliases greatly speeds up xarray coordinate generation
        return hash(self.path) ^ hash(self.store) ^ _dict_hash(self.coord_aliases)


class Sweep(msgspec.Struct, forbid_unknown_fields=True):
    captures: tuple[RadioCapture, ...]
    radio_setup: RadioSetup = msgspec.field(default_factory=RadioSetup)
    defaults: RadioCapture = msgspec.field(default_factory=RadioCapture)
    channel_analysis: dict = msgspec.field(default_factory=_make_default_analysis)
    description: Description = msgspec.field(default_factory=Description)
    output: Output = msgspec.field(default_factory=Output)


@functools.lru_cache
def get_attrs(struct: type[msgspec.Struct], field: str) -> dict[str, str]:
    """get an attrs dict for xarray based on Annotated type hints with `meta`"""
    hints = typing.get_type_hints(struct, include_extras=True)

    try:
        metas = hints[field].__metadata__
    except (AttributeError, KeyError):
        return {}

    if len(metas) == 0:
        return {}
    elif len(metas) == 1 and isinstance(metas[0], msgspec.Meta):
        return metas[0].extra
    else:
        raise TypeError(
            'Annotated[] type hints must contain exactly one msgspec.Meta object'
        )
