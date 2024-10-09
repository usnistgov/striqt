"""data structures for configuration of radio hardware, captures, and sweeps"""

from __future__ import annotations
from frozendict import frozendict
import functools
import msgspec
import typing
from typing import Annotated, Optional, Literal, Any

from . import util

import channel_analysis
import channel_analysis._api.filters
from channel_analysis._api.structs import (
    meta,
    ChannelAnalysis,
    struct_to_builtins,
    builtins_to_struct,
    copy_struct,
)

if typing.TYPE_CHECKING:
    import pandas as pd
else:
    # this is needed to resolve the TimestampType stub at runtime
    pd = util.lazy_import('pandas')

_TShift = Literal['left', 'right', 'none']


def _make_default_analysis():
    return channel_analysis.filters.as_registered_channel_analysis.spec_type()()


class RadioCapture(channel_analysis.Capture, forbid_unknown_fields=True):
    """configuration for a single waveform capture"""

    # RF and leveling
    center_frequency: Annotated[float, meta('RF center frequency', 'Hz', gt=0)] = 3710e6
    channel: Annotated[int, meta('Input port index', ge=0)] = 0
    gain: Annotated[float, meta('Gain setting', 'dB')] = -10

    # acquisition
    start_time: Optional[
        Annotated[channel_analysis.TimestampType, meta('Acquisition start time')]
    ] = None
    duration: Annotated[float, meta('Acquisition duration', 's', gt=0)] = 0.1
    sample_rate: Annotated[float, meta('Sample rate', 'S/s', gt=0)] = 15.36e6

    # filtering and resampling
    analysis_bandwidth: Optional[
        Annotated[float, meta('Waveform filter bandwidth', 'Hz', gt=0)]
    ] = 10e6
    lo_shift: Annotated[_TShift, meta('LO shift direction')] = 'none'
    host_resample: bool = True


class RadioSetup(msgspec.Struct, forbid_unknown_fields=True):
    """run-time characteristics of the radio that are left invariant during a sweep"""

    driver: str = 'Air7201B'
    resource: Any = None
    gps: bool = False
    time_source: Literal['internal', 'external'] = 'internal'
    periodic_trigger: Optional[float] = None
    calibration: Optional[str] = None


class Description(msgspec.Struct, forbid_unknown_fields=True):
    summary: Optional[str] = None
    location: Optional[tuple[float, float, float]] = None
    signal_chain: tuple[str, ...] = ()


class Output(msgspec.Struct, forbid_unknown_fields=True):
    path: Optional[str] = None
    store: typing.Union[Literal['zip'],Literal['directory']] = 'zip'
    coord_aliases: dict[str, dict[str, dict[str, Any]]] = {}


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


def reset_nonsampling_fields(capture: RadioCapture) -> RadioCapture:
    """return a struct containing only the sampling-related fields of the RadioCapture"""

    SAMPLING_FIELDS = (
        'duration',
        'sample_rate',
        'analysis_bandwidth',
        'lo_shift',
        'host_resample',
    )

    mapping = struct_to_builtins(capture)
    mapping = {k: mapping[k] for k in SAMPLING_FIELDS}

    return msgspec.convert(mapping, RadioCapture)
