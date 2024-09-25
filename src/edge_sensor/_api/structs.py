"""data structures for configuration of radio hardware, captures, and sweeps"""

from __future__ import annotations
from frozendict import frozendict
import functools
import msgspec
from typing import Annotated, Optional, Literal, Any

import channel_analysis
import channel_analysis._api.filters
from channel_analysis._api.structs import (
    meta,
    ChannelAnalysis,
    struct_to_builtins,
    builtins_to_struct,
    copy_struct,
)


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
    start_time: Optional[Annotated[channel_analysis.TimestampType, meta('Acquisition start time')]] = None
    duration: Annotated[float, meta('Acquisition duration', 's', gt=0)] = 0.1
    sample_rate: Annotated[float, meta('Sample rate', 'S/s', gt=0)] = 15.36e6

    # filtering and resampling
    analysis_bandwidth: Optional[
        Annotated[float, meta('Waveform filter bandwidth', 'Hz', gt=0)]
    ] = 10e6
    lo_shift: Annotated[_TShift, meta('LO shift direction')] = 'none'
    gpu_resample: bool = True

    # hooks for external devices (switches, noise diodes, etc)
    external: Annotated[frozendict[str, Any], meta('External device states')] = (
        frozendict()
    )


class RadioSetup(msgspec.Struct, forbid_unknown_fields=True):
    """run-time characteristics of the radio that are left invariant during a sweep"""

    driver: str = 'Air7201B'
    resource: Any = None
    gps: bool = False
    timebase: Literal['internal', 'gpsdo'] = 'internal'
    periodic_trigger: Optional[float] = None
    calibration: Optional[str] = None

    # external frequency conversion disabled when if_frequency is None
    preselect_if_frequency: Optional[
        Annotated[float, meta('preselector IF filter center frequency', gt=0)]
    ] = None  # Hz (or none, for no ext frontend)
    preselect_lo_gain: Optional[
        Annotated[float, meta('preselector LO path gain setting', 'dB')]
    ] = 0  # dB (ignored when if_frequency is None)
    preselect_rf_gain: Optional[
        Annotated[float, meta('preselector RF path gain setting', 'dB')]
    ] = 0


class Description(msgspec.Struct, forbid_unknown_fields=True):
    summary: Optional[str] = None
    location: Optional[tuple[float, float, float]] = None
    signal_chain: tuple[str, ...] = ()


class Sweep(msgspec.Struct, forbid_unknown_fields=True):
    captures: tuple[RadioCapture, ...]
    radio_setup: RadioSetup = msgspec.field(default_factory=RadioSetup)
    defaults: RadioCapture = msgspec.field(default_factory=RadioCapture)
    channel_analysis: dict = msgspec.field(default_factory=_make_default_analysis)
    description: Description = msgspec.field(default_factory=Description)


@functools.lru_cache
def get_shared_capture_fields(captures: tuple[RadioCapture, ...]):
    base = set(RadioCapture.__struct_fields__) - {'external'}
    external_keys = (set(c.external.keys()) for c in captures)
    external = set.intersection(*external_keys)

    return tuple(sorted(base | external))
