"""data structures for configuration of hardware and experiments"""

from __future__ import annotations
from frozendict import frozendict
import functools
import msgspec
from typing import Optional, Literal, Any
from typing import Annotated as A
from pathlib import Path
from msgspec import convert

import channel_analysis
import channel_analysis.dataarrays
from channel_analysis.structs import meta, get_attrs, ChannelAnalysis, to_builtins


_TShift = Literal['left', 'right', 'none']


def _make_default_analysis():
    return channel_analysis.dataarrays.as_registered_channel_analysis.spec_type()()


class RadioCapture(channel_analysis.Capture, forbid_unknown_fields=True):
    """configuration for a single waveform capture"""

    # RF and leveling
    center_frequency: A[float, meta('RF center frequency', 'Hz')] = 3710e6
    channel: A[int, meta('Input port index')] = 0
    gain: A[float, meta('Gain setting', 'dB')] = -10

    # acquisition
    duration: A[float, meta('duration of the capture', 's')] = 0.1
    sample_rate: A[float, meta('Sample rate', 'S/s')] = 15.36e6

    # filtering and resampling
    analysis_bandwidth: A[Optional[float], meta('Waveform filter bandwidth', 'Hz')] = (
        10e6
    )
    lo_shift: A[_TShift, meta('LO shift direction')] = 'none'
    gpu_resample: bool = True

    # hooks for external devices (switches, noise diodes, etc)
    external: A[frozendict[str, Any], meta('External device states')] = frozendict()


class RadioSetup(msgspec.Struct, forbid_unknown_fields=True):
    """run-time characteristics of the radio that are invariant during a test"""

    driver: str = 'Air7201B'
    resource: Any = None
    gps: bool = False
    timebase: Literal['internal', 'gpsdo'] = 'internal'
    periodic_trigger: Optional[float] = None
    calibration: Optional[str] = None

    # external frequency conversion disabled when if_frequency is None
    preselect_if_frequency: A[
        Optional[float], meta('preselector IF filter center frequency')
    ] = None  # Hz (or none, for no ext frontend)
    preselect_lo_gain: A[
        Optional[float], meta('preselector LO path gain setting', 'dB')
    ] = 0  # dB (ignored when if_frequency is None)
    preselect_rf_gain: A[
        Optional[float], meta('preselector RF path gain setting', 'dB')
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
