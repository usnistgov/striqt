"""data structures for configuration of hardware and experiments"""

from __future__ import annotations
import msgspec
from typing import Optional, Literal, Any
from typing import Annotated as A
from pathlib import Path
from msgspec import to_builtins, convert

import channel_analysis
import channel_analysis.waveform
from channel_analysis.structs import meta, get_attrs, ChannelAnalysis


def make_default_analysis():
    return channel_analysis.waveform.registry.spec_type()()


def describe_capture(capture: RadioCapture, swept_fields):
    return ', '.join([f'{k}={getattr(capture, k)}' for k in swept_fields])


_TShift = Literal['left', 'right', 'none']


class RadioCapture(channel_analysis.Capture):
    """configuration for a single waveform capture"""

    # RF and leveling
    center_frequency: A[float, meta('RF center frequency', 'Hz')] = 3710e6
    channel: A[int, meta('RX hardware input port')] = 0
    gain: A[float, meta('internal gain setting inside the radio', 'dB')] = -10

    # acquisition
    duration: A[float, meta('duration of the capture', 's')] = 0.1
    sample_rate: A[float, meta('IQ sample rate', 'S/s')] = 15.36e6

    # filtering and resampling
    analysis_bandwidth: A[Optional[float], meta('DSP filter bandwidth', 'Hz')] = 10e6
    lo_shift: A[_TShift, meta('direction of the LO shift')] = 'none'
    gpu_resample: bool = True
    lo_filter: bool = False


class RadioSetup(msgspec.Struct):
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


class Description(msgspec.Struct):
    summary: Optional[str] = None
    location: Optional[tuple[float, float, float]] = None
    signal_chain: tuple[str, ...] = ()


class Sweep(msgspec.Struct):
    captures: tuple[RadioCapture, ...]
    radio_setup: RadioSetup = msgspec.field(default_factory=RadioSetup)
    defaults: RadioCapture = msgspec.field(default_factory=RadioCapture)
    channel_analysis: dict = msgspec.field(
        default_factory=make_default_analysis
    )
    description: Description = msgspec.field(default_factory=Description)
