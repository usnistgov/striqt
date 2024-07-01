"""mgspec structs for configuration"""


from __future__ import annotations
from msgspec import Struct, field
from typing import Literal, Optional
from channel_analysis.waveform import _registry as analysis_registry
from channel_analysis.waveform import _ConfigStruct as ChannelAnalysis

from pathlib import Path

def analysis_factory():
    return analysis_registry.tospec()()


class State(Struct):
    # RF and leveling
    center_frequency: float = 3710e6
    channel: int = 0
    gain: float = -10
    calibrated: bool = True

    # acquisition
    duration: float = 0.1
    sample_rate: float = 15.36e6

    # filtering and resampling
    analysis_bandwidth: float = 10e6
    lo_shift: Optional['left'|'right'] = 'left' # shift the LO outside the acquisition band
    window: 'hamming'|'blackman'|'blackmanharris' = 'hamming' # the COLA spectral window to use

    # external frequency conversion support
    if_frequency: Optional[float] = None # Hz (or none, for no IF frontend)
    lo_gain: Optional[float] = None# dB


class Acquisition(Struct):
    location: Optional[tuple[str,str,str]] = None
    timebase: 'builtin'|'gps' = 'builtin'
    cyclic_trigger: bool|float = False
    calibration_path: Optional[str] = None
    timeout: float = 5
    defaults: State = field(default_factory=State)


class Runner(Struct):
    acquisition: Acquisition = field(default_factory=Acquisition)
    sweep: list[State] = field(default_factory=lambda: [State()])
    channel_analysis: ChannelAnalysis = field(default_factory=analysis_factory)
