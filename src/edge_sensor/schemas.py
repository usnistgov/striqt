"""mgspec structs for configuration"""


from __future__ import annotations
import msgspec
from typing import Literal, Optional
from channel_analysis import waveform
from pathlib import Path


class State(msgspec.Struct):
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
    if_frequency: Optional[float] = None # Hz (or none, for no IF frontend)
    lo_shift: Optional['left'|'right'] = 'left' # shift the LO outside the acquisition band
    window: 'hamming'|'blackman'|'blackmanharris' = 'hamming' # the COLA spectral window to use


class Acquisition(msgspec.Struct):
    location: Optional[tuple[str,str,str]] = None
    timebase: Literal['builtin']|Literal['gps'] = 'builtin'
    cyclic_trigger: bool|float = False
    calibration_path: Optional[str] = None
    defaults: State = msgspec.field(default_factory=State)


class Runner(msgspec.Struct, omit_defaults=True):
    acquisition: Acquisition = msgspec.field(default_factory=Acquisition)
    sweep: list[State] = msgspec.field(default_factory=lambda: [State()])
    channel_analysis: waveform._ConfigStruct = \
        msgspec.field(default_factory=lambda: waveform._registry.tospec()())
