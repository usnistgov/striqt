"""mgspec structs for use as configuration schemas"""

from __future__ import annotations
from msgspec import Struct, field
from typing import Optional, Literal
from channel_analysis import waveform


def make_default_analysis():
    return waveform.analysis_registry.tostruct()()


class Capture(Struct):
    """configuration for a single waveform capture"""

    # RF and leveling
    center_frequency: float = 3710e6
    channel: int = 0
    gain: float = -10

    # acquisition
    duration: float = 0.1
    sample_rate: float = 15.36e6

    # filtering and resampling
    analysis_bandwidth: float = 10e6
    lo_shift: Literal['left','right',None] = 'left'  # shift the LO outside the acquisition band

    # external frequency conversion support
    if_frequency: Optional[float] = None  # Hz (or none, for no ext frontend)
    lo_gain: Optional[float] = 0  # dB (ignored when if_frequency is None)
    rf_gain: Optional[float] = 0 # dB (ignored when if_frequency is None)


class System(Struct):
    location: Optional[tuple[str, str, str]] = None
    timebase: Literal['builtin','gps'] = 'builtin'
    cyclic_trigger: bool | float = False
    calibration_path: Optional[str] = None
    defaults: Capture = field(default_factory=Capture)


class Run(Struct):
    acquisition: System = field(default_factory=System)
    sweep: list[Capture] = field(default_factory=lambda: [Capture()])
    channel_analysis: waveform._ConfigStruct = field(
        default_factory=make_default_analysis
    )
