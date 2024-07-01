"""mgspec structs for use as configuration schemas"""


from __future__ import annotations
from msgspec import Struct, field
from typing import Optional
from channel_analysis import waveform


def _analysis_factory():
    return waveform._registry.tospec()()


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
    lo_gain: Optional[float] = 0 # dB (ignored for no IF frontend)


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
    channel_analysis: waveform._ConfigStruct = field(default_factory=_analysis_factory)
