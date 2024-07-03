"""mgspec structs for radio and test configuration"""

from __future__ import annotations
import msgspec
from typing import Optional, Literal, Any
from channel_analysis import structs


def make_default_analysis():
    return structs.analysis_registry.tostruct()()


class RadioCapture(structs.Capture):
    """configuration for a single waveform capture"""

    # defaults have been added here

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


class Radio(msgspec.Struct):
    """characteristics of the radio"""
    driver: str = 'AirT7201'
    resource: Any = None
    gps: bool = False
    location: Optional[tuple[float, float, float]] = None
    timebase: Literal['internal','gps'] = 'internal'
    cyclic_trigger: Optional[float] = None
    calibration_path: Optional[str] = None


class Sweep(msgspec.Struct):
    captures: list[RadioCapture]
    radio: Radio = msgspec.field(default_factory=Radio)
    defaults: RadioCapture = msgspec.field(default_factory=RadioCapture)
    channel_analysis: Any = msgspec.field(default_factory=make_default_analysis)


def read_yaml_sweep(path) -> tuple[Sweep, tuple[str, ...]]:
    """build a Sweep object from the contents of specified yaml file"""

    with open(path, 'rb') as fd:
        text = fd.read()

    # validate first
    msgspec.yaml.decode(text, type=Sweep, strict=False)

    # build a dict to extract the list of sweep fields and apply defaults
    tree = msgspec.yaml.decode(text, type=dict, strict=False)
    sweep_fields = sorted(set.union(*[set(c) for c in tree['sweep']]))

    # apply default capture settings
    defaults = tree['defaults']
    tree['sweep'] = [dict(defaults, **c) for c in tree['sweep']]

    run = msgspec.convert(tree, type=Sweep, strict=False)
    return run, sweep_fields
