"""mgspec structs for radio and test configuration"""

from __future__ import annotations
import msgspec
from typing import Optional, Literal, Any
import channel_analysis
from pathlib import Path


def make_default_analysis():
    return channel_analysis.waveforms.analysis_registry.tostruct()()


class RadioCapture(channel_analysis.Capture):
    """configuration for a single waveform capture"""

    # RF and leveling
    center_frequency: float = 3710e6
    channel: int = 0
    gain: float = -10

    # acquisition
    duration: float = 0.1
    sample_rate: float = 15.36e6

    # filtering and resampling
    analysis_bandwidth: Optional[float] = 10e6 # None for no bandpass filter
    lo_shift: Literal['left', 'right', 'none'] = 'left'

    # external frequency conversion support
    preselect_if_frequency: Optional[float] = None  # Hz (or none, for no ext frontend)
    preselect_lo_gain: Optional[float] = 0  # dB (ignored when if_frequency is None)
    preselect_rf_gain: Optional[float] = 0  # dB (ignored when if_frequency is None)


class Radio(msgspec.Struct):
    """run-time characteristics of the radio that are invariant during a test"""

    driver: str = 'AirT7201'
    resource: Any = None
    gps: bool = False
    location: Optional[tuple[float, float, float]] = None
    timebase: Literal['internal', 'gpsdo'] = 'internal'
    cyclic_trigger: Optional[float] = None
    calibration_path: Optional[str] = None


class Sweep(msgspec.Struct):
    captures: list[RadioCapture]
    radio: Radio = msgspec.field(default_factory=Radio)
    defaults: RadioCapture = msgspec.field(default_factory=RadioCapture)
    channel_analysis: Any = msgspec.field(default_factory=make_default_analysis)


def to_calibration_capture(c: RadioCapture, duration=0.1) -> RadioCapture:
    """return a capture configured as a calibration with the specified duration"""

    d = msgspec.to_builtins(c)
    d['duration'] = duration
    return msgspec.convert(d, type=RadioCapture)


def read_yaml_sweep(path: str | Path) -> tuple[Sweep, tuple[str, ...]]:
    """build a Sweep struct from the contents of specified yaml file"""

    with open(path, 'rb') as fd:
        text = fd.read()

    # validate first
    msgspec.yaml.decode(text, type=Sweep, strict=False)

    # build a dict to extract the list of sweep fields and apply defaults
    tree = msgspec.yaml.decode(text, type=dict, strict=False)
    sweep_fields = sorted(set.union(*[set(c) for c in tree['captures']]))

    # apply default capture settings
    defaults = tree['defaults']
    tree['sweep'] = [dict(defaults, **c) for c in tree['captures']]

    run = msgspec.convert(tree, type=Sweep, strict=False)
    return run, sweep_fields
