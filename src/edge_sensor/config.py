"""mgspec structs for use as configuration schemas"""

from __future__ import annotations
import msgspec
from typing import Optional, Literal, Union, Any
from channel_analysis import config as _waveform_config


def make_default_analysis():
    return _waveform_config.analysis_registry.tostruct()()


class Capture(msgspec.Struct, omit_defaults=True):
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


class System(msgspec.Struct):
    location: Optional[tuple[float, float, float]] = None
    timebase: Literal['builtin','gps'] = 'builtin'
    cyclic_trigger: Optional[float] = None
    calibration_path: Optional[str] = None


class Run(msgspec.Struct):
    sweep: list[Capture]
    system: System = msgspec.field(default_factory=System)
    defaults: Capture = msgspec.field(default_factory=Capture)
    channel_analysis: Any = msgspec.field(default_factory=make_default_analysis)


def read_yaml_runner(path) -> (Run, tuple):
    with open(path, 'rb') as fd:
        text = fd.read()

    # validate first
    msgspec.yaml.decode(text, type=Run, strict=False)

    # build a dict to extract the list of sweep fields and apply defaults
    tree = msgspec.yaml.decode(text, type=dict, strict=False)
    sweep_fields = sorted(set.union(*[set(c) for c in tree['sweep']]))

    # apply default capture settings
    defaults = tree['defaults']
    tree['sweep'] = [dict(defaults, **c) for c in tree['sweep']]

    run = msgspec.convert(tree, type=Run, strict=False)
    return run, sweep_fields
