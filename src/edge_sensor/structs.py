"""mgspec structs for radio and test configuration"""

from __future__ import annotations
import msgspec
from typing import Optional, Literal, Any
import channel_analysis
from channel_analysis.structs import meta, get_attrs
from pathlib import Path
from msgspec import Meta
import typing
from typing import Annotated as A


def make_default_analysis():
    return channel_analysis.waveforms.analysis_registry.tostruct()()


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
    lo_shift: A[_TShift, meta('direction of the LO shift')] = 'left'

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
    tree['captures'] = [dict(defaults, **c) for c in tree['captures']]

    run = msgspec.convert(tree, type=Sweep, strict=False)
    return run, sweep_fields
