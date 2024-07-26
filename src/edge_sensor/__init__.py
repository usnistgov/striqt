"""this module deals with the integration of sensor operation on sensor hardware"""

# work around a dynamic library loading packaging quirk on jetson aarch64
import iqwaveform

del iqwaveform

from . import radio, structs

from channel_analysis import load, dump
from .structs import read_yaml_sweep