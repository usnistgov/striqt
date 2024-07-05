"""this module deals with the integration of sensor operation on sensor hardware"""

# work around a dynamic library loading packaging quirk on jetson aarch64
import iqwaveform

del iqwaveform

from . import host, radio

from channel_analysis import load, dump
