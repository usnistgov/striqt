"""integrate channel_analysis into captures on sensor hardware"""

# start with this to work around a dynamic library loading
# packaging quirk on jetson aarch64
from .api import calibration, io, iq_corrections, structs, util

from .api.controller import connect, start_server, SweepController
from .api.io import DataStoreManager, dump, load, open_store, read_yaml_sweep
from .api.calibration import read_calibration_corrections
from .api.sources import (
    RadioSource,
    NullSource,
    design_capture_filter,
)
from .api.structs import RadioCapture, RadioSetup, Sweep, Description
from .api.sweeps import iter_sweep
from .api.xarray_ops import CAPTURE_DIM, concat_time_dim, analyze_capture

# support legacy namespacing until the _api -> api change propagates through dependent libraries
from . import api
