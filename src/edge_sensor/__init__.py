"""integrate channel_analysis into captures on sensor hardware"""

# start with this to work around a dynamic library loading
# packaging quirk on jetson aarch64
from .api import calibration, io, peripherals, structs, util, writers

from .api.controller import connect, start_server, SweepController
from .api.io import dump, load, open_store, read_yaml_sweep
from .api.calibration import read_calibration_corrections
from .api.iq_corrections import resampling_correction
from .api.sources import (
    SourceBase,
    NullSource,
    design_capture_filter,
)
from .api.structs import RadioCapture, RadioSetup, Sweep, Description
from .api.sweeps import iter_sweep
from .api.xarray_ops import CAPTURE_DIM, concat_time_dim, analyze_capture

# support legacy namespacing until the _api -> api change propagates through dependent libraries
from . import api
