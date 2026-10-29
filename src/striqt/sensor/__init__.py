# lib must load before specs: specs.structs imports lib.typing, and lib.compute
# reads specs.SensorCapture at import time
from . import lib

# isort: split
from . import bindings, calibration, io, peripherals, sinks, specs, util
from .__about__ import __version__
from .io import (
    open_store,
    read_calibration,
    read_json_spec,
    read_yaml_spec,
    read_zarr_spec,
)
from .lib import sources, typing
from .lib.actions import iterate_sweep
from .lib.resources import open_resources
