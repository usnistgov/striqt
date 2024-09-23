"""evaluation of IQ data into analysis products packaged into xarray.DataArray and xarray.Dataset"""

from . import io

from . import dataarrays, source, structs, type_stubs, util

from .io import load, dump, open_store

from .xarray_wrappers import *

from .dataarrays import (
    analyze_by_spec,
)

from .source import simulated_awgn, filter_iq_capture

from .structs import Capture, FilteredCapture
from .type_stubs import ArrayType, DatasetType