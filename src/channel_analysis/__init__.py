"""evaluation of IQ data into analysis products packaged into xarray.DataArray and xarray.Dataset"""

from . import dataarrays, source, structs, type_stubs

from .io import load, dump, open_store
from .structs import Capture, FilteredCapture
from .dataarrays import (
    amplitude_probability_distribution,
    analyze_by_spec,
    cyclic_channel_power,
    persistence_spectrum,
    power_time_series,
)
from .factories import channel_power_distribution
