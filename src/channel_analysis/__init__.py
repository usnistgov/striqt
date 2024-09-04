"""evaluation of IQ data into analysis products packaged into xarray.DataArray and xarray.Dataset"""

from . import dataarrays, source, structs, type_stubs

from .io import load, dump, open_store
from .structs import Capture, FilteredCapture
from .factories import (
    channel_power_distribution,
    channel_power_time_series,
    cyclic_channel_power,
    iq_waveform,
    persistence_spectrum,
)

from .dataarrays import (
    analyze_by_spec,
    iir_filter,
    ola_filter
)