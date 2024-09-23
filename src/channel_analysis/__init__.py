"""evaluation of IQ data into analysis products packaged into xarray.DataArray and xarray.Dataset"""

from . import io

from . import dataarrays, source, structs, type_stubs

from .io import load, dump, open_store

from .xarray_wrappers import (
    cellular_cyclic_autocorrelation,
    channel_power_ccdf,
    channel_power_time_series,
    cyclic_channel_power,
    iq_waveform,
    persistence_spectrum,
    spectrogram_ccdf,
)

from .dataarrays import (
    analyze_by_spec,
)

from .source import simulated_awgn, filter_iq_capture

from .structs import Capture, FilteredCapture
from .type_stubs import ArrayType, DatasetType