"""Factory functions that produce channel analysis products packaged as xarray.DataArray objects"""

from ._cellular_cyclic_autocorrelation import cellular_cyclic_autocorrelation
from ._channel_power_ccdf import channel_power_ccdf
from ._channel_power_time_series import channel_power_time_series
from ._cyclic_channel_power import cyclic_channel_power
from ._iq_waveform import iq_waveform
from ._persistence_spectrum import persistence_spectrum
from ._spectrogram_ccdf import spectrogram_ccdf
