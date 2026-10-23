from .cellular import (
    cellular_5g_pss_correlation,
    cellular_5g_pss_sync,
    cellular_5g_sss_correlation,
    cellular_5g_ssb_spectrogram,
    cellular_cyclic_autocorrelation,
    cellular_resource_power_histogram,
)
from .power import (
    channel_power_histogram,
    channel_power_time_series,
    cyclic_channel_power,
)
from .waveforms import iq_waveform
from .spectrum import (
    power_spectral_density,
    spectrogram,
    spectrogram_histogram,
    spectrogram_ratio_histogram,
)
from .shared import registry
from .. import specs
from ..lib import register

# every measurement and trigger registered above must have its field on the bundles
register.check_group_fields(
    specs.BuiltinAnalyses, {info.name: spec_type for spec_type, info in registry.items()}
)
register.check_group_fields(
    specs.BuiltinTriggers,
    {
        info.name: info.meas_spec_type
        for key, info in registry.signal_trigger.items()
        if isinstance(key, str)
    },
)

__all__ = [
    'cellular_5g_pss_correlation',
    'cellular_5g_pss_sync',
    'cellular_5g_ssb_spectrogram',
    'cellular_5g_sss_correlation',
    'cellular_cyclic_autocorrelation',
    'cellular_resource_power_histogram',
    'channel_power_histogram',
    'channel_power_time_series',
    'cyclic_channel_power',
    'iq_waveform',
    'power_spectral_density',
    'registry',
    'spectrogram',
    'spectrogram_histogram',
    'spectrogram_ratio_histogram',
]
