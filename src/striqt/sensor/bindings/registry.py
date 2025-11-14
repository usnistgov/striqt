"""
A registry of data class <-> IQ acquisition control class bindings.
These are taken to define sensors. Extension classes can implement
peripherals and expanded data fields elsewhere.
"""

from ..lib import sources as _sources
from ..lib import specs as _specs
from . import deepwave
from ._util import SensorBinding, bind_sensor

#%% Synthetic data sources for testing, warmup, and post-analysis
warmup = bind_sensor(
    'warmup',
    SensorBinding(
        source_spec=_specs.NullSourceSpec,
        capture_spec=_specs.CaptureSpec,
        source=_sources.WarmupSource,
    ),
)

#%% File sources
file = bind_sensor(
    'file',
    SensorBinding(
        source_spec=_sources.FileSourceSpec,
        capture_spec=_specs.FileCaptureSpec,
        source=_sources.FileSource,
    ),
)

tdms_file = bind_sensor(
    'tdms_file',
    SensorBinding(
        source_spec=_sources.TDMSSourceSpec,
        capture_spec=_specs.FileCaptureSpec,
        source=_sources.TDMSFileSource,
    ),
)

zarr_iq = bind_sensor(
    'zarr_iq',
    SensorBinding(
        source_spec=_sources.ZarrFileSourceSpec,
        capture_spec=_specs.FileCaptureSpec,
        source=_sources.ZarrIQSource,
    ),
)

# %% Function generator sources
noise = bind_sensor(
    'noise',
    SensorBinding(
        source_spec=_sources.FunctionSourceSpec,
        capture_spec=_sources.NoiseCaptureSpec,
        source=_sources.NoiseSource,
    ),
)

dirac_delta = bind_sensor(
    'dirac_delta',
    SensorBinding(
        source_spec=_sources.FunctionSourceSpec,
        capture_spec=_sources.DiracDeltaCaptureSpec,
        source=_sources.DiracDeltaSource,
    ),
)

single_tone = bind_sensor(
    'single_tone',
    SensorBinding(
        source_spec=_sources.FunctionSourceSpec,
        capture_spec=_sources.SingleToneCaptureSpec,
        source=_sources.SingleToneSource,
    ),
)

sawtooth = bind_sensor(
    'sawtooth',
    SensorBinding(
        source_spec=_sources.FunctionSourceSpec,
        capture_spec=_sources.SawtoothCaptureSpec,
        source=_sources.SawtoothSource,
    ),
)


# %% Hardware data sources
air7101b = bind_sensor(
    'air7101b',
    SensorBinding(
        source_spec=deepwave.Air7101BSourceSpec,
        capture_spec=_specs.SoapyCaptureSpec,
        source=deepwave.Airstack1Source,
    ),
)

air7201b = bind_sensor(
    'air7201b',
    SensorBinding(
        source_spec=deepwave.Air7201BSourceSpec,
        capture_spec=_specs.SoapyCaptureSpec,
        source=deepwave.Airstack1Source,
    ),
)

air8201b = bind_sensor(
    'air8201b',
    SensorBinding(
        source_spec=deepwave.Air8201BSourceSpec,
        capture_spec=_specs.SoapyCaptureSpec,
        source=deepwave.Airstack1Source,
    ),
)