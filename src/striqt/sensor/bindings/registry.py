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
Warmup = bind_sensor(
    'Warmup',
    SensorBinding(
        source_spec=_specs.NullSourceSpec,
        capture_spec=_specs.CaptureSpec,
        source=_sources.WarmupSource,
    ),
)

#%% File sources
File = bind_sensor(
    'File',
    SensorBinding(
        source_spec=_sources.FileSourceSpec,
        capture_spec=_specs.FileCaptureSpec,
        source=_sources.FileSource,
    ),
)

TDMSFile = bind_sensor(
    'TDMSFile',
    SensorBinding(
        source_spec=_sources.TDMSSourceSpec,
        capture_spec=_specs.FileCaptureSpec,
        source=_sources.TDMSFileSource,
    ),
)

ZarrIQFile = bind_sensor(
    'ZarrIQFile',
    SensorBinding(
        source_spec=_sources.ZarrFileSourceSpec,
        capture_spec=_specs.FileCaptureSpec,
        source=_sources.ZarrIQSource,
    ),
)

# %% Function generator sources
Noise = bind_sensor(
    'Noise',
    SensorBinding(
        source_spec=_sources.FunctionSourceSpec,
        capture_spec=_sources.NoiseCaptureSpec,
        source=_sources.NoiseSource,
    ),
)

DiractDeltaFunction = bind_sensor(
    'DiracDeltaFunction',
    SensorBinding(
        source_spec=_sources.FunctionSourceSpec,
        capture_spec=_sources.DiracDeltaCaptureSpec,
        source=_sources.DiracDeltaSource,
    ),
)

SingleToneFunction = bind_sensor(
    'SingleToneFunction',
    SensorBinding(
        source_spec=_sources.FunctionSourceSpec,
        capture_spec=_sources.SingleToneCaptureSpec,
        source=_sources.SingleToneSource,
    ),
)

SawtoothFunction = bind_sensor(
    'SawtoothFunction',
    SensorBinding(
        source_spec=_sources.FunctionSourceSpec,
        capture_spec=_sources.SawtoothCaptureSpec,
        source=_sources.SawtoothSource,
    ),
)


# %% Hardware data sources
Air7101B = bind_sensor(
    'Air7101B',
    SensorBinding(
        source_spec=deepwave.Air7101BSourceSpec,
        capture_spec=_specs.SoapyCaptureSpec,
        source=deepwave.Airstack1Source,
    ),
)

Air7201B = bind_sensor(
    'Air7201B',
    SensorBinding(
        source_spec=deepwave.Air7201BSourceSpec,
        capture_spec=_specs.SoapyCaptureSpec,
        source=deepwave.Airstack1Source,
    ),
)

Air8201b = bind_sensor(
    'Air8201B',
    SensorBinding(
        source_spec=deepwave.Air8201BSourceSpec,
        capture_spec=_specs.SoapyCaptureSpec,
        source=deepwave.Airstack1Source,
    ),
)
