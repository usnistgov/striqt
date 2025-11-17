"""
A registry of data class <-> IQ acquisition control class bindings.
These are taken to define sensors. Extension classes can implement
peripherals and expanded data fields elsewhere.
"""

from .lib import sources, specs
from .lib.sources import deepwave
from .lib.bindings import SensorBinding, bind_sensor

# %% Synthetic data sources for testing, warmup, and post-analysis
warmup = bind_sensor(
    'warmup',
    SensorBinding(
        source_spec=specs.NullSourceSpec,
        capture_spec=specs.WaveformCaptureSpec,
        source=sources.WarmupSource,
    ),
)

# %% File sources
file = bind_sensor(
    'file',
    SensorBinding(
        source_spec=sources.FileSourceSpec,
        capture_spec=specs.FileCaptureSpec,
        source=sources.FileSource,
    ),
)

tdms_file = bind_sensor(
    'tdms_file',
    SensorBinding(
        source_spec=sources.TDMSSourceSpec,
        capture_spec=specs.FileCaptureSpec,
        source=sources.TDMSFileSource,
    ),
)

zarr_iq = bind_sensor(
    'zarr_iq',
    SensorBinding(
        source_spec=sources.ZarrFileSourceSpec,
        capture_spec=specs.FileCaptureSpec,
        source=sources.ZarrIQSource,
    ),
)

# %% Function generator sources
noise = bind_sensor(
    'noise',
    SensorBinding(
        source_spec=sources.FunctionSourceSpec,
        capture_spec=sources.NoiseCaptureSpec,
        source=sources.NoiseSource,
    ),
)

dirac_delta = bind_sensor(
    'dirac_delta',
    SensorBinding(
        source_spec=sources.FunctionSourceSpec,
        capture_spec=sources.DiracDeltaCaptureSpec,
        source=sources.DiracDeltaSource,
    ),
)

single_tone = bind_sensor(
    'single_tone',
    SensorBinding(
        source_spec=sources.FunctionSourceSpec,
        capture_spec=sources.SingleToneCaptureSpec,
        source=sources.SingleToneSource,
    ),
)

sawtooth = bind_sensor(
    'sawtooth',
    SensorBinding(
        source_spec=sources.FunctionSourceSpec,
        capture_spec=sources.SawtoothCaptureSpec,
        source=sources.SawtoothSource,
    ),
)


# %% Hardware data sources
air7101b = bind_sensor(
    'air7101b',
    SensorBinding(
        source_spec=deepwave.Air7101BSourceSpec,
        capture_spec=specs.SoapyCaptureSpec,
        source=deepwave.Airstack1Source,
    ),
)

air7201b = bind_sensor(
    'air7201b',
    SensorBinding(
        source_spec=deepwave.Air7201BSourceSpec,
        capture_spec=specs.SoapyCaptureSpec,
        source=deepwave.Airstack1Source,
    ),
)

air8201b = bind_sensor(
    'air8201b',
    SensorBinding(
        source_spec=deepwave.Air8201BSourceSpec,
        capture_spec=specs.SoapyCaptureSpec,
        source=deepwave.Airstack1Source,
    ),
)


del sources, specs, deepwave
