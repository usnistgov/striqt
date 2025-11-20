"""
A registry of data class <-> IQ acquisition control class bindings.
These are taken to define sensors. Extension classes can implement
peripherals and expanded data fields elsewhere.
"""

from .lib import peripherals, sinks, sources, specs
from .lib.sources import deepwave
from .lib.bindings import Sensor, Schema, bind_sensor
from .lib.calibration import bind_manual_yfactor_calibration

# %% Synthetic data sources for testing/warmup/post-analysis
warmup = bind_sensor(
    'warmup',
    Sensor(
        source=sources.NoSource,
        sink=sinks.NoSink,
    ),
    Schema(
        source=specs.NoSource,
        capture=specs.ResampledCapture,
        peripherals=specs.NoPeripherals,
    ),
)

file = bind_sensor(
    'file',
    Sensor(source=sources.FileSource),
    Schema(
        source=sources.FileSourceSpec,
        capture=specs.FileCapture,
        peripherals=specs.NoPeripherals,
    ),
)

tdms_file = bind_sensor(
    'tdms_file',
    Sensor(source=sources.TDMSFileSource),
    Schema(
        source=sources.TDMSFileSourceSpec,
        capture=specs.FileCapture,
        peripherals=specs.NoPeripherals,
    ),
)


zarr_iq = bind_sensor(
    'zarr_iq',
    Sensor(source=sources.ZarrIQSource),
    Schema(
        source=sources.ZarrIQSourceSpec,
        capture=specs.FileCapture,
        peripherals=specs.NoPeripherals,
    ),
)


# %% Function generator sources
noise = bind_sensor(
    'noise',
    Sensor(source=sources.NoiseSource),
    Schema(
        source=sources.FunctionSourceSpec,
        capture=sources.NoiseCaptureSpec,
        peripherals=specs.NoPeripherals,
    ),
)

dirac_delta = bind_sensor(
    'dirac_delta',
    Sensor(source=sources.DiracDeltaSource),
    Schema(
        source=sources.FunctionSourceSpec,
        capture=sources.DiracDeltaCaptureSpec,
        peripherals=specs.NoPeripherals,
    ),
)

single_tone = bind_sensor(
    'single_tone',
    Sensor(source=sources.SingleToneSource),
    Schema(
        source=sources.FunctionSourceSpec,
        capture=sources.SingleToneCaptureSpec,
        peripherals=specs.NoPeripherals,
    ),
)

sawtooth = bind_sensor(
    'sawtooth',
    Sensor(source=sources.SawtoothSource),
    Schema(
        source=sources.FunctionSourceSpec,
        capture=sources.SawtoothCaptureSpec,
        peripherals=specs.NoPeripherals,
    ),
)


# %% Hardware data sources
air7101b = bind_sensor(
    'air7101b',
    Sensor(source=deepwave.Airstack1Source),
    Schema(
        source=deepwave.Air7101BSourceSpec,
        capture=specs.SoapyCapture,
        peripherals=specs.NoPeripherals,
    ),
)

air7101b_calibration = bind_manual_yfactor_calibration('air7101b_calibration', air7101b)

air7201b = bind_sensor(
    'air7201b',
    Sensor(source=deepwave.Airstack1Source),
    Schema(
        source=deepwave.Air7201BSourceSpec,
        capture=specs.SoapyCapture,
        peripherals=specs.NoPeripherals,
    ),
)

air7201b_calibration = bind_manual_yfactor_calibration('air7201b_calibration', air7201b)

air8201b = bind_sensor(
    'air8201b',
    Sensor(source=deepwave.Airstack1Source),
    Schema(
        source=deepwave.Air8201BSourceSpec,
        capture=specs.SoapyCapture,
        peripherals=specs.NoPeripherals,
    ),
)

air8201b_calibration = bind_manual_yfactor_calibration('air8201b_calibration', air8201b)


del peripherals, sources, specs
