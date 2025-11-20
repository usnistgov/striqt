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
        source=sources.WarmupSource,
        peripherals=peripherals.NoPeripherals,
        sink=sinks.ZarrCaptureSink,
    ),
    Schema(
        source=specs.NullSource,
        capture=specs.ResampledCapture,
        peripherals=specs.NoPeripherals,
    ),
)


# file = bind_sensor(
#     'file',
#     SensorBinding(
#         source_spec=sources.FileSourceSpec,
#         capture_spec=specs.FileCapture,
#         peripherals_spec=specs.NoPeripherals,
#         peripherals=peripherals.NoPeripherals,
#         source=sources.FileSource,
#         sweep_spec=specs.Sweep
#     ),
# )

# tdms_file = bind_sensor(
#     'tdms_file',
#     SensorBinding(
#         source_spec=sources.TDMSSourceSpec,
#         capture_spec=specs.FileCapture,
#         peripherals_spec=specs.NoPeripherals,
#         peripherals=peripherals.NoPeripherals,
#         source=sources.TDMSFileSource,
#         sweep_spec=specs.Sweep
#     ),
# )

# zarr_iq = bind_sensor(
#     'zarr_iq',
#     SensorBinding(
#         source_spec=sources.ZarrFileSourceSpec,
#         capture_spec=specs.FileCapture,
#         peripherals_spec=specs.NoPeripherals,
#         peripherals=peripherals.NoPeripherals,
#         source=sources.ZarrIQSource,
#         sweep_spec=specs.Sweep
#     ),
# )

# # %% Function generator sources
# noise = bind_sensor(
#     'noise',
#     SensorBinding(
#         source_spec=sources.FunctionSourceSpec,
#         capture_spec=sources.NoiseCaptureSpec,
#         peripherals_spec=specs.NoPeripherals,
#         peripherals=peripherals.NoPeripherals,
#         source=sources.NoiseSource,
#         sweep_spec=specs.Sweep
#     ),
# )

# dirac_delta = bind_sensor(
#     'dirac_delta',
#     SensorBinding(
#         source_spec=sources.FunctionSourceSpec,
#         capture_spec=sources.DiracDeltaCaptureSpec,
#         peripherals_spec=specs.NoPeripherals,
#         peripherals=peripherals.NoPeripherals,
#         source=sources.DiracDeltaSource,
#         sweep_spec=specs.Sweep
#     ),
# )

# single_tone = bind_sensor(
#     'single_tone',
#     SensorBinding(
#         source_spec=sources.FunctionSourceSpec,
#         capture_spec=sources.SingleToneCaptureSpec,
#         peripherals_spec=specs.NoPeripherals,
#         peripherals=peripherals.NoPeripherals,
#         source=sources.SingleToneSource,
#         sweep_spec=specs.Sweep
#     ),
# )

# sawtooth = bind_sensor(
#     'sawtooth',
#     SensorBinding(
#         source_spec=sources.FunctionSourceSpec,
#         capture_spec=sources.SawtoothCaptureSpec,
#         peripherals_spec=specs.NoPeripherals,
#         peripherals=peripherals.NoPeripherals,
#         source=sources.SawtoothSource,
#         sweep_spec=specs.Sweep
#     ),
# )


# # %% Hardware data sources
# air7101b = bind_sensor(
#     'air7101b',
#     SensorBinding(
#         source_spec=deepwave.Air7101BSourceSpec,
#         capture_spec=specs.SoapyCapture,
#         peripherals_spec=specs.NoPeripherals,
#         peripherals=peripherals.NoPeripherals,
#         source=deepwave.Airstack1Source,
#         sweep_spec=specs.Sweep
#     ),
# )

# air7101b_calibration = bind_manual_yfactor_calibration('air7101b_calibration', air7101b)

air7201b = bind_sensor(
    'air7201b',
    Sensor(
        peripherals=peripherals.NoPeripherals,
        source=deepwave.Airstack1Source,
        sweep_spec=specs.Sweep,
    ),
    Schema(
        source=deepwave.Air7201BSourceSpec,
        capture=specs.SoapyCapture,
        peripherals=specs.NoPeripherals,
    ),
)

# air7201b_calibration = bind_manual_yfactor_calibration('air7201b_calibration', air7201b)

# air8201b = bind_sensor(
#     'air8201b',
#     SensorBinding(
#         source_spec=deepwave.Air8201BSourceSpec,
#         capture_spec=specs.SoapyCapture,
#         peripherals_spec=specs.NoPeripherals,
#         peripherals=peripherals.NoPeripherals,
#         source=deepwave.Airstack1Source,
#         sweep_spec=specs.Sweep
#     ),
# )

# air8201b_calibration = bind_manual_yfactor_calibration('air8201b_calibration', air8201b)


# del peripherals, sources, specs
