# striqt.sensor.specs

::: striqt.sensor.specs
    options:
      members: false

## Sweeps and captures

Spec structs for sweeps, actions, capture settings, and the loops that expand one capture into several.

::: striqt.sensor.specs
    options:
      extra:
        module_docstring: false
      members:
        - Sweep
        - SweepOptions
        - ActionSpec
        - Cellular5GNRSweep
        - ManualYFactorSpec
        - SensorCapture
        - SoapyCapture
        - FileCapture
        - NoiseCapture
        - SingleToneCapture
        - SawtoothCapture
        - DiracDeltaCapture
        - BurstOptions
        - CaptureRemap
        - CapturePlan
        - LoopBase
        - Repeat
        - Range
        - FrequencyBinRange
        - CellChannel
        - Cellular5GNRAnalysis
        - Description
        - Extension
        - Schema
        - SpecValidationError

## Sources, sinks, and peripherals

Spec structs for IQ sources, data sinks, peripherals, and the acquisition metadata recorded with each capture.

::: striqt.sensor.specs
    options:
      extra:
        module_docstring: false
      members:
        - Source
        - SoapySource
        - ZarrIQSource
        - MATSource
        - TDMSSource
        - FunctionSource
        - NoSource
        - SourceInfo
        - Sink
        - ZarrSink
        - DataOptions
        - PlotOptions
        - SharedPlotOptions
        - Peripherals
        - NoPeripherals
        - ManualYFactorCalibration
        - AcquisitionInfo
        - SoapyAcquisitionInfo
        - FileAcquisitionInfo
        - AcquiredIQ
        - PendingExtraData
