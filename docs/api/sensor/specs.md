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
        - ActionSpec
        - BurstOptions
        - CapturePlan
        - CaptureRemap
        - CellChannel
        - Cellular5GNRAnalysis
        - Cellular5GNRSweep
        - CellularChannelPowerHistogramSpec
        - CellularChannelPowerTimeSeriesSpec
        - CellularCyclicAutocorrelationSpec
        - CellularCyclicChannelPowerSpec
        - CellularIQWaveformSpec
        - CellularPowerSpectralDensitySpec
        - CellularPSSCorrelationSpec
        - CellularPSSSyncSpec
        - CellularResourcePowerHistogramSpec
        - CellularSpectrogramSpec
        - CellularSSBSpectrogramSpec
        - Description
        - DiracDeltaCapture
        - Extension
        - FileCapture
        - FrequencyBinRange
        - List
        - LoopBase
        - ManualYFactorCalibration
        - NoiseCapture
        - PlotOptions
        - Range
        - Repeat
        - SawtoothCapture
        - Schema
        - SensorCapture
        - SingleToneCapture
        - SoapyCapture
        - SpecValidationError
        - Sweep
        - SweepOptions

## Sources, sinks, and peripherals

Spec structs for IQ sources, data sinks, peripherals, and the acquisition metadata recorded with each capture.

::: striqt.sensor.specs
    options:
      extra:
        module_docstring: false
      members:
        - AcquiredIQ
        - AcquisitionInfo
        - DataOptions
        - FileAcquisitionInfo
        - FunctionSource
        - ManualYFactorSpec
        - MATSource
        - NoPeripherals
        - NoSource
        - PendingExtraData
        - Peripherals
        - SharedPlotOptions
        - Sink
        - SoapyAcquisitionInfo
        - SoapySource
        - Source
        - SourceInfo
        - TDMSSource
        - ZarrIQSource
        - ZarrSink
