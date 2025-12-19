from ..lib.register import (
    AlignmentSourceRegistry,
    MeasurementRegistry,
    to_analysis_spec_type,
)

measurements = MeasurementRegistry()
coordinates = measurements.coordinates
trigger_source = measurements.trigger_source
