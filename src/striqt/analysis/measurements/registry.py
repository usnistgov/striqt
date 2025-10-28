from ..lib.register import (
    MeasurementRegistry,
    AlignmentSourceRegistry,
    to_analysis_spec_type,
)

measurements = MeasurementRegistry()
coordinates = measurements.coordinates
channel_sync_source = measurements.channel_sync_source
