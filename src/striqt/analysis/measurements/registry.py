from ..lib.register import (
    AlignmentSourceRegistry,
    MeasurementRegistry,
    to_analysis_spec_type,
)

measurements = MeasurementRegistry()
coordinates = measurements.coordinates
channel_sync_source = measurements.channel_sync_source
