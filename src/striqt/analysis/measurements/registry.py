from ..lib.register import (
    AlignmentSourceRegistry,
    AnalysisRegistry,
    to_analysis_spec_type,
)

measurements = AnalysisRegistry()
coordinates = measurements.coordinates
signal_trigger = measurements.signal_trigger
