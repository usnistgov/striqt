"""A registry of sensor data <-> source class bindings"""

from . import deepwave
from ..lib import specs as _specs
from ..lib import sources as _sources
from ._util import (
    bind_sensor,
    SensorBinding,
    get_registry,
    get_binding,
    get_tagged_sweep_spec,
)

#%% Synthetic data sources for testing, warmup, and post-analysis
Warmup = bind_sensor(
    'Warmup',
    SensorBinding(
        source_spec=_specs.NullSourceSpec,
        capture_spec=_specs.CaptureSpec,
        source=_sources.NullSource,
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

#%% Hardware data sources
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
