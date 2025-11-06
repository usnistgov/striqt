from . import deepwave
from ..lib import specs as _specs
from ..lib.sources import testing as _testing
from ._base import bind, SensorBinding, get, tagged_union_spec


zarr_file = bind(
    'zarr_file',
    SensorBinding(
        source_spec=_testing.ZarrFileSourceSpec,
        capture_spec=_testing.FileCaptureSpec,
        source=_testing.ZarrIQSource,
    ),
)

air7101b = bind(
    'air7101b',
    SensorBinding(
        source_spec=deepwave.Air7101BSourceSpec,
        capture_spec=_specs.SoapyCaptureSpec,
        source=deepwave.Airstack1Source,
    ),
)

air7201b = bind(
    'air7201B',
    SensorBinding(
        source_spec=deepwave.Air7201BSourceSpec,
        capture_spec=_specs.SoapyCaptureSpec,
        source=deepwave.Airstack1Source,
    ),
)

air8201b = bind(
    'air8201B',
    SensorBinding(
        source_spec=deepwave.Air8201BSourceSpec,
        capture_spec=_specs.SoapyCaptureSpec,
        source=deepwave.Airstack1Source,
    ),
)
