from __future__ import annotations
import typing

from ..._api import structs, util
from .xarray import evaluate_channel_analysis, package_channel_analysis
from .registry import ChannelAnalysisRegistryDecorator


if typing.TYPE_CHECKING:
    import iqwaveform
    import xarray as xr
else:
    iqwaveform = util.lazy_import('iqwaveform')
    xr = util.lazy_import('xarray')


as_registered_channel_analysis = ChannelAnalysisRegistryDecorator(
    structs.ChannelAnalysis
)


def analyze_by_spec(
    iq: 'iqwaveform.util.Array',
    capture: structs.Capture,
    *,
    spec: str | dict | structs.ChannelAnalysis,
) -> 'xr.DatasetType':
    """evaluate a set of different channel analyses on the iq waveform as specified by spec"""

    results = evaluate_channel_analysis(
        iq, capture, spec=spec, registry=as_registered_channel_analysis
    )
    return package_channel_analysis(capture, results)
