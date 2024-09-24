from __future__ import annotations

from ..._api import structs
from .capture import evaluate_channel_analysis, package_channel_analysis
from .registry import ChannelAnalysisRegistryDecorator
from ..._api import type_stubs


as_registered_channel_analysis = ChannelAnalysisRegistryDecorator(
    structs.ChannelAnalysis
)


def analyze_by_spec(
    iq: type_stubs.ArrayType,
    capture: structs.Capture,
    *,
    spec: str | dict | structs.ChannelAnalysis,
) -> type_stubs.DatasetType:
    """evaluate a set of different channel analyses on the iq waveform as specified by spec"""

    results = evaluate_channel_analysis(
        iq, capture, spec=spec, registry=as_registered_channel_analysis
    )
    return package_channel_analysis(capture, results)
