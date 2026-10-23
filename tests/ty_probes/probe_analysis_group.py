"""the analysis and trigger bundles expose their measurement fields to the type checker"""

from __future__ import annotations

from typing import Optional

from typing_extensions import assert_type

import striqt.analysis as sa
import striqt.sensor as ss


def probe(source: ss.specs.Source) -> None:
    analysis = sa.specs.BuiltinAnalyses(iq_waveform=sa.specs.IQWaveform())
    assert_type(analysis.iq_waveform, Optional[sa.specs.IQWaveform])
    assert_type(
        ss.specs.BundledAnalysis.from_dict({'iq_waveform': {}}),
        sa.specs.BuiltinAnalyses,
    )
    assert_type(analysis.replace(spectrogram=None), sa.specs.BuiltinAnalyses)
    if isinstance(source.signal_trigger, sa.specs.BuiltinTriggers):
        assert_type(
            source.signal_trigger.cellular_5g_pss_sync,
            Optional[sa.specs.Cellular5GNPSSSync],
        )
