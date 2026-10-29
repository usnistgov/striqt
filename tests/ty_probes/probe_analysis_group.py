"""the analysis and trigger groups expose the measurement fields to the type checker"""

from __future__ import annotations

from typing import Optional

from typing_extensions import assert_type

import striqt.analysis as sa
import striqt.sensor as ss


class Quick(sa.specs.AnalysisGroup, frozen=True, kw_only=True, omit_defaults=True):
    spectrogram: Optional[sa.specs.Spectrogram] = None


def probe_schema() -> None:
    schema = ss.specs.Schema(
        source=ss.specs.FunctionSource,
        capture=ss.specs.SingleToneCapture,
        peripherals=ss.specs.Peripherals,
        init_like=ss.specs.FunctionSource,
        arm_like=ss.specs.SingleToneCapture,
        analysis=Quick,
    )
    assert_type(schema.analysis, type[Quick])


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
