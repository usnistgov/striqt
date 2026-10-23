"""tests for striqt.analysis.lib.dataarrays: the acquired-IQ container and waveform selection"""

from __future__ import annotations

import numpy as np
import pytest

import striqt.analysis as sa


# %% select_waveform
@pytest.mark.parametrize('prefer_filtered', [True, False])
@pytest.mark.parametrize('prefer_aligned', [True, False])
def test_select_waveform_returns_an_array_unchanged(prefer_filtered, prefer_aligned):
    x = np.zeros((2, 8), dtype='complex64')
    out = sa.select_waveform(
        x, prefer_filtered=prefer_filtered, prefer_aligned=prefer_aligned
    )
    assert out is x


# %% AcquiredIQ


@pytest.mark.parametrize('prefer_filtered', [True, False])
@pytest.mark.parametrize('prefer_aligned', [True, False])
def test_the_base_select_returns_x_for_every_preference(
    prefer_filtered, prefer_aligned
):
    x = np.zeros((2, 8), dtype='complex64')
    iq = sa.AcquiredIQ(x=x, capture=None)
    prefs = {'prefer_filtered': prefer_filtered, 'prefer_aligned': prefer_aligned}
    assert iq.select(**prefs) is x
    assert sa.select_waveform(iq, **prefs) is x


# %% evaluate_by_spec


def test_evaluate_by_spec_hands_the_acquisition_to_the_measurement():
    """a measurement given an AcquiredIQ measures the array that `select` returns"""
    capture = sa.specs.Capture(duration=16e-6, sample_rate=1e6)
    x = np.arange(32, dtype='complex64').reshape(2, 16)
    iq = sa.AcquiredIQ(x=x, capture=capture)
    spec = sa.specs.BuiltinAnalyses(iq_waveform=sa.specs.IQWaveform())
    options = sa.EvaluationOptions(as_xarray='delayed', registry=sa.registry)

    results = sa.analyze_by_spec(iq, spec, capture, options)

    assert np.array_equal(results['iq_waveform'].result, x)
