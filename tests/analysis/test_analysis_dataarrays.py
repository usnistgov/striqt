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
