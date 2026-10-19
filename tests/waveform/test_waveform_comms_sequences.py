"""striqt.waveform.lib.comms.sequences: the linear-feedback shift register generator
behind the 5G NR sync sequences, checked against the properties that maximal-length
sequence theory guarantees for a degree-7 generator (period 2**7 - 1 = 127).
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from striqt.waveform.lib.comms.sequences import m_sequence

PERIOD = 2**7 - 1

# the PSS generator of TS 38.211 7.4.2.2.1 and the SSS x_1 generator of 7.4.2.3.1
GENERATORS = [([0, 1, 1, 0, 1, 1, 1], (3, 7)), ([1, 0, 0, 0, 0, 0, 0], (6, 7))]


@pytest.mark.parametrize('initial, taps', GENERATORS, ids=['pss', 'sss_x1'])
def test_balance(initial, taps):
    """a maximal-length sequence of degree 7 holds 2**6 ones and 2**6 - 1 zeros"""
    x = np.array(m_sequence(initial, taps, PERIOD), dtype=int)
    assert x.shape == (PERIOD,)
    assert x.sum() == 64
    assert (x == 0).sum() == 63


@pytest.mark.parametrize('initial, taps', GENERATORS, ids=['pss', 'sss_x1'])
def test_two_level_autocorrelation(initial, taps):
    """the periodic autocorrelation of a bipolar maximal-length sequence is the period
    at zero lag and exactly -1 at every other lag"""
    b = 1 - 2 * np.array(m_sequence(initial, taps, PERIOD), dtype=int)
    autocorr = np.array([np.sum(b * np.roll(b, k)) for k in range(PERIOD)])
    expected = np.full(PERIOD, -1, dtype=int)
    expected[0] = PERIOD
    assert_array_equal(autocorr, expected)
