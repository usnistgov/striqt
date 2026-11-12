"""striqt.waveform.lib.comms.phy_802_16: the IEEE 802.16 OFDMA sample rate"""

from __future__ import annotations

import pytest

from striqt.waveform.lib.comms import phy_802_16

# %% Phy802_16.__init__

# IEEE Std 802.16-2017 §8.4.2.4: the sampling factor n is 8/7 for a channel bandwidth
# that is a multiple of 1.75 MHz, 28/25 for a multiple of 1.25, 1.5, 2, or 2.75 MHz,
# and 8/7 otherwise; the sample rate is floor(n * BW / 8000) * 8000
STANDARD_SAMPLE_RATES = {
    1.25e6: 1.4e6,
    1.75e6: 2e6,
    2.5e6: 2.8e6,
    3.5e6: 4e6,
    5e6: 5.6e6,
    7e6: 8e6,
    8.75e6: 10e6,
    10e6: 11.2e6,
    14e6: 16e6,
    20e6: 22.4e6,
}


@pytest.mark.parametrize(
    'channel_bandwidth, sample_rate',
    STANDARD_SAMPLE_RATES.items(),
    ids=[f'{bw / 1e6:g}MHz' for bw in STANDARD_SAMPLE_RATES],
)
def test_sample_rate_follows_the_sampling_factor_of_the_standard(
    channel_bandwidth: float, sample_rate: float
):
    """the standard rate of each profile bandwidth; 14 MHz is a multiple of both
    1.75 MHz and 2 MHz, and the 1.75 MHz rule takes precedence"""
    phy = phy_802_16.Phy802_16(channel_bandwidth)
    assert phy.sample_rate == sample_rate
    assert phy.sampling_factor == pytest.approx(sample_rate / channel_bandwidth)
