"""OFDM frame models for 3GPP and IEEE 802.16, the cyclic-prefix correlator, and 5G NR
synchronization.

The frame models (`Phy3GPP`, `Phy802_16`) give the sample indices of cyclic prefixes
and symbols, `corr_at_indices` correlates a waveform at those indices, and the
`sync_5gnr` names build PSS/SSS templates and locate synchronization signal blocks
in captured IQ.
"""

from .core import PhyOFDM
from .cp import corr_at_indices, corr_atol
from .phy_3gpp import Phy3GPP, get_3gpp_phy, slot_period
from .phy_802_16 import Phy802_16, get_802_16_phy
from .sequences import m_sequence
from .sync_5gnr import (
    COARSE_LAG_DIM,
    FINE_LAG_DIM,
    SyncParams,
    choose_ssb_offset,
    correlate_sync_sequence,
    get_5g_ssb_iq,
    index_pss_symbols,
    pss_5g_nr,
    pss_params,
    sss_5g_nr,
    sss_params,
    sync_frame_count,
    weighted_ssb_detect,
)
