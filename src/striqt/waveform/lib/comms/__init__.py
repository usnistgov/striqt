"""OFDM frame models for 3GPP and IEEE 802.16, the cyclic-prefix correlator, and 5G NR
synchronization.

The frame models in `phy_3gpp` and `phy_802_16` give the sample indices of cyclic
prefixes and symbols, `cp.correlate_cyclic_shift` correlates a waveform at those
indices, and `sync_5gnr` builds PSS/SSS templates and locates synchronization signal
blocks in captured IQ.
"""

# cp_cpu_jit and cp_cuda_jit are left out so importing the package does not import numba
from . import core, cp, phy_3gpp, phy_802_16, sequences, sync_5gnr
