"""LTE frame model (via `Phy3GPP` with ``generation='4G'``)."""

from .lib.comms.cp import correlate_cyclic_shift
from .lib.comms.phy_3gpp import Phy3GPP, get_3gpp_phy, slot_period
