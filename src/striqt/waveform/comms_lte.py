"""LTE frame model (via `Phy3GPP` with ``generation='4G'``)."""

from .lib.comms import (
    Phy3GPP,
    correlate_cyclic_shift,
    get_3gpp_phy,
    slot_period,
)

__all__ = [
    'Phy3GPP',
    'correlate_cyclic_shift',
    'get_3gpp_phy',
    'slot_period',
]
