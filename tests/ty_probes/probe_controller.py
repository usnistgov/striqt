"""a sensor binding exposes its source and capture spec fields as keyword parameters"""

from __future__ import annotations

import striqt.sensor as ss

src = ss.bindings.single_tone(num_rx_ports=1, master_clock_rate=1e6)

ss.bindings.single_tone(num_rx_ports=1, master_clock_rate=1e6, bogus=1)  # ty: ignore[unknown-argument]
ss.bindings.single_tone(num_rx_ports='x', master_clock_rate=1e6)  # ty: ignore[invalid-argument-type]
ss.bindings.single_tone(num_rx_ports=1)  # ty: ignore[missing-argument]

src.arm(port=0, bogus=1)  # ty: ignore[unknown-argument]
src.arm(port=0, frequency_offset='x')  # ty: ignore[invalid-argument-type]
