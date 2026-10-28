from __future__ import annotations as __

from typing import Any, Iterable
from .. import specs
from .typing import Peripherals, SP, SC


class PeripheralsBase(Peripherals[SP, SC]):
    """base `Peripherals` implementation that opens on construction.

    Construction reads the `peripherals` block of the sweep and calls `open`. Exit
    from the context manager calls `close`.
    """

    spec: SP

    def __init__(self, spec: specs.Sweep[Any, SP, SC, Any]):
        self.spec = spec.peripherals
        self.open()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class NoPeripherals(PeripheralsBase[SP, SC]):
    def open(self):
        return

    def close(self):
        return

    def setup(self, captures: Iterable[SC], loops: Iterable[specs.LoopBase]):
        return

    def arm(self, capture):
        pass

    def acquire(self, capture):
        return {}
