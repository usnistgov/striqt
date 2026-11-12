from __future__ import annotations as __

from collections.abc import Iterable
from typing import Any

from .. import specs
from .typing import SC, SP, Peripherals


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
    """a `Peripherals` implementation for a sensor without peripheral hardware.

    `open`, `close`, `setup`, and `arm` return None and access no device, and
    `acquire` returns an empty dict.
    """

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
