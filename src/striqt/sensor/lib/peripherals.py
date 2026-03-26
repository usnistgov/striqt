from __future__ import annotations as __

from typing import Any, Generic, Iterable
from .. import specs
from .typing import Peripherals, SP, SC, SPC


class PeripheralsBase(Peripherals[SP, SC]):
    """dunder-method convenience implementation for the Peripherals protocol"""

    spec: SP

    def __init__(self, spec: specs.Sweep[Any, SP, SC]):
        self.spec = spec.peripherals
        self.open()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class CalibrationPeripheralsBase(Peripherals[SP, SC], Generic[SP, SC, SPC]):
    """base class defining the object protocol peripheral hardware support.

    This is implemented primarily through connection management and callback
    methods for arming and acquisition.
    """

    calibration_spec: SPC | None

    def __init__(self, spec: specs.CalibrationSweep[Any, SP, SC, SPC]):
        self.calibration_spec = spec.calibration
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
