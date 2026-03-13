from __future__ import annotations as __

from typing import Any, Generic, Iterable
from .. import specs
from .typing import PeripheralsProtocol, TP, TC, TPC


class PeripheralsBase(Generic[TP, TC], PeripheralsProtocol[TC]):
    """base class defining the object protocol peripheral hardware support.

    This is implemented primarily through connection management and callback
    methods for arming and acquisition.
    """

    spec: TP

    def __init__(self, spec: specs.Sweep[Any, TP, TC]):
        self.spec = spec.peripherals
        self.open()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class CalibrationPeripheralsBase(PeripheralsBase[TP, TC], Generic[TP, TC, TPC]):
    """base class defining the object protocol peripheral hardware support.

    This is implemented primarily through connection management and callback
    methods for arming and acquisition.
    """

    calibration_spec: TPC | None

    def __init__(self, spec: specs.CalibrationSweep[Any, TP, TC, TPC]):
        self.calibration_spec = spec.calibration
        self.open()


class NoPeripherals(PeripheralsBase[TP, TC]):
    def open(self):
        return

    def close(self):
        return

    def setup(self, captures: Iterable[TC], loops: Iterable[specs.LoopBase]):
        return

    def arm(self, capture):
        pass

    def acquire(self, capture):
        return {}
