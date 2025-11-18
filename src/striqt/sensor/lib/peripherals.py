from __future__ import annotations

import typing

from . import specs
from .specs import _TS, _TPC

_TC = typing.TypeVar('_TC', bound=specs.ResampledCapture, contravariant=True)
_TP = typing.TypeVar('_TP', bound=specs.Peripheral)


class PeripheralsProtocol(typing.Protocol[_TC]):
    """a peripherals extension class must implement these"""

    def open(self): ...

    def close(self): ...

    def setup(self): ...

    def arm(self, capture: _TC): ...

    def acquire(self, capture: _TC) -> dict[str, typing.Any]: ...


class PeripheralsBase(typing.Generic[_TS, _TP, _TC], PeripheralsProtocol[_TC]):
    """base class defining the object protocol peripheral hardware support.

    This is implemented primarily through connection management and callback
    methods for arming and acquisition.
    """

    spec: _TP | None

    def __init__(self, spec: specs.Sweep[_TS, _TP, _TC]):
        self.spec = spec.peripherals
        self.open()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class CalibrationPeripheralsBase(
    PeripheralsBase[_TS, _TP, _TC],
    typing.Generic[_TS, _TP, _TC, _TPC],
):
    """base class defining the object protocol peripheral hardware support.

    This is implemented primarily through connection management and callback
    methods for arming and acquisition.
    """

    calibration_spec: _TPC | None

    def __init__(self, spec: specs.CalibrationSweep[_TS, _TP, _TC, _TPC]):
        self.calibration_spec = spec.calibration_peripherals
        self.open()


class NoPeripherals(PeripheralsBase[specs._TS, specs.NoPeripheral, _TC]):
    def open(self):
        return

    def close(self):
        return

    def setup(self):
        return

    def arm(self, capture):
        pass

    def acquire(self, capture):
        return {}
