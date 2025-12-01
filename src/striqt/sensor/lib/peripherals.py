from __future__ import annotations

import typing

from . import specs
from .specs import _TS, _TPC

_TC = typing.TypeVar('_TC', bound=specs.ResampledCapture, contravariant=True)
_TP = typing.TypeVar('_TP', bound=specs.Peripherals)


class PeripheralsProtocol(typing.Protocol[_TC]):
    """a peripherals extension class must implement these"""

    def open(self): ...

    def close(self): ...

    def setup(
        self, captures: typing.Sequence[_TC], loops: typing.Sequence[specs.LoopSpec]
    ): ...

    def arm(self, capture: _TC): ...

    def acquire(self, capture: _TC) -> dict[str, typing.Any]: ...


class PeripheralsBase(typing.Generic[_TP, _TC], PeripheralsProtocol[_TC]):
    """base class defining the object protocol peripheral hardware support.

    This is implemented primarily through connection management and callback
    methods for arming and acquisition.
    """

    spec: _TP

    def __init__(self, spec: specs.Sweep[typing.Any, _TP, _TC]):
        self.spec = spec.peripherals
        self.open()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class CalibrationPeripheralsBase(
    PeripheralsBase[_TP, _TC],
    typing.Generic[_TP, _TC, _TPC],
):
    """base class defining the object protocol peripheral hardware support.

    This is implemented primarily through connection management and callback
    methods for arming and acquisition.
    """

    calibration_spec: _TPC | None

    def __init__(self, spec: specs.CalibrationSweep[typing.Any, _TP, _TC, _TPC]):
        self.calibration_spec = spec.calibration
        self.open()


class NoPeripherals(PeripheralsBase[_TP, _TC]):
    def open(self):
        return

    def close(self):
        return

    def setup(self, captures: typing.Iterable[_TC]):
        return

    def arm(self, capture):
        pass

    def acquire(self, capture):
        return {}
