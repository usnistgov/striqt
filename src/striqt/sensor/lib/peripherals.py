from __future__ import annotations

import typing

from . import specs


_TC = typing.TypeVar('_TC', bound=specs.ResampledCapture, contravariant=True)
_TP = typing.TypeVar('_TP', bound=specs.Peripheral)


class PeripheralsProtocol(typing.Protocol[_TC]):
    """a peripherals extension class must implement these"""

    def open(self): ...

    def close(self): ...

    def setup(self): ...

    def arm(self, capture: specs._TC): ...

    def acquire(self, capture: specs._TC) -> dict[str, typing.Any]: ...


class PeripheralsBase(typing.Generic[_TP, _TC], PeripheralsProtocol[_TC]):
    """base class defining the object protocol peripheral hardware support.

    This is implemented primarily through connection management and callback
    methods for arming and acquisition.
    """

    spec: _TP

    def __init__(self, spec: _TP):
        self.spec = spec
        self.open()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class NoPeripherals(PeripheralsBase):
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
