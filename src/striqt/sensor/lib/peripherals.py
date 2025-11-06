from __future__ import annotations
from . import specs, sources
import typing


_TC = typing.TypeVar('_TC', bound=specs.CaptureSpec, contravariant=True)
_TS = typing.TypeVar('_TS', bound=specs.SourceSpec, covariant=True)


class PeripheralsProtocol(typing.Protocol[_TC]):
    """the methods that a peripherals extension class should implement"""

    def open(self): ...

    def close(self): ...

    def setup(self) -> None: ...

    def arm(self, capture: specs._TC) -> None: ...

    def acquire(self, capture: specs._TC) -> dict[str, typing.Any]: ...


class PeripheralsBase(typing.Generic[_TS, _TC], PeripheralsProtocol[_TC]):
    """base class defining the object protocol peripheral hardware support.

    This is implemented primarily through connection management and callback
    methods for arming and acquisition.
    """

    sweep: specs.SweepSpec[_TS, _TC]
    source: sources.SourceBase[_TS, _TC]

    def __init__(
        self, sweep: specs.SweepSpec[_TS, _TC], source: sources.SourceBase[_TS, _TC]
    ):
        self.sweep = sweep
        self.source = source

    def __enter__(self):
        self.open()
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
