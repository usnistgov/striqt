from __future__ import annotations

import typing

from . import sources, specs

_TC = typing.TypeVar('_TC', bound=specs.CaptureSpec)
_TS = typing.TypeVar('_TS', bound=specs.SourceSpec)


class PeripheralsProtocol(typing.Protocol[_TS, _TC]):
    """a peripherals extension class must implement these"""

    def open(self): ...

    def close(self): ...

    def setup(self, source: sources.SourceBase[_TS, _TC]) -> None: ...

    def arm(self, capture: specs._TC) -> None: ...

    def acquire(self, capture: specs._TC) -> dict[str, typing.Any]: ...


class PeripheralsBase(PeripheralsProtocol[_TS, _TC]):
    """base class defining the object protocol peripheral hardware support.

    This is implemented primarily through connection management and callback
    methods for arming and acquisition.
    """

    sweep: specs.SweepSpec[_TS, _TC]
    source: sources.SourceBase[_TS, _TC]

    def __init__(self, sweep: specs.SweepSpec[_TS, _TC]):
        self.sweep = sweep
        self.open()

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

    def setup(self, source):
        return

    def arm(self, capture):
        pass

    def acquire(self, capture):
        return {}
