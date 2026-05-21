from __future__ import annotations as __
from abc import abstractmethod

import dataclasses
import functools
import typing
from typing_extensions import ParamSpec, Self, TypeAlias, TypeVar, Unpack
from typing import (
    Any,
    Callable,
    Generic,
    Protocol,
    Sequence,
    TypedDict,
    TypeVar,
    TYPE_CHECKING,
    runtime_checkable,
)

import striqt.analysis as sa
import striqt.waveform as sw

if TYPE_CHECKING:
    from .. import specs


# these are evaluated at runtime, so they need to be here
P = ParamSpec('P')
PS = ParamSpec('PS')
PC = ParamSpec('PC')
R = TypeVar('R')

S = TypeVar('S', bound='specs.SpecBase')
SS = TypeVar('SS', bound='specs.Source')
SC = TypeVar('SC', bound='specs.SensorCapture')
SP = TypeVar('SP', bound='specs.Peripherals')
SPC = TypeVar('SPC', bound='specs.Peripherals')


# %% compute.py
from striqt.analysis.lib.typing import TAR


_SC = TypeVar('_SC', bound='specs.SensorCapture', contravariant=True)
_SP = TypeVar('_SP', bound='specs.Peripherals', covariant=True)


# %% peripherals.py
@runtime_checkable
class Peripherals(Protocol[_SP, _SC]):
    @abstractmethod
    def __init__(self, spec: 'specs.Sweep[Any, _SP, _SC]'): ...

    def open(self): ...

    def close(self): ...

    def setup(self, captures: 'Sequence[_SC]', loops: 'Sequence[specs.LoopSpec]'): ...

    def arm(self, capture: _SC): ...

    def acquire(self, capture: _SC) -> dict[str, typing.Any]: ...

    def __enter__(self) -> Self: ...

    def __exit__(self, exc_type, exc_val, exc_tb): ...


# %% sources/base.py
@runtime_checkable
class SourceBackend(Protocol[SS, SC]):
    @abstractmethod
    def __init__(self, spec): ...

    @abstractmethod
    def get_info(self) -> specs.SourceInfo: ...

    @abstractmethod
    def get_id(self) -> str: ...

    @abstractmethod
    def get_resampler(self, capture: SC) -> ResamplerDesign: ...

    @abstractmethod
    def setup(self, *, rx_ports: tuple[int, ...] | None = None) -> None: ...

    @abstractmethod
    def arm(self, capture: SC) -> SC | None: ...

    @abstractmethod
    def trigger(self, overlaps: tuple[int, int] = (0, 0)) -> None: ...

    @abstractmethod
    def read(
        self,
        buffers: list[Array],
        offset: int,
        count: int,
        timeout_sec: float | None,
        *,
        on_overflow: specs.types.OnOverflow = 'except',
    ) -> tuple[int, int]: ...

    def prepare_retrigger(self):
        pass

    def package_iq(
        self,
        iq: 'specs.AcquiredIQ',
        samples: Array,
        time_ns: int | None,
    ) -> 'specs.AcquiredIQ':
        return iq

    def close(self):
        pass


if typing.TYPE_CHECKING:
    from . import sources
    from striqt.analysis.lib.typing import Array, FileStream, TAR, ZarrStore
    from striqt.waveform.fourier import ResamplerDesign

    PassThroughWrapper: TypeAlias = typing.Callable[[Callable[P, R]], Callable[P, R]]

    # %% base.py

    class ResamplerKws(TypedDict, total=False):
        bw_lo: float
        min_oversampling: float
        window: str
        min_fft_size: int

    class _CallableWithCaptureArg(Protocol):
        def __call__(
            self, capture: specs.SensorCapture, *args, **kws
        ) -> typing.Any: ...

    _TAC = TypeVar('_TAC', bound=_CallableWithCaptureArg)
    CaptureConverterWrapper = Callable[[_TAC], _TAC]

    # %% compute.py
    WarmupSweep: TypeAlias = specs.Sweep[specs.NoSource, specs.NoPeripherals, SC]

    # %% resources.py
    class SourceOpenCallback(Protocol):
        def __call__(self, sweep: specs.Sweep, source_id: str) -> None: ...
