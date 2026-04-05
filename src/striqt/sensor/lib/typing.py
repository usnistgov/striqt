from __future__ import annotations as __

import typing
from typing_extensions import ParamSpec, Self, TypeAlias, TypeVar, Unpack
from typing import (
    Any,
    Callable,
    Protocol,
    Sequence,
    TypedDict,
    TypeVar,
    TYPE_CHECKING,
    runtime_checkable,
)

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
class Source(Protocol[SS, SC, PS, PC]):
    __setup__: SS
    _capture: typing.Optional[SC]

    def __init__(
        self,
        _setup: SS | None = None,
        /,
        reuse_iq=False,
        *args: PS.args,
        **kwargs: PS.kwargs,
    ): ...

    @classmethod
    def from_spec(
        cls,
        spec: SS,
        *,
        captures: tuple[SC, ...] | None = None,
        loops: tuple[specs.LoopSpec, ...] | None = None,
        reuse_iq: bool = False,
    ) -> Self: ...

    def read_iq(
        self, overlaps: tuple[int, int] = (0, 0)
    ) -> 'tuple[Array, int|None]': ...

    def _connect(self, spec: SS) -> None: ...

    def _apply_setup(
        self,
        spec: SS,
        *,
        captures: tuple[SC, ...] | None = None,
        loops: 'tuple[specs.LoopSpec, ...] | None' = None,
    ) -> None: ...

    def arm(self, *args: PC.args, **kwargs: PC.kwargs): ...

    def arm_spec(self, spec: SC): ...

    def acquire(self, overlaps: tuple[int, int] = (0, 0)) -> sources.AcquiredIQ: ...

    @property
    def capture_spec(self) -> SC: ...

    def _prepare_capture(self, capture: SC) -> SC | None: ...

    def get_resampler(self, capture: 'SC | None' = None) -> ResamplerDesign: ...


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
