import typing
from typing_extensions import ParamSpec, Self, TypeVar, Unpack
from typing import (
    Any,
    Callable,
    Protocol,
    Sequence,
    TypeAlias,
    TypedDict,
    TypeVar,
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from .. import specs


# these are evaluated at runtime, so they need to be here
P = ParamSpec('P')
PS = ParamSpec('PS')
PC = ParamSpec('PC')
R = TypeVar('R')

TS = TypeVar('TS', bound='specs.Source')
TC = TypeVar('TC', bound='specs.SensorCapture')
TB = TypeVar('TB', bound='specs.SpecBase')
TP = TypeVar('TP', bound='specs.Peripherals')
TPC = TypeVar('TPC', bound='specs.Peripherals')


# %% compute.py
from striqt.analysis.lib.typing import TAR


_TC = TypeVar('_TC', bound='specs.SensorCapture', contravariant=True)


# %% peripherals.py
class PeripheralsProtocol(Protocol[_TC]):
    def __init__(self, spec: 'specs.Sweep[Any, TP, TC]'): ...

    def open(self): ...

    def close(self): ...

    def setup(self, captures: 'Sequence[_TC]', loops: 'Sequence[specs.LoopSpec]'): ...

    def arm(self, capture: _TC): ...

    def acquire(self, capture: _TC) -> dict[str, typing.Any]: ...


# %% sources/base.py
class HasSetup(Protocol[TS, PS]):
    __setup__: TS

    def __init__(
        self,
        _setup: TS | None = None,
        /,
        reuse_iq=False,
        *args: PS.args,
        **kwargs: PS.kwargs,
    ): ...

    @classmethod
    def from_spec(
        cls,
        spec: TS,
        *,
        captures: tuple[TC, ...] | None = None,
        loops: tuple[specs.LoopSpec, ...] | None = None,
        reuse_iq: bool = False,
    ) -> Self: ...

    def _connect(self, spec: TS) -> None: ...

    def _apply_setup(
        self,
        spec: TS,
        *,
        captures: tuple[TC, ...] | None = None,
        loops: 'tuple[specs.LoopSpec, ...] | None' = None,
    ) -> None: ...


class HasCapture(typing.Protocol[TC, PC]):
    _capture: typing.Optional[TC]

    def arm(self, *args: PC.args, **kwargs: PC.kwargs): ...

    def arm_spec(self, spec: TC): ...

    def acquire(
        self,
        *,
        correction: bool = True,
        alias_func: 'specs.helpers.PathAliasFormatter | None' = None,
    ) -> sources.AcquiredIQ: ...

    @property
    def capture_spec(self) -> TC: ...

    def _prepare_capture(self, capture: TC) -> TC | None: ...

    def get_resampler(self, capture: 'TC | None' = None) -> ResamplerDesign: ...


if typing.TYPE_CHECKING:
    from . import sources
    from striqt.analysis.lib.typing import Array, FileStream, TAR, ZarrStore
    from striqt.waveform.fourier import ResamplerDesign

    GenericWrapper: TypeAlias = typing.Callable[[Callable[P, R]], Callable[P, R]]

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
    WarmupSweep: TypeAlias = specs.Sweep[specs.NoSource, specs.NoPeripherals, TC]

    # %% resources.py
    class SourceOpenCallback(Protocol):
        def __call__(self, sweep: specs.Sweep, source_id: str) -> None: ...
