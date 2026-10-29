from __future__ import annotations as __

import dataclasses
import functools
import typing
from abc import abstractmethod
from collections.abc import Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Protocol,
    TypedDict,
    TypeVar,
    runtime_checkable,
)

from typing_extensions import ParamSpec, Self, TypeAlias, TypeVar, Unpack

import striqt.analysis as sa
import striqt.waveform as sw

if TYPE_CHECKING:
    from .. import specs
    from .actions.core import ActionRunner


# these are evaluated at runtime, so they need to be here
P = ParamSpec('P')
PS = ParamSpec('PS')
PC = ParamSpec('PC')
R = TypeVar('R')

S = TypeVar('S', bound='specs.SpecBase')
SS = TypeVar('SS', bound='specs.Source')
SC = TypeVar('SC', bound='specs.SensorCapture')
SP = TypeVar('SP', bound='specs.Peripherals')
SA = TypeVar('SA', bound='sa.specs.AnalysisGroup')
SPC = TypeVar('SPC', bound='specs.Peripherals')
_SS = TypeVar('_SS', bound='specs.Source', covariant=True)
TA = TypeVar('TA', bound='specs.ActionSpec')
TR = TypeVar('TR', bound='ActionRunner[Any, Any, Any, Any, Any, Any, Any]')

# %% compute.py
from striqt.analysis.lib.typing import TAR

_SC = TypeVar('_SC', bound='specs.SensorCapture', contravariant=True)
_SP = TypeVar('_SP', bound='specs.Peripherals', covariant=True)


# %% peripherals.py
@runtime_checkable
class Peripherals(Protocol[_SP, _SC]):
    @abstractmethod
    def __init__(self, spec: specs.Sweep[Any, _SP, _SC, Any]): ...

    def open(self): ...

    def close(self): ...

    def setup(self, captures: Sequence[_SC], loops: Sequence[specs.LoopSpec]): ...

    def arm(self, capture: _SC): ...

    def acquire(self, capture: _SC) -> dict[str, typing.Any]: ...

    def __enter__(self) -> Self: ...

    def __exit__(self, exc_type, exc_val, exc_tb): ...


# %% sources/base.py
@runtime_checkable
class SourceBackend(Protocol[_SS, SC]):
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
    ) -> tuple[int, int]:
        """fill `count` samples of the armed capture into each buffer from `offset`.

        The first read of a capture begins one lead overlap before the record.

        Args:
            buffers: one array per port of the armed capture, in capture port order
            offset: the index in each buffer of the first sample written
            count: the number of samples requested per port
            timeout_sec: the maximum wait (in s) for `count` samples, or None to
                wait indefinitely
            on_overflow: ``'except'`` to raise on a stream overflow, ``'ignore'`` to
                continue

        Returns:
            the number of samples written per port, and the time (in ns) of the
            sample at ``buffers[i][offset]``

        Raises:
            OverflowError: if the stream overflowed and `on_overflow` is ``'except'``
        """
        ...

    def prepare_retrigger(self):
        pass

    def package_iq(
        self,
        iq: specs.AcquiredIQ,
        samples: Array,
        time_ns: int | None,
    ) -> specs.AcquiredIQ:
        return iq

    @abstractmethod
    def close(self):
        pass


if typing.TYPE_CHECKING:
    from striqt.analysis.lib.typing import TAR, Array, FileStream, ZarrStore
    from striqt.waveform.fourier import ResamplerDesign

    PassThroughWrapper: TypeAlias = typing.Callable[[Callable[P, R]], Callable[P, R]]

    # %% base.py

    class ResamplerKws(TypedDict, total=False):
        bw_lo: float
        min_oversampling: float
        window: str
        min_fft_size: int

    # %% resources.py
    from striqt.analysis.lib.register import KwArgCache

    from .compute.datasets import DelayedDataset
    from .resources import Resources

    class SourceOpenCallback(Protocol):
        def __call__(self, sweep: specs.Sweep, source_id: str) -> None: ...

    AnalyzedCallback: TypeAlias = Callable[[DelayedDataset], None]

    class CachedCallback(Protocol):
        def __call__(
            self,
            resources: Resources,
            cache: KwArgCache,
            capture: specs.SensorCapture,
            result: Any,
            **kwargs: Any,
        ) -> None: ...
