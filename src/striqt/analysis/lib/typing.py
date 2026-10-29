from __future__ import annotations as __

from collections.abc import Iterable
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Optional,
    TypeVar,
    Union,
    overload,
)

# %% dataarrays.py
AnalysisReturnFlag = Literal[True, False, 'delayed']
TAR = TypeVar('TAR', bound=AnalysisReturnFlag)

if TYPE_CHECKING:
    from typing import Callable

    import xarray as xr
    import zarr
    import zarr.storage
    from typing_extensions import (
        ParamSpec,
        Protocol,
        Self,
        TypeAlias,
        TypedDict,
        TypeVar,
    )

    from striqt.waveform.lib.typing import Array, ArrayBackend

    from .. import specs
    from . import dataarrays

    P = ParamSpec('P')
    R = TypeVar('R', infer_variance=True)
    TC = TypeVar('TC', bound=specs.Capture, infer_variance=True)
    TM = TypeVar('TM', bound=specs.Analysis, infer_variance=True)

    # %% io.py

    class FileStream(Protocol):
        def close(self): ...

        def read(self, count: int) -> Array: ...

        def seek(self, pos: int): ...

        def get_capture_fields(self) -> dict: ...

    ChunksSize = Optional[Union[int, Literal['auto'], tuple[int, ...]]]

    if hasattr(zarr.storage, 'Store'):
        # zarr 2.x
        ZarrStore: TypeAlias = zarr.storage.Store
    elif hasattr(zarr, 'abc'):
        # zarr 3.x; the `hasattr` checks let ty resolve each branch under both the
        # zarr 2 and zarr 3 stubs. `Any` applies where neither layout resolves.
        ZarrStore: TypeAlias = zarr.abc.store.Store
    else:
        ZarrStore: TypeAlias = Any

    ZarrFormat: TypeAlias = Union[str, Literal[2, 3]]

    # %% register.py
    Measurement: TypeAlias = Union[Array, tuple[Array, dict[str, Any]]]

    class ToleranceKws(TypedDict, total=False):
        array_backend: ArrayBackend
        input_error: float

    RM = TypeVar('RM', bound=Measurement)

    class AnalysisFunc(Protocol[P, R]):
        __name__: str

        def __call__(
            self,
            x: Array,
            capture: specs.Capture,
            *args: P.args,
            **kwargs: P.kwargs,
        ) -> R: ...

    class WrappedAnalysis(Protocol[P, R]):
        __name__: str

        @overload
        def __call__(
            self,
            iq: Union[Array, dataarrays.AcquiredIQ],
            capture: specs.Capture,
            as_xarray: Literal[True] = ...,
            *args: P.args,
            **kwargs: P.kwargs,
        ) -> xr.DataArray: ...

        @overload
        def __call__(
            self,
            iq: Union[Array, dataarrays.AcquiredIQ],
            capture: specs.Capture,
            as_xarray: Literal[False],
            *args: P.args,
            **kwargs: P.kwargs,
        ) -> R: ...

        @overload
        def __call__(
            self,
            iq: Union[Array, dataarrays.AcquiredIQ],
            capture: specs.Capture,
            as_xarray: Literal['delayed'],
            *args: P.args,
            **kwargs: P.kwargs,
        ) -> dataarrays.DelayedDataArray: ...

        def __call__(
            self,
            iq: Union[Array, dataarrays.AcquiredIQ],
            capture: specs.Capture,
            as_xarray: specs.types.AsXArray = True,
            *args: P.args,
            **kwargs: P.kwargs,
        ) -> R | dataarrays.DelayedDataArray | xr.DataArray: ...

    AnalysisFuncWrapper: TypeAlias = Callable[
        [AnalysisFunc[P, Measurement]], WrappedAnalysis[P, Measurement]
    ]

    class CoordFunc(Protocol[TC, TM, R]):
        __name__: str

        # positional-only: `specs.helpers.lru_cache_on_converted` converts the
        # leading positional arguments before the cache lookup
        def __call__(self, capture: TC, spec: TM, /) -> R: ...

    CoordFuncWrapper: TypeAlias = Callable[[CoordFunc[TC, TM, R]], CoordFunc[TC, TM, R]]
