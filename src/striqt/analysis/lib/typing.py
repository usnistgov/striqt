from __future__ import annotations as __
from typing import Any, Iterable, Literal, overload, TypeVar, TYPE_CHECKING


# %% dataarrays.py
AnalysisReturnFlag = Literal[True, False, 'delayed']
TAR = TypeVar('TAR', bound=AnalysisReturnFlag)

if TYPE_CHECKING:
    from typing_extensions import (
        Callable,
        ParamSpec,
        Protocol,
        Self,
        TypeAlias,
        TypedDict,
        TypeVar,
    )

    from . import dataarrays
    from striqt.waveform.lib.typing import Array, ArrayBackend
    from .. import specs

    import xarray as xr
    import zarr
    import zarr.storage

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

    ChunksSize = int | Literal['auto'] | tuple[int, ...] | None

    if hasattr(zarr.storage, 'Store'):
        # zarr 2.x
        ZarrStore: TypeAlias = zarr.storage.Store
    elif hasattr(zarr, 'abc'):
        # zarr 3.x; the hasattr narrowing is what lets ty resolve this branch
        # under both the zarr 2 and zarr 3 stubs, and the fallback keeps the
        # alias gradual where neither layout resolves
        ZarrStore: TypeAlias = zarr.abc.store.Store
    else:
        ZarrStore: TypeAlias = Any

    ZarrFormat: TypeAlias = str | Literal[2, 3]

    # %% register.py
    Measurement: TypeAlias = Array | tuple[Array, dict[str, Any]]

    class ToleranceKws(TypedDict, total=False):
        array_backend: ArrayBackend
        input_error: float

    RM = TypeVar('RM', bound=Measurement)

    class AnalysisFunc(Protocol[P, R]):
        __name__: str

        def __call__(
            self,
            iq: Array,
            capture: specs.Capture,
            *args: P.args,
            **kwargs: P.kwargs,
        ) -> R: ...

    class WrappedAnalysis(Protocol[P, R]):
        __name__: str

        @overload
        def __call__(
            self,
            iq: Array,
            capture: specs.Capture,
            as_xarray: Literal[True] = ...,
            *args: P.args,
            **kwargs: P.kwargs,
        ) -> xr.DataArray: ...

        @overload
        def __call__(
            self,
            iq: Array,
            capture: specs.Capture,
            as_xarray: Literal[False],
            *args: P.args,
            **kwargs: P.kwargs,
        ) -> R: ...

        @overload
        def __call__(
            self,
            iq: Array,
            capture: specs.Capture,
            as_xarray: Literal['delayed'],
            *args: P.args,
            **kwargs: P.kwargs,
        ) -> dataarrays.DelayedDataArray: ...

        def __call__(
            self,
            iq: Array,
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

        # positional-only: factories cached with `specs.helpers.lru_cache_on_converted`
        # convert these arguments before the cache lookup, which needs them by position
        def __call__(self, capture: TC, spec: TM, /) -> R: ...

    CoordFuncWrapper: TypeAlias = Callable[[CoordFunc[TC, TM, R]], CoordFunc[TC, TM, R]]
