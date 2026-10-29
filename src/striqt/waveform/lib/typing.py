from __future__ import annotations as __

import typing
from typing import Literal, Union

CellSSBIndexes = Union[
    Literal['auto', 'a', 'b', 'c', 'd', 'e', 'f', 'g'], tuple[int, ...]
]

if typing.TYPE_CHECKING:
    from numbers import Number
    from types import ModuleType
    from typing import Any, Callable, Optional, Protocol, TypeVar, Union

    from typing_extensions import ParamSpec, TypeAlias, TypeIs

    # the cupy import is confined to TYPE_CHECKING: at runtime, a failed cupy import
    # would lengthen the import of striqt.waveform
    try:
        import cupy as cp  # ty: ignore[unresolved-import]

        TypeIsCupy = TypeIs[cp.ndarray]
    except ModuleNotFoundError:
        import numpy as cp

    import array_api_compat.cupy
    import array_api_compat.numpy
    import numpy as np
    import pandas as pd
    import xarray as xr
    from numpy.typing import DTypeLike
    from xarray.core.types import Dims

    # union of supported array types
    Array: TypeAlias = 'cp.ndarray|np.ndarray'

    # pandas types
    DataFrameType: TypeAlias = 'pd.DataFrame'
    SeriesType: TypeAlias = 'pd.Series'

    # xarray types
    DataArrayType: TypeAlias = 'xr.DataArray'
    DatasetType: TypeAlias = 'xr.Dataset'

    ArrayLike: TypeAlias = Union[Array, SeriesType, DataFrameType, DataArrayType]

    ShiftType = typing.Literal['left', 'right', 'none', False]

    WindowSpecType: TypeAlias = Union[str, tuple[str, Any], tuple[str, Any, Any]]
    WindowType: TypeAlias = Union[Array, WindowSpecType]

    XpType: TypeAlias = Optional[ModuleType]
    ArrayBackend: TypeAlias = typing.Literal['numpy', 'cupy']

    _ALN = TypeVar('_ALN', bound=Union[ArrayLike, Number])
    _AL = TypeVar('_AL', bound=ArrayLike)
    _AT = TypeVar('_AT', bound=Array)
    P = ParamSpec('P')
    R = TypeVar('R')

    class LRUWrapped(Protocol[P, R]):
        __name__: str
        __wrapped__: Callable[P, R]

        def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R: ...
        def cache_clear(self) -> None: ...
        def cache_info(self) -> Any: ...
