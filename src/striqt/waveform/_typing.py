from __future__ import annotations as __

import typing

if typing.TYPE_CHECKING:
    from typing_extensions import TypeAlias, ParamSpec
    from typing import Any, Callable, Protocol, TypeVar, Union
    from numbers import Number
    from types import ModuleType

    # bury this type checking in here to avoid lengthening the import time of iqwaveform
    # if cupy isn't installed
    try:
        import cupy as cp  # type: ignore
    except ModuleNotFoundError:
        import numpy as cp

    import array_api_compat.numpy, array_api_compat.cupy
    import numpy as np
    import pandas as pd
    import xarray as xr

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

    XpType: TypeAlias = ModuleType | None

    _ALN = TypeVar('_ALN', bound=Union[ArrayLike, Number])
    _AL = TypeVar('_AL', bound=ArrayLike)
    _AT = TypeVar('_AT', bound=Array)
    P = ParamSpec('P')
    R = TypeVar('R')

    class CachedCallable(Protocol[P, R]):
        __wrapped__: Callable[P, R]

        def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R: ...
