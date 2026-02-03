from __future__ import annotations as __

import typing

if typing.TYPE_CHECKING:
    from typing_extensions import TypeAlias
    from numbers import Number

    # bury this type checking in here to avoid lengthening the import time of iqwaveform
    # if cupy isn't installed
    try:
        import cupy as cp  # type: ignore
    except ModuleNotFoundError:
        import numpy as cp
    import numpy as np
    import pandas as pd
    import xarray as xr

    # union of supported array types
    ArrayType: TypeAlias = 'cp.ndarray|np.ndarray'

    # pandas types
    DataFrameType: TypeAlias = 'pd.DataFrame'
    SeriesType: TypeAlias = 'pd.Series'

    # xarray types
    DataArrayType: TypeAlias = 'xr.DataArray'
    DatasetType: TypeAlias = 'xr.Dataset'

    # Union types
    ArrayLike: TypeAlias = typing.Union[
        ArrayType, SeriesType, DataFrameType, DataArrayType
    ]

    WindowSpecType: TypeAlias = typing.Union[str, tuple[str, typing.Any], tuple[str, typing.Any, typing.Any]]
    WindowType: TypeAlias = typing.Union[ArrayType, WindowSpecType]

    _ALN = typing.TypeVar('_ALN', bound=typing.Union[ArrayLike, Number]) 
    _AL = typing.TypeVar('_AL', bound=ArrayLike)
    _AT = typing.TypeVar('_AT', bound=ArrayType)
