from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from typing_extensions import TypeAlias

    # bury this type checking in here to avoid lengthening the import time of iqwaveform
    # if cupy isn't installed
    try:
        import cupy as cp # type: ignore
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
