from __future__ import annotations
import typing

if typing.TYPE_CHECKING:
    from typing_extensions import TypeAlias

    # bury this type checking in here to avoid lengthening the import time of iqwaveform
    # if cupy isn't installed
    try:
        import cupy as cp
    except ModuleNotFoundError:
        import numpy as cp
    import numpy as np

    import pandas as pd
    from matplotlib import axes
    import matplotlib.ticker
    import xarray as xr

    # union of supported array types
    ArrayType: TypeAlias = 'cp.ndarray|np.ndarray'

    # pandas types
    DataFrameType: TypeAlias = 'pd.DataFrame'
    SeriesType: TypeAlias = 'pd.Series'
    IndexType: TypeAlias = 'pd.Index'

    # xarray types
    DataArrayType: TypeAlias = 'xr.DataArray'
    DatasetType: TypeAlias = 'xr.Dataset'

    # Matplotlib types
    AxisType: TypeAlias = 'axes.Axes'
    LocatorType: TypeAlias = 'matplotlib.ticker.MaxNLocator'

    # Union types
    ArrayLike: TypeAlias = typing.Union[ArrayType, SeriesType, DataFrameType, DataArrayType]
