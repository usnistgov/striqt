"""typing stub definition aliases that avoid expensive imports"""

from __future__ import annotations
import typing

if typing.TYPE_CHECKING:
    from typing_extensions import TypeAlias

    import xarray as xr
    import iqwaveform
    import pandas as pd

DataArrayType: TypeAlias = 'xr.DataArray'
DatasetType: TypeAlias = 'xr.Dataset'
CooordinatesType: TypeAlias = 'xr.Coordinates'
TimestampType: TypeAlias = 'pd.Timestamp'
ArrayType: TypeAlias = 'iqwaveform.util.Array'
StatisticListType: TypeAlias = tuple[typing.Union[str, float], ...]
