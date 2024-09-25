"""typing stub definition aliases that avoid expensive imports"""

from __future__ import annotations
import typing
from . import util

if typing.TYPE_CHECKING:
    import xarray as xr
    import iqwaveform
    import pandas as pd
else:
    pd = util.lazy_import('pandas')
    iqwaveform = util.lazy_import('iqwaveform')
    xr = util.lazy_import('xarray')

DataArrayType: typing.TypeAlias = 'xr.DataArray'
DatasetType: typing.TypeAlias = 'xr.Dataset'
CooordinatesType: typing.TypeAlias = 'xr.Coordinates'
TimestampType: typing.TypeAlias = 'pd.Timestamp'
ArrayType: typing.TypeAlias = 'iqwaveform.util.Array'
StatisticListType: typing.TypeAlias = tuple[typing.Union[str, float], ...]
