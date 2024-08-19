"""typing stub definition aliases that avoid expensive imports"""

from __future__ import annotations
import typing

if typing.TYPE_CHECKING:
    from typing_extensions import TypeAlias

    import xarray as xr
    import iqwaveform

    DataArrayType: TypeAlias = 'xr.DataArray'
    DatasetType: TypeAlias = 'xr.Dataset'
    CooordinatesType: TypeAlias = 'xr.Coordinates'
    ArrayType: TypeAlias = 'iqwaveform.util.Array'