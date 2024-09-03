from __future__ import annotations
import xarray as xr
import matplotlib as mpl
from typing import Optional
from .structs import Capture


def summarize_metadata(
    source: xr.Dataset,
    capture_type: type[Capture],
    array: xr.DataArray = None,
    *,
    as_str: bool = False,
):
    meta = {
        k: source.attrs[k] for k in capture_type.__struct_fields__ if k in source.attrs
    }

    if array is not None:
        meta.update(
            {
                name: coord.item()
                for name, coord in array.coords.items()
                if coord.size == 1
            }
        )

    if as_str:
        return '\n'.join([f'{k}: {v}' for k, v in meta.items()])
    else:
        return meta


def label_axis(
    axis: mpl.axis.Axis,
    a: xr.DataArray | xr.Dataset,
    dimension: Optional[str] = None,
    tick_units=True,
):
    """apply axis labeling based on label and unit metadata in the specified dimension of `a`.

    If dimension is None, then labeling is applied from metadata in a.attrs
    """

    if dimension is None:
        # label = a.attrs.get('label', None)
        units = a.attrs.get('units', None)
    else:
        # label = a[dimension].attrs.get('label', None)
        units = a[dimension].attrs.get('units', None)

    # if label is not None:
    #     if units is not None and not tick_units:
    #         label = f'{label} ({units})'
    #     axis.set_label_text(label)
    if units is not None and tick_units:
        axis.set_major_formatter(mpl.ticker.EngFormatter(unit=units))


def label_legend(
    ax: mpl.axes._axes.Axes,
    a: xr.DataArray | xr.Dataset,
    dimension: str,
    tick_units=True,
):
    """apply legend labeling based on label and unit metadata in the specified dimension of `a`"""

    standard_name = a[dimension].attrs.get('standard_name', None)
    units = a[dimension].attrs.get('units', None)
    values = a[dimension].values

    if standard_name is not None:
        if units is not None and not tick_units:
            standard_name = f'{standard_name} ({units})'
    if units is not None:
        # TODO: implement tick_units
        formatter = mpl.ticker.EngFormatter(unit=units)
        values = [formatter(v) for v in values]

    ax.legend(values, title=standard_name)
