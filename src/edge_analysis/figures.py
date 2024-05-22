from __future__ import annotations
import xarray as xr
import matplotlib as mpl

def label_axis(axis: mpl.axis.Axis, a: xr.DataArray|xr.Dataset, dimension: str=None, tick_units=True):
    """apply axis labeling based on label and unit metadata in the specified dimension of `a`.
    
    If dimension is None, then labeling is applied from metadata in a.attrs
    """

    if dimension is None:
        label = a.attrs.get('label', None)
        units = a.attrs.get('units', None)
    else:
        label = a[dimension].attrs.get('label', None)
        units = a[dimension].attrs.get('units', None)

    if label is not None:
        if units is not None and not tick_units:
            label = f'{label} ({units})'
        axis.set_label_text(label)
    if units is not None and tick_units:
        axis.set_major_formatter(mpl.ticker.EngFormatter(unit=units))

def label_legend(ax: mpl.axes._axes.Axes, a: xr.DataArray|xr.Dataset, dimension: str, tick_units=True):
    """apply legend labeling based on label and unit metadata in the specified dimension of `a`"""

    label = a[dimension].attrs.get('label', None)
    units = a[dimension].attrs.get('units', None)
    values = a[dimension].values

    if label is not None:
        if units is not None and not tick_units:
            label = f'{label} ({units})'
    if units is not None:
        # TODO: implement tick_units
        pass

    ax.legend(values, title=label)
