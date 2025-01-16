from __future__ import annotations

import typing

from .api.structs import Capture
from .api import util

if typing.TYPE_CHECKING:
    import matplotlib as mpl
    from matplotlib import pyplot as plt
    import numpy as np
    import xarray as xr
    import iqwaveform
else:
    xr = util.lazy_import('xarray')
    mpl = util.lazy_import('matplotlib')
    plt = util.lazy_import('matplotlib.pyplot')
    np = util.lazy_import('numpy')
    iqwaveform = util.lazy_import('iqwaveform')


def summarize_metadata(
    source: 'xr.Dataset',
    capture_type: type[Capture],
    array: 'xr.DataArray ' = None,
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


def plot_cyclic_channel_power(
    cyclic_channel_power: 'xr.DataArray',
    center_statistic='mean',
    bound_statistics=('min', 'max'),
    dB=True,
    ax=None,
    colors=None,
):
    if ax is None:
        _, ax = plt.subplots()

    time = cyclic_channel_power.cyclic_lag

    for i, detector in enumerate(cyclic_channel_power.power_detector.values):
        a = cyclic_channel_power.sel(power_detector=detector).squeeze('capture')

        if not dB:
            a = iqwaveform.dBtopow(a)

        ax.plot(
            time,
            (a.sel(cyclic_statistic=center_statistic)),
            color=f'C{i}' if colors is None else colors[i],
        )

    for i, detector in enumerate(cyclic_channel_power.power_detector.values):
        a = cyclic_channel_power.sel(power_detector=detector).squeeze('capture')

        if not dB:
            a = iqwaveform.dBtopow(a)

        ax.fill_between(
            time,
            a.sel(cyclic_statistic=bound_statistics[0]),
            a.sel(cyclic_statistic=bound_statistics[1]),
            color=f'C{i}' if colors is None else colors[i],
            alpha=0.25,
            lw=0,
            rasterized=True,
        )

    label_axis('x', cyclic_channel_power.cyclic_lag, ax=ax)
    label_axis('y', cyclic_channel_power, tick_units=False, ax=ax)
    label_legend(cyclic_channel_power.power_detector, ax=ax)


def label_axis(
    which_axis: typing.Literal['x'] | typing.Literal['y'],
    data: typing.Union['xr.DataArray', 'xr.Dataset'],
    *,
    coord_name: typing.Optional['xr.Coordinates'] = None,
    tick_units=True,
    ax: typing.Optional['mpl.axes.Ax'] = None,
):
    """apply axis labeling based on label and unit metadata in the specified dimension of `a`.

    If dimension is None, then labeling is applied from metadata in a.attrs
    """

    if ax is None:
        ax = plt.gca()

    if which_axis == 'x':
        axis = ax.xaxis
    elif which_axis == 'y':
        axis = ax.yaxis

    if coord_name is None:
        # label = a.attrs.get('standard_name', None)
        units = data.attrs.get('units', None)
    else:
        # label = a[dimension].attrs.get('label', None)
        units = data[coord_name].attrs.get('units', None)

    # if label is not None:
    #     if units is not None and not tick_units:
    #         label = f'{label} ({units})'
    #     axis.set_label_text(label)
    if units is not None and tick_units:
        axis.set_major_formatter(mpl.ticker.EngFormatter(unit=units))
        axis.set_label_text(data.standard_name or data.name)
    elif units is not None:
        axis.set_label_text(f'{data.standard_name or data.name} ({units})')
    else:
        axis.set_label_text(data.standard_name or data.name)


def label_legend(
    data: typing.Union['xr.DataArray', 'xr.Dataset'],
    *,
    coord_name: str = None,
    tick_units=True,
    ax: typing.Optional['mpl.axes._axes.Axes'] = None,
):
    """apply legend labeling based on label and unit metadata in the specified dimension of `a`"""

    if ax is None:
        ax = plt.gca()

    if coord_name is None:
        obj = data
    else:
        obj = data[coord_name]
    standard_name = obj.attrs.get('standard_name', None)
    units = obj.attrs.get('units', None)
    values = obj.values

    if standard_name is not None:
        if units is not None and not tick_units:
            standard_name = f'{standard_name} ({units})'
    if units is not None:
        # TODO: implement tick_units
        formatter = mpl.ticker.EngFormatter(unit=units)
        values = [formatter(v) for v in values]

    ax.legend(values, title=standard_name)


def label_selection(
    sel: typing.Union['xr.DataArray', 'xr.Dataset'],
    ax: typing.Optional['mpl.axes._axes.Axes'] = None,
    attrs=True,
):
    if ax is None:
        ax = plt.gca()
    coord_names = {}
    for name, coord in sel.coords.items():
        if name == 'capture' or name in sel.indexes or coord.values.size == 0:
            continue

        units = coord.attrs.get('units', None)

        label = coord.attrs.get('standard_name', coord.attrs.get('name', name))
        values = np.atleast_1d(coord.values)
        if units is not None:
            formatter = mpl.ticker.EngFormatter(unit=units)
            coord_names[label] = ', '.join([formatter(v) for v in values])
        else:
            coord_names[label] = ', '.join([str(v) for v in values])

    coord_title = ', '.join(f'{k}: {v}' for k, v in coord_names.items())

    if attrs:
        attr_title = ', '.join(
            [
                f'{k}: {v}'
                for k, v in sel.attrs.items()
                if k not in ('units', 'name', 'standard_name')
            ]
        )
    else:
        attr_title = ''

    if len(attr_title) > 0:
        attr_title = f'\nAnalysis: {attr_title}'
    ax.set_title(f'{coord_title}{attr_title}')
