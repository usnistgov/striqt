from __future__ import annotations as __

import math
import numbers
import typing

import striqt.analysis as sa

from matplotlib import pyplot as plt
import numpy as np
import xarray as xr


if typing.TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.legend
    import matplotlib.figure


def _show_xarray_units_in_parentheses():
    """change xarray plots to "Label ({units})" to match IEEE style guidelines"""

    from xarray.plot.utils import _get_units_from_attrs

    code = _get_units_from_attrs.__code__
    consts = tuple([' ({})' if c == ' [{}]' else c for c in code.co_consts])
    _get_units_from_attrs.__code__ = code.replace(co_consts=consts)


_show_xarray_units_in_parentheses()


def label_by_coord(data: xr.DataArray, fmt: str, *, title_case=True, **extra_fields):
    coords = _coords_to_dicts(data, title_case=title_case)
    return [fmt.format(**(extra_fields | c)) for c in coords]


def summarize_metadata(
    source: xr.Dataset,
    capture_type: type[sa.specs.Capture],
    array: 'xr.DataArray|None' = None,
    *,
    as_str: bool = False,
):
    meta: dict = {
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


def _coords_to_dicts(da: xr.DataArray, title_case=False) -> list[dict[str, typing.Any]]:
    UNIT_PREFIXES = {'center_frequency': 'M'}

    result = [{} for _ in range(len(da.coords['port']))]
    for k, values in da.port.coords.variables.items():
        attrs = values.attrs
        if np.ndim(values) == 0:
            values = [values] * len(result)
        else:
            values = values.values.tolist()
        for d, item in zip(result, values):
            if isinstance(item, numbers.Number):
                prefix = UNIT_PREFIXES.get(k, None)
                d[k] = sa.dataarrays.describe_value(item, attrs, unit_prefix=prefix)
            elif isinstance(item, str):
                d[k] = item.replace('_', ' ')
                if title_case:
                    d[k] = d[k].title()
            else:
                d[k] = item
    return result


def _get_fig_center_x(fig: 'matplotlib.figure.Figure') -> float:
    left = math.inf
    right = -math.inf

    for ax in fig.get_axes():
        if hasattr(ax, '_colorbar'):
            continue

        pos = ax.get_position()
        if pos.x0 < left:
            left = pos.x0
        if pos.x1 > right:
            right = pos.x1

    return (left + right) / 2


def _get_fig_center_y(fig: 'matplotlib.figure.Figure') -> float:
    bot = math.inf
    top = -math.inf

    for ax in fig.get_axes():
        pos = ax.get_position()
        if pos.y0 < bot:
            bot = pos.y0
        if pos.y1 > top:
            top = pos.y1

    return (top + bot) / 2


def label_axis(
    which_axis: typing.Literal['x', 'y', 'colorbar'],
    ax_data: typing.Union[xr.DataArray, xr.Dataset],
    *,
    coord_name: typing.Optional['xr.Coordinates'] = None,
    tick_units: bool | typing.Literal['auto'] = 'auto',
    short=False,
    ax: typing.Optional['matplotlib.axes.Axes' | list['matplotlib.axes.Axes']] = None,
    fig: typing.Optional['matplotlib.figure.Figure'] = None,
):
    """apply axis labeling based on label and unit metadata in the specified dimension of `a`.

    If dimension is None, then labeling is applied from metadata in a.attrs
    """

    from . import ticker

    if which_axis not in ('x', 'y', 'colorbar'):
        raise ValueError("which_axis must be one of 'x', 'y', 'colorbar'")

    if tick_units == 'auto':
        if len(ax_data.attrs.get('units', '')) in (1, 2):
            tick_units = True
        else:
            tick_units = False

    if fig is None:
        fig = plt.gcf()
        do_suplabel = False
    else:
        do_suplabel = False if which_axis == 'colorbar' else True

    if ax is not None:
        pass
    elif fig.axes is not None:
        assert isinstance(fig.axes, list)
        if which_axis == 'colorbar':
            colorbars = [ax for ax in fig.axes if 'colorbar' in repr(ax)]
            if len(colorbars) == 0:
                raise ValueError('no colorbars found')
            else:
                ax = colorbars[0]
        else:
            if do_suplabel:
                ax = fig.axes
            else:
                ax = fig.gca()
    else:
        raise ValueError('found no axes to label')

    if isinstance(ax, (tuple, list)):
        axs = list(ax)
    else:
        axs = [ax]

    if which_axis == 'x':
        target_axs = [a.xaxis for a in axs if not hasattr(a, '_colorbar')]
    elif which_axis == 'y':
        target_axs = [a.yaxis for a in axs if not hasattr(a, '_colorbar')]
    elif which_axis == 'colorbar':
        target_axs = [a.yaxis for a in axs if hasattr(a, '_colorbar')]

    if coord_name is None:
        # label = a.attrs.get('standard_name', None)
        units = ax_data.attrs.get('units', None)
    else:
        # label = a[dimension].attrs.get('label', None)
        units = ax_data[coord_name].attrs.get('units', None)

    standard_name = ax_data.attrs.get('standard_name', None) or ax_data.name
    long_name = ax_data.attrs.get('long_name', None) or standard_name

    if short:
        desc_text = standard_name
    else:
        desc_text = long_name

    if units is not None:
        formatter = ticker.EngFormatter(unit=units, unitInTick=tick_units)

        for target in target_axs:
            target.set_major_formatter(formatter)

        ax_finite_data = ax_data.data[np.isfinite(ax_data.data)]
        if len(ax_finite_data) > 0:
            unit_suffix = formatter.get_axis_unit_suffix(
                ax_finite_data.min(), ax_finite_data.max()
            )
            label_str = f'{desc_text}{unit_suffix}'
        else:
            label_str = None
    else:
        label_str = f'{desc_text}'

    if label_str is None:
        return

    if do_suplabel:
        if which_axis == 'x':
            assert fig is not None
            fig.supxlabel(label_str, x=_get_fig_center_x(fig))
        elif which_axis == 'y':
            assert fig is not None
            fig.supylabel(label_str, y=_get_fig_center_y(fig))

        for target in target_axs:
            target.label.set_visible(False)
    else:
        for target in target_axs:
            target.set_label_text(label_str)


def label_legend(
    data: typing.Union[xr.DataArray, xr.Dataset],
    *,
    coord_name: str | None = None,
    tick_units=True,
    ax: typing.Optional['matplotlib.axes.Axes'] = None,
) -> 'matplotlib.legend.Legend':
    """apply legend labeling based on label and unit metadata in the specified dimension of `a`"""
    from . import ticker

    if ax is None:
        ax = plt.gca()

    if coord_name is None:
        obj = data
    else:
        obj = data[coord_name]
    standard_name = obj.attrs.get('standard_name', None)
    units = obj.attrs.get('units', None)
    data = obj.data

    if standard_name is not None:
        if units is not None and not tick_units:
            standard_name = f'{standard_name} ({units})'
    if units is not None:
        # TODO: implement tick_units
        formatter = ticker.EngFormatter(unit=units, unitInTick=True)
        legend_data = [formatter(v) for v in data]  # type: ignore
    else:
        legend_data = [str(v) for v in data]

    return ax.legend(legend_data, title=standard_name)


def label_selection(
    sel: typing.Union[xr.DataArray, xr.Dataset],
    ax: 'matplotlib.axes.Axes',
    attrs=True,
):
    from . import ticker

    if ax is None:
        ax = plt.gca()
    coord_names = {}
    for name, coord in sel.coords.items():
        if name == sa.dataarrays.CAPTURE_DIM:
            continue
        elif name in sel.indexes or coord.data.size == 0:
            continue

        units = coord.attrs.get('units', None)

        label = coord.attrs.get('standard_name', coord.attrs.get('name', name))
        values = np.atleast_1d(coord.data)
        if units is not None:
            formatter = ticker.EngFormatter(unit=units)
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
