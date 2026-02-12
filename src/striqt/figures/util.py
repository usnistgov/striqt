from __future__ import annotations as __

import math
import numbers
import typing

import striqt.waveform as sw
import striqt.analysis as sa

from matplotlib import pyplot as plt
from matplotlib import colors, ticker
import numpy as np
import xarray as xr


if typing.TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.legend
    import matplotlib.figure


class EngFormatter(ticker.EngFormatter):
    """Behave as mpl.ticker.EngFormatter, but also support an
    invariant the unit suffix across the entire axis"""

    _usetex: bool
    _useMathText: bool

    def __init__(
        self,
        unit='',
        unitInTick=True,
        places=None,
        sep=None,
        *,
        usetex=None,
        useMathText=None,
        **kws,
    ):
        self.unitInTick = unitInTick

        if sep is not None:
            pass
        if unit is None or (len(unit) == 1 and not unit.isalnum()):
            sep = ''
        else:
            sep = ' '

        super().__init__(
            unit,
            places,
            sep,
            usetex=usetex,
            useMathText=useMathText,
            **kws,
        )

    def format_data(self, value):
        sign = 1
        fmt = 'g' if self.places is None else f'.{self.places:d}f'

        if value < 0:
            sign = -1
            value = -value

        elif value == float('inf'):
            return '∞'

        elif value == float('-inf'):
            return '-∞'

        if value != 0:
            pow10 = int(math.floor(math.log10(value) / 3) * 3)
        else:
            pow10 = 0
            # Force value to zero, to avoid inconsistencies like
            # format_eng(-0) = "0" and format_eng(0.0) = "0"
            # but format_eng(-0.0) = "-0.0"
            value = 0.0

        if self.unitInTick:
            pow10 = np.clip(pow10, min(self.ENG_PREFIXES), max(self.ENG_PREFIXES))
        else:
            pow10 = self.orderOfMagnitude

        mant = sign * value / (10.0**pow10)
        # Taking care of the cases like 999.9..., which may be rounded to 1000
        # instead of 1 k.  Beware of the corner case of values that are beyond
        # the range of SI prefixes (i.e. > 'Y').
        if (
            abs(float(format(mant, fmt))) >= 1000
            and pow10 < max(self.ENG_PREFIXES)
            and self.unitInTick
        ):
            mant /= 1000
            pow10 += 3

        unit_prefix = self.ENG_PREFIXES[int(pow10)]
        if self.unitInTick and (self.unit or unit_prefix):
            suffix = f'{self.sep}{unit_prefix}{self.unit}'
        else:
            suffix = ''

        if self._usetex or self._useMathText:
            return f'${mant:{fmt}}${suffix}'
        else:
            return f'{mant:{fmt}}{suffix}'

    def get_axis_unit_suffix(self, vmin, vmax):
        if self.unitInTick:
            return ''

        orderOfMagnitude = math.floor(math.log(vmax - vmin, 1000)) * 3
        unit_prefix = self.ENG_PREFIXES[int(orderOfMagnitude)]
        return f'{self.sep}({unit_prefix}{self.unit})'


def label_by_coord(data: xr.DataArray, fmt: str, *, title_case=True, **extra_fields):
    coords = _capture_to_dicts(data.capture, title_case=title_case)
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


def _capture_to_dicts(
    capture: xr.DataArray, title_case=False
) -> list[dict[str, typing.Any]]:
    UNIT_PREFIXES = {'center_frequency': 'M'}

    if capture.ndim > 0:
        return [_capture_to_dicts(c, title_case)[0] for c in capture]

    coords = capture.coords.to_dataset().to_dict('list')['coords']
    d = {}
    for k, v in coords.items():
        if isinstance(v['data'], numbers.Number):
            prefix = UNIT_PREFIXES.get(k, None)
            d[k] = sa.dataarrays.describe_value(
                v['data'], v['attrs'], unit_prefix=prefix
            )
        elif isinstance(v['data'], str):
            d[k] = v['data'].replace('_', ' ')
            if title_case:
                d[k] = d[k].title()
        else:
            d[k] = v['data']

    return [d]


def _color_levels(data: xr.DataArray, vmin, vmax, vstep):
    if vmin is None:
        vmin = float(data.min())

    if vmax is None:
        vmax = float(data.max())

    vmax_edge = math.ceil(vmax / vstep) * vstep
    vmin_edge = math.floor(vmin / vstep) * vstep

    n = round((vmax_edge - vmin_edge) / vstep) + 1

    levels = np.linspace(vmin_edge, vmax_edge, n).tolist()

    return levels


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


def get_system_noise(data: xr.Dataset, var_name: str, margin=None):
    da = data[var_name]
    if 'system_noise' not in data:
        return None

    if 'noise_bandwidth' in da.attrs:
        B = da.attrs['noise_bandwidth']
    elif da.attrs.get('units', None) != 'dBm':
        return None
    elif 'analysis_bandwidth' in data.coords:
        B = data.coords['analysis_bandwidth']
    else:
        return None

    noise = data.system_noise + sw.powtodB(B)

    if margin is None:
        return noise
    else:
        return float(noise.min() - margin)


def select_histogram_bins(data, var_name, bin_name, pad_low=12, pad_hi=3):
    xmin = get_system_noise(data, bin_name, margin=pad_low)
    sub = data[var_name]
    xmax = sub.cumsum(bin_name).idxmax(bin_name).max() + pad_hi
    return sub.sel({bin_name: slice(xmin, xmax)})


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
        formatter = EngFormatter(unit=units, unitInTick=tick_units)

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
        formatter = EngFormatter(unit=units, unitInTick=True)
        legend_data = [formatter(v) for v in data]  # type: ignore
    else:
        legend_data = [str(v) for v in data]

    return ax.legend(legend_data, title=standard_name)


def label_selection(
    sel: typing.Union[xr.DataArray, xr.Dataset],
    ax: 'matplotlib.axes.Axes',
    attrs=True,
):
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
            formatter = EngFormatter(unit=units)
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


class _HeatmapKwArgs(typing.TypedDict):
    norm: colors.BoundaryNorm | None
    cmap: colors.Colormap | str


class _BasePlotKws(typing.TypedDict):
    x: str
    y: typing.NotRequired[str | None]
    hue: typing.NotRequired[str]


class _PlotKwArgs(_BasePlotKws):
    col: str
    col_wrap: typing.NotRequired[int | None]
    row: typing.NotRequired[str | None]
    figsize: typing.NotRequired[list[float]]


def quantize_heatmap_kws(
    data,
    cmap: str = 'cubehelix',
    vmin: float | None = None,
    vmax: float | None = None,
    vstep: float | None = None,
) -> _HeatmapKwArgs:
    if vstep is None:
        norm = None
        _cmap = cmap
    else:
        levels = _color_levels(data, vmin, vmax, vstep)
        _cmap = plt.get_cmap(cmap, len(levels) - 1)
        norm = colors.BoundaryNorm(levels, ncolors=_cmap.N)

    return _HeatmapKwArgs(norm=norm, cmap=_cmap)
