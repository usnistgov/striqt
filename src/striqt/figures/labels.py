from __future__ import annotations as __

import dataclasses
import functools
import math
import typing

import striqt.analysis as sa
import striqt.waveform as sw
from striqt.analysis.lib.util import np, xr

if typing.TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure
    import matplotlib.legend


@functools.cache
def _monkeypatch_xarray_unit_labels():
    """change xarray plots to "Label ({units})" to match IEEE style guidelines"""

    from xarray.plot.utils import _get_units_from_attrs

    code = _get_units_from_attrs.__code__
    consts = tuple([' ({})' if c == ' [{}]' else c for c in code.co_consts])
    _get_units_from_attrs.__code__ = code.replace(co_consts=consts)


def label_by_coord(
    data: xr.DataArray,
    fmt: str,
    coord_or_dim: str = 'port',
    *,
    title_case: bool = True,
    **extra_fields: typing.Any,
) -> list[str]:
    """return `fmt` formatted once per value of `coord_or_dim` in `data`.

    The fields of `fmt` are the coordinates on the dimension of `coord_or_dim` and
    `extra_fields`. A coordinate field takes precedence over an extra field of the
    same name.

    Args:
        data: the array with the coordinates
        fmt: a `str.format` template; a numeric field accepts a prefix symbol spec
            (``'{center_frequency:M}'``) or a float spec (``'{center_frequency:.0f}'``)
        coord_or_dim: the coordinate or dimension that selects one entry per value
        title_case: True title-cases a lowercase string coordinate value
        **extra_fields: further fields of `fmt`

    Returns:
        one string per value of `coord_or_dim`
    """
    _monkeypatch_xarray_unit_labels()
    coords = _coords_to_dicts(data, coord_or_dim=coord_or_dim, title_case=title_case)
    return [fmt.format(**(extra_fields | c)) for c in coords]


@dataclasses.dataclass(frozen=True)
class _FacetValue:
    """a numeric coordinate value that is formatted in its units by a template field.

    The format spec ``''`` or ``'auto'`` applies `shared`; a spec that is one SI
    prefix symbol (``'M'``, ``'G'``, ``'k'``, ``'u'``) forces that prefix; any other
    spec formats the bare number through `format`.
    """

    value: typing.Union[int, float]
    """the coordinate value in `units`"""
    units: typing.Union[str, None]
    """the units string of the coordinate, or None"""
    shared: sw.units.SIPrefix
    """the prefix shared by every value of the coordinate"""

    def __format__(self, spec: str) -> str:
        if spec in ('', 'auto'):
            return sa.labels.describe_value(self.value, self.units, prefix=self.shared)
        if spec == 'u' or spec in sw.units.PREFIXES.values():
            return sa.labels.describe_value(self.value, self.units, prefix=spec)
        return format(self.value, spec)

    def __str__(self) -> str:
        return self.__format__('')


def _python_scalar(value: typing.Any) -> typing.Any:
    return value.item() if isinstance(value, np.generic) else value


def _numbers(values: typing.Iterable[typing.Any]) -> list[float]:
    """return the finite numbers among `values` and the members of its tuples."""
    out = []
    for v in values:
        if isinstance(v, tuple):
            out.extend(_numbers(v))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out.append(float(v))
    return out


def _coords_to_dicts(
    da: xr.DataArray, coord_or_dim: str = 'port', title_case: bool = False
) -> list[dict[str, typing.Any]]:
    """return the coordinates on the dimension of `coord_or_dim`, one dict per value.

    A datetime64 value is text to the millisecond. A string is written with ``_``
    replaced by a space, title-cased when `title_case` is True and the string is
    lowercase. A bool is ``'True'`` or ``'False'``. A number is a `_FacetValue` with the
    prefix shared by every value of the coordinate. Any other object, such as a tuple
    or None, is text from `striqt.analysis.labels.describe_value`.
    """
    coord = da.coords[coord_or_dim]
    result: list[dict[str, typing.Any]] = [{} for _ in range(len(coord))]
    for k, var in coord.coords.variables.items():
        values = np.atleast_1d(var.values)
        if var.ndim == 0:
            values = np.repeat(values, len(result))
        units = var.attrs.get('units', None)
        scalars = [_python_scalar(v) for v in values]
        shared = sw.units.si_prefix(_numbers(scalars), units)
        for d, raw, v in zip(result, values, scalars):
            if np.issubdtype(values.dtype, np.datetime64):
                d[k] = np.datetime_as_string(raw, unit='ms').item()
            elif isinstance(v, str):
                d[k] = v.replace('_', ' ')
                if title_case and d[k] == d[k].lower():
                    d[k] = d[k].title()
            elif isinstance(v, bool):
                d[k] = str(v)
            elif isinstance(v, (int, float)):
                d[k] = _FacetValue(v, units, shared)
            else:
                d[k] = sa.labels.describe_value(v, units, prefix=shared)
    return result


def _get_fig_center_x(fig: matplotlib.figure.Figure) -> float:
    left = math.inf
    right = -math.inf

    for ax in fig.get_axes():
        if hasattr(ax, '_colorbar'):
            continue

        pos = ax.get_position()
        left = min(left, pos.x0)
        right = max(right, pos.x1)

    return (left + right) / 2


def _get_fig_center_y(fig: matplotlib.figure.Figure) -> float:
    bot = math.inf
    top = -math.inf

    for ax in fig.get_axes():
        pos = ax.get_position()
        bot = min(bot, pos.y0)
        top = max(top, pos.y1)

    return (top + bot) / 2


def label_axis(
    which_axis: typing.Literal['x', 'y', 'colorbar'],
    ax_data: typing.Union[xr.DataArray, xr.Dataset],
    *,
    coord_name: typing.Optional[str] = None,
    tick_units: bool | typing.Literal['auto'] = 'auto',
    short: bool = False,
    ax: typing.Optional[matplotlib.axes.Axes | list[matplotlib.axes.Axes]] = None,
    fig: typing.Optional[matplotlib.figure.Figure] = None,
) -> None:
    """label an axis with the quantity and units named by the attrs of `ax_data`.

    The label text is `striqt.analysis.labels.quantity_label` of the attrs. With a
    `units` attr, each target axis gets an `EngFormatter` with one SI prefix, frozen
    from the extent of the finite values as `si_prefix` selects it for a sequence
    (the magnitude of the largest value, lowered to the span by at most one prefix).
    The prefixed units appear in
    every tick label (`tick_units` True) or once in the axis label (`tick_units`
    False), never in both. A label is set with or without finite values.

    Args:
        which_axis: ``'x'``, ``'y'``, or ``'colorbar'``; the colorbar axis is the
            first colorbar of the figure
        ax_data: the array with the attrs that name the quantity and the values that
            select the prefix
        coord_name: a coordinate of `ax_data` that supplies the attrs and values
            instead, or None
        tick_units: True places the prefixed units in each tick label and False in
            the axis label; ``'auto'`` is True for a `units` attr of 1 or 2 characters
        short: True labels with `standard_name` instead of `long_name`
        ax: the axes, or list of axes, to label; None labels the current axes, or
            every axes of `fig` when `fig` is given
        fig: when given with `ax` None, an x or y label is set once as a figure
            super-label and the per-axes labels are hidden

    Raises:
        ValueError: if `which_axis` is not one of the 3 values, if a colorbar is
            requested and the figure has none, or if there are no axes to label
    """

    import matplotlib.pyplot as plt

    from . import ticker

    _monkeypatch_xarray_unit_labels()

    if which_axis not in ('x', 'y', 'colorbar'):
        raise ValueError("which_axis must be one of 'x', 'y', 'colorbar'")

    target = ax_data if coord_name is None else ax_data[coord_name]
    m = sa.labels.measurand_from_attrs(target.attrs, getattr(target, 'name', None))

    if tick_units == 'auto':
        tick_units = len(m.units or '') in (1, 2)

    if fig is None:
        fig = plt.gcf()
        do_suplabel = False
    else:
        do_suplabel = which_axis != 'colorbar'

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

    extent: tuple[float, ...] = ()
    if m.units is not None and isinstance(target, xr.DataArray):
        values = target.to_numpy()
        finite = values[np.isfinite(values)]
        if finite.size:
            extent = (float(finite.min()), float(finite.max()))
    # the extent selects the prefix as a sequence does, so a 3.69 to 3.71 GHz axis
    # reads in MHz, and the ticks keep the prefix through a zoom
    p = sw.units.si_prefix(extent, m.units)

    if m.units is not None:
        formatter = ticker.EngFormatter(
            sa.labels.units_label(m), unit_in_tick=tick_units, exponent=p.exponent
        )
        for target_ax in target_axs:
            target_ax.set_major_formatter(formatter)

    label_str = sa.labels.quantity_label(
        m, short=short, with_units=not tick_units, prefix=p.symbol
    )

    if do_suplabel:
        if which_axis == 'x':
            assert fig is not None
            fig.supxlabel(label_str, x=_get_fig_center_x(fig))
        elif which_axis == 'y':
            assert fig is not None
            fig.supylabel(label_str, y=_get_fig_center_y(fig))

        for target_ax in target_axs:
            target_ax.label.set_visible(False)
    else:
        for target_ax in target_axs:
            target_ax.set_label_text(label_str)


def label_legend(
    data: typing.Union[xr.DataArray, xr.Dataset],
    *,
    coord_name: str | None = None,
    tick_units: bool = True,
    ax: typing.Optional[matplotlib.axes.Axes] = None,
) -> matplotlib.legend.Legend:
    """add a legend with one entry per value of `data` and a title naming the quantity.

    Numeric entries share one SI prefix. With `tick_units` True each entry ends with
    the prefixed units and the title has none; with False each entry is a bare
    mantissa and the title ends with the prefixed units in parentheses. A string value
    is written as is.

    Args:
        data: the array or dataset with the attrs and values that label the legend
        coord_name: a coordinate of `data` that supplies the attrs and values, or
            None for `data` itself
        tick_units: True puts the units in each entry, False in the title
        ax: the axes that receives the legend; None uses the current axes

    Returns:
        the `matplotlib.legend.Legend` added to `ax`
    """
    import matplotlib.pyplot as plt

    _monkeypatch_xarray_unit_labels()

    if ax is None:
        ax = plt.gca()

    obj = data if coord_name is None else data[coord_name]
    m = sa.labels.measurand_from_attrs(obj.attrs, getattr(obj, 'name', None))
    values = [_python_scalar(v) for v in np.atleast_1d(obj.to_numpy())]
    p = sw.units.si_prefix(_numbers(values), m.units)

    entries = []
    for v in values:
        if isinstance(v, str) or m.units is None:
            entries.append(str(v))
        elif tick_units or not isinstance(v, (int, float)):
            entries.append(sa.labels.describe_value(v, m, prefix=p))
        else:
            entries.append(sw.units.format_quantity(v, None, prefix=p))

    title = sa.labels.quantity_label(
        m, short=True, with_units=not tick_units, prefix=p.symbol
    )
    return ax.legend(entries, title=title)
