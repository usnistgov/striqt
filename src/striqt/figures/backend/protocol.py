"""the drawing contract that the plotting backends implement.

The grid type `T` is the object a backend returns from each drawing method and
accepts back as `target`: a matplotlib facet grid or a plotly figure.
"""

from __future__ import annotations as __

import typing

from striqt.analysis.lib.util import xr

from .. import specs

if typing.TYPE_CHECKING:
    from pathlib import Path

    from xarray.core.types import ScaleOptions

    _T = typing.TypeVar('_T', bound=xr.DataArray)

T = typing.TypeVar('T')

StepStyle = typing.Literal['pre', 'post', 'mid']
"""where the step of a stepped line falls between 2 samples: ``'pre'`` steps at
the left sample, ``'post'`` at the right sample, and ``'mid'`` halfway"""


class PlotBackend(typing.Protocol[T]):
    """the drawing methods of a plotting backend with grid type `T`.

    An implementation lays out one grid of facets per drawing call from the `col`,
    `row`, and `col_wrap` fields of `opts`, and returns the grid. A later call with
    that grid as `target` draws onto the same facets, and `finish` labels the grid.
    """

    opts: specs.SharedPlotOptions
    """the facet layout and label formats shared by every figure"""

    def line(
        self,
        data: xr.DataArray,
        *,
        x: str,
        hue: str | None = None,
        yscale: ScaleOptions = 'linear',
        sharex: bool = True,
        ylim: tuple[float | None, float | None] | None = None,
        step: StepStyle | None = None,
        target: T | None = None,
    ) -> T:
        """draw one line per `hue` value of `data` on each facet.

        Args:
            data: the values on the y axis, with dimensions `x`, `hue` (when set), and
                the facet dimensions of `opts`
            x: the dimension or coordinate of `data` on the x axis
            hue: the dimension of `data` that selects one line per value, with a
                legend; `None` draws one line per facet
            yscale: the y axis scale, ``'linear'`` or ``'log'``
            sharex: whether the facets share x limits
            ylim: the y limits; `None` in either position leaves that limit to the
                data extent
            step: the placement of the step between samples when each value is held
                until the next sample; `None` connects the samples
            target: `None` builds a new grid; the lines are drawn onto a grid returned
                by an earlier call of this backend, with facet coordinates equal to
                those of `data`

        Returns:
            the new grid, or `target` when given

        Raises:
            ValueError: if `target` is a grid with facets that `data` does not have
        """
        ...

    def fill_between(
        self,
        low: xr.DataArray,
        high: xr.DataArray,
        *,
        x: str,
        hue: str | None = None,
        alpha: float = 0.25,
        step: StepStyle | None = None,
        target: T | None = None,
    ) -> T:
        """shade the band from `low` to `high` per `hue` value on each facet.

        A band and a line drawn with the same `hue` value on one grid have the same
        color.

        Args:
            low: the lower edge of the band, with dimensions `x`, `hue` (when set),
                and the facet dimensions of `opts`
            high: the upper edge of the band, with the dimensions of `low`
            x: the dimension or coordinate of `low` and `high` on the x axis
            hue: the dimension that selects one band per value; `None` shades one
                band per facet
            alpha: the opacity of the band, in [0, 1]
            step: the placement of the step between samples when each edge is held
                until the next sample; `None` connects the samples
            target: `None` builds a new grid; the bands are drawn onto a grid returned
                by an earlier call of this backend, with facet coordinates equal to
                those of `low`

        Returns:
            the new grid, or `target` when given

        Raises:
            ValueError: if `target` is a grid with facets that `low` does not have
        """
        ...

    def heatmap(
        self,
        data: xr.DataArray,
        *,
        x: str,
        y: str,
        cmap: str = 'cubehelix',
        vmin: float | None = None,
        vmax: float | None = None,
        vstep: float | None = None,
    ) -> T:
        """draw `data` as a color image with a colorbar on each facet.

        Args:
            data: the values mapped to color, with dimensions `x`, `y`, and the facet
                dimensions of `opts`
            x: the dimension or coordinate of `data` on the x axis
            y: the dimension or coordinate of `data` on the y axis
            cmap: a matplotlib colormap name
            vmin: the value at the low end of the colormap; `None` selects the data
                minimum
            vmax: the value at the high end of the colormap; `None` selects the data
                maximum
            vstep: the width (in the units of `data`) of the color bands that
                quantize the colormap between `vmin` rounded down and `vmax` rounded
                up to multiples of `vstep`; `None` maps colors continuously

        Returns:
            a new grid
        """
        ...

    def finish(
        self,
        target: T,
        xticklabelunits: bool | typing.Literal['auto'] = 'auto',
    ) -> Path | None:
        """label `target` from the attrs of the drawn data and the formats of `opts`.

        Args:
            target: the grid returned by a drawing method of this backend
            xticklabelunits: whether each x tick label ends with the units;
                ``'auto'`` leaves the choice to the backend

        Returns:
            the path of the saved figure file, or `None` when no file is written
        """
        ...

    def limit_facets(
        self,
        target: T,
        low: xr.DataArray | float | None,
        high: float | None,
    ) -> None:
        """set the x limits of each facet of `target` to (`low`, `high`).

        Args:
            target: the grid to limit
            low: the lower limit, or a DataArray with one value per facet; `None`
                leaves the lower limit to the data extent
            high: the upper limit; `None` leaves the upper limit to the data extent
        """
        ...

    def mark_noise_level(
        self,
        data: xr.Dataset,
        var_name: str,
        grid: T,
        where: typing.Literal['x', 'y', 'colorbar'],
    ) -> None:
        """draw the system noise level (in dBm) for `var_name` on each facet of `grid`.

        The level is `system_noise` in the noise bandwidth of `var_name`, drawn on
        each facet at the entry for the coordinate value of that facet. Nothing is
        drawn when `data` has no `system_noise` variable or `var_name` has no noise
        bandwidth.

        Args:
            data: the dataset with the `system_noise` variable (dBm/Hz) and `var_name`
            var_name: the data variable or power-bin coordinate that sets the noise
                bandwidth
            grid: the grid to draw on
            where: ``'x'`` draws a horizontal line for power on the y axis, ``'y'`` a
                vertical line for power on the x axis, and ``'colorbar'`` a mark on
                the colorbar of a heatmap

        Raises:
            TypeError: if `where` is ``'colorbar'`` and `grid` has no colorbar, or if
                `where` is none of the 3 values
        """
        ...


def coerce_column(data: _T, plotter: PlotBackend[typing.Any]) -> _T:
    """add the size-1 facet dimension `_view` to `data` of an unfaceted plot.

    A plot is unfaceted when `plotter.opts` sets neither `col` nor `row`. With `col`
    or `row` set, `data` is returned unchanged.
    """
    opts = plotter.opts
    if opts.col is None and opts.row is None:
        return data.expand_dims(_view=[''])
    else:
        return data
