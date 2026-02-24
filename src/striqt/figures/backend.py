from __future__ import annotations as __
from pathlib import Path
import typing

from . import specs, util

import striqt.analysis as sa
import striqt.waveform as sw

if typing.TYPE_CHECKING:
    import matplotlib as mpl
    import xarray as xr
    import xarray.plot
    import xarray.core.types

    _T = typing.TypeVar('_T', bound=xr.DataArray)

else:
    mpl = sw.util.lazy_import('matplotlib')
    xr = sw.util.lazy_import('xarray')
    np = sw.util.lazy_import('numpy')


def _select_dpi(grid, x_data, y_data: 'xr.DataArray | None', min_=0, max_=300):
    dpi = min_
    scale_inv = grid.fig.dpi_scale_trans.inverted()
    for ax in grid.fig.axes:
        if grid.cbar is not None and ax is grid.cbar.ax:
            continue
        bbox = ax.get_window_extent().transformed(scale_inv)
        width, height = bbox.width, bbox.height
        x_dpi = x_data.size / width
        dpi = max([dpi, x_dpi] + [] if y_data is None else [y_data.size / height])

    return min(dpi, max_)


def coerce_column(data: '_T', plotter: 'PlotBackend') -> '_T':
    """xarray plot routines only return FacetGrid objects"""
    opts = plotter.opts
    if opts.col is None and opts.row is None:
        return data.extend_dims(_view=[''])
    else:
        return data


class _DimKws(typing.TypedDict):
    x: str
    y: typing.NotRequired[str]
    hue: typing.NotRequired[str]


class _LayoutKwArgs(_DimKws):
    col: str | None
    col_wrap: typing.NotRequired[int | None]
    row: typing.NotRequired[str | None]
    figsize: typing.NotRequired[list[float]]


def plot_cyclic_channel_power(
    cyclic_channel_power: xr.DataArray,
    center_statistic='mean',
    bound_statistics=('min', 'max'),
    dB=True,
    ax=None,
    colors=None,
    steps=True,
    plot_kws={},
):
    from . import labels
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()

    time = cyclic_channel_power.cyclic_lag

    fill_kws = {}
    if steps:
        plot_kws = plot_kws | {'drawstyle': 'steps-post'}
        fill_kws['step'] = 'post'

    if sa.dataarrays.CAPTURE_DIM in cyclic_channel_power.dims:
        cyclic_channel_power = cyclic_channel_power.squeeze(sa.dataarrays.CAPTURE_DIM)

    if not dB:
        cyclic_channel_power = sw.dBtopow(cyclic_channel_power)

    for i, detector in enumerate(cyclic_channel_power.power_detector.data):
        a = cyclic_channel_power.sel(power_detector=detector)

        ax.plot(
            time,
            (a.sel(cyclic_statistic=center_statistic)),
            color=f'C{i}' if colors is None else colors[i],
            **plot_kws,
        )

    for i, detector in enumerate(cyclic_channel_power.power_detector.data):
        a = cyclic_channel_power.sel(power_detector=detector)

        ax.fill_between(
            time,
            a.sel(cyclic_statistic=bound_statistics[0]),
            a.sel(cyclic_statistic=bound_statistics[1]),
            color=f'C{i}' if colors is None else colors[i],
            alpha=0.25,
            lw=0,
            rasterized=True,
            **fill_kws,
        )

    labels.label_axis('x', cyclic_channel_power.cyclic_lag, ax=ax)
    labels.label_axis('y', cyclic_channel_power, tick_units=False, ax=ax)
    labels.label_legend(cyclic_channel_power.power_detector, ax=ax)


class PlotBackend:
    opts: specs.SharedPlotOptions

    def __init__(
        self,
        opts: specs.SharedPlotOptions,
        output_dir: Path | None,
        *,
        interactive: bool = False,
    ):
        self.opts = opts
        self.output_dir = output_dir
        self.interactive = interactive

        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    @sw.util.lru_cache()
    def _coord_kws(self, **base: typing.Unpack[_DimKws]) -> _LayoutKwArgs:
        if self.opts.col is None and self.opts.row is None:
            col = '_view'
        else:
            col = self.opts.col

        return {
            **base,
            'figsize': mpl.rcParams['figure.figsize'],
            'col_wrap': self.opts.col_wrap,
            'row': self.opts.row,
            'col': col,
        }

    def heatmap(
        self,
        data: 'xr.DataArray',
        *,
        cmap: str = 'cubehelix',
        vmin: float | None = None,
        vmax: float | None = None,
        vstep: float | None = None,
        **kwargs: typing.Unpack[_DimKws],
    ):
        from matplotlib import colors

        coords = self._coord_kws(**kwargs)
        rasterized = data.size > 10000

        if vstep is None:
            _cmap = cmap
            norm = None
        else:
            levels = util.quantized_value_range(data, vmin, vmax, vstep)
            _cmap = mpl.colormaps.get_cmap(cmap).resampled(len(levels) - 1)
            norm = colors.BoundaryNorm(levels, ncolors=_cmap.N)

        grid = data.plot.imshow(
            **coords,
            cmap=_cmap,
            norm=norm,
            rasterized=rasterized,
            interpolation='nearest',
        )

        for ax in grid.axs.flat:
            ax.grid(False)

        if grid.cbar is not None and grid.cbar.solids is not None:
            grid.cbar.solids.set_rasterized(rasterized)

        x = grid._x_var = kwargs['x']  # type: ignore
        y = grid._y_var = kwargs.get('y', None)  # type: ignore
        ydata = data if y is None else data[y]
        grid._dpi = _select_dpi(grid, data[x], ydata, 100, 300)  # type: ignore

        return grid

    def line(
        self,
        data: 'xr.DataArray',
        yscale: 'xarray.core.types.ScaleOptions' = 'linear',
        **kwargs: typing.Unpack[_DimKws],
    ):
        coords = self._coord_kws(**kwargs)
        rasterized = data.size > 10000

        grid = data.plot.line(**coords, yscale=yscale, rasterized=rasterized)

        grid._x_var = kwargs['x']  # type: ignore
        grid._y_var = None
        grid._dpi = _select_dpi(grid, data[grid._x_var], None, min_=100, max_=300)  # type: ignore

        return grid

    def finish(
        self,
        grid: 'xarray.plot.FacetGrid',
        # coords: _LayoutKwArgs,
        xticklabelunits: bool | typing.Literal['auto'] = 'auto',
    ):
        from . import labels
        import matplotlib.pyplot as plt

        data = grid.data
        x = grid._x_var
        y = grid._y_var
        dpi: int = getattr(grid, '_dpi', 150)

        if self.opts.suptitle_fmt is not None:
            suptitle = labels.label_by_coord(
                data,
                self.opts.suptitle_fmt,
                title_case=True,
                coord_or_dim=self.opts.col or 'port',
                name=data.name,
                meta=data.attrs,
            )[0]
            grid.fig.suptitle(
                suptitle, x=labels._get_fig_center_x(grid.fig), ha='center', va='bottom'
            )

        # axes
        labels.label_axis('x', data[x], fig=grid.fig, tick_units=xticklabelunits)
        if y is None:
            labels.label_axis('y', data, fig=grid.fig, tick_units=False)
        else:
            labels.label_axis('y', data[y], fig=grid.fig, tick_units=False)

        # row/col titles
        grid.set_titles('{value}')
        if self.opts.col is None and self.opts.col_label_format is not None:
            for ax in grid.fig.axes:
                if grid.cbar is None or ax is not grid.cbar.ax:
                    ax.title.set_visible(False)
        else:
            col_text = labels.label_by_coord(
                data,
                self.opts.col_label_format,
                coord_or_dim=self.opts.col or 'port',
                title_case=True,
                name=data.name,
                **data.attrs,
            )
            for s, ax in zip(col_text, grid.axs.flat):
                ax.set_title(s)
        if self.opts.row is not None and self.opts.row_label_format is not None:
            row_text = labels.label_by_coord(
                data,
                self.opts.row_label_format,
                title_case=True,
                coord_or_dim=self.opts.col or 'port',
                name=data.name,
                **data.attrs,
            )
            for s, label in zip(row_text, grid.row_labels):
                assert label is not None
                label.set_text(s)

        if grid.cbar is not None:
            ylabel = grid.cbar.ax.get_ylabel().replace('\n', ' ')
            grid.cbar.ax.set_ylabel(ylabel, rotation=90)

        if self.output_dir is not None:
            filename = labels.label_by_coord(
                data,
                self.opts.filename_fmt,
                coord_or_dim=self.opts.col or 'port',
                name=data.name,
                **data.attrs,
            )
            path = Path(self.output_dir) / filename[0]
            grid.fig.savefig(path, dpi=dpi)

        if not self.interactive:
            plt.close(grid.fig)

    def mark_noise_level(
        self,
        data: 'xr.Dataset',
        var_name: str,
        grid: 'xarray.plot.FacetGrid',
        horizontal=True,
    ):
        noise = util.get_system_noise(data, var_name)

        if noise is None:
            return
        assert not isinstance(noise, (int, float))
        for pow, ax in zip(noise.values.flat, grid.axs.flat):
            if horizontal:
                ax.axhline(pow, color='k', linestyle=':')
            else:
                ax.axvline(pow, color='k', linestyle=':')
