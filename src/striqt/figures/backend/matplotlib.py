from __future__ import annotations as __

import contextlib
import functools
import io
import sys
import typing
from pathlib import Path

import striqt.analysis as sa
import striqt.waveform as sw
from striqt.analysis.lib.util import np, xr

from .. import specs, util
from .protocol import PlotBackend

if typing.TYPE_CHECKING:
    from collections.abc import Generator, Iterator
    from multiprocessing.synchronize import RLock

    import matplotlib as mpl
    import matplotlib.axes
    import xarray.plot
    from typing_extensions import Self
    from xarray.core.types import ScaleOptions

    from .protocol import StepStyle

else:
    mpl = sw.util.lazy_import('matplotlib')

TerminalGraphics = typing.Literal['sixel', 'kitcat', 'kitty']


class _LayoutKws(typing.TypedDict):
    col: str | None
    row: str | None
    col_wrap: int | None
    figsize: list[float]


def select_mpl_backend(style: str | None, interactive: TerminalGraphics | None) -> None:
    """select a matplotlib backend, with the terminal style when `interactive` is set.

    Args:
        style: a matplotlib style name applied with the terminal style, or None
        interactive: ``'sixel'``, ``'kitcat'``, or ``'kitty'`` selects a terminal
            graphics backend; None selects ``'agg'``

    Raises:
        ValueError: if `interactive` is none of None, ``'sixel'``, ``'kitcat'``, or
            ``'kitty'``
    """

    import matplotlib as mpl
    from matplotlib import pyplot as plt

    styles = [style] if style else []

    plt.ioff()
    if interactive is None:
        mpl.use('agg')
    elif interactive == 'sixel':
        mpl.use('module://matplotlib-backend-sixel')
        styles.append('striqt.figures.terminal')
    elif interactive == 'kitcat' or interactive == 'kitty':
        mpl.use('kitcat')
        styles.append('striqt.figures.terminal')
    else:
        raise ValueError('interactive argument must be "kitcat", "sixel", or None')

    if styles:
        plt.style.use(styles)


def term_graphics_notice(interactive: str | None) -> None:
    if interactive is None:
        pass
    elif interactive == 'sixel':
        print(
            "🪧 if plots don't appear in your terminal, ensure your terminal "
            'sixel graphics protocol (wezterm, tabby, ...) '
            'and that you are disconnected from screen or tmux'
        )
    else:
        print(
            "🪧 if plots don't appear in your terminal, ensure your terminal "
            'supports the kitty graphics protocol (wezterm, iTerm2, ...) '
            'and that you are disconnected from screen or tmux'
        )


class _FakeLock:
    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        pass


@contextlib.contextmanager
def batch_term_images(
    interactive: TerminalGraphics | None,
) -> Generator[None, None, None]:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        yield
    image_payload = buffer.getvalue()

    if interactive == 'kitcat' or interactive == 'kitty':
        image_payload = '\033_Ga=d,d=a\033\\\033[H' + image_payload

    sys.stdout.write(image_payload)
    sys.stdout.flush()


class MatplotlibBackend(PlotBackend['xarray.plot.FacetGrid']):
    """a plotting backend that draws with matplotlib, one `FacetGrid` per figure.

    Each drawing method lays out the facets from `opts` and returns the grid.
    `finish` writes and shows the figure as `output_dir` and `interactive` select.
    Artists with more than 10000 values and all bands are rasterized, and lines
    drawn onto a `target` add no legend.
    """

    output_dir: Path | None
    """the directory that `finish` writes figure files in; `None` writes no file"""
    interactive: bool | str | None
    """True leaves pyplot interactive after `finish`; a `TerminalGraphics` name
    shows each figure in the terminal; False or `None` does neither"""
    lock: RLock | _FakeLock
    """serializes the terminal image writes of `finish` across processes"""
    last_grid: xarray.plot.FacetGrid | None = None
    """the grid of the most recent `finish` call, or `None` before the first"""

    def __init__(
        self,
        opts: specs.SharedPlotOptions,
        output_dir: Path | None,
        *,
        interactive: bool | str | None = None,
        lock: RLock | None = None,
    ) -> None:
        """store the options and create `output_dir` with its parents.

        Args:
            opts: the facet layout and label formats of every figure
            output_dir: the directory for the figure files; `None` writes no file
            interactive: True leaves pyplot interactive after each `finish`; a
                `TerminalGraphics` name shows each figure in the terminal; False or
                `None` does neither
            lock: serializes the terminal image writes across processes; `None`
                locks nothing
        """
        self.opts = opts
        self.output_dir = output_dir
        self.interactive = interactive

        if lock is None:
            self.lock = _FakeLock()
        else:
            self.lock = lock

        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    @functools.lru_cache  # noqa: B019 one backend instance per process
    def _coord_kws(self) -> _LayoutKws:
        """return the facet layout keywords of the xarray plot calls from `opts`.

        Without `col` or `row` in `opts`, `col` is the size-1 `_view` dimension that
        `coerce_column` adds to the data.
        """
        if self.opts.col is None and self.opts.row is None:
            col = '_view'
        else:
            col = self.opts.col

        return {
            'figsize': mpl.rcParams['figure.figsize'],
            'col_wrap': self.opts.col_wrap,
            'row': self.opts.row,
            'col': col,
        }

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
    ) -> xarray.plot.FacetGrid:
        from matplotlib import colors

        rasterized = data.size > 10000

        if vstep is None:
            _cmap = cmap
            norm = None
        else:
            levels = util.quantized_value_range(data, vmin, vmax, vstep)
            _cmap = mpl.colormaps.get_cmap(cmap).resampled(len(levels) - 1)
            norm = colors.BoundaryNorm(levels, ncolors=_cmap.N)
            vmin = vmax = None

        grid = data.plot.imshow(
            **self._coord_kws(),
            x=x,
            y=y,
            cmap=_cmap,
            norm=norm,
            vmin=vmin,
            vmax=vmax,
            rasterized=rasterized,
            interpolation='nearest',
        )
        grid = typing.cast('xarray.plot.FacetGrid', grid)

        for ax in grid.axs.flat:
            ax.grid(False)

        if grid.cbar is not None and grid.cbar.solids is not None:
            grid.cbar.solids.set_rasterized(rasterized)

        _set_axis_vars(grid, x, y)

        return grid

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
        target: xarray.plot.FacetGrid | None = None,
    ) -> xarray.plot.FacetGrid:
        kwargs: dict[str, typing.Any] = {'rasterized': data.size > 10000}
        if step is not None:
            kwargs['drawstyle'] = f'steps-{step}'
        if ylim is not None:
            kwargs['ylim'] = ylim

        if target is None:
            grid = data.plot.line(
                **self._coord_kws(),
                x=x,
                hue=hue,
                yscale=yscale,
                sharex=sharex,
                **kwargs,
            )
            grid = typing.cast('xarray.plot.FacetGrid', grid)
            _set_axis_vars(grid, x, None)
            return grid

        from xarray.plot import dataarray_plot

        for ax, sub in _facet_subsets(target, data):
            dataarray_plot.line(
                sub,
                x=x,
                hue=hue,
                ax=ax,
                yscale=yscale,
                add_legend=False,
                _labels=False,
                **kwargs,
            )

        return target

    def fill_between(
        self,
        low: xr.DataArray,
        high: xr.DataArray,
        *,
        x: str,
        hue: str | None = None,
        alpha: float = 0.25,
        step: StepStyle | None = None,
        target: xarray.plot.FacetGrid | None = None,
    ) -> xarray.plot.FacetGrid:
        if target is None:
            from xarray.plot import FacetGrid

            grid = FacetGrid(low, sharex=True, sharey=True, **self._coord_kws())
            _set_axis_vars(grid, x, None)
            for ax, name_dict in zip(grid.axs.flat, grid.name_dicts.flat):
                if name_dict is None:
                    ax.set_visible(False)
            grid.fig.tight_layout()
        else:
            grid = target

        for (ax, lo), (_, hi) in zip(
            _facet_subsets(grid, low), _facet_subsets(grid, high)
        ):
            if hue is None:
                bands = [(0, lo, hi)]
            else:
                bands = [
                    (i, lo.sel({hue: value}), hi.sel({hue: value}))
                    for i, value in enumerate(lo[hue].values)
                ]

            for i, lo_i, hi_i in bands:
                ax.fill_between(
                    lo_i[x].values,
                    lo_i.values,
                    hi_i.values,
                    color=f'C{i}',
                    alpha=alpha,
                    lw=0,
                    rasterized=True,
                    step=step,
                )

        return grid

    def finish(
        self,
        target: xarray.plot.FacetGrid,
        xticklabelunits: bool | typing.Literal['auto'] = 'auto',
    ) -> Path | None:
        """label `target` from the attrs of the drawn data and the formats of `opts`.

        The figure file is written to `output_dir` at the dpi between 120 and 300
        that resolves each sample of the drawn data, and a record is logged on the
        ``analysis`` logger at INFO. Interactive mode is turned on when `interactive`
        is True. The figure is shown in the terminal and closed when `interactive`
        names a terminal graphics protocol. `target` is kept as `last_grid`.

        Returns:
            the path of the figure file, or `None` when `output_dir` is `None`
        """
        import matplotlib.pyplot as plt

        from .. import labels

        grid = target
        data = grid.data
        x = grid._x_var
        y = grid._y_var

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

        labels.label_axis('x', data[x], fig=grid.fig, tick_units=xticklabelunits)
        if y is None:
            labels.label_axis('y', data, fig=grid.fig, tick_units=False)
        else:
            labels.label_axis('y', data[y], fig=grid.fig, tick_units=False)

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
            labels.label_axis('colorbar', grid.data, fig=grid.fig)

        if self.output_dir is not None:
            filename = labels.label_by_coord(
                data,
                self.opts.filename_fmt,
                coord_or_dim=self.opts.col or 'port',
                name=data.name,
                **data.attrs,
            )
            path = Path(self.output_dir) / filename[0]
            ydata = None if y is None else data[y]
            dpi = _select_dpi(grid, data[x], ydata, min_=120, max_=300)
            grid.fig.savefig(path, dpi=dpi)
            sa.util.get_logger('analysis').info(f"💾 '{path!s}'")
        else:
            path = None

        if self.interactive is True:
            plt.ion()
        elif self.interactive:
            with self.lock:
                grid.fig.show()  # writes the terminal image payload to stdout
                plt.close(grid.fig)

        self.last_grid = grid

        return path

    def limit_facets(
        self,
        target: xarray.plot.FacetGrid,
        low: xr.DataArray | float | None,
        high: float | None,
    ) -> None:
        if isinstance(low, xr.DataArray):
            lows = _facet_values(low, target)
        else:
            lows = [low] * target.axs.size
        for ax, lo in zip(target.axs.flat, lows):
            ax.set_xlim(lo, high)

    def mark_noise_level(
        self,
        data: xr.Dataset,
        var_name: str,
        grid: xarray.plot.FacetGrid,
        where: typing.Literal['x', 'y', 'colorbar'],
    ) -> None:
        noise = util.get_system_noise(data, var_name)

        if noise is None:
            return
        assert not isinstance(noise, (int, float))
        for pow, ax in zip(_facet_values(noise, grid), grid.axs.flat):
            if pow is None:
                continue
            if where == 'x':
                ax.axhline(pow, color='k', linestyle=':')
            elif where == 'y':
                ax.axvline(pow, color='k', linestyle=':')
            elif where == 'colorbar':
                cbar = grid.cbar
                if cbar is None:
                    raise TypeError('no colorbar on facet grid')

                if cbar.orientation == 'vertical':
                    long_axis = cbar.ax.yaxis
                else:
                    long_axis = cbar.ax.xaxis

                meanloc = noise.values.mean()
                for i, n in enumerate(noise.values.tolist()):
                    labels: list[typing.Any] = long_axis.get_ticklabels()
                    long_axis.set_ticks(
                        cbar.get_ticks().tolist() + [n],
                        labels + ['$k T B$' if i == 0 else ''],
                    )
                    label = long_axis.get_ticklabels()[-1]
                    label.set_color('#eec009')
                    label.set_y(meanloc)
                    long_axis.get_ticklines()[-1].set_markeredgecolor('#eec009')
                ind = np.argmin(np.abs(cbar.get_ticks() - meanloc))
                labels = list(long_axis.get_ticklabels())
                labels[ind] = ''
                long_axis.set_ticklabels(labels)

            else:
                raise TypeError('where must be "x", "y", or "colorbar"')


def _set_axis_vars(grid: xarray.plot.FacetGrid, x: str, y: str | None) -> None:
    """set the x and y variable names of `grid`."""
    # xarray annotates both attributes as None and assigns Hashable values to them
    grid._x_var = x  # ty: ignore[invalid-assignment]
    grid._y_var = y  # ty: ignore[invalid-assignment]


def _facet_subsets(
    grid: xarray.plot.FacetGrid, data: xr.DataArray
) -> Iterator[tuple[matplotlib.axes.Axes, xr.DataArray]]:
    """yield each named facet axis of `grid` with the subset of `data` on that facet.

    The subset is `data` indexed by the `grid.name_dicts` entry of the facet, by
    position on a dimension without coordinate values. An axis without a name (the
    padding of a wrapped grid) is skipped.

    Raises:
        ValueError: if `data` lacks a facet dimension or value of `grid`
    """
    for name_dict, ax in zip(grid.name_dicts.flat, grid.axs.flat):
        if name_dict is None:
            continue
        try:
            sub = data.loc[name_dict]
        except (KeyError, IndexError, ValueError) as ex:
            raise ValueError(
                f'{data.name!r} has no facet {name_dict!r} of the target grid'
            ) from ex
        yield ax, sub


def _facet_values(
    values: xr.DataArray, grid: xarray.plot.FacetGrid
) -> list[float | None]:
    """return one value of `values` per facet of `grid`, in `grid.axs.flat` order.

    The value of a facet is the minimum of `values` selected by the `grid.name_dicts`
    entry of the facet. A facet without a name, which is an unused axis of a wrapped
    grid, gets None. Without `name_dicts`, `values` are returned in order, one per
    facet, with None for each remaining axis.
    """
    name_dicts = getattr(grid, 'name_dicts', None)
    if name_dicts is None:
        flat = [float(v) for v in values.values.flat]
        return flat + [None] * (grid.axs.size - len(flat))

    out: list[float | None] = []
    for name_dict in name_dicts.flat:
        if name_dict is None:
            out.append(None)
            continue
        sel = values
        for key, value in name_dict.items():
            if key in sel.indexes:
                sel = sel.sel({key: value})
            elif key in sel.coords:
                sel = sel.where(sel.coords[key] == value, drop=True)
            elif key in sel.dims:
                # a dimension without coordinate values is faceted by position
                sel = sel.isel({key: int(value)})
        if sel.size == 0:
            out.append(None)
        else:
            out.append(float(sel.min()))
    return out


def _select_dpi(
    grid: xarray.plot.FacetGrid,
    x_data: xr.DataArray,
    y_data: xr.DataArray | None,
    min_: float = 0,
    max_: float = 300,
) -> float:
    """return the dpi in [`min_`, `max_`] that resolves one sample per pixel per facet.

    The dpi of a facet is the sample count of `x_data` per inch of axis width or,
    when `y_data` is set and larger, the sample count of `y_data` per inch of axis
    height. The result is the largest facet dpi, clipped to [`min_`, `max_`].
    """
    dpi = min_
    scale_inv = grid.fig.dpi_scale_trans.inverted()
    for ax in grid.fig.axes:
        if grid.cbar is not None and ax is grid.cbar.ax:
            continue
        bbox = ax.get_window_extent().transformed(scale_inv)
        width, height = bbox.width, bbox.height
        candidates = [dpi, x_data.size / width]
        if y_data is not None:
            candidates.append(y_data.size / height)
        dpi = max(candidates)

    return min(max(dpi, min_), max_)
