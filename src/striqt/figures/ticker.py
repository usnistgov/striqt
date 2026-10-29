from __future__ import annotations as __

import math
import typing

import numpy as np
import pandas as pd
from matplotlib import axis as _axis, scale as _scale, ticker as _ticker

import striqt.waveform as sw

from ..waveform.lib.util import lru_cache


def round_places(x, digits):
    scale = 10 ** (np.ceil(np.log10(x)))
    return np.round(x / scale, digits) * scale


def is_decade(x, **kwargs):
    y = np.log10(x)
    return np.isclose(y, np.round(y), **kwargs)


@lru_cache()
def _log_tick_range(vlo, vhi, count, subs=(1.0,)):
    """return up to `count` tick values in ``[vlo, vhi)`` on decades scaled by `subs`.

    The ticks are those of `matplotlib.ticker.LogLocator`, and are not evenly spaced.
    """
    locator = _ticker.LogLocator(base=10.0, subs=subs, numticks=count)
    ticks = locator.tick_values(vlo, vhi)
    return ticks[(ticks >= vlo) & (ticks < vhi)]


@lru_cache()
def _linear_tick_range(vlo, vhi, count, steps=(1.0,)):
    """return up to `count` tick values in ``[vlo, vhi)`` on round multiples of `steps`.

    The ticks are those of `matplotlib.ticker.MaxNLocator`.
    """
    locator = _ticker.MaxNLocator(nbins=count, steps=steps)
    ticks = locator.tick_values(vlo, vhi)
    return ticks[(ticks >= vlo) & (ticks < vhi)]


@lru_cache()
def _prune_ticks(ticks: tuple, count: int, prefer: tuple = ()) -> np.ndarray:
    """remove the most closely spaced ticks from `ticks` until `count` remain.

    Args:
        ticks: the candidate tick values
        count: the number of ticks to keep
        prefer: tick values exempt from removal, in order of preference; only the
            first `count` entries are exempt

    Returns:
        the remaining ticks in the order of `ticks`, as a 1-D array of `count`
        values, or more when every remaining tick is exempt
    """

    x = np.array(ticks).copy()
    prefer = tuple(prefer)
    while count < len(x):
        diffs = np.nanmin(
            np.vstack([np.diff(x, prepend=np.nan), np.diff(x, append=np.nan)]),
            axis=0,
        )

        for i in np.argsort(diffs):
            if x[i] not in prefer[: min(len(prefer), count)]:
                x = np.delete(x, i)
                break
        else:
            break

    return x


class EngFormatter(_ticker.EngFormatter):
    """a `matplotlib.ticker.EngFormatter` with the option of one unit prefix per axis.

    With ``unitInTick=False``, each tick is a mantissa at one shared order of
    magnitude, and `get_axis_unit_suffix` returns the prefixed unit for the axis
    label.
    """

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
        # a mantissa such as 999.9 may round to 1000 and is then carried to the
        # next prefix, except beyond the largest SI prefix ('Y')
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
            return f'{mant:{fmt}}{suffix}'
        else:
            return f'{mant:{fmt}}{suffix}'

    def get_axis_unit_suffix(self, vmin, vmax):
        if self.unitInTick:
            return ''

        orderOfMagnitude = math.floor(math.log(vmax - vmin, 1000)) * 3
        unit_prefix = self.ENG_PREFIXES[int(orderOfMagnitude)]
        return f'{self.sep}({unit_prefix}{self.unit})'


class GammaMaxNLocator(_ticker.MaxNLocator):
    """The ticker locator for linearized gamma-distributed survival functions"""

    _nbins: int
    axis: _axis.Axis

    # avoid removing these quantiles when selecting ticks
    PREFER_TICKS: typing.ClassVar[list[float]] = [
        0.5,
        0.9,
        0.1,
        0.99,
        1 - 1e-3,
        1 - 1e-4,
        0.95,
        1e-4,
        0.8,
        1 - 1e-5,
        0.98,
        1e-2,
        1 - 1e-6,
        1e-5,
        1e-3,
        1 - 1e-7,
        1 - 1e-8,
        1 - 1e-9,
        1e-7,
        1e-9,
        1e-8,
    ]

    def __init__(self, transform: _scale.FuncTransform, nbins=None, minor=False):
        self._transform = transform
        self._minor = minor
        super().__init__(nbins)

    def __call__(self) -> typing.Sequence[float]:
        dmin, dmax = self.axis.get_data_interval()
        vmin, vmax = self.axis.get_view_interval()
        return self.tick_values(max(vmin, dmin), min(vmax, dmax))

    def tick_values(self, vmin, vmax) -> typing.Sequence[float]:
        vmin, vmax = min((vmin, vmax)), max((vmin, vmax))
        vmin, vmax = self.limit_range_for_scale(vmin, vmax, 1e-9)

        # thresholds for scaling regimes used to select tick count and placement
        vth_lo = 0.15
        vth_hi = 0.85

        # generate candidate values for ticks
        maybe_ticks = []
        maybe_ticks.extend(_log_tick_range(vmin, vth_lo, self._nbins, subs=(1.0,)))
        maybe_ticks.extend(
            _linear_tick_range(vth_lo, vth_hi, self._nbins, steps=(1, 5, 10))
        )
        maybe_ticks.extend(
            1 - _log_tick_range(1 - vmax, 1 - vth_hi, self._nbins, subs=(1.0, 2, 3, 5))
        )
        maybe_ticks.extend([0.9, 0.95])
        maybe_ticks = np.sort(np.unique(maybe_ticks))

        # transform the scale by quantile
        tr_ticks = self._transform.transform(maybe_ticks)
        tr_prefer = self._transform.transform(
            np.array(self.PREFER_TICKS + [vmin] + [vmax])
        )
        tr_ticks = _prune_ticks(tuple(tr_ticks), self._nbins, tuple(tr_prefer))
        ticks = self._transform.inverted().transform(tr_ticks)
        return np.sort(ticks)  # type: ignore

    def get_transform(self):
        return self._transform

    def limit_range_for_scale(self, vmin, vmax, minpos):
        """Limit the domain to positive values."""
        vmin, vmax = min((vmin, vmax)), max((vmin, vmax))

        if not np.isfinite(minpos):
            minpos = 1e-12  # arbitrary floor well below any plotted probability

        ret = (
            max(minpos, vmin),
            1.0 - minpos if vmax >= 1 - minpos else vmax,
        )

        self.axis.set_view_interval(ret[1], ret[0], True)
        return ret

    def view_limits(self, dmin: float, dmax: float) -> tuple[float, float]:
        vmin, vmax = self.nonsingular(dmin, dmax)
        return vmin, vmax


class GammaLogitFormatter(_ticker.LogitFormatter):
    """A text formatter for probability labels on the GammaCCDF scale"""

    _minor: bool
    _one_half: str
    _labelled: set

    def __call__(self, x, pos=None):
        if self._minor and x not in self._labelled:
            return ''
        if x <= 0 or x >= 1:
            return ''
        if math.isclose(2 * x, round(2 * x)) and round(2 * x) == 1:
            s = self._one_half
        elif np.any(np.isclose(x, np.array([0.9, 0.99]), rtol=1e-5)):
            if x < 0.15:
                s = f'{round_places(x, 1):f}'
            else:
                s = str(x)
        elif x < 0.1 and is_decade(x, rtol=1e-5):
            exponent = round(np.log10(x))
            s = '10^{%d}' % exponent  # noqa: UP031
        elif x > 0.9 and is_decade(1 - x, rtol=1e-5):
            exponent = round(np.log10(1 - x))
            s = self._one_minus('10^{%d}' % exponent)  # type: ignore  # noqa: UP031
        elif x < 0.05:
            s = self._format_value(x, self.locs)  # type: ignore
        elif x > 0.98:
            s = self._one_minus(self._format_value(1 - x, 1 - self.locs))  # type: ignore
        else:
            s = self._format_value(x, self.locs, sci_notation=False)  # type: ignore
        return r'$\mathdefault{%s}$' % s  # noqa: UP031


class GammaQQScale(_scale.FuncScale):
    """an axis scale that linearizes the Gamma survival function of power in dB.

    `k` is the Gamma shape parameter, which is the number of averaged power samples
    in a power measurement.

    Examples:
        >>> ax.plot(10 * np.log10(bins), sf)
        >>> ax.set_yscale('gamma-qq', k=10)
    """

    name = 'gamma-qq'

    def __init__(
        self,
        axis,
        *,
        k,
        major_ticks=10,
        minor_ticks=None,
        vmin=None,
        vmax=None,
        db_ordinal=True,
    ):
        from scipy import stats

        def forward(q):
            x = stats.gamma.isf(q, a=k, scale=1)
            if db_ordinal:
                x = sw.powtodB(x)
            return x

        def inverse(x):
            if db_ordinal:
                x = sw.dBtopow(x)
            q = stats.gamma.sf(x, a=k, scale=1)
            return q

        transform = _scale.FuncTransform(forward=forward, inverse=inverse)
        self._major_locator = GammaMaxNLocator(transform=transform, nbins=major_ticks)

        if minor_ticks is not None:
            self._minor_locator = GammaMaxNLocator(
                transform=transform, nbins=minor_ticks, minor=True
            )
            self._minor_locator = None

        super().__init__(axis, (forward, inverse))

    def set_default_locators_and_formatters(self, axis):
        axis.set_major_locator(self._major_locator)

        # if self._minor_locator is not None:
        # axis.set_minor_locator(self._minor_locator)

        axis.set_major_formatter(GammaLogitFormatter(one_half='0.5'))


_scale.register_scale(GammaQQScale)


def contiguous_segments(df, index_level, threshold=7, relative=True):
    """split `df` into DataFrames at each gap in the index level `index_level`.

    A gap is a step between consecutive index values greater than `threshold`,
    scaled by the median step when `relative` is True.
    """
    delta = pd.Series(df.index.get_level_values(index_level)).diff()
    if relative:
        threshold = threshold * delta.median()
    sel = delta[delta > threshold]
    assert isinstance(sel, (pd.Series, pd.DataFrame))
    i_gaps = sel.index.values
    i_segments = [[0] + list(i_gaps), list(i_gaps) + [None]]

    return [df.iloc[i0:i1] for i0, i1 in zip(*i_segments)]


def _has_tick_label_collision(ax, which: str, spacing_threshold=10):
    """return whether adjacent tick labels along an axis are closer than `spacing_threshold`.

    Args:
        ax: the matplotlib axes that owns the tick labels
        which: ``'x'`` or ``'y'``
        spacing_threshold: the minimum gap between adjacent label boxes, in display
            pixels

    Returns:
        True when the smallest gap between adjacent tick label boxes is below
        `spacing_threshold`

    Raises:
        ValueError: if `which` is neither ``'x'`` nor ``'y'``
    """
    fig = ax.get_figure()

    if which == 'x':
        the_ax = ax.xaxis
    elif which == 'y':
        the_ax = ax.yaxis
    else:
        raise ValueError(f'"which" must be "x" or "y", but got "{which!r}"')

    boxen = [
        t.get_tightbbox(fig.canvas.get_renderer()) for t in the_ax.get_ticklabels()
    ]

    if which == 'x':
        boxen = np.array([(b.x0, b.x1) for b in boxen])
    else:
        boxen = np.array([(b.y0, b.y1) for b in boxen])

    spacing = boxen[1:, 0] - boxen[:-1, 1]

    return np.min(spacing) < spacing_threshold


def rotate_ticklabels_on_collision(ax, which: str, angles: list, spacing_threshold=3):
    # lazy_import does not work for matplotlib submodules
    from matplotlib import pyplot as plt

    def set_rotation(the_ax, angle):
        for label in the_ax.get_ticklabels():
            label.set_rotation(angle)
            if which == 'y' and angle == 90:
                label.set_verticalalignment('center')
            elif which == 'x' and angle == 90:
                label.set_horizontalalignment('right')

    if which == 'x':
        the_ax = ax.xaxis
    elif which == 'y':
        the_ax = ax.yaxis
    else:
        raise ValueError(f'"which" argument must be "x" or "y", but got "{which!r}"')

    set_rotation(the_ax, angles[0])
    if len(angles) == 1:
        return angles[0]

    a = angles[0]
    for angle in angles[1:]:
        plt.draw()

        if _has_tick_label_collision(ax, which, spacing_threshold):
            a = angle
            set_rotation(the_ax, angle)
        else:
            break
    return a


def xaxis_concise_dates(fig, ax, adjacent_offset: bool = True):
    """format the x axis of `ax` with `matplotlib.dates.ConciseDateFormatter`.

    With `adjacent_offset` True, the tick labels are drawn unrotated and the date
    offset is prepended to the first tick label. The figure is redrawn.
    """

    # lazy_import does not work for matplotlib submodules
    from matplotlib import dates, pyplot as plt, transforms

    formatter = dates.ConciseDateFormatter(dates.AutoDateLocator(), show_offset=True)

    if adjacent_offset:
        plt.xticks(rotation=0, ha='right')
    ax.xaxis.set_major_formatter(formatter)

    plt.draw()

    if adjacent_offset:
        labels = [item.get_text() for item in ax.get_xticklabels()]
        labels[0] = f'{formatter.get_offset()} {labels[0]}'
        ax.set_xticklabels(labels)

        dx = 5 / 72.0
        dy = 0 / 72.0
        offset = transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
        for label in ax.get_xticklabels():
            label.set_transform(label.get_transform() + offset)

    return ax
