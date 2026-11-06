"""plotting backends that draw the figures of each data variable."""

from .matplotlib import (
    PlotBackend,
    TerminalGraphics,
    batch_term_images,
    coerce_column,
    plot_cyclic_channel_power,
    select_mpl_backend,
    term_graphics_notice,
)
from .plotly import WebPlotBackend

__all__ = [
    'PlotBackend',
    'TerminalGraphics',
    'WebPlotBackend',
    'batch_term_images',
    'coerce_column',
    'plot_cyclic_channel_power',
    'select_mpl_backend',
    'term_graphics_notice',
]
