"""plotting backends that draw the figures of each data variable."""

from .matplotlib import (
    MatplotlibBackend,
    TerminalGraphics,
    batch_term_images,
    select_mpl_backend,
    term_graphics_notice,
)
from .plotly import PlotlyBackend
from .protocol import PlotBackend, StepStyle, coerce_column
