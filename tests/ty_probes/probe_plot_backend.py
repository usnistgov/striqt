"""each plotting backend satisfies the `PlotBackend` protocol with its grid type"""

from __future__ import annotations

from typing import Any, cast

import plotly.graph_objects as go
from xarray.plot import FacetGrid

from striqt.figures.backend import MatplotlibBackend, PlotBackend, PlotlyBackend

opts = cast(Any, None)

mpl: PlotBackend[FacetGrid] = MatplotlibBackend(opts, None)
web: PlotBackend[go.Figure] = PlotlyBackend(opts)
