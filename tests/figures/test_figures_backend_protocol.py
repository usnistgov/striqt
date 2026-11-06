"""striqt.figures.backend.protocol: the facet dimension that unfaceted options add"""

from __future__ import annotations

import types

import pytest
from figures_datasets import make_dataset

import striqt.sensor as ss
from striqt.figures.backend import coerce_column

# %% coerce_column


@pytest.mark.parametrize(
    ('col', 'row', 'adds_view'),
    [(None, None, True), ('capture', None, False), (None, 'capture', False)],
    ids=['unfaceted', 'col', 'row'],
)
def test_coerce_column_adds_a_view_dim_only_without_col_and_row(
    col: str | None, row: str | None, adds_view: bool
) -> None:
    """a size-1 ``_view`` dimension is prepended when neither `col` nor `row` is set,
    and the values are unchanged"""
    # striqt.figures.specs.SharedPlotOptions rejects col=None in __post_init__; the
    # sensor base struct has no validation
    plotter = types.SimpleNamespace(opts=ss.specs.SharedPlotOptions(col=col, row=row))
    data = make_dataset()['channel_power_time_series']

    out = coerce_column(data, plotter)

    if adds_view:
        assert out.dims == ('_view', *data.dims)
        assert out.sizes['_view'] == 1
        assert out.isel(_view=0, drop=True).equals(data)
    else:
        assert out.dims == data.dims
        assert out.equals(data)
