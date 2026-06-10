from __future__ import annotations as __

import msgspec
import typing
import string
import functools
import striqt.sensor as ss


class SharedPlotOptions(
    ss.specs.SharedPlotOptions, frozen=True, kw_only=True, forbid_unknown_fields=True
):
    def __post_init__(self):
        from .util import get_format_fields

        if self.row is not None:
            assert self.row != self.col
            assert self.row not in get_format_fields(self.filename_fmt)
            assert self.row not in get_format_fields(self.suptitle_fmt)
            assert self.row not in get_format_fields(self.col_label_format)
            assert self.row_label_format is not None
            assert self.col not in get_format_fields(self.row_label_format)
            if self.col_wrap is not None:
                raise msgspec.ValidationError(
                    'col_wrap must be None if row is specified'
                )

        assert self.col not in get_format_fields(self.filename_fmt)
        assert self.col not in get_format_fields(self.suptitle_fmt)
        assert self.col


class DataOptions(
    ss.specs.DataOptions, frozen=True, kw_only=True, forbid_unknown_fields=True
):
    def __post_init__(self):
        from .util import literal_eval

        for k, v in list(self.select.items()):
            if isinstance(v, str):
                self.select[k] = literal_eval(v)


class PlotOptions(
    ss.specs.PlotOptions, frozen=True, kw_only=True, forbid_unknown_fields=True
):
    data: DataOptions
    plotter: SharedPlotOptions

    def __post_init__(self):
        from .data_vars import _data_plots
        import inspect

        for name, kwargs in self.variables.items():
            if name not in _data_plots:
                raise KeyError(
                    f'not such plot func {name!r}. must be one of {_data_plots!r}'
                )
            sig = inspect.signature(_data_plots[name])
            try:
                sig.bind(None, None, **kwargs)
            except Exception as ex:
                ex.args = (f'.variables[{name!r}]: function ' + ex.args[0],) + ex.args[
                    1:
                ]
                raise ex

        for n in self.data.groupby_dims:
            if n in (self.plotter.row, self.plotter.col):
                raise msgspec.ValidationError(
                    'data.groupby_dims overlaps with plotter.row or plotter.col'
                )
