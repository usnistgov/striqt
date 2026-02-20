from __future__ import annotations as __

import msgspec
import typing
import string
import functools


@functools.lru_cache()
def get_format_fields(fmt, exclude=()):
    all_exclude = exclude + (None,)
    return [l[1] for l in string.Formatter().parse(fmt) if l[1] not in all_exclude]


class SharedPlotOptions(msgspec.Struct, kw_only=True, forbid_unknown_fields=True):
    style: str | None = None
    col: str | None = 'port'
    row: str | None = None
    col_label_format: str = 'Port {port}'
    row_label_format: str | None = None
    col_wrap: int | None = None
    suptitle_fmt: str = '{start_time}'
    filename_fmt: str = '{name} {start_time}.svg'

    def __post_init__(self):
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


class DataOptions(msgspec.Struct, kw_only=True, forbid_unknown_fields=True):
    groupby_dims: tuple[str, ...] = ()
    sweep_index: int
    query: str | None = None
    select: dict[str, typing.Any] = msgspec.field(default_factory=dict)

    def __post_init__(self):
        for k, v in list(self.select.items()):
            if isinstance(v, str):
                self.select[k] = eval(v, {}, {'slice': slice})


class PlotOptions(msgspec.Struct, kw_only=True, forbid_unknown_fields=True):
    data: DataOptions
    plotter: SharedPlotOptions
    variables: dict[str, dict[str, typing.Any]] = msgspec.field(default_factory=dict)

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
