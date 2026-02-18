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


class PlotOptions(msgspec.Struct, kw_only=True, forbid_unknown_fields=True):
    data: DataOptions
    plotter: SharedPlotOptions
    variables: dict[str, dict[str, typing.Any]] = msgspec.field(default_factory=dict)

    def __post_init__(self):
        from .plots import data_var_plotters

        for name, kwargs in list(self.variables.items()):
            if name not in data_var_plotters:
                raise KeyError(
                    f'not such plot func {name!r}. must be one of {data_var_plotters!r}'
                )
            for k, v in list(kwargs.items()):
                if isinstance(v, str) and v.startswith('slice'):
                    self.variables[name][k] = eval(v, {}, {'slice': slice})

        for n in self.data.groupby_dims:
            if n in (self.plotter.row, self.plotter.col):
                raise msgspec.ValidationError(
                    'data.groupby_dims overlaps with plotter.row or plotter.col'
                )
