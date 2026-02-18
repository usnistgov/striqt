import msgspec
import typing


class SharedPlotOptions(msgspec.Struct, kw_only=True, forbid_unknown_fields=True):
    style: str | None = None
    col: str | None = 'port'
    row: str | None = None
    col_label_format: str = 'Port {port}'
    row_label_format: str | None = None
    col_wrap: int = 2
    title_fmt: str = 'Port {port}'
    suptitle_fmt: str = '{center_frequency}'
    filename_fmt: str = '{name} {center_frequency}.svg'


class DataOptions(msgspec.Struct, kw_only=True, forbid_unknown_fields=True):
    index_dims: tuple[str, ...] = 'sweep_start_time', 'start_time', 'port'
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
