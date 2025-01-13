"""wrap channel_analysis for _radio_ captures, together with utility functions for RadioCapture"""

from __future__ import annotations
import dataclasses
import functools
import msgspec
import numbers
import pickle
import typing

from . import iq_corrections, structs, util
from . import radio

if typing.TYPE_CHECKING:
    import labbench as lb
    import matplotlib
    import numpy as np
    import pandas as pd
    import xarray as xr
    import channel_analysis
else:
    lb = util.lazy_import('labbench')
    matplotlib = util.lazy_import('matplotlib')
    np = util.lazy_import('numpy')
    pd = util.lazy_import('pandas')
    xr = util.lazy_import('xarray')
    channel_analysis = util.lazy_import('channel_analysis')


CAPTURE_DIM = 'capture'
SWEEP_TIMESTAMP_NAME = 'sweep_start_time'
RADIO_ID_NAME = 'radio_id'


@functools.lru_cache
def _get_unit_formatter(units: str) -> 'matplotlib.ticker.EngFormatter':
    return matplotlib.ticker.EngFormatter(unit=units)


def _describe_field(capture: 'channel_analysis.Capture', name: str):
    meta = channel_analysis.structs.get_capture_type_attrs(type(capture))
    attrs = meta[name]
    value = getattr(capture, name)

    if value is None:
        value_str = 'None'
    elif attrs.get('units', None) is not None and np.isfinite(value):
        if isinstance(value, tuple):
            value_tup = [_get_unit_formatter(attrs['units'])(v) for v in value]
            value_str = f"({', '.join(value_tup)})"
        else:
            value_str = _get_unit_formatter(attrs['units'])(value)
    else:
        value_str = repr(value)

    return f'{name}={value_str}'


def concat_time_dim(datasets: list['xr.Dataset'], time_dim: str) -> 'xr.Dataset':
    """concatenate captured datasets into one along a time axis.

    This can be used to e.g. transform a contiguous sequence
    of spectrogram captures into a single spectrogram.

    Preconditions:
    - all datasets share the same dimension and type.
    - time coordinates based on time_dim are uniformly spaced

    """
    pad_dims = {time_dim: (0, len(datasets[0][time_dim]) * (len(datasets) - 1))}
    ds = datasets[0].pad(pad_dims, constant_values=float('nan'))

    for data_name, var in ds.data_vars.items():
        if time_dim not in var.dims:
            continue
        else:
            axis = var.dims[1:].index(time_dim)

        values = np.concatenate(
            [sub[data_name].isel(capture=0).values for sub in datasets], axis=axis
        )
        var.values[:] = values

    for coord_name, coord in ds.coords.items():
        if time_dim not in coord.dims:
            continue
        time_step = float(coord[1] - coord[0])
        ds[coord_name] = pd.RangeIndex(ds.sizes[coord_name]) * time_step

    return ds


def describe_capture(
    this: structs.RadioCapture | None,
    prev: structs.RadioCapture | None = None,
    *,
    index: int,
    count: int,
) -> str:
    """generate a description of a capture"""
    if this is None:
        if prev is None:
            return 'saving last analysis'
        else:
            return 'performing last analysis'

    diffs = []

    for name in type(this).__struct_fields__:
        if name == 'start_time':
            continue
        value = getattr(this, name)
        if prev is None or value != getattr(prev, name):
            diffs.append(_describe_field(this, name))

    capture_diff = ', '.join(diffs)

    if index is not None:
        progress = str(index + 1)

        if count is not None:
            progress = f'{progress}/{count}'

        progress = progress + ' '
    else:
        progress = ''

    return progress + capture_diff


@functools.lru_cache
def coord_template(
    capture_cls: type[structs.RadioCapture],
    channels: tuple[int, ...],
    **alias_dtypes: dict[str, type],
):
    """returns a cached xr.Coordinates object to use as a template for data results"""

    capture = capture_cls()
    vars = {}

    for field in capture_cls.__struct_fields__:
        if field == 'channel':
            value = list(channels)
        else:
            entry = getattr(capture, field)
            (value,) = broadcast_to_channels(channels, entry, allow_mismatch=True)

        vars[field] = xr.Variable(
            (CAPTURE_DIM,),
            value,
            fastpath=True,
            attrs=structs.get_attrs(capture_cls, field),
        )

    for field, dtype in alias_dtypes.items():
        vars[field] = xr.Variable(
            (CAPTURE_DIM,),
            broadcast_to_channels(channels, ('',))[0],
            fastpath=True,
        ).astype(dtype)

    vars[SWEEP_TIMESTAMP_NAME] = xr.Variable(
        (CAPTURE_DIM,),
        broadcast_to_channels(channels, (pd.Timestamp('now'),))[0],
        fastpath=True,
        attrs={'standard_name': 'Sweep start time'},
    )
    vars[RADIO_ID_NAME] = xr.Variable(
        (CAPTURE_DIM,),
        broadcast_to_channels(channels, ('unspecified-radio',))[0],
        fastpath=True,
        attrs={'standard_name': 'Radio hardware ID'},
    ).astype('object')

    return xr.Coordinates(vars)


def _get_capture_field(
    name,
    capture: structs.Capture,
    radio_id: str,
    alias_hits: dict,
    sweep_time=None,
):
    if isinstance(capture.channel, tuple):
        raise ValueError('split the capture before the call to _get_capture_field')

    # aliases = output.coord_aliases
    # if len(aliases) > 0:
    #     alias_hits = _evaluate_aliases(capture, radio_id, output)

    if hasattr(capture, name):
        value = getattr(capture, name)
        if isinstance(value, tuple):
            value = value[0]
    elif name in alias_hits:
        # default_type = type(next(iter(aliases[name].values())))
        value = alias_hits[name]
    elif name == 'radio_id':
        value = radio_id
    elif name == SWEEP_TIMESTAMP_NAME:
        value = sweep_time
    else:
        raise KeyError
    return value


@functools.lru_cache
def _get_alias_dtypes(output: structs.Output):
    aliases = output.coord_aliases

    alias_dtypes = {}
    for field, entries in aliases.items():
        alias_dtypes[field] = np.array(list(entries.keys())).dtype
    return alias_dtypes


@functools.lru_cache(10000)
def broadcast_to_channels(
    channels: int | tuple[int, ...], *params, allow_mismatch=False
) -> list[list]:
    """broadcast sequences in each element in `params` up to the
    length of capture.channel.
    """

    res = []
    if isinstance(channels, numbers.Number):
        count = 1
    else:
        count = len(channels)

    for p in params:
        if not isinstance(p, (tuple, list)):
            res.append((p,) * count)
        elif len(p) == count:
            res.append(tuple(p))
        elif allow_mismatch:
            res.append(tuple(p[:1]) * count)
        else:
            raise ValueError(
                f'cannot broadcast tuple of length {len(p)} to {count} channels'
            )

    return res


def build_coords(
    capture: structs.RadioCapture, output: structs.Output, radio_id: str, sweep_time
):
    alias_dtypes = _get_alias_dtypes(output)

    if isinstance(capture.channel, numbers.Number):
        channels = (capture.channel,)
    else:
        channels = tuple(capture.channel)

    coords = coord_template(type(capture), channels, **alias_dtypes).copy(deep=True)

    updates = {}

    for c in split_capture_channels(capture):
        alias_hits = _evaluate_aliases(c, radio_id, output)

        for field in coords.keys():
            if field == RADIO_ID_NAME:
                updates.setdefault(field, []).append(radio_id)
                continue

            try:
                value = _get_capture_field(field, c, radio_id, alias_hits, sweep_time)
            except KeyError:
                if field in output.coord_aliases:
                    lb.logger.warning(f'warning: no alias name matches in "{field}"')
                continue

            updates.setdefault(field, []).append(value)

    for field, values in updates.items():
        coords[field].values[:] = np.array(values)

    return coords


def _alias_is_in_coord(dataset, alias_spec) -> bool:
    """return whether the given mapping matches coordinate values in dataset"""
    for match_name, match_value in alias_spec.items():
        if match_name in dataset.coords:
            match_coord = dataset.coords[match_name]
        else:
            raise KeyError

        if match_coord.values[0] != match_value:
            # no match
            return False
    else:
        return False


def _assign_alias_coords(capture_data: 'xr.Dataset', aliases):
    for coord_name, coord_spec in aliases.items():
        for alias_value, alias_spec in coord_spec.items():
            if _alias_is_in_coord(capture_data, alias_spec):
                capture_data = capture_data.assign_coords(
                    {coord_name: (CAPTURE_DIM, [alias_value])}
                )
                break

    return capture_data


def _match_capture_fields(
    capture: structs.RadioCapture, fields: dict[str], radio_id: str
):
    if isinstance(capture.channel, tuple):
        raise ValueError('split the capture to evaluate alias matches')

    for name, value in fields.items():
        if name == 'radio_id' and value == radio_id:
            continue

        if not hasattr(capture, name):
            return False

        capture_value = getattr(capture, name)

        if isinstance(capture_value, tuple):
            capture_value = capture_value[0]

        if capture_value != value:
            return False

    return True


@functools.lru_cache()
def _evaluate_aliases(
    capture: structs.RadioCapture, radio_id: str, output: structs.Output
):
    """evaluate the field values"""

    ret = {}

    for coord_name, coord_spec in output.coord_aliases.items():
        for alias_value, field_spec in coord_spec.items():
            if _match_capture_fields(capture, field_spec, radio_id):
                ret[coord_name] = alias_value
                break
    return ret


@functools.lru_cache()
def split_capture_channels(capture: structs.RadioCapture) -> list[structs.RadioCapture]:
    """split a multi-channel capture into a list of single-channel captures.

    If capture is not a multi-channel capture (its channel field is just a number),
    then the returned list will be [capture].
    """

    if isinstance(capture.channel, numbers.Number):
        return [capture]

    remaps = [dict() for i in range(len(capture.channel))]

    for field in capture.__struct_fields__:
        values = getattr(capture, field)
        if not isinstance(values, tuple):
            continue

        for remap, value in zip(remaps, values):
            remap[field] = value

    return [msgspec.structs.replace(capture, **remap) for remap in remaps]


def capture_fields_with_aliases(
    capture: structs.RadioCapture, radio_id: str, output: structs.Output
) -> dict:
    attrs = structs.struct_to_builtins(capture)
    c = split_capture_channels(capture)[0]
    aliases = _evaluate_aliases(c, radio_id, output)

    return dict(attrs, **aliases)


@dataclasses.dataclass
class ChannelAnalysisWrapper:
    """Inject radio device and capture metadata and coordinates into a channel analysis result"""

    __name__ = 'analyze'

    radio: radio.RadioDevice
    sweep: structs.Sweep
    analysis_spec: list[channel_analysis.ChannelAnalysis]
    extra_attrs: dict[str, typing.Any] | None = None

    def __call__(
        self,
        iq: 'channel_analysis.ArrayType',
        sweep_time,
        capture: structs.RadioCapture,
        pickled=False,
    ) -> 'xr.Dataset':
        """Inject radio device and capture info into a channel analysis result."""

        with lb.stopwatch('analysis', logger_level='debug'):
            with lb.stopwatch('analysis: resample/calibrate', logger_level='debug'):
                iq = iq_corrections.resampling_correction(iq, capture, self.radio)

            with lb.stopwatch('build coords', threshold=10e-3, logger_level='debug'):
                coords = build_coords(
                    capture,
                    output=self.sweep.output,
                    radio_id=self.radio.id,
                    sweep_time=sweep_time,
                )

            analysis = channel_analysis.analyze_by_spec(
                iq, capture, spec=self.analysis_spec, expand_dims=(CAPTURE_DIM,)
            )

            analysis = analysis.assign_coords(coords)

            # these are coordinates - drop from attrs
            for name in coords.keys():
                analysis.attrs.pop(name, None)

        if self.extra_attrs is not None:
            analysis.attrs.update(self.extra_attrs)

        analysis[SWEEP_TIMESTAMP_NAME].attrs.update(label='Sweep start time')

        if pickled:
            return pickle.dumps(analysis)
        else:
            return analysis
