"""wrap channel_analysis for _radio_ captures, together with utility functions for RadioCapture"""

from __future__ import annotations
import dataclasses
import functools
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
def coord_template(capture_cls: type[structs.RadioCapture], aliases: tuple[str, ...]):
    """returns a cached xr.Coordinates object to use as a template for data results"""

    capture = capture_cls()
    vars = {}

    for field in capture_cls.__struct_fields__:
        value = getattr(capture, field)

        vars[field] = xr.Variable(
            (CAPTURE_DIM,),
            [value],
            fastpath=True,
            attrs=structs.get_attrs(capture_cls, field),
        )

    for field in aliases:
        vars[field] = xr.Variable(
            (CAPTURE_DIM,),
            [''],
            fastpath=True,
        ).astype('object')

    vars[SWEEP_TIMESTAMP_NAME] = xr.Variable(
        (CAPTURE_DIM,),
        [pd.Timestamp('now')],
        fastpath=True,
        attrs={'standard_name': 'Sweep start time'},
    )
    vars[RADIO_ID_NAME] = xr.Variable(
        (CAPTURE_DIM,),
        ['unspecified-radio'],
        fastpath=True,
        attrs={'standard_name': 'Radio hardware ID'},
    ).astype('object')

    return xr.Coordinates(vars)


def _get_capture_field(
    name,
    capture: structs.Capture,
    radio_id: str,
    aliases: dict[str, typing.Any],
    sweep_time=None,
):
    if hasattr(capture, name):
        value = getattr(capture, name)
    elif name in aliases:
        value = aliases[name]
    elif name == 'radio_id':
        value = radio_id
    elif name == SWEEP_TIMESTAMP_NAME:
        value = sweep_time
    else:
        raise KeyError
    return value


def build_coords(
    capture: structs.RadioCapture, aliases: dict, radio_id: str, sweep_time
):
    coords = coord_template(type(capture), tuple(aliases.keys())).copy(deep=True)

    for field in coords.keys():
        value = _get_capture_field(field, capture, radio_id, aliases, sweep_time)

        if isinstance(value, str):
            # to coerce strings as variable-length types later for storage
            coords[field] = coords[field].astype('object')

        coords[field].values[:] = value

    coords[RADIO_ID_NAME].values[:] = radio_id

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


def _alias_is_in_capture(
    capture: structs.Capture, radio_id: str, aliases: dict[str, typing.Any], alias_spec
) -> bool:
    """return whether or not the given mapping matches coordinate values in dataset"""

    for match_name, match_value in alias_spec.items():
        capture_value = _get_capture_field(match_name, capture, radio_id, aliases)
        if capture_value != match_value:
            # no match
            return False
    else:
        return True


def _evaluate_aliases(capture: structs.Capture, radio_id: str, alias_spec: dict):
    """evaluate the field values"""
    ret = {}
    for coord_name, coord_spec in alias_spec.items():
        for alias_value, alias_spec in coord_spec.items():
            if _alias_is_in_capture(capture, radio_id, ret, alias_spec):
                print('get alias: ', coord_name, alias_value)
                ret[coord_name] = alias_value
                break
    print('got: ', ret)
    return ret


def capture_fields_with_aliases(
    capture: structs.Capture, radio_id: str, alias_spec: dict
) -> dict:
    attrs = structs.struct_to_builtins(capture)
    aliases = _evaluate_aliases(capture, radio_id, alias_spec)

    return dict(attrs, **aliases)


@dataclasses.dataclass
class ChannelAnalysisWrapper:
    """Inject radio device and capture metadata and coordinates into a channel analysis result"""

    __name__ = 'analyze'

    radio: radio.RadioDevice
    sweep: structs.Sweep
    analysis_spec: list[channel_analysis.ChannelAnalysis]
    extra_attrs: dict[str, typing.Any] | None = None
    calibration: typing.Optional['xr.Dataset'] = None

    def __call__(
        self,
        iq: 'channel_analysis.ArrayType',
        sweep_time,
        capture: structs.RadioCapture,
        pickled=False,
    ) -> 'xr.Dataset':
        """Inject radio device and capture info into a channel analysis result."""

        with lb.stopwatch('analysis', logger_level='debug'):
            iq = iq_corrections.resampling_correction(
                iq, capture, self.radio, force_calibration=self.calibration
            )

            coords = build_coords(
                capture,
                aliases=self.sweep.output.coord_aliases,
                radio_id=self.radio.id,
                sweep_time=sweep_time,
            )

            analysis = channel_analysis.analyze_by_spec(
                iq, capture, spec=self.analysis_spec
            )

            analysis = analysis.expand_dims((CAPTURE_DIM,)).assign_coords(coords)
            analysis = _assign_alias_coords(analysis, self.sweep.output.coord_aliases)

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
